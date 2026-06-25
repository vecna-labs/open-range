"""Auto-evolve: families enumerate mutations, core picks one based on signal."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from graphschema import (
    Edge,
    Issue,
    Node,
    Ontology,
    WorldGraph,
    apply_patch,
)
from openrange_pack_sdk import (
    Backing,
    Builder,
    BuildResult,
    EpisodeReportLike,
    LLMBackend,
    Manifest,
    Mutation,
    Pack,
    PackPrior,
    RuntimeHandle,
    TaskFamily,
    TaskSpec,
)

from openrange.core.episode import EpisodeService

if TYPE_CHECKING:
    from openrange_pack_sdk import Snapshot


Direction = Literal["harden", "soften", "diversify"]

CurriculumPolicy = Callable[[Sequence[EpisodeReportLike]], "Direction | None"]

_DIFFICULTY_STEP: dict[str, float] = {"harden": 0.2, "soften": -0.2}


def direction_from_reports(
    reports: Sequence[EpisodeReportLike],
    *,
    harden_threshold: float = 0.66,
    soften_threshold: float = 0.33,
) -> Direction | None:
    """Pass-rate decides direction. Returns `None` when there are no reports."""
    if not reports:
        return None
    passed = sum(1 for r in reports if _report_passed(r))
    pass_rate = passed / len(reports)
    if pass_rate >= harden_threshold:
        return "harden"
    if pass_rate <= soften_threshold:
        return "soften"
    return "diversify"


def _report_passed(report: EpisodeReportLike) -> bool:
    try:
        return bool(report.passed)
    except AttributeError:
        return False


EvolutionGate = Callable[["Snapshot", Mutation], bool]

SeedGate = Callable[["Snapshot"], bool]


def _verify_realized(
    pack: Pack,
    root: Path,
    snapshot: Snapshot,
    accept: Callable[[Snapshot, str], bool],
) -> bool:
    task = next((t for t in snapshot.tasks if t.entrypoints), None)
    if task is None:
        return True
    svc = EpisodeService(pack, root)
    try:
        handle = svc.start_episode(snapshot, task.id)
        return accept(snapshot, str(svc.surface(handle)["base_url"]))
    finally:
        svc.close()


def consequence_gate(
    pack: Pack,
    workdir: str | Path,
    accept: Callable[[Snapshot, str], bool],
) -> EvolutionGate:
    """An :data:`EvolutionGate` that realizes each evolved world and keeps it only when
    ``accept(snapshot, base_url)`` confirms it — a pack-supplied check run against the
    realized world. ``accept`` is the pack's verdict, so core needs no pack import. A
    world with no realizable task can't be checked and passes through."""
    root = Path(workdir)

    def gate(evolved: Snapshot, mutation: Mutation) -> bool:
        del mutation
        return _verify_realized(pack, root, evolved, accept)

    return gate


def consequence_seed_gate(
    pack: Pack,
    workdir: str | Path,
    accept: Callable[[Snapshot, str], bool],
) -> SeedGate:
    """A :data:`SeedGate` that applies the same ``accept`` verdict as
    :func:`consequence_gate`, but at pool construction — so an initial world seeds the
    pool only if its reference breach actually leaks against the realized world."""
    root = Path(workdir)
    return lambda snapshot: _verify_realized(pack, root, snapshot, accept)


def auto_evolve(
    snapshot: Snapshot,
    *reports: EpisodeReportLike,
    pack: Pack,
    policy: CurriculumPolicy = direction_from_reports,
    llm: LLMBackend | None = None,
    max_repairs: int = 2,
    gate: EvolutionGate | None = None,
) -> Snapshot | None:
    """Pick an evolution for ``direction`` and re-admit it.

    Tries the families' patch mutations first; if none admit to a distinct
    world, falls back to a builder *grow* (re-running the builder with a
    difficulty-stepped prior). Returns the next Snapshot, or ``None`` if nothing
    admits.

    ``gate``, when given, vets each admitted candidate against its claimed
    ``direction`` (see :data:`EvolutionGate`); a rejected candidate is skipped
    so a mislabelled mutation never lands.
    """
    if not reports:
        return None

    direction = policy(reports)
    if direction is None:
        return None

    for chosen in _patch_candidates(pack, snapshot, reports, direction, llm=llm):
        try:
            evolved = _evolve_snapshot(snapshot, pack, chosen, max_repairs=max_repairs)
        except Exception:  # noqa: BLE001
            continue
        if evolved is None or evolved.snapshot_id == snapshot.snapshot_id:
            continue
        if gate is not None:
            try:
                if not gate(evolved, chosen):
                    continue
            except Exception:  # noqa: BLE001
                continue
        return evolved

    # Intentionally outside the gate: a pack using grow for a monotone frontier
    # must keep ``default_prior`` None so this fallback never fires there.
    try:
        return _grow_snapshot(snapshot, pack, direction, max_repairs=max_repairs)
    except Exception:  # noqa: BLE001 — pack-supplied code is untrusted
        return None


def _patch_candidates(
    pack: Pack,
    snapshot: Snapshot,
    reports: Sequence[EpisodeReportLike],
    direction: Direction,
    *,
    llm: LLMBackend | None,
) -> list[Mutation]:
    options = _enumerate_options(pack, snapshot, reports, llm=llm)
    return sorted(
        (o for o in options if o.direction == direction and o.relevance > 0.0),
        key=lambda o: o.relevance,
        reverse=True,
    )


def _enumerate_options(
    pack: Pack,
    snapshot: Snapshot,
    reports: Sequence[EpisodeReportLike],
    *,
    llm: LLMBackend | None,
) -> list[Mutation]:
    options: list[Mutation] = []
    for family in pack.task_families():
        options.extend(family.available_mutations(snapshot, reports, llm=llm))
    return options


def _grow_snapshot(
    snapshot: Snapshot,
    pack: Pack,
    direction: Direction,
    *,
    max_repairs: int,
) -> Snapshot | None:
    from openrange.core.admit import AdmissionFailure, admit

    baseline = pack.default_prior()
    if baseline is None:
        return None
    step = _DIFFICULTY_STEP.get(direction)
    if step is None:
        return None
    carried = snapshot.lineage.get("curriculum_difficulty")
    difficulty = (
        dict(carried) if isinstance(carried, dict) else dict(baseline.difficulty)
    )
    stepped = {k: min(1.0, max(0.0, float(v) + step)) for k, v in difficulty.items()}
    if not stepped or stepped == difficulty:
        return None

    grown_prior = PackPrior(
        source="curriculum.grow",
        ontology=baseline.ontology,
        topology=baseline.topology,
        task_seeds=baseline.task_seeds,
        difficulty=stepped,
        coverage=baseline.coverage,
    )
    manifest_in = snapshot.lineage.get("manifest", {})
    manifest = dict(manifest_in) if isinstance(manifest_in, dict) else {}
    result = admit(pack, manifest, prior=grown_prior, max_repairs=max_repairs)
    if isinstance(result, AdmissionFailure):
        return None
    grown = _with_grow_lineage(result, snapshot, direction, stepped)
    if grown.snapshot_id == snapshot.snapshot_id:
        return None
    return grown


def _evolve_block(
    *,
    parent_snapshot_id: str,
    direction: str,
    kind: str,
    relevance: float | None = None,
    family: str | None = None,
    note: str = "",
) -> dict[str, object]:
    return {
        "parent_snapshot_id": parent_snapshot_id,
        "direction": direction,
        "kind": kind,
        "relevance": relevance,
        "family": family,
        "note": note,
    }


def _with_grow_lineage(
    result: Snapshot,
    parent: Snapshot,
    direction: Direction,
    difficulty: dict[str, float],
) -> Snapshot:
    from openrange_pack_sdk import BuildEvent
    from openrange_pack_sdk import Snapshot as _Snapshot

    event = BuildEvent(
        seq=len(result.history),
        phase="evolve",
        detail=f"grew from {parent.snapshot_id} via {direction} (regime)",
        refs=(parent.snapshot_id,),
    )
    return _Snapshot(
        snapshot_id=result.snapshot_id,
        ontology_id=result.ontology_id,
        graph=result.graph,
        tasks=result.tasks,
        lineage={
            **dict(result.lineage),
            "curriculum_difficulty": difficulty,
            "_evolve": _evolve_block(
                parent_snapshot_id=parent.snapshot_id,
                direction=direction,
                kind="grow",
            ),
        },
        history=(*result.history, event),
    )


def _evolve_snapshot(
    snapshot: Snapshot,
    pack: Pack,
    mutation: Mutation,
    *,
    max_repairs: int,
) -> Snapshot | None:
    from openrange_pack_sdk import Snapshot as _Snapshot

    from openrange.core.admit import AdmissionFailure, admit

    builder = pack.make_builder(None)
    patch = builder.evolve(snapshot, mutation)

    evolved_graph = _clone_graph(snapshot.graph)
    apply_patch(evolved_graph, patch)

    manifest_in: object = snapshot.lineage.get("manifest", {})
    base_manifest: dict[str, object] = (
        dict(manifest_in) if isinstance(manifest_in, dict) else {}
    )

    regenerated: list[TaskSpec] = []
    for family in pack.task_families():
        regenerated.extend(family.generate(evolved_graph, base_manifest, None))

    wrapped = _PreBuiltPack(pack, evolved_graph, regenerated)
    result = admit(wrapped, manifest=dict(base_manifest), max_repairs=max_repairs)
    if isinstance(result, AdmissionFailure):
        return None
    assert isinstance(result, _Snapshot)
    from openrange_pack_sdk import BuildEvent

    evolve_event = BuildEvent(
        seq=len(result.history),
        phase="evolve",
        detail=(
            f"evolved from {snapshot.snapshot_id} via "
            f"{mutation.family}/{mutation.direction} "
            f"(relevance={mutation.relevance:.2f})"
        ),
        refs=(snapshot.snapshot_id,),
    )
    carried = snapshot.lineage.get("curriculum_difficulty")
    lineage = dict(result.lineage)
    if isinstance(carried, dict):
        lineage.setdefault("curriculum_difficulty", carried)
    lineage["_evolve"] = _evolve_block(
        parent_snapshot_id=snapshot.snapshot_id,
        direction=mutation.direction,
        kind="patch",
        relevance=mutation.relevance,
        family=mutation.family,
        note=mutation.note,
    )
    return _Snapshot(
        snapshot_id=result.snapshot_id,
        ontology_id=result.ontology_id,
        graph=result.graph,
        tasks=result.tasks,
        lineage=lineage,
        history=(*result.history, evolve_event),
    )


def _clone_graph(graph: WorldGraph) -> WorldGraph:
    cloned = WorldGraph(ontology=graph.ontology, meta=dict(graph.meta))
    for nid, n in graph.nodes.items():
        cloned.nodes[nid] = Node(
            id=n.id,
            kind=n.kind,
            attrs=dict(n.attrs),
            roles=set(n.roles),
            visibility=n.visibility,
            runtime=dict(n.runtime),
            meta=dict(n.meta),
        )
    for eid, e in graph.edges.items():
        cloned.edges[eid] = Edge(
            id=e.id,
            kind=e.kind,
            src=e.src,
            dst=e.dst,
            attrs=dict(e.attrs),
        )
    return cloned


class _OneShotBuilder(Builder):
    def __init__(
        self,
        graph: WorldGraph,
        tasks: list[TaskSpec],
    ) -> None:
        self._graph = graph
        self._tasks = tasks

    def build(self, manifest: Manifest) -> BuildResult:
        del manifest
        return BuildResult(
            graph=self._graph,
            tasks=list(self._tasks),
            admission_meta={"builder": "core.curriculum._PreBuiltPack"},
        )


class _PreBuiltPack(Pack):
    def __init__(
        self,
        inner: Pack,
        graph: WorldGraph,
        tasks: list[TaskSpec],
    ) -> None:
        self._inner = inner
        self._graph = graph
        self._tasks = tasks
        self.id = inner.id
        self.version = inner.version

    def ontology(self) -> Ontology:
        return self._inner.ontology()

    def invariants(self) -> list[Callable[[WorldGraph], list[Issue]]]:
        return self._inner.invariants()

    def make_builder(self, prior: PackPrior | None) -> Builder:
        del prior
        return _OneShotBuilder(self._graph, self._tasks)

    def realize(self, graph: WorldGraph, backing: Backing) -> RuntimeHandle:
        return self._inner.realize(graph, backing)

    def task_families(self) -> list[TaskFamily]:
        return self._inner.task_families()
