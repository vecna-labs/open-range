"""Auto-evolve: families enumerate mutations, core picks one based on signal."""

from __future__ import annotations

from collections.abc import Callable, Sequence
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

if TYPE_CHECKING:
    from openrange_pack_sdk import Snapshot


Direction = Literal["harden", "soften", "diversify"]

CurriculumPolicy = Callable[[Sequence[EpisodeReportLike]], "Direction | None"]


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


def auto_evolve(
    snapshot: Snapshot,
    *reports: EpisodeReportLike,
    pack: Pack,
    policy: CurriculumPolicy = direction_from_reports,
    llm: LLMBackend | None = None,
    max_repairs: int = 2,
) -> Snapshot | None:
    """Pick a Mutation by `policy`, apply it, re-admit. Returns the next
    Snapshot, or `None` if no candidate survives admission."""
    if not reports:
        return None

    options = _enumerate_options(pack, snapshot, reports, llm=llm)
    if not options:
        return None

    direction = policy(reports)
    if direction is None:
        return None

    candidates = sorted(
        (o for o in options if o.direction == direction and o.relevance > 0.0),
        key=lambda o: o.relevance,
        reverse=True,
    )
    if not candidates:
        return None

    for chosen in candidates:
        try:
            evolved = _evolve_snapshot(
                snapshot,
                pack,
                chosen,
                max_repairs=max_repairs,
            )
        except Exception:  # noqa: BLE001 — pack-supplied code is untrusted
            continue
        if evolved is None:
            continue
        return evolved
    return None


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

    # Wrap the pack so admission sees the pre-evolved graph + tasks
    # while ontology / invariants / families flow through unchanged.
    wrapped = _PreBuiltPack(pack, evolved_graph, list(snapshot.tasks))
    manifest_in: object = snapshot.lineage.get("manifest", {})
    base_manifest: dict[str, object] = (
        dict(manifest_in) if isinstance(manifest_in, dict) else {}
    )
    evolved_manifest = {
        **base_manifest,
        "_evolve": {
            "parent_snapshot_id": snapshot.snapshot_id,
            "direction": mutation.direction,
            "relevance": mutation.relevance,
            "family": mutation.family,
            "note": mutation.note,
        },
    }
    result = admit(wrapped, manifest=evolved_manifest, max_repairs=max_repairs)
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
    return _Snapshot(
        snapshot_id=result.snapshot_id,
        ontology_id=result.ontology_id,
        graph=result.graph,
        tasks=result.tasks,
        lineage=result.lineage,
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
