"""Auto-evolve: families enumerate mutations, core picks one based on signal.

Each ``TaskFamily`` lists the evolution moves available from a given
snapshot, tagging each ``Mutation`` with a direction (harden / soften /
diversify) and a relevance score (0..1) reflecting how well the move
responds to recent agent behavior. Core aggregates proposals across
every family the pack offers, applies a policy to derive a direction
from the report set, picks the highest-relevance candidate in that
direction, and forwards to the pack's ``Builder.evolve`` to obtain a
``GraphPatch``. The patch is applied to a copy of the snapshot's graph
and the result is re-admitted to produce the next snapshot.

Boundary:
    PACK owns per-family enumeration / tagging and the ``evolve``
    refinement step.
    CORE owns the aggregation loop, the direction policy, the relevance
    tie-break, the patch application, and the re-admission of the
    evolved graph.

Shape change vs the pre-refactor module:
    - ``Mutation`` now lives in ``openrange.core.contracts`` and carries
      a ``GraphPatch`` + ``family`` tag (not an opaque ``directive``
      mapping).
    - The pack-level ``Pack.available_mutations`` is gone. Aggregation
      happens here by iterating ``pack.task_families()`` and asking each
      ``family.available_mutations(snapshot, reports, llm=)``.
    - ``auto_evolve`` no longer resolves the pack from a registry. The
      caller passes the concrete ``Pack`` instance as a keyword
      argument. This keeps the curriculum module decoupled from the
      pack-registry seam while it's still being wired up (and arguably
      cleaner regardless — the runtime already holds the pack).
    - ``direction_from_reports`` reads ``report.passed`` (the slice
      declared on ``EpisodeReportLike``) rather than poking into a
      verifier-result mapping.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Literal

from openrange.core.contracts import (
    Backing,
    Builder,
    BuildResult,
    EpisodeReportLike,
    LLMBackendLike,
    Manifest,
    Mutation,
    Pack,
    PackPrior,
    RuntimeHandle,
    TaskFamily,
    TaskSpec,
)
from openrange.world_ir import (
    Edge,
    Issue,
    Node,
    Ontology,
    WorldGraph,
    apply_patch,
)

if TYPE_CHECKING:
    from openrange.core.admit import Snapshot


Direction = Literal["harden", "soften", "diversify"]

CurriculumPolicy = Callable[[Sequence[EpisodeReportLike]], "Direction | None"]


# ---------------------------------------------------------------------------
# Policy: derive a direction from the report set
# ---------------------------------------------------------------------------


def direction_from_reports(
    reports: Sequence[EpisodeReportLike],
    *,
    harden_threshold: float = 0.66,
    soften_threshold: float = 0.33,
) -> Direction | None:
    """Default policy: pass-rate across reports decides direction.

    The protocol slice ``EpisodeReportLike`` is intentionally narrow —
    just ``.passed``. A report whose attribute access raises is treated
    as a failure (so an opt-in widening to a richer report type degrades
    gracefully if a field disappears).

    Returns ``None`` when there are no reports — no signal to act on.
    """
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
    """Read ``report.passed`` defensively.

    A report missing the attribute (or one that errors on access) is
    treated as a failure rather than crashing the curriculum policy.
    """
    try:
        return bool(report.passed)
    except AttributeError:
        return False


# ---------------------------------------------------------------------------
# Orchestration: aggregate proposals, pick one, apply + re-admit
# ---------------------------------------------------------------------------


def auto_evolve(
    snapshot: Snapshot,
    *reports: EpisodeReportLike,
    pack: Pack,
    policy: CurriculumPolicy = direction_from_reports,
    llm: LLMBackendLike | None = None,
    max_repairs: int = 2,
) -> Snapshot | None:
    """Pick a mutation based on agent performance and apply it.

    Asks every ``TaskFamily`` on ``pack`` to enumerate mutations against
    ``snapshot`` and the given ``reports`` (with optional LLM
    enrichment), aggregates the proposals, applies ``policy`` to pick a
    direction, walks candidates in that direction by descending
    relevance, and forwards each to ``pack.make_builder().evolve()`` to
    produce a ``GraphPatch``. The patch is applied to a copy of the
    snapshot's graph and re-admitted through ``admit()`` to produce the
    next ``Snapshot``.

    Each candidate is tried in order; if applying / re-admitting one
    fails (e.g. the patch violates an ontology constraint), the next
    candidate is tried. ``None`` is returned when there is no signal to
    act on (no reports, no proposals, no direction, no positive-relevance
    candidate in that direction, or every candidate failed admission).
    Callers loop until ``None`` to walk the curriculum naturally.

    The ``pack`` argument is REQUIRED. Earlier shapes resolved the pack
    from a registry attached to the snapshot's manifest; that registry
    isn't wired in the new core (yet), and threading the pack
    explicitly keeps the module decoupled regardless — runtime layers
    already hold the pack at hand.
    """
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
            # Skip this candidate; a single broken proposal must not
            # tear down the curriculum loop. The next candidate gets
            # a turn.
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
    llm: LLMBackendLike | None,
) -> list[Mutation]:
    """Aggregate ``available_mutations`` across every family on ``pack``.

    The order of options is family-iteration order followed by each
    family's own ordering — relevance sorting in ``auto_evolve`` makes
    the aggregate order observable only when two families tie on the
    same direction + relevance.
    """
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
    """Apply a chosen ``Mutation`` and re-admit the resulting graph.

    Steps:
      1. Spin up a fresh builder (``pack.make_builder(None)``) and ask
         it to ``evolve(snapshot, mutation) -> GraphPatch``. The default
         returns the mutation's patch verbatim; a pack may override to
         refine the patch (e.g. mint deterministic ids that fit the
         existing graph).
      2. Apply the patch to a *copy* of the snapshot's graph. The
         snapshot itself stays frozen — re-admission must always start
         from a graph the pack will accept, and mutating the original
         in place would leak edits into anyone still holding the
         snapshot.
      3. Re-admit through an in-process pack wrapper that returns the
         pre-built (evolved graph, original tasks) pair. The full
         admission gate runs: structural + conformance + invariants +
         task-binding + feasibility. Re-using the original ``tasks``
         keeps the curriculum focused on the world: a task that survives
         the world's edit is still valid; one that doesn't will fail
         feasibility and the next candidate runs.

    Returns ``None`` if re-admission rejects the evolved graph (the
    caller treats this as "skip this candidate").
    """
    from openrange.core.admit import AdmissionFailure, admit
    from openrange.core.admit import Snapshot as _Snapshot

    builder = pack.make_builder(None)
    patch = builder.evolve(snapshot, mutation)

    evolved_graph = _clone_graph(snapshot.graph)
    apply_patch(evolved_graph, patch)

    # Wrap the pack so admission sees the pre-evolved graph + existing
    # tasks. Everything else on the pack (ontology, invariants,
    # families) flows through unchanged so the full admission gate
    # still runs.
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
    return result


def _clone_graph(graph: WorldGraph) -> WorldGraph:
    """Make a shallow-but-detached copy of ``graph`` for patch application.

    The nodes and edges themselves are cloned (so patch updates that
    replace whole nodes don't leak), but the ontology id and meta dict
    are reused. The original snapshot's graph stays untouched.
    """
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


# ---------------------------------------------------------------------------
# Internal: a one-shot Pack wrapper that returns a pre-built BuildResult
# ---------------------------------------------------------------------------


class _OneShotBuilder(Builder):
    """Builder that always returns a pre-built ``BuildResult``.

    Used by ``_PreBuiltPack`` during re-admission of an evolved
    snapshot: the curriculum has already produced the evolved graph
    plus the task list, so the builder's job is just to hand them
    back. The admission loop still runs the full validate +
    feasibility gate, so a bad evolution still gets caught.
    """

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
    """A wrapping ``Pack`` whose builder returns a fixed ``BuildResult``.

    All non-builder methods delegate to the wrapped pack so admission's
    ontology / invariants / family-feasibility gates run unchanged. The
    only override is ``make_builder``: the returned builder always emits
    the pre-evolved graph + tasks, regardless of manifest.

    Lives at module scope rather than nested inside ``_evolve_snapshot``
    so mypy can verify it satisfies the ``Pack`` ABC.
    """

    def __init__(
        self,
        inner: Pack,
        graph: WorldGraph,
        tasks: list[TaskSpec],
    ) -> None:
        self._inner = inner
        self._graph = graph
        self._tasks = tasks
        # Match the wrapped pack's identity so lineage / debugging
        # reads still see the original pack.
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


__all__ = [
    "CurriculumPolicy",
    "Direction",
    "auto_evolve",
    "direction_from_reports",
]
