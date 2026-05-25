"""Pack, Builder, TaskFamily — the binding surface between OpenRange and a domain.

This module owns the protocols core depends on plus the plain-data shapes
that flow through them. The two boundaries:

    CORE  owns TYPES and GENERIC ALGORITHMS — things identical for every
          domain. It never names a domain concept.

    PACK  owns VALUES and DOMAIN FUNCTIONS — the ontology value, the builder,
          the realizer, the task families, the checks. Everything
          domain-specific.

The split is enforceable: a CI check (`scripts/check_boundary.sh`) greps
this file (and all of core) for the forbidden domain vocabulary defined
in that script. There should be zero hits.

The seam to the BBG world is `PackPrior`. A `PackPrior` arrives at
`Pack.make_builder(prior=...)` from either:

    - `openrange.distill(graph, status_log)`  (the flywheel path: distilled
      from a real agent's spatial memory dump)
    - a hand-authored default shipped with the pack  (the boot path:
      `make_builder(prior=None)` falls back to the pack's hand-authored
      `PackPrior`)

The builder has one code path; it never knows which source produced the
prior. That's what keeps the bootstrap-to-flywheel transition seamless.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from openrange.world_ir import GraphPatch, Issue, Ontology, WorldGraph

if TYPE_CHECKING:
    from openrange.core.admit import Snapshot


# ---------------------------------------------------------------------------
# Backing — how a realized world runs
# ---------------------------------------------------------------------------


class Backing(StrEnum):
    """The runtime substrate a Pack realizes its world against.

    A pack may support one backing or several; the same `Pack.realize()` call
    receives the choice. Choosing here is a runtime decision (laptop vs.
    container vs. compute farm) that does not affect the world's graph.

    PROCESS    : in-process simulation (NPCs as threads, file artifacts in /tmp)
    CONTAINER  : docker / podman / k8s per service
    SIMULATOR  : a pack-provided simulator (no real services)
    HYBRID     : a mix — process for cheap parts, container for expensive ones
    """

    PROCESS = "process"
    CONTAINER = "container"
    SIMULATOR = "simulator"
    HYBRID = "hybrid"


# ---------------------------------------------------------------------------
# Generic wire shapes — declared here so packs and core agree on the types
# that cross the boundary
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TaskSpec:
    """One task an agent can attempt against a world.

    Three parts: an instruction (what to do), entrypoints (where it starts
    acting — node-ids in the world graph), and goal_nodes (what counts as
    completion — also node-ids; may be HIDDEN).

    `feasibility_check` and `success_check` are HANDLES, not exec'd source.
    They name a TaskFamily; the pack's `task_family(name)` resolves them to
    a class that runs the check as a method.

    Entrypoints and goal_nodes live HERE, on the task, never as node roles.
    Two tasks against the same world may entrypoint different nodes (a build
    task entrypoints the repo; a pentest task entrypoints the endpoint) —
    that is task-relative, not world-absolute.
    """

    id: str
    instruction: str
    entrypoints: tuple[str, ...]
    goal_nodes: tuple[str, ...]
    feasibility_check: str  # "webapp.pentest" — a TaskFamily.id
    success_check: str  # same handle conventionally
    meta: Mapping[str, Any] = field(default_factory=dict)
    # `meta` carries family + difficulty + any pack-shaped task provenance.


@dataclass(frozen=True)
class TaskSeed:
    """One distilled hint at where a task could be authored.

    Emitted by `distill()` (one per thought-cluster) and consumed by a
    TaskFamily's `generate()`. Carries ANCHORS (kinds of things the
    cluster anchored on) and SUGGESTED GOAL KINDS (kinds of things at the
    sinks of productive paths). A TaskFamily can use these as seeds for
    its own task synthesis — or ignore them and generate from manifest +
    graph alone.

    `family` is OPTIONAL — `distill()` never tags seeds with a family.
    A harness with that knowledge may attach one; otherwise TaskFamilies
    self-select by `anchor_kinds`.
    """

    theme: str  # cluster identifier, e.g. "cluster-0"
    anchor_kinds: tuple[str, ...]  # kinds of nodes the cluster anchored on
    suggested_goal_kinds: tuple[str, ...]
    difficulty: float  # 0..1
    evidence: int = 1  # how many trajectories agreed
    family: str | None = None


@dataclass(frozen=True)
class PackPrior:
    """The generation prior flowing from BBG → Builder.

    Carries ONLY generic graph statistics. The builder INTERPRETS these
    into domain decisions; the prior never tells the builder what to do.
    This is the one rule that keeps `distill()` reusable across domains.

    `ontology` may be the target pack's ontology (refinement path) or an
    induced ontology that `distill` proposed (bootstrap path with no
    existing pack).

    `topology` keys are a fixed generic set:
      - `node_kind_freq` : count of each kind in the source graph
      - `salient_kind_freq` : count of each kind where status=salient
      - `dead_end_ratio` : fraction of traversed edges with outcome=dead_end
      - `hidden_signal` : per-kind count of confirmed-thought anchors
    """

    source: str  # e.g. "bbg@0.1.0 :: sha256:..."
    ontology: Ontology
    topology: Mapping[str, Any]
    task_seeds: tuple[TaskSeed, ...] = ()
    difficulty: Mapping[str, float] = field(default_factory=dict)
    coverage: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class BuildResult:
    """What `Builder.build()` returns: the candidate world + its tasks.

    `admission_meta` is the builder's own provenance (LLM prompts, sampling
    seed, prior summary). It rides into `Snapshot.lineage` and is otherwise
    opaque to core.
    """

    graph: WorldGraph
    tasks: list[TaskSpec]
    admission_meta: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FeasibilityVerdict:
    """A TaskFamily's verdict on whether a task is solvable in this world."""

    feasible: bool
    reason: str = ""


@dataclass(frozen=True)
class EpisodeResult:
    """What an episode's success-check returns.

    Structured — NOT a scalar reward. A harness-side training adapter maps
    this into whatever signal a training setup needs. OpenRange does not
    shape rewards.
    """

    success: bool
    subgoals: Mapping[str, bool] = field(default_factory=dict)
    reason: str = ""


@dataclass(frozen=True)
class Mutation:
    """One curriculum move proposed by a TaskFamily.

    Carries a `GraphPatch` (the universal diff type), a direction tag
    (`harden` / `soften` / `diversify`), a relevance score (0..1)
    reflecting how well the move responds to recent episode reports,
    the family that proposed it, and an optional note.

    A future curriculum policy in the runtime layer will pick a
    direction from aggregate pass-rate, score candidate `Mutation`s
    against it, and apply the winning patch via
    `Builder.evolve(snapshot, mutation)`. Today's foundation ships the
    type and the `evolve()` contract; the policy lands when the runtime
    layer is re-wired.
    """

    patch: GraphPatch
    direction: str  # "harden" | "soften" | "diversify"
    relevance: float
    family: str
    note: str = ""


# ---------------------------------------------------------------------------
# RuntimeHandle — what `Pack.realize` returns
# ---------------------------------------------------------------------------


@runtime_checkable
class RuntimeHandle(Protocol):
    """A running realized world. Owned by the pack; consumed by the episode.

    The handle is the seam between the realizer (pack territory: knows how
    to start services, seed databases, render templates) and the episode
    loop (core territory: doesn't care how, only that the agent can act and
    the final state can be read).

    `surface()` returns whatever the agent's IO needs — base URLs, file
    roots, MCP endpoints. The shape is pack-defined; the harness binds
    against keys it expects.

    `collect()` returns the structured final state at episode end. A
    TaskFamily's `check_success` reads from this dict against world graph
    + task to decide success. Shape is pack-defined; the family knows it
    because the family and the pack are siblings.
    """

    def reset(self) -> None: ...
    def surface(self) -> Mapping[str, Any]: ...
    def collect(self) -> Mapping[str, Any]: ...
    def stop(self) -> None: ...


# ---------------------------------------------------------------------------
# Manifest — declared loosely so packs can ship their own typed shapes
# ---------------------------------------------------------------------------

Manifest = Mapping[str, Any]
"""A manifest is a free-form dict. Packs document their expected keys; core
never branches on a manifest field."""


# ---------------------------------------------------------------------------
# The three core protocols
# ---------------------------------------------------------------------------


class TaskFamily(ABC):
    """A domain of tasks posed against a Pack's world.

    A Pack owns a world-family (e.g. `webapp`); a TaskFamily owns one
    DOMAIN of tasks against that world (e.g. `webapp.build`,
    `webapp.pentest`). The same world graph can serve multiple families
    with different entrypoints, different goals, and different success
    criteria — *this* is where the word "domain" lives, not on Pack.

    A TaskFamily owns:
      - task generation (instruction text, entrypoint selection, goal
        selection from the graph)
      - feasibility checking (against the world graph, no runtime)
      - success checking (against the world graph + the realizer's
        collected final state)
      - curriculum mutations (what graph changes would harden / soften
        / diversify *this* family's tasks specifically)

    Concrete TaskFamilies are pack-side classes. Core never imports a
    specific family; it looks them up by `id` through the Pack.
    """

    id: str = ""  # "webapp.build", "webapp.pentest"
    pack_id: str = ""  # back-pointer to the Pack.id

    @abstractmethod
    def generate(
        self, graph: WorldGraph, manifest: Manifest, prior: PackPrior | None
    ) -> list[TaskSpec]:
        """Generate one or more TaskSpecs against this world.

        The family selects entrypoints (which node-ids the agent acts
        from), goal-nodes (which node-ids count as completion), and
        composes the instruction string. `prior.task_seeds` is the
        distill output, available as a hint; the family may ignore it.
        """

    @abstractmethod
    def check_feasibility(
        self, graph: WorldGraph, task: TaskSpec
    ) -> FeasibilityVerdict:
        """Decide whether `task` is actually solvable in `graph`.

        Pure-graph reasoning — no realizer, no runtime. The family walks
        the graph and decides. (Schema correctness is already covered by
        admission's structural+conformance tiers; this is the
        domain-meaning check that only the family knows how to do.)
        """

    @abstractmethod
    def check_success(
        self, graph: WorldGraph, task: TaskSpec, final_state: Mapping[str, Any]
    ) -> EpisodeResult:
        """Read the realizer's final-state mapping and decide success.

        `final_state` is the dict `RuntimeHandle.collect()` returned. Its
        keys are a pack/family convention. The family compares against
        the world graph + the task and returns a structured result —
        success bool, optional subgoals, optional reason.
        """

    def available_mutations(
        self,
        snapshot: Snapshot,
        reports: Sequence[Any],
        *,
        llm: Any | None = None,
    ) -> tuple[Mutation, ...]:
        """Propose curriculum mutations that would harden / soften / diversify
        this family's tasks specifically.

        Default returns `()` — families without curriculum support opt out
        cleanly. `llm` is offered so a family can re-score relevance with
        a semantic pass; families that don't use LLMs ignore it.

        `reports` and `llm` are typed as `Sequence[Any]` / `Any | None`
        on purpose: the concrete `EpisodeReport` and `LLMBackend` types
        live in the runtime layer which is currently being re-wired
        against this shape. When that layer lands, the annotation
        narrows to those concrete types without changing this signature.
        """
        del snapshot, reports, llm
        return ()


class Builder(ABC):
    """The pack-side machinery that produces a candidate `BuildResult`.

    A Pack constructs its Builder via `make_builder(prior)`. The same
    Builder can be invoked many times with different manifests; it should
    be deterministic in `(manifest, prior)` modulo any builder-internal
    seed.

    `build()` returns a candidate world + tasks. Admission validates them
    and, on failure, calls `repair(prev, errors, infeasible)` — up to the
    configured budget. After admission, `evolve(snapshot, mutation)`
    applies a curriculum move as a `GraphPatch`.
    """

    @abstractmethod
    def build(self, manifest: Manifest) -> BuildResult: ...

    def repair(
        self, prev: BuildResult, errors: list[Issue], infeasible: list[str]
    ) -> BuildResult:
        """Optional repair hook. Default: regenerate from scratch.

        A procedural builder may resample; an LLM builder may patch the
        offending piece. Core supplies the failures and asks for a new
        candidate; the builder decides how to respond.
        """
        del prev, errors, infeasible
        raise NotImplementedError(
            "this Builder did not implement repair(); admission will not "
            "retry. Override repair() to participate in the admission loop."
        )

    def evolve(self, snapshot: Snapshot, mutation: Mutation) -> GraphPatch:
        """Apply a curriculum mutation, returning a `GraphPatch`.

        Default: return the mutation's patch verbatim. A pack that wants
        to refine the patch (e.g. adjust LLM-generated artifacts to fit
        the existing world) overrides this.
        """
        del snapshot
        return mutation.patch


class Pack(ABC):
    """The pack-side contract core depends on.

    A Pack ships:
      - an `Ontology` declaring the world's node/edge kinds + attrs
      - pack invariants (Tier-3 callables core's `validate()` runs)
      - a `Builder` factory accepting a `PackPrior | None`
      - a `realize()` that turns an admitted graph into a `RuntimeHandle`
      - one or more `TaskFamily` classes

    Core depends on this Protocol; it never imports a concrete pack.
    Packs ship as their own Python packages and register via the
    `openrange.packs` entry point group declared in their `pyproject.toml`.
    """

    id: str = ""  # e.g. "webapp"
    version: str = ""

    @abstractmethod
    def ontology(self) -> Ontology: ...

    def invariants(self) -> list[Callable[[WorldGraph], list[Issue]]]:
        """Tier-3 invariants the validator runs on every candidate.

        Returns a list of plain functions, each `(graph) -> list[Issue]`.
        Default empty — a pack without graph-wide invariants opts out
        cleanly.
        """
        return []

    @abstractmethod
    def make_builder(self, prior: PackPrior | None) -> Builder:
        """Construct a fresh Builder for this pack with the given prior.

        `prior=None` is the boot path: the pack falls back to a
        hand-authored default `PackPrior` (typically returned by a
        `default_prior()` helper inside the pack).
        """

    @abstractmethod
    def realize(self, graph: WorldGraph, backing: Backing) -> RuntimeHandle: ...

    def task_families(self) -> list[TaskFamily]:
        """Every TaskFamily this pack offers. Default empty.

        A pack with no task families won't admit anything (admission
        requires at least one task), so packs that admit at all must
        return at least one family here.
        """
        return []

    def task_family(self, family_id: str) -> TaskFamily | None:
        """Look up a TaskFamily by its `id`. Default: linear scan."""
        for fam in self.task_families():
            if fam.id == family_id:
                return fam
        return None
