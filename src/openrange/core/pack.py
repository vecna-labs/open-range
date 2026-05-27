"""Pack / Builder / TaskFamily Protocols + wire shapes. See DESIGN.md §2."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from graphschema import GraphPatch, Issue, Ontology, WorldGraph

from openrange.core.errors import PackError

if TYPE_CHECKING:
    from openrange.core.admit import Snapshot


class Backing(StrEnum):
    PROCESS = "process"
    CONTAINER = "container"
    SIMULATOR = "simulator"
    HYBRID = "hybrid"


@dataclass(frozen=True)
class TaskSpec:
    """Goal nodes may be HIDDEN; entrypoints may not."""

    id: str
    instruction: str
    entrypoints: tuple[str, ...]
    goal_nodes: tuple[str, ...]
    feasibility_check: str
    success_check: str
    meta: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class TaskSeed:
    """A hint a TaskFamily's `generate()` may consult. Mutable so callers
    can re-tag `family` after the seed is produced."""

    theme: str
    anchor_kinds: list[str]
    suggested_goal_kinds: list[str]
    difficulty: float
    evidence: int = 1
    family: str | None = None


@dataclass
class PackPrior:
    """Generic graph statistics the Builder INTERPRETS; never dictates outputs."""

    source: str
    ontology: Ontology
    topology: Mapping[str, Any]
    task_seeds: list[TaskSeed] = field(default_factory=list)
    difficulty: Mapping[str, float] = field(default_factory=dict)
    coverage: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class BuildResult:
    """The candidate world + tasks from `Builder.build()`. `admission_meta`
    rides into `Snapshot.lineage`; opaque to core."""

    graph: WorldGraph
    tasks: list[TaskSpec]
    admission_meta: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FeasibilityVerdict:
    feasible: bool
    reason: str = ""


@dataclass(frozen=True)
class EpisodeResult:
    """Structured outcome — never a scalar reward. Harness-side
    training adapters do the shaping."""

    success: bool
    subgoals: Mapping[str, bool] = field(default_factory=dict)
    reason: str = ""


@dataclass(frozen=True)
class Mutation:
    """One curriculum move. `direction` ∈ {"harden","soften","diversify"};
    `relevance` ∈ [0,1]."""

    patch: GraphPatch
    direction: str
    relevance: float
    family: str
    note: str = ""


@runtime_checkable
class RuntimeHandle(Protocol):
    """A running realized world. Eight-method lifecycle; see CONTRACTS.md §9."""

    def reset(self) -> None: ...
    def surface(self) -> Mapping[str, Any]: ...
    def poll_events(self) -> tuple[Mapping[str, Any], ...]: ...
    def terminal(self) -> tuple[bool, str | None]: ...
    def checkpoint(self) -> Any: ...
    def restore(self, state: Any) -> None: ...
    def collect(self) -> Mapping[str, Any]: ...
    def stop(self) -> None: ...


Manifest = Mapping[str, Any]


@runtime_checkable
class EpisodeReportLike(Protocol):
    """Slice of `EpisodeReport` curriculum policies read."""

    @property
    def passed(self) -> bool: ...


@runtime_checkable
class LLMBackendLike(Protocol):
    """Duck-typed slice of `openrange.llm.LLMBackend`. `request` is `Any` so
    core does not import `openrange.llm`."""

    def complete(self, request: Any) -> Any: ...


class TaskFamily(ABC):
    """One domain of tasks against a Pack's world. See DESIGN.md §4."""

    id: str = ""
    pack_id: str = ""

    @abstractmethod
    def generate(
        self,
        graph: WorldGraph,
        manifest: Manifest,
        prior: PackPrior | None,
    ) -> list[TaskSpec]: ...

    @abstractmethod
    def check_feasibility(
        self,
        graph: WorldGraph,
        task: TaskSpec,
    ) -> FeasibilityVerdict: ...

    @abstractmethod
    def check_success(
        self,
        graph: WorldGraph,
        task: TaskSpec,
        final_state: Mapping[str, Any],
    ) -> EpisodeResult: ...

    def available_mutations(
        self,
        snapshot: Snapshot,
        reports: Sequence[EpisodeReportLike],
        *,
        llm: LLMBackendLike | None = None,
    ) -> tuple[Mutation, ...]:
        """Default opts out; families with curriculum support override."""
        del snapshot, reports, llm
        return ()


class Builder(ABC):
    """Produces a `BuildResult`. Deterministic in `(manifest, prior)`."""

    @abstractmethod
    def build(self, manifest: Manifest) -> BuildResult: ...

    def repair(
        self,
        prev: BuildResult,
        errors: list[Issue],
        infeasible: list[str],
    ) -> BuildResult:
        """Default raises; override to participate in admission's repair loop."""
        del prev, errors, infeasible
        raise NotImplementedError(
            "this Builder did not implement repair(); admission will not "
            "retry. Override repair() to participate in the admission loop."
        )

    def evolve(
        self,
        snapshot: Snapshot,
        mutation: Mutation,
    ) -> GraphPatch:
        """Default returns the mutation's patch verbatim."""
        del snapshot
        return mutation.patch


class Pack(ABC):
    """The pack-side contract core depends on. See DESIGN.md §2."""

    id: str = ""
    version: str = ""

    @abstractmethod
    def ontology(self) -> Ontology: ...

    def invariants(self) -> list[Callable[[WorldGraph], list[Issue]]]:
        return []

    @abstractmethod
    def make_builder(self, prior: PackPrior | None) -> Builder: ...

    @abstractmethod
    def realize(
        self,
        graph: WorldGraph,
        backing: Backing,
    ) -> RuntimeHandle: ...

    def task_families(self) -> list[TaskFamily]:
        return []

    def task_family(self, family_id: str) -> TaskFamily | None:
        for fam in self.task_families():
            if fam.id == family_id:
                return fam
        return None


PACK_ENTRY_POINT_GROUP = "openrange.packs"


class PackRegistry:
    """Registry of Pack instances by id, discovered via the
    `openrange.packs` entry-point group when `autodiscover=True`."""

    def __init__(self, *, autodiscover: bool = False) -> None:
        self._packs: dict[str, Pack] = {}
        self._autodiscover = autodiscover
        self._discovered = False

    def register(self, pack: Pack) -> None:
        self._packs[pack.id] = pack

    def resolve(self, pack_id: str) -> Pack:
        self._ensure_discovered()
        try:
            return self._packs[pack_id]
        except KeyError as exc:
            raise PackError(f"unknown pack {pack_id!r}") from exc

    def resolve_class(self, pack_id: str) -> type[Pack]:
        return type(self.resolve(pack_id))

    def ids(self) -> tuple[str, ...]:
        self._ensure_discovered()
        return tuple(sorted(self._packs))

    def discover(self) -> None:
        self._ensure_discovered(force=True)

    def _ensure_discovered(self, *, force: bool = False) -> None:
        if not self._autodiscover and not force:
            return
        if self._discovered and not force:
            return
        self._discovered = True
        from openrange.core._registry import iter_entry_points

        for name, value in iter_entry_points(
            PACK_ENTRY_POINT_GROUP,
            error_cls=PackError,
            kind="pack",
        ):
            if name in self._packs and not force:
                continue
            pack = value() if callable(value) else value
            if not isinstance(pack, Pack):
                raise PackError(
                    f"entry point {name!r} did not return a Pack",
                )
            if pack.id != name:
                raise PackError(
                    f"entry point name {name!r} does not match pack.id {pack.id!r}",
                )
            self._packs[pack.id] = pack


PACKS = PackRegistry(autodiscover=True)
