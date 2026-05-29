"""Protocols and ABCs a pack implements + the runtime consumes."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from typing import Any, ClassVar, Protocol, runtime_checkable

from graphschema import GraphPatch, Issue, Node, Ontology, WorldGraph

from openrange_pack_sdk._errors import AgentBackendError
from openrange_pack_sdk._types import (
    Backing,
    BuildResult,
    EpisodeResult,
    FeasibilityVerdict,
    LLMRequest,
    LLMResult,
    Manifest,
    Mutation,
    PackPrior,
    Snapshot,
    TaskSpec,
)

_log = logging.getLogger(__name__)


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


@runtime_checkable
class EpisodeReportLike(Protocol):
    """The slice of an episode report that families read.

    ``passed`` is the gate the curriculum policy reads. ``final_state`` is
    the runtime-emitted bag of facts the family's mutation/grading logic
    interrogates (which records were touched, which requests fired, etc).
    The contract is intentionally narrow — anything beyond these two
    fields is a concrete-report concern and packs reach for it at their
    own risk.
    """

    @property
    def passed(self) -> bool: ...

    @property
    def final_state(self) -> Mapping[str, Any]: ...


@runtime_checkable
class LLMBackend(Protocol):
    """A typed LLM transport. Concrete backends (Codex, fakes, custom
    HTTP) implement ``complete``; packs consume only the Protocol.

    Concrete backends MAY also define a ``preflight()`` method to validate
    binaries/credentials once. Callers that need preflight (e.g.,
    CodexAgentBackend) check for it via ``hasattr`` and call it
    defensively — the Protocol does not require it so a minimal in-process
    fake stays free of boilerplate."""

    def complete(self, request: LLMRequest) -> LLMResult: ...


AgentSession = Callable[[str], Any]
"""A callable an ``AgentBackend.build_agent`` returns. The return is
intentionally ``Any`` because different backends hand back different
shapes; callers normalize per-backend at the call site."""


@runtime_checkable
class AgentBackend(Protocol):
    """Factory for agent sessions. `preflight` validates dependencies;
    `build_agent` returns a callable session. Backends without tool
    dispatch must raise on non-empty `tools`."""

    def preflight(self) -> None: ...

    def build_agent(
        self,
        *,
        system_prompt: str,
        tools: Sequence[Callable[..., Any]] = (),
    ) -> AgentSession: ...


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
        llm: LLMBackend | None = None,
    ) -> tuple[Mutation, ...]:
        """Default opts out; families with curriculum support override."""
        del snapshot, reports, llm
        return ()

    def make_task(
        self,
        *,
        instruction: str,
        entrypoints: str | tuple[str, ...],
        goal_nodes: str | tuple[str, ...] = (),
        index: int | str = 0,
        difficulty: float = 0.5,
        meta: Mapping[str, Any] | None = None,
    ) -> TaskSpec:
        """Build a TaskSpec wired to this family.

        Derives ``id = f"{self.id}.{index}"``, ``feasibility_check`` /
        ``success_check`` from ``self.id``, and seeds ``meta`` with
        ``family = self.id`` + ``difficulty``. Pass a single node id as
        ``entrypoints`` / ``goal_nodes`` for the common one-node case;
        pass a tuple for multi-node tasks. ``index`` accepts ``int`` or
        ``str`` so families that produce per-instance tasks can use
        meaningful labels (``index="alice"`` → ``self.id + ".alice"``).
        Extra task metadata goes through the explicit ``meta`` mapping —
        no kwargs splat, so future ``make_task`` parameters can be added
        without breaking callers.
        """
        entry_tuple = (
            (entrypoints,) if isinstance(entrypoints, str) else tuple(entrypoints)
        )
        goal_tuple = (goal_nodes,) if isinstance(goal_nodes, str) else tuple(goal_nodes)
        merged_meta: dict[str, Any] = {"family": self.id, "difficulty": difficulty}
        if meta:
            merged_meta.update(meta)
        return TaskSpec(
            id=f"{self.id}.{index}",
            instruction=instruction,
            entrypoints=entry_tuple,
            goal_nodes=goal_tuple,
            feasibility_check=self.id,
            success_check=self.id,
            meta=merged_meta,
        )

    def make_mutation(
        self,
        *,
        direction: str,
        relevance: float,
        patch: Any,
        note: str = "",
    ) -> Mutation:
        """Build a Mutation tagged with ``family = self.id``.

        ``direction`` ∈ {"harden", "soften", "diversify"}; ``relevance`` ∈
        [0, 1]. ``patch`` is a ``GraphPatch``.
        """
        return Mutation(
            patch=patch,
            direction=direction,
            relevance=relevance,
            family=self.id,
            note=note,
        )

    def bump_scalar_attr(
        self,
        node: Node,
        key: str,
        new_value: Any,
        *,
        direction: str,
        relevance: float,
        note: str = "",
    ) -> Mutation:
        """Curriculum move that rewrites one scalar attr on ``node``.

        The common difficulty knob (cyber's build level, trading's return
        target / risk limit): a single ``GraphPatch`` replacing ``node`` with a
        copy whose ``attrs[key]`` is ``new_value``, every other field preserved.
        ``direction`` ∈ {"harden", "soften", "diversify"}.
        """
        updated = replace(node, attrs={**node.attrs, key: new_value})
        return self.make_mutation(
            direction=direction,
            relevance=relevance,
            patch=GraphPatch(nodes_updated=[updated]),
            note=note or f"{key}={new_value} on {node.id}",
        )


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
    ) -> Any:
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

    def default_prior(self) -> PackPrior | None:
        """Baseline prior for curriculum *grow* moves; ``None`` (default) opts
        out, so ``auto_evolve`` only patches. A pack whose builder reads the
        prior's ``difficulty`` returns its default prior here, letting core step
        that difficulty up/down and re-admit a freshly-built world."""
        return None


class NPC(ABC):
    """Subclasses implement ``step``; ``start``/``stop`` are no-op hooks.

    Set ``requires_llm = True`` to receive an :class:`AgentBackend` under
    ``context["agent_backend"]`` at ``start()`` (or ``None`` if the runtime
    wasn't configured with one). NPCs that don't opt in pay nothing.

    Broken-state: an NPC that cannot run sets ``self.broken_reason`` to a
    human-readable string and short-circuits ``step``. The episode service
    polls it and surfaces the transition to the dashboard so a silent NPC
    never goes unnoticed.
    """

    requires_llm: ClassVar[bool] = False
    broken_reason: str | None = None

    @property
    def actor_id(self) -> str:
        explicit = getattr(self, "_actor_id", None)
        if isinstance(explicit, str) and explicit:
            return explicit
        return f"{type(self).__name__}-{id(self) & 0xFFFF:04x}"

    @abstractmethod
    def step(self, interface: Mapping[str, Any]) -> None:
        """One tick. Decide whether to act based on internal cadence; swallow
        failures so the episode keeps running."""

    def start(self, context: Mapping[str, Any]) -> None:
        del context

    def stop(self) -> None:  # noqa: B027 — intentional default no-op
        pass


class AgentNPC(NPC):
    """An NPC backed by an LLM agent loop with a tool surface.

    Subclasses provide a ``system_prompt`` and a ``_build_tools(interface)``
    hook returning tool callables bound over the runtime backing. Agent
    dispatch is delegated to an :class:`AgentBackend`; constructor-supplied
    backends win over the runtime's ``context["agent_backend"]``.

    Failure model: init failure (preflight, missing backend, tool builder
    raises) marks the NPC permanently broken with one ``WARNING``. Per-tick
    LLM failures log at ``DEBUG`` and retry next cadence window.
    """

    requires_llm: ClassVar[bool] = True

    def __init__(
        self,
        *,
        system_prompt: str,
        cadence_ticks: int = 5,
        agent_backend: AgentBackend | None = None,
    ) -> None:
        if not system_prompt:
            raise ValueError("system_prompt must be non-empty")
        if cadence_ticks < 1:
            raise ValueError("cadence_ticks must be >= 1")
        self._system_prompt = system_prompt
        self._cadence_ticks = cadence_ticks
        self._backend_override = agent_backend
        self._runtime_backend: AgentBackend | None = None
        self._cooldown = 0
        self._agent: Any = None
        self._broken = False
        # Preflight at construction (when we have a backend already) so a
        # missing SDK / binary surfaces as soon as the manifest resolves,
        # not on the first acting tick. Runtime-supplied backends preflight
        # in ``start()`` instead.
        if agent_backend is not None:
            try:
                agent_backend.preflight()
            except Exception as exc:
                self._mark_broken(f"backend preflight failed: {exc}", exc=exc)

    def start(self, context: Mapping[str, Any]) -> None:
        if self._broken:
            return
        runtime_backend = context.get("agent_backend")
        if runtime_backend is not None:
            self._runtime_backend = runtime_backend
        backend = self._backend_override or self._runtime_backend
        if backend is None:
            self._mark_broken(
                "no AgentBackend configured (set RunConfig.npc_agent_backend "
                "or pass agent_backend to the NPC constructor)",
            )
            return
        if self._backend_override is None:
            try:
                backend.preflight()
            except Exception as exc:
                self._mark_broken(
                    f"runtime backend preflight failed: {exc}",
                    exc=exc,
                )

    def step(self, interface: Mapping[str, Any]) -> None:
        if self._broken:
            return
        if self._cooldown > 0:
            self._cooldown -= 1
            return
        self._cooldown = self._cadence_ticks - 1
        if self._agent is None:
            try:
                tools = list(self._build_tools(interface))
                self._agent = self._build_agent(tools)
            except Exception as exc:
                self._mark_broken(f"failed to construct agent: {exc}", exc=exc)
                self._agent = None
                return
        try:
            self._invoke_agent(self._user_prompt(interface))
        except Exception:
            _log.debug(
                "NPC %s tick failed; will retry next cadence window",
                type(self).__name__,
                exc_info=True,
            )
            return

    def _mark_broken(self, reason: str, *, exc: BaseException | None = None) -> None:
        if self._broken:
            return
        self._broken = True
        self.broken_reason = reason
        # exc_info=exc (not =True) — for broken-by-config cases there is no
        # in-flight exception, and ``=True`` would grab whatever
        # ``sys.exc_info`` returns from an unrelated traceback.
        _log.warning(
            "NPC %s is permanently broken (%s); the rest of the episode runs "
            "without it",
            type(self).__name__,
            reason,
            exc_info=exc,
        )

    def stop(self) -> None:
        self._agent = None

    @abstractmethod
    def _build_tools(
        self,
        interface: Mapping[str, Any],
    ) -> Sequence[Callable[..., Any]]: ...

    def _user_prompt(self, interface: Mapping[str, Any]) -> str:
        del interface
        return (
            "Take one realistic action consistent with your role. "
            "Use the available tools. Keep it short."
        )

    def _build_agent(self, tools: Sequence[Callable[..., Any]]) -> Any:
        backend = self._backend_override or self._runtime_backend
        if backend is None:
            raise AgentBackendError(
                "no AgentBackend available — start() did not capture one",
            )
        return backend.build_agent(
            system_prompt=self._system_prompt,
            tools=list(tools),
        )

    def _invoke_agent(self, prompt: str) -> None:
        self._agent(prompt)
