"""NPC contract and registry.

An NPC is an autonomous actor that runs alongside the agent during an
episode, receiving the same ``interface`` mapping the verifier and
admission probe see. Two shapes ship: ``NPC`` for scripted actors and
``AgentNPC`` for LLM-backed loops with tools.

Manifest schema::

    npc:
      - type: cyber.browsing_user      # NPCRegistry id
        count: 3                        # default 1
        config:                         # default {}
          cadence_ticks: 2
          paths: ["/search?q=alpha"]

Factories are registered via the ``openrange.npcs`` entry-point group;
the registry builds each NPC fresh per episode.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from typing import Any, ClassVar

from openrange.agent_backend import AgentBackend, AgentBackendError
from openrange.core.errors import OpenRangeError

_log = logging.getLogger(__name__)

NPC_ENTRY_POINT_GROUP = "openrange.npcs"

NPCFactory = Callable[[Mapping[str, object]], "NPC"]


class NPCError(OpenRangeError):
    pass


class NPC(ABC):
    """Subclasses implement ``step``; ``start``/``stop`` are no-op hooks.

    Set ``requires_llm = True`` to receive an
    :class:`~openrange.agent_backend.AgentBackend` under
    ``context["agent_backend"]`` at ``start()`` (or ``None`` if the
    runtime wasn't configured with one). NPCs that don't opt in pay
    nothing.

    Broken-state: an NPC that cannot run sets ``self.broken_reason``
    to a human-readable string and short-circuits ``step``. The
    episode service polls it and surfaces the transition to the
    dashboard so a silent NPC never goes unnoticed.
    """

    requires_llm: ClassVar[bool] = False
    broken_reason: str | None = None

    @property
    def actor_id(self) -> str:
        # Override or set self._actor_id for a real display name; default
        # is class name + short instance hash to disambiguate count > 1.
        explicit = getattr(self, "_actor_id", None)
        if isinstance(explicit, str) and explicit:
            return explicit
        return f"{type(self).__name__}-{id(self) & 0xFFFF:04x}"

    @abstractmethod
    def step(self, interface: Mapping[str, Any]) -> None:
        """One tick. Decide whether to act based on internal cadence;
        swallow failures so the episode keeps running."""

    def start(self, context: Mapping[str, Any]) -> None:
        """Optional setup. ``context`` carries
        ``{episode_id, snapshot_id, task_id, base_url, record_action}``;
        NPCs with ``requires_llm = True`` also receive ``agent_backend``.
        ``record_action(action, *, target=None, observation=None)``
        publishes a dashboard event tagged with this NPC's ``actor_id``.
        """
        del context

    def stop(self) -> None:  # noqa: B027 — intentional default no-op
        pass


class AgentNPC(NPC):
    """An NPC backed by an LLM agent loop with a tool surface.

    Subclasses provide a ``system_prompt`` and a ``_build_tools(interface)``
    hook returning tool callables bound over the runtime backing.
    Agent dispatch is delegated to an
    :class:`~openrange.agent_backend.AgentBackend`; constructor-supplied
    backends win over the runtime's ``context["agent_backend"]``.

    Failure model: init failure (preflight, missing backend, tool
    builder raises) marks the NPC permanently broken with one
    ``WARNING``. Per-tick LLM failures log at ``DEBUG`` and retry next
    cadence window.
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
        # Preflight at construction (when we have a backend already) so
        # a missing SDK / binary surfaces as soon as the manifest
        # resolves, not on the first acting tick. Runtime-supplied
        # backends preflight in ``start()`` instead.
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
                "no AgentBackend configured "
                "(set RunConfig.npc_agent_backend or pass agent_backend "
                "to the NPC constructor)",
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
            # Transient (rate limits, timeouts) — DEBUG only; operator view stays clean.
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
        # exc_info=exc (not =True) — for broken-by-config cases there
        # is no in-flight exception, and ``=True`` would grab whatever
        # ``sys.exc_info`` returns from an unrelated traceback.
        _log.warning(
            "NPC %s is permanently broken (%s); "
            "the rest of the episode runs without it",
            type(self).__name__,
            reason,
            exc_info=exc,
        )

    def stop(self) -> None:
        self._agent = None

    # -- subclass extension points ----------------------------------------

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
        if self._agent is None:
            return
        self._agent(prompt)


class NPCRegistry:
    """Registry of NPC factories by id.

    ``autodiscover=False`` (default) gives tests a clean slate; the
    global ``NPCS`` autodiscovers from the ``openrange.npcs`` entry-point
    group on first access.
    """

    def __init__(self, *, autodiscover: bool = False) -> None:
        self._factories: dict[str, NPCFactory] = {}
        self._autodiscover = autodiscover
        self._discovered = False

    def register(self, npc_id: str, factory: NPCFactory) -> None:
        self._factories[npc_id] = factory

    def resolve(self, npc_id: str, config: Mapping[str, object]) -> NPC:
        self._ensure_discovered()
        try:
            factory = self._factories[npc_id]
        except KeyError as exc:
            raise NPCError(f"unknown NPC {npc_id!r}") from exc
        npc = factory(config)
        if not isinstance(npc, NPC):
            raise NPCError(
                f"NPC factory {npc_id!r} did not return an NPC instance",
            )
        return npc

    def ids(self) -> tuple[str, ...]:
        self._ensure_discovered()
        return tuple(sorted(self._factories))

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
            NPC_ENTRY_POINT_GROUP,
            error_cls=NPCError,
            kind="NPC",
        ):
            if name in self._factories and not force:
                continue
            if not callable(value):
                raise NPCError(
                    f"entry point {name!r} did not yield a callable",
                )
            self._factories[name] = value


NPCS = NPCRegistry(autodiscover=True)


def resolve_manifest_npcs(
    npc_entries: tuple[Mapping[str, object], ...],
    *,
    registry: NPCRegistry | None = None,
) -> list[NPC]:
    """Construct NPC instances from manifest entries.

    Each entry is a mapping with ``type`` (required), ``count`` (default
    1), and ``config`` (default empty). Returns a flat list of NPCs —
    one per spawn slot, so the caller can iterate and step uniformly.
    """
    reg = registry if registry is not None else NPCS
    npcs: list[NPC] = []
    for entry in npc_entries:
        npc_type = entry.get("type")
        if not isinstance(npc_type, str) or not npc_type:
            raise NPCError("manifest npc entry must carry a non-empty 'type'")
        count_raw = entry.get("count", 1)
        if not isinstance(count_raw, int) or count_raw < 0:
            raise NPCError(
                f"manifest npc entry 'count' must be a non-negative int "
                f"(got {count_raw!r})",
            )
        config_raw = entry.get("config", {})
        if not isinstance(config_raw, Mapping):
            raise NPCError(
                f"manifest npc entry 'config' must be a mapping for {npc_type!r}",
            )
        for _ in range(count_raw):
            npcs.append(reg.resolve(npc_type, config_raw))
    return npcs
