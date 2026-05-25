"""Episode service: the agent harness's seam into running worlds.

This is the second of the two CORE/PACK boundaries (the first is
``admit``). The pack ships a :class:`~openrange.core.contracts.Pack`
whose ``realize(graph, backing)`` produces a
:class:`~openrange.core.contracts.RuntimeHandle` implementing the
eight-method Protocol. Core's :class:`EpisodeService` drives that
handle through the episode lifecycle:

  ``reset()``       prepare a clean run state
  ``surface()``     return the agent-facing IO surface (HTTP base_url,
                    file roots, MCP endpoints, NPC adapter dicts) —
                    consumed by the agent harness and the NPCs
  ``poll_events()`` drain side-effect events the realized world
                    produced; forwarded to the dashboard
  ``terminal()``    has the agent finished? ``(done, reason)``
  ``checkpoint()``  opaque pack-defined state for counterfactual replay
  ``restore(state)``reverse a checkpoint payload
  ``collect()``     structured final-state mapping for ``check_success``
  ``stop()``        tear the realized world down

Episode end dispatches to ``pack.task_family(task.success_check)`` so
the family decides success against ``(graph, task, final_state)`` and
returns an :class:`~openrange.core.contracts.EpisodeResult`. Core never
inspects the structured fields itself — :class:`EpisodeReport` carries
the result through; whoever consumes it (curriculum, training loops,
tests) reads the typed ``success`` / ``subgoals`` / ``reason``.

The agent acts on the world through whatever surface the pack exposes.
OpenRange does not own agent action: ``record_turn`` is observational
only. ``tick`` drains events + decides terminal; ``advance`` is the
multi-tick wrapper. ``checkpoint`` / ``restore`` / ``fork`` enable
counterfactual training by riding the handle's opaque state payload.
"""

from __future__ import annotations

import atexit
import contextlib
import threading
import uuid
import weakref
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal

from openrange.agent_backend import AgentBackend, StrandsAgentBackend
from openrange.core.admit_loop import Snapshot
from openrange.core.contracts import (
    Backing,
    EpisodeResult,
    Pack,
    RuntimeHandle,
    TaskSpec,
)
from openrange.core.errors import OpenRangeError
from openrange.core.turn import ActorTurn
from openrange.npc import NPC, resolve_manifest_npcs

if TYPE_CHECKING:
    from openrange.dashboard import DashboardView


class EpisodeError(OpenRangeError):
    """Raised when an episode operation cannot proceed."""


# ---------------------------------------------------------------------------
# Public data shapes
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EpisodeHandle:
    """Identifies one in-flight episode.

    Carried out across the public surface (``start_episode``,
    ``stop_episode``, ``observe`` ...) — the rest of the per-episode
    state hides behind :class:`EpisodeService`. The shape is unchanged
    from the pre-refactor API so harnesses upgrade without rebinding.
    """

    id: str
    snapshot_id: str
    task_id: str


@dataclass(frozen=True, slots=True)
class Observation:
    """One observation pulled from the runtime.

    ``events`` originates from :meth:`RuntimeHandle.poll_events`;
    ``visible_state`` is the static keys an observer expects (base
    URL, agent root). The mapping is pack-defined; the harness picks
    what it needs.
    """

    visible_state: Mapping[str, Any] = field(default_factory=dict)
    events: tuple[Mapping[str, Any], ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AgentTurn:
    """A note from the agent harness about what the agent just did.

    Observational only — OpenRange does not enforce that the agent's
    action match the tool calls listed here. The harness recording is
    enough for the dashboard timeline.
    """

    message: str | None = None
    tool_calls: tuple[Mapping[str, Any], ...] = ()
    tool_results: tuple[Mapping[str, Any], ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TickRequest:
    """Knobs passed to :meth:`EpisodeService.tick`.

    The fields are pack-agnostic: every realizer drains events, the
    NPC pass is per-tick, and timers are reserved for a future
    timer-driven mutator. Today ``process_timers`` is unused but kept
    in the public shape so consumers don't rewrite call sites later.
    """

    max_events: int | None = None
    process_npcs: bool = True
    process_timers: bool = True


@dataclass(frozen=True, slots=True)
class TickResult:
    """Outcome of one tick: any events, plus terminal hint."""

    events: tuple[Mapping[str, Any], ...] = ()
    done: bool = False
    terminal_reason: str | None = None


@dataclass(frozen=True, slots=True)
class AdvanceRequest:
    """Knobs passed to :meth:`EpisodeService.advance`.

    ``until`` decides when the multi-tick loop yields:
    ``"observation"`` returns at the first poll that produces events,
    ``"event"`` is its synonym (kept for harness compatibility),
    ``"terminal"`` keeps ticking until the handle reports terminal,
    ``"idle"`` is reserved for a future "until-nothing-happens" mode.
    """

    until: Literal["observation", "event", "terminal", "idle"] = "observation"
    max_ticks: int = 16
    timeout_seconds: float | None = None


@dataclass(frozen=True, slots=True)
class EpisodeUpdate:
    """Aggregate of one :meth:`advance` call.

    ``observation`` is set when the loop yielded on an event;
    ``events`` collects every event drained during the multi-tick
    window; ``done`` / ``terminal_reason`` mirror the handle's
    last :meth:`RuntimeHandle.terminal` call.
    """

    observation: Observation | None = None
    events: tuple[Mapping[str, Any], ...] = ()
    done: bool = False
    terminal_reason: str | None = None


@dataclass(frozen=True, slots=True)
class EpisodeReport:
    """The terminal artifact from a stopped episode.

    Wraps the family's :class:`EpisodeResult` rather than a raw dict —
    the typed shape (``success: bool``, ``subgoals: Mapping[str, bool]``,
    ``reason: str``) is the new contract with curriculum / training
    consumers. ``final_state`` is the dict :meth:`RuntimeHandle.collect`
    returned and is exposed verbatim so a dashboard can show what the
    family read.

    Implements the :class:`EpisodeReportLike` Protocol via the
    ``passed`` property, so curriculum's
    :func:`~openrange.core.curriculum.direction_from_reports` can read
    pass-rate without a separate adapter.
    """

    snapshot_id: str
    task_id: str
    episode_result: EpisodeResult
    final_state: Mapping[str, Any] = field(default_factory=dict)
    agent_summary: str = ""

    @property
    def passed(self) -> bool:
        """Adapter for :class:`EpisodeReportLike` — the family decided."""
        return self.episode_result.success

    def as_dict(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "task_id": self.task_id,
            "episode_result": {
                "success": self.episode_result.success,
                "subgoals": dict(self.episode_result.subgoals),
                "reason": self.episode_result.reason,
            },
            "final_state": dict(self.final_state),
            "agent_summary": self.agent_summary,
        }


@dataclass(frozen=True, slots=True)
class EpisodeCheckpoint:
    """Captured state for a running episode.

    Wraps the opaque payload :meth:`RuntimeHandle.checkpoint` returned.
    The shape is pack-defined; core just shuttles it back into
    :meth:`RuntimeHandle.restore`. The four identifier fields are
    enough to recover the (snapshot, task, episode) context without
    digging into the opaque state.
    """

    id: str
    episode_id: str
    snapshot_id: str
    task_id: str
    state: Any
    metadata: Mapping[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Internal per-episode state
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _RunningEpisode:
    """Per-episode state owned by :class:`EpisodeService`.

    Holds the handle plus the cached surface (so ``observe`` /
    ``base_url`` / ``agent_root`` don't re-call ``surface()`` for
    every read), plus the dashboard + NPC bookkeeping.
    """

    handle: EpisodeHandle
    snapshot: Snapshot
    task: TaskSpec
    runtime: RuntimeHandle
    run_root: Path
    surface_cache: Mapping[str, Any]
    dashboard: DashboardView | None = None
    agent_summary: str = ""
    final_state: Mapping[str, Any] | None = None
    episode_result: EpisodeResult | None = None
    tick_thread: threading.Thread | None = None
    tick_stop: threading.Event | None = None
    npcs: list[NPC] = field(default_factory=list)
    stopped: bool = False


# ---------------------------------------------------------------------------
# EpisodeService
# ---------------------------------------------------------------------------


class EpisodeService:
    """Owns running worlds; provides start, observe, advance, checkpoint, fork.

    One :class:`EpisodeService` runs against one :class:`Pack` — the
    pack is fixed at construction (resolved design Q1) so a service
    can never realize a snapshot built by a different pack. A run that
    needs multiple packs constructs one service per pack.

    The constructor pre-resolves the NPC agent backend: an explicit
    backend wins; the ``npc_llm_model`` shorthand auto-promotes to a
    :class:`StrandsAgentBackend`. Both unset means LLM-backed NPCs go
    broken at start with a clear ``no backend configured`` reason.
    """

    def __init__(
        self,
        pack: Pack,
        run_root: str | Path,
        *,
        dashboard: DashboardView | None = None,
        npc_agent_backend: AgentBackend | None = None,
        npc_llm_model: str | None = None,
        backing: Backing = Backing.PROCESS,
    ) -> None:
        self.pack = pack
        self.run_root = Path(run_root)
        self.run_root.mkdir(parents=True, exist_ok=True)
        self.dashboard = dashboard
        self.backing = backing
        if npc_agent_backend is not None and npc_llm_model is not None:
            raise EpisodeError(
                "EpisodeService: pass either 'npc_agent_backend' or "
                "'npc_llm_model', not both",
            )
        if npc_agent_backend is not None:
            self.npc_agent_backend: AgentBackend | None = npc_agent_backend
        elif npc_llm_model is not None:
            self.npc_agent_backend = StrandsAgentBackend(model=npc_llm_model)
        else:
            self.npc_agent_backend = None
        self._episodes: dict[str, _RunningEpisode] = {}
        # Backstop: if the caller's try/finally misses ``close()``
        # (KeyboardInterrupt mid-cleanup, uncaught exception, etc.)
        # this still tells live handles to stop so their subprocesses
        # don't get reparented to PID 1.
        atexit.register(_atexit_stop_episodes, weakref.ref(self))

    # -- lifecycle ----------------------------------------------------------

    def start_episode(
        self,
        snapshot: Snapshot,
        task_id: str | None = None,
    ) -> EpisodeHandle:
        """Realize ``snapshot.graph`` and prepare a running episode.

        The pack's :meth:`~openrange.core.contracts.Pack.realize`
        returns a handle, ``reset()`` boots the world, ``surface()``
        seeds the IO surface cached for ``observe`` / ``base_url`` /
        ``agent_root`` reads. NPCs are constructed from the manifest
        (if any) and started against the surface. Auto-tick mode is
        wired from the manifest too — if no manifest knobs are
        present, neither feature engages and the episode runs purely
        under the harness's explicit ``tick`` / ``advance`` calls.
        """
        task = _resolve_task(snapshot, task_id)
        if not task.entrypoints:
            raise EpisodeError(f"task {task.id!r} has no entrypoints")

        episode_id = uuid.uuid4().hex[:12]
        # First episode for this task in this run uses the bare task.id;
        # forks / restores / parallel episodes append the episode id.
        candidate = self.run_root / task.id
        episode_root = (
            candidate
            if not candidate.exists()
            else self.run_root / f"{task.id}-{episode_id}"
        )
        episode_root.mkdir(parents=True)

        runtime = self.pack.realize(snapshot.graph, self.backing)
        try:
            runtime.reset()
            surface_mapping = MappingProxyType(dict(runtime.surface()))
        except Exception:
            # Reset failed mid-flight: tear down whatever the pack
            # started so the test/harness sees a clean failure rather
            # than a leaked subprocess.
            with contextlib.suppress(Exception):  # best-effort cleanup
                runtime.stop()
            raise

        handle = EpisodeHandle(episode_id, snapshot.snapshot_id, task.id)
        running = _RunningEpisode(
            handle=handle,
            snapshot=snapshot,
            task=task,
            runtime=runtime,
            run_root=episode_root,
            surface_cache=surface_mapping,
            dashboard=self.dashboard,
        )
        self._episodes[handle.id] = running
        self._record_system(
            running,
            {"reset": True},
            state={"run_root": str(episode_root)},
        )
        self._record_system(
            running,
            {"start": "runtime"},
            observation=_observation_metadata(surface_mapping),
        )
        self._start_npcs(running)
        rate = _manifest_auto_tick_rate(snapshot)
        if rate is not None:
            self._start_auto_tick(running, rate)
        return handle

    def stop_episode(self, episode: EpisodeHandle) -> EpisodeReport:
        """Stop the runtime, run the success check, return the report.

        Idempotent only in the sense that a second call returns the
        cached report — the second call does not re-stop the handle
        (the handle's own ``stop()`` is the source of idempotency for
        the realized world).
        """
        running = self._require(episode)
        if running.episode_result is not None and running.stopped:
            return self._cached_report(running)
        self._stop_auto_tick(running)
        self._stop_npcs(running)
        # Drain any final events the world produced between the last
        # poll and the stop call. The dashboard timeline wants them
        # before the "finish" turn lands.
        self._drain_events(running)
        final_state: Mapping[str, Any] = MappingProxyType(
            dict(running.runtime.collect()),
        )
        running.final_state = final_state
        episode_result = self._check_success(running, final_state)
        running.episode_result = episode_result
        try:
            running.runtime.stop()
        except Exception as exc:  # noqa: BLE001
            # A failed stop must not mask the agent's result. Surface
            # the failure on the dashboard so a human can investigate.
            self._record_system(
                running,
                {"stop_error": type(exc).__name__},
                observation={"reason": str(exc)},
            )
        running.stopped = True
        self._record_system(
            running,
            {"finish": True},
            state=dict(final_state),
        )
        return self._cached_report(running)

    def check_episode(self, episode: EpisodeHandle) -> EpisodeReport:
        """Idempotent: returns the report from a stopped episode.

        If the episode is still live, runs :meth:`stop_episode` first.
        Useful as the canonical "give me the report" call when the
        harness doesn't track whether it has already stopped.
        """
        running = self._require(episode)
        if running.episode_result is None or not running.stopped:
            return self.stop_episode(episode)
        return self._cached_report(running)

    def surface(self, episode: EpisodeHandle) -> Mapping[str, Any]:
        """The pack-defined IO surface dict for this episode.

        Cached at start; not re-polled, since the surface is meant to
        be stable for the episode's life. A pack whose surface drifts
        (e.g. a port that changes) can call ``reset()`` mid-episode
        and the next ``observe`` reflects it.
        """
        return self._require(episode).surface_cache

    def base_url(self, episode: EpisodeHandle) -> str:
        """The HTTP base URL, when the surface declares one.

        Convenience for HTTP-backed packs (the cyber webapp's
        :class:`WebappRuntimeHandle` surfaces this key); raises if the
        surface omits it so a typo doesn't return an empty string.
        """
        surface = self._require(episode).surface_cache
        value = surface.get("base_url")
        if not isinstance(value, str):
            raise EpisodeError(
                f"episode {episode.id!r} surface does not expose 'base_url'",
            )
        return value

    def agent_root(self, episode: EpisodeHandle) -> Path:
        """The agent's working directory, when the surface declares one.

        Convenience for filesystem-backed packs that hand the agent a
        scratch dir. Raises if the surface omits the key.
        """
        surface = self._require(episode).surface_cache
        value = surface.get("agent_root")
        if not isinstance(value, (str, Path)):
            raise EpisodeError(
                f"episode {episode.id!r} surface does not expose 'agent_root'",
            )
        return Path(value)

    # -- agent / world flow -------------------------------------------------

    def observe(self, episode: EpisodeHandle) -> Observation:
        """Drain pending events and return them with surface metadata."""
        running = self._require(episode)
        events = self._drain_events(running)
        return Observation(
            visible_state=running.surface_cache,
            events=events,
            metadata=_observation_metadata(running.surface_cache),
        )

    def record_turn(self, episode: EpisodeHandle, turn: AgentTurn) -> None:
        """Note an agent turn — observational only.

        The agent's tool calls happen against the surface directly;
        this call lets the harness leave a breadcrumb on the
        dashboard timeline. The latest non-empty ``message`` ends up
        in :attr:`EpisodeReport.agent_summary`.
        """
        running = self._require(episode)
        if turn.message:
            running.agent_summary = turn.message

    def tick(
        self,
        episode: EpisodeHandle,
        request: TickRequest | None = None,
    ) -> TickResult:
        """One tick: drive NPCs, drain events, check terminal."""
        req = request or TickRequest()
        running = self._require(episode)
        if req.process_npcs:
            self._step_npcs(running)
        events = self._drain_events(running)
        done, reason = self._terminal_state(running)
        return TickResult(events=events, done=done, terminal_reason=reason)

    def advance(
        self,
        episode: EpisodeHandle,
        request: AdvanceRequest | None = None,
    ) -> EpisodeUpdate:
        """Tick up to :attr:`AdvanceRequest.max_ticks` times.

        Yields early on terminal (always) or on the first event burst
        when ``until`` is ``"observation"`` / ``"event"``. ``"terminal"``
        keeps ticking until the handle reports done. ``"idle"`` is
        reserved for a future quiescence-detection mode and currently
        behaves like ``"observation"``.
        """
        req = request or AdvanceRequest()
        running = self._require(episode)
        all_events: list[Mapping[str, Any]] = []
        for _ in range(req.max_ticks):
            events = self._drain_events(running)
            all_events.extend(events)
            done, reason = self._terminal_state(running)
            if done:
                return EpisodeUpdate(
                    observation=Observation(
                        visible_state=running.surface_cache,
                        events=tuple(events),
                    ),
                    events=tuple(all_events),
                    done=True,
                    terminal_reason=reason,
                )
            if req.until in ("observation", "event", "idle") and events:
                return EpisodeUpdate(
                    observation=Observation(
                        visible_state=running.surface_cache,
                        events=tuple(events),
                    ),
                    events=tuple(all_events),
                    done=False,
                )
        return EpisodeUpdate(
            events=tuple(all_events),
            done=False,
            terminal_reason="max_ticks",
        )

    # -- counterfactual support --------------------------------------------

    def checkpoint(self, episode: EpisodeHandle) -> EpisodeCheckpoint:
        """Capture an opaque pack-defined snapshot of episode state.

        The payload is whatever :meth:`RuntimeHandle.checkpoint`
        returned. Core never inspects it; the only thing it knows is
        the four identifier fields. The caller passes the checkpoint
        back to :meth:`restore` or stashes it for counterfactual
        comparison later.
        """
        running = self._require(episode)
        state = running.runtime.checkpoint()
        return EpisodeCheckpoint(
            id=uuid.uuid4().hex[:12],
            episode_id=episode.id,
            snapshot_id=running.snapshot.snapshot_id,
            task_id=running.task.id,
            state=state,
        )

    def restore(self, checkpoint: EpisodeCheckpoint) -> EpisodeHandle:
        """Spin up a fresh episode whose handle is restored from state.

        The original episode must still be registered with the
        service (so we can find its snapshot/task); the new episode
        gets a fresh runtime handle, fresh ``reset()``, then the
        opaque payload is replayed via :meth:`RuntimeHandle.restore`.
        Process-state semantics are the pack's call: cheap-checkpoint
        packs replay filesystem state, stateful-backing packs replay
        the in-process state machine.
        """
        running = self._episodes.get(checkpoint.episode_id)
        if running is None:
            raise EpisodeError(
                f"original episode {checkpoint.episode_id!r} not active",
            )
        new_handle = self.start_episode(running.snapshot, running.task.id)
        new_running = self._require(new_handle)
        try:
            new_running.runtime.restore(checkpoint.state)
        except Exception:
            # restore is the user-driven step; surface its failure but
            # leave the new episode running so the caller can decide
            # whether to stop it.
            self._record_system(
                new_running,
                {"restore_error": True},
                observation={"reason": "runtime.restore() raised"},
            )
            raise
        # Refresh the cached surface in case restore() rebound the
        # underlying transport (a checkpoint that re-spawns the
        # subprocess will hand back a different base_url).
        new_running.surface_cache = MappingProxyType(
            dict(new_running.runtime.surface()),
        )
        return new_handle

    def fork(self, episode: EpisodeHandle) -> EpisodeHandle:
        """Spin up a sibling episode from the current point.

        Equivalent to checkpoint+restore on a single line; differs
        only in not exposing a :class:`EpisodeCheckpoint` artifact to
        the caller.
        """
        checkpoint = self.checkpoint(episode)
        return self.restore(checkpoint)

    # -- internals ----------------------------------------------------------

    def _require(self, episode: EpisodeHandle) -> _RunningEpisode:
        running = self._episodes.get(episode.id)
        if running is None:
            raise EpisodeError(f"unknown episode {episode.id!r}")
        return running

    def _cached_report(self, running: _RunningEpisode) -> EpisodeReport:
        # ``running.episode_result`` is set by ``stop_episode`` before
        # ``stopped`` flips; callers reach this only via the typed
        # idempotency check above, so the assertion documents the
        # invariant rather than guards against a runtime path.
        assert running.episode_result is not None
        assert running.final_state is not None
        return EpisodeReport(
            snapshot_id=running.snapshot.snapshot_id,
            task_id=running.task.id,
            episode_result=running.episode_result,
            final_state=running.final_state,
            agent_summary=running.agent_summary,
        )

    def _check_success(
        self,
        running: _RunningEpisode,
        final_state: Mapping[str, Any],
    ) -> EpisodeResult:
        """Dispatch to ``pack.task_family(task.success_check)``.

        A task naming a family the pack does not declare is treated
        as a hard failure — the pack is the authority on what
        families it owns, and a missing one is a config bug, not a
        silently-zero result.
        """
        family = self.pack.task_family(running.task.success_check)
        if family is None:
            return EpisodeResult(
                success=False,
                reason=(
                    f"pack {self.pack.id!r} has no TaskFamily "
                    f"{running.task.success_check!r}"
                ),
            )
        return family.check_success(running.snapshot.graph, running.task, final_state)

    def _terminal_state(
        self,
        running: _RunningEpisode,
    ) -> tuple[bool, str | None]:
        if running.stopped:
            return True, "stopped"
        return running.runtime.terminal()

    def _drain_events(
        self,
        running: _RunningEpisode,
    ) -> tuple[Mapping[str, Any], ...]:
        try:
            events = running.runtime.poll_events()
        except Exception:  # noqa: BLE001
            # A broken poll mustn't sink the loop — the realizer logs
            # its own failures and the next poll may recover. Yield
            # no events for this tick.
            return ()
        for event in events:
            self._record_world_event(running, event)
        return tuple(events)

    def _record_system(
        self,
        running: _RunningEpisode,
        action: Mapping[str, object],
        *,
        observation: Mapping[str, object] | None = None,
        state: Mapping[str, object] | None = None,
    ) -> None:
        if running.dashboard is None:
            return
        running.dashboard.record_turn(
            ActorTurn(
                running.task.id,
                "runtime",
                "system",
                "environment",
                action,
                observation=observation,
                state=state,
            ),
        )

    def _record_world_event(
        self,
        running: _RunningEpisode,
        event: Mapping[str, Any],
    ) -> None:
        if running.dashboard is None:
            return
        # Events are pack-shaped — core can't know whether a key like
        # ``path`` is meaningful. Forward them as the action body so
        # the dashboard timeline carries them verbatim; the dashboard
        # is responsible for any pack-specific rendering.
        target = running.task.entrypoints[0] if running.task.entrypoints else "world"
        action = {str(k): v for k, v in event.items()}
        running.dashboard.record_turn(
            ActorTurn(
                running.task.id,
                "agent",
                "agent",
                target,
                action,
                metadata={"source": "runtime_event"},
            ),
        )

    def _start_npcs(self, running: _RunningEpisode) -> None:
        npc_entries = _manifest_npc_entries(running.snapshot)
        if not npc_entries:
            return
        # Manifest-shape errors (unknown type, malformed config) still
        # propagate from ``resolve_manifest_npcs`` — those are config
        # mistakes the operator needs to fix. Per-NPC SDK / preflight
        # failures are caught inside the NPC and surfaced via
        # ``broken_reason`` (recorded below as a dashboard event).
        npcs = resolve_manifest_npcs(npc_entries)
        if not npcs:
            return
        base_context: dict[str, Any] = {
            "episode_id": running.handle.id,
            "snapshot_id": running.snapshot.snapshot_id,
            "task_id": running.task.id,
        }
        # Surface keys flow into the NPC context so the pre-refactor
        # NPC contract keeps working: NPCs that bound to ``base_url``
        # / ``http_get`` etc. still find them.
        for key, value in running.surface_cache.items():
            base_context.setdefault(str(key), value)
        for npc in npcs:
            ctx = dict(base_context)
            ctx["record_action"] = self._make_npc_recorder(running, npc)
            if npc.requires_llm:
                ctx["agent_backend"] = self.npc_agent_backend
            npc.start(MappingProxyType(ctx))
            if npc.broken_reason is not None:
                self._record_npc_broken(running, npc)
        running.npcs = npcs

    def _step_npcs(self, running: _RunningEpisode) -> None:
        if not running.npcs:
            return
        interface = running.surface_cache
        for npc in running.npcs:
            already_broken = npc.broken_reason is not None
            try:
                npc.step(interface)
            except Exception:  # noqa: BLE001
                continue
            if not already_broken and npc.broken_reason is not None:
                self._record_npc_broken(running, npc)

    def _make_npc_recorder(
        self,
        running: _RunningEpisode,
        npc: NPC,
    ) -> Callable[..., None]:
        """Build the per-NPC ``record_action`` callable handed via context.

        Returns a closure tagged with the NPC's ``actor_id`` so events
        flow into the dashboard with consistent attribution. Errors
        (e.g. dashboard offline) are silent — recording is
        observational and must never sink an NPC tick.
        """

        def record(
            action: Mapping[str, object],
            *,
            target: str | None = None,
            observation: Mapping[str, object] | None = None,
        ) -> None:
            if running.dashboard is None:
                return
            try:
                running.dashboard.record_turn(
                    ActorTurn(
                        running.task.id,
                        npc.actor_id,
                        "npc",
                        target if target is not None else "world",
                        action,
                        observation=observation,
                    ),
                )
            except Exception:  # noqa: BLE001 — observational, never raise
                return

        return record

    def _record_npc_broken(self, running: _RunningEpisode, npc: NPC) -> None:
        """Surface an NPC's transition to broken on the dashboard."""
        self._record_system(
            running,
            {"npc_broken": type(npc).__name__},
            observation={"reason": npc.broken_reason or ""},
        )

    def _stop_npcs(self, running: _RunningEpisode) -> None:
        for npc in running.npcs:
            try:
                npc.stop()
            except Exception:  # noqa: BLE001
                continue
        running.npcs = []

    def _start_auto_tick(self, running: _RunningEpisode, rate_hz: float) -> None:
        running.tick_stop = threading.Event()
        running.tick_thread = threading.Thread(
            target=_auto_tick_loop,
            args=(self, running, rate_hz),
            daemon=True,
        )
        running.tick_thread.start()

    def _stop_auto_tick(self, running: _RunningEpisode) -> None:
        if running.tick_thread is None or running.tick_stop is None:
            return
        running.tick_stop.set()
        running.tick_thread.join(timeout=5)
        running.tick_thread = None
        running.tick_stop = None

    def close(self) -> None:
        """Stop all live episodes.

        Best-effort cleanup — every handle's ``stop()`` is wrapped so
        one pack misbehaving doesn't leave others running.
        """
        for running in list(self._episodes.values()):
            self._stop_auto_tick(running)
            self._stop_npcs(running)
            if not running.stopped:
                with contextlib.suppress(Exception):
                    running.runtime.stop()
                running.stopped = True
        self._episodes.clear()


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _resolve_task(snapshot: Snapshot, task_id: str | None) -> TaskSpec:
    """Find the task by id, defaulting to the first one in the snapshot.

    The new :class:`Snapshot` carries a flat ``tasks: tuple[TaskSpec,
    ...]`` — no helper method on the dataclass — so resolution lives
    here at the call site rather than on the snapshot itself.
    """
    if not snapshot.tasks:
        raise EpisodeError(
            f"snapshot {snapshot.snapshot_id!r} has no tasks",
        )
    if task_id is None:
        return snapshot.tasks[0]
    for task in snapshot.tasks:
        if task.id == task_id:
            return task
    raise EpisodeError(
        f"snapshot {snapshot.snapshot_id!r} has no task {task_id!r}",
    )


def _observation_metadata(surface: Mapping[str, Any]) -> Mapping[str, Any]:
    """Pull the stable, stringly-typed surface keys into observation metadata.

    The full surface (which may carry callables — ``http_get`` etc.)
    is not safe to ship through the dashboard JSON serializer.
    Stringly-typed keys (``base_url``, ``agent_root``) are; that's the
    slice we expose as observation metadata.
    """
    out: dict[str, Any] = {}
    for key in ("base_url", "agent_root"):
        value = surface.get(key)
        if isinstance(value, str):
            out[key] = value
        elif isinstance(value, Path):
            out[key] = str(value)
    return MappingProxyType(out)


def _manifest_mapping(snapshot: Snapshot) -> Mapping[str, Any]:
    """Pull ``snapshot.lineage["manifest"]`` if present and a mapping.

    The new :class:`Snapshot` stores the build manifest under
    ``lineage`` as a free-form dict — there is no typed manifest
    object anymore. NPCs and the runtime tick mode are read out
    defensively: missing/malformed shapes degrade to no-NPCs and
    no-auto-tick rather than raising, so a manifest that omits
    runtime knobs is just a quiet manifest.
    """
    manifest = snapshot.lineage.get("manifest")
    if isinstance(manifest, Mapping):
        return manifest
    return {}


def _manifest_npc_entries(snapshot: Snapshot) -> tuple[Mapping[str, Any], ...]:
    """Read the ``npc:`` list from the manifest, defensively.

    The list is whatever ``resolve_manifest_npcs`` accepts — each
    entry a mapping with ``type`` / ``count`` / ``config`` keys. A
    missing list, a non-list value, or non-mapping entries all
    degrade to ``()`` so the rest of the system can rely on the
    return shape.
    """
    raw = _manifest_mapping(snapshot).get("npc")
    if not isinstance(raw, (list, tuple)):
        return ()
    entries: list[Mapping[str, Any]] = []
    for item in raw:
        if isinstance(item, Mapping):
            entries.append(item)
    return tuple(entries)


def _manifest_auto_tick_rate(snapshot: Snapshot) -> float | None:
    """Read ``runtime.tick`` knobs from the manifest, returning the rate.

    Schema (mirrors the pre-refactor typed shape):

        runtime:
          tick:
            mode: "auto" | "manual"
            rate_hz: 4.0

    Returns ``None`` when the manifest omits runtime tick config or
    sets ``mode`` to anything other than ``auto``. ``rate_hz`` must
    parse as a positive float; non-positive values disable auto-tick
    silently so a typo doesn't spin a 0-second loop.
    """
    runtime_cfg = _manifest_mapping(snapshot).get("runtime")
    if not isinstance(runtime_cfg, Mapping):
        return None
    tick_cfg = runtime_cfg.get("tick")
    if not isinstance(tick_cfg, Mapping):
        return None
    mode = tick_cfg.get("mode")
    if mode != "auto":
        return None
    rate_raw = tick_cfg.get("rate_hz")
    if isinstance(rate_raw, (int, float)) and not isinstance(rate_raw, bool):
        rate = float(rate_raw)
        if rate > 0:
            return rate
    return None


def _atexit_stop_episodes(service_ref: weakref.ref[EpisodeService]) -> None:
    service = service_ref()
    if service is None:
        return
    for running in list(service._episodes.values()):
        if running.stopped:
            continue
        try:
            running.runtime.stop()
        except Exception:  # noqa: BLE001 — best-effort cleanup
            continue
        running.stopped = True


def _auto_tick_loop(
    service: EpisodeService,
    running: _RunningEpisode,
    rate_hz: float,
) -> None:
    if running.tick_stop is None:
        return
    interval = 1.0 / rate_hz
    while not running.tick_stop.wait(interval):
        try:
            service.tick(running.handle)
        except EpisodeError:
            return  # episode was stopped
