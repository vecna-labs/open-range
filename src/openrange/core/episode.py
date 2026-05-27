"""EpisodeService — the agent harness's seam into running worlds."""

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

from openrange_pack_sdk import (
    NPC,
    AgentBackend,
    Backing,
    EpisodeResult,
    OpenRangeError,
    Pack,
    RuntimeHandle,
    Snapshot,
    TaskSpec,
)

from openrange.agent_backend import StrandsAgentBackend
from openrange.core.turn import ActorTurn
from openrange.npc import resolve_manifest_npcs

if TYPE_CHECKING:
    from openrange.dashboard import DashboardView


class EpisodeError(OpenRangeError):
    pass


@dataclass(frozen=True, slots=True)
class EpisodeHandle:
    id: str
    snapshot_id: str
    task_id: str


@dataclass(frozen=True, slots=True)
class Observation:
    """`events` from `RuntimeHandle.poll_events`; `visible_state` is the
    pack-defined static surface mapping."""

    visible_state: Mapping[str, Any] = field(default_factory=dict)
    events: tuple[Mapping[str, Any], ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AgentTurn:
    """Harness-supplied note. Observational only."""

    message: str | None = None
    tool_calls: tuple[Mapping[str, Any], ...] = ()
    tool_results: tuple[Mapping[str, Any], ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TickRequest:
    max_events: int | None = None
    process_npcs: bool = True
    process_timers: bool = True


@dataclass(frozen=True, slots=True)
class TickResult:
    events: tuple[Mapping[str, Any], ...] = ()
    done: bool = False
    terminal_reason: str | None = None


@dataclass(frozen=True, slots=True)
class AdvanceRequest:
    """`until` decides when the multi-tick loop yields. `"observation"`
    and `"event"` return at the first poll that produces events;
    `"terminal"` ticks until the handle reports terminal; `"idle"`
    behaves like `"observation"`."""

    until: Literal["observation", "event", "terminal", "idle"] = "observation"
    max_ticks: int = 16
    timeout_seconds: float | None = None


@dataclass(frozen=True, slots=True)
class EpisodeUpdate:
    observation: Observation | None = None
    events: tuple[Mapping[str, Any], ...] = ()
    done: bool = False
    terminal_reason: str | None = None


@dataclass(frozen=True, slots=True)
class EpisodeReport:
    """Terminal artifact from a stopped episode. Implements `EpisodeReportLike`."""

    snapshot_id: str
    task_id: str
    episode_result: EpisodeResult
    final_state: Mapping[str, Any] = field(default_factory=dict)
    agent_summary: str = ""

    @property
    def passed(self) -> bool:
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
    """Pack-defined opaque blob from `RuntimeHandle.checkpoint()`."""

    id: str
    episode_id: str
    snapshot_id: str
    task_id: str
    state: Any
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class _RunningEpisode:
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


class EpisodeService:
    """Owns running worlds against a fixed `Pack`. One per `Pack`.

    `npc_agent_backend` and `npc_llm_model` are mutually exclusive
    shorthands; both unset means LLM-backed NPCs mark themselves broken
    at start.
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
        # Cached reports for stopped episodes — populated by
        # ``stop_episode`` so ``check_episode`` keeps working after the
        # running entry is evicted.
        self._reports: dict[str, EpisodeReport] = {}
        # Backstop if a caller skips `close()` — keeps subprocesses
        # from reparenting to PID 1.
        atexit.register(_atexit_stop_episodes, weakref.ref(self))

    def start_episode(
        self,
        snapshot: Snapshot,
        task_id: str | None = None,
    ) -> EpisodeHandle:
        task = _resolve_task(snapshot, task_id)
        if not task.entrypoints:
            raise EpisodeError(f"task {task.id!r} has no entrypoints")

        episode_id = uuid.uuid4().hex[:12]
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
            with contextlib.suppress(Exception):
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
        A second call returns the cached report; does not re-stop. The
        running entry is evicted from ``_episodes`` once stopped so a
        long-running harness does not accumulate dead handles."""
        cached = self._reports.get(episode.id)
        if cached is not None and episode.id not in self._episodes:
            return cached
        running = self._require(episode)
        if running.episode_result is not None and running.stopped:
            report = self._cached_report(running)
            self._reports[episode.id] = report
            self._episodes.pop(episode.id, None)
            return report
        self._stop_auto_tick(running)
        self._stop_npcs(running)
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
            # A failed stop must not mask the agent's result.
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
        report = self._cached_report(running)
        self._reports[episode.id] = report
        self._episodes.pop(episode.id, None)
        return report

    def check_episode(self, episode: EpisodeHandle) -> EpisodeReport:
        """Return the report from a stopped episode, stopping first if live."""
        cached = self._reports.get(episode.id)
        if cached is not None:
            return cached
        running = self._require(episode)
        if running.episode_result is None or not running.stopped:
            return self.stop_episode(episode)
        return self._cached_report(running)

    def surface(self, episode: EpisodeHandle) -> Mapping[str, Any]:
        """The pack-defined IO surface dict for this episode."""
        return self._require(episode).surface_cache

    def base_url(self, episode: EpisodeHandle) -> str:
        """The base URL of the agent-facing IO surface, when one is declared."""
        surface = self._require(episode).surface_cache
        value = surface.get("base_url")
        if not isinstance(value, str):
            raise EpisodeError(
                f"episode {episode.id!r} surface does not expose 'base_url'",
            )
        return value

    def agent_root(self, episode: EpisodeHandle) -> Path:
        """The agent's working directory, when the surface declares one."""
        surface = self._require(episode).surface_cache
        value = surface.get("agent_root")
        if not isinstance(value, (str, Path)):
            raise EpisodeError(
                f"episode {episode.id!r} surface does not expose 'agent_root'",
            )
        return Path(value)

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
        """Observational breadcrumb. The latest non-empty `message` lands
        in `EpisodeReport.agent_summary`."""
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
        """Tick up to `request.max_ticks` times. Yields early on terminal,
        or on the first event burst when `until` is observation/event/idle."""
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

    def checkpoint(self, episode: EpisodeHandle) -> EpisodeCheckpoint:
        """Capture an opaque pack-defined snapshot of episode state."""
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
        """Start a fresh episode and `RuntimeHandle.restore` it from
        `checkpoint.state`. The originating episode must still be live.
        If ``runtime.restore`` raises, the freshly-started runtime is
        stopped and its ``_episodes`` entry is popped before re-raising —
        otherwise a failed restore would leak a subprocess + dict entry."""
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
            self._record_system(
                new_running,
                {"restore_error": True},
                observation={"reason": "runtime.restore() raised"},
            )
            self._stop_auto_tick(new_running)
            self._stop_npcs(new_running)
            with contextlib.suppress(Exception):
                new_running.runtime.stop()
            new_running.stopped = True
            self._episodes.pop(new_handle.id, None)
            raise
        # restore() may have rebound transport (e.g. new port).
        new_running.surface_cache = MappingProxyType(
            dict(new_running.runtime.surface()),
        )
        return new_handle

    def fork(self, episode: EpisodeHandle) -> EpisodeHandle:
        """Spin up a sibling episode from the current point."""
        checkpoint = self.checkpoint(episode)
        return self.restore(checkpoint)

    def _require(self, episode: EpisodeHandle) -> _RunningEpisode:
        running = self._episodes.get(episode.id)
        if running is None:
            raise EpisodeError(f"unknown episode {episode.id!r}")
        return running

    def _cached_report(self, running: _RunningEpisode) -> EpisodeReport:
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
        # Manifest-shape errors propagate; per-NPC failures land in
        # `broken_reason`.
        npcs = resolve_manifest_npcs(npc_entries)
        if not npcs:
            return
        base_context: dict[str, Any] = {
            "episode_id": running.handle.id,
            "snapshot_id": running.snapshot.snapshot_id,
            "task_id": running.task.id,
        }
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
        """Best-effort stop of all live episodes."""
        for running in list(self._episodes.values()):
            self._stop_auto_tick(running)
            self._stop_npcs(running)
            if not running.stopped:
                with contextlib.suppress(Exception):
                    running.runtime.stop()
                running.stopped = True
        self._episodes.clear()


def _resolve_task(snapshot: Snapshot, task_id: str | None) -> TaskSpec:
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
    # Surface may carry callables the dashboard JSON serializer can't handle.
    out: dict[str, Any] = {}
    for key in ("base_url", "agent_root"):
        value = surface.get(key)
        if isinstance(value, str):
            out[key] = value
        elif isinstance(value, Path):
            out[key] = str(value)
    return MappingProxyType(out)


def _manifest_mapping(snapshot: Snapshot) -> Mapping[str, Any]:
    manifest = snapshot.lineage.get("manifest")
    if isinstance(manifest, Mapping):
        return manifest
    return {}


def _manifest_npc_entries(snapshot: Snapshot) -> tuple[Mapping[str, Any], ...]:
    raw = _manifest_mapping(snapshot).get("npc")
    if not isinstance(raw, (list, tuple)):
        return ()
    entries: list[Mapping[str, Any]] = []
    for item in raw:
        if isinstance(item, Mapping):
            entries.append(item)
    return tuple(entries)


def _manifest_auto_tick_rate(snapshot: Snapshot) -> float | None:
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


def _atexit_stop_episodes(svc_ref: weakref.ref[EpisodeService]) -> None:
    svc = svc_ref()
    if svc is None:
        return
    for running in list(svc._episodes.values()):
        if running.stopped:
            continue
        try:
            running.runtime.stop()
        except Exception:  # noqa: BLE001 — best-effort cleanup
            continue
        running.stopped = True


def _auto_tick_loop(
    svc: EpisodeService,
    running: _RunningEpisode,
    rate_hz: float,
) -> None:
    if running.tick_stop is None:
        return
    interval = 1.0 / rate_hz
    while not running.tick_stop.wait(interval):
        try:
            svc.tick(running.handle)
        except EpisodeError:
            return
