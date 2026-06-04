"""User-facing convenience wrapper around admit + EpisodeService + dashboard."""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openrange_pack_sdk import AgentBackend, Pack, Snapshot, TaskSpec

from openrange.core.admit import AdmissionFailure, admit
from openrange.core.episode import AgentTurn, EpisodeError, EpisodeService
from openrange.core.errors import EpisodeRuntimeError as EpisodeRuntimeError
from openrange.core.pack import PACKS
from openrange.dashboard import (
    DashboardArtifactLog,
    DashboardHTTPServer,
    DashboardView,
)
from openrange.training import EpisodeRun


@dataclass(frozen=True, slots=True)
class RunConfig:
    root: Path
    dashboard: bool = True
    reset_dashboard: bool = True
    dashboard_host: str = "127.0.0.1"
    dashboard_port: int | None = None
    npc_agent_backend: AgentBackend | None = None
    npc_llm_model: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.root, Path):
            object.__setattr__(self, "root", Path(self.root))  # type: ignore[unreachable]


@dataclass(frozen=True, slots=True)
class DashboardServerHandle:
    server: DashboardHTTPServer
    thread: threading.Thread

    @property
    def url(self) -> str:
        host = str(self.server.server_address[0])
        return f"http://{host}:{self.server.server_address[1]}"

    def close(self) -> None:
        if self.server.view is not None:
            self.server.view.close()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)

    def __enter__(self) -> DashboardServerHandle:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


@dataclass(frozen=True, slots=True)
class EpisodeContext:
    """What a :data:`Solver` sees for one running episode: the resolved ``task``
    (whose ``instruction`` is the problem statement) and the pack-defined IO
    ``surface``, plus accessors for the two surface keys a solver usually wants —
    the editable working directory and the server URL. Both raise if the world
    doesn't declare them, so a solver asks only for what its world provides.
    """

    task: TaskSpec
    surface: Mapping[str, Any]

    @property
    def root(self) -> Path:
        value = self.surface.get("solver_root")
        if not isinstance(value, (str, Path)):
            raise EpisodeError("episode surface does not expose 'solver_root'")
        return Path(value)

    @property
    def base_url(self) -> str:
        value = self.surface.get("base_url")
        if not isinstance(value, str):
            raise EpisodeError("episode surface does not expose 'base_url'")
        return value


# Return-based, not handed the service: a solver returning its turn(s) is what
# lets ``run_episode`` own the start/record/stop/close lifecycle.
Solver = Callable[[EpisodeContext], "AgentTurn | Sequence[AgentTurn] | None"]


class OpenRangeRun:
    def __init__(self, config: str | Path | RunConfig) -> None:
        self.config = (
            config if isinstance(config, RunConfig) else RunConfig(Path(config))
        )
        if (
            self.config.npc_agent_backend is not None
            and self.config.npc_llm_model is not None
        ):
            raise ValueError(
                "RunConfig: pass either 'npc_agent_backend' or "
                "'npc_llm_model', not both",
            )
        self.root = self.config.root
        self.root.mkdir(parents=True, exist_ok=True)
        self._dashboard = (
            None
            if not self.config.dashboard
            else DashboardArtifactLog(
                self.root / "dashboard.events.jsonl",
                self.root / "dashboard.json",
                reset=self.config.reset_dashboard,
            )
        )
        self._dashboard_view: DashboardView | None = None

    def build(
        self,
        manifest: Mapping[str, Any],
        *,
        max_repairs: int = 2,
    ) -> Snapshot:
        pack = _resolve_pack(manifest)
        result = admit(pack, manifest, max_repairs=max_repairs)
        if isinstance(result, AdmissionFailure):
            raise EpisodeRuntimeError(
                f"admission failed after {result.attempts} attempt(s): "
                f"{len(result.issues)} error(s), "
                f"{len(result.infeasible_tasks)} infeasible task(s)",
            )
        if self._dashboard is not None:
            self._dashboard.record_builder_step(
                "builder_finished",
                {
                    "snapshot_id": result.snapshot_id,
                    "task_count": len(result.tasks),
                },
            )
        return result

    def _ensure_dashboard_view(self, snapshot: Snapshot) -> DashboardView | None:
        if not self.config.dashboard:
            return None
        if self._dashboard_view is None:
            self._dashboard_view = DashboardView(
                snapshot,
                event_log_path=self.root / "dashboard.events.jsonl",
                state_path=self.root / "dashboard.json",
                reset_artifacts=False,
            )
        return self._dashboard_view

    def episode_service(self, snapshot: Snapshot) -> EpisodeService:
        """Pack is resolved from `snapshot.lineage["pack"]` so replayed
        snapshots from another run still work."""
        pack = _resolve_pack_from_snapshot(snapshot)
        view = self._ensure_dashboard_view(snapshot)
        return EpisodeService(
            pack,
            self.root,
            dashboard=view,
            npc_agent_backend=self.config.npc_agent_backend,
            npc_llm_model=self.config.npc_llm_model,
        )

    def run_episode(
        self,
        snapshot: Snapshot,
        solver: Solver,
        *,
        task_id: str | None = None,
    ) -> EpisodeRun:
        """Run one episode end to end and return the graded result.

        Realizes the world, hands ``solver`` an :class:`EpisodeContext`, records
        the turn(s) it returns, grades at stop, and bundles the report with the
        turns as an :class:`~openrange.training.EpisodeRun` (``.trajectory`` /
        ``.reward`` / ``.success``). This is the
        ``episode_service → start_episode → record_turn → stop_episode → close``
        loop every harness would otherwise hand-roll.

        ``solver`` raising propagates (after the runtime is torn down); a solver
        that wants a failed backend *graded* — against whatever it left in the
        workspace — catches its own error and returns a turn instead of raising.
        """
        svc = self.episode_service(snapshot)
        try:
            handle = svc.start_episode(snapshot, task_id)
            task = next(t for t in snapshot.tasks if t.id == handle.task_id)
            turns = _normalize_turns(
                solver(EpisodeContext(task=task, surface=svc.surface(handle))),
            )
            for turn in turns:
                svc.record_turn(handle, turn)
            report = svc.stop_episode(handle)
        finally:
            svc.close()
        return EpisodeRun(report=report, turns=tuple(turns))

    def serve_dashboard(
        self,
        snapshot: Snapshot,
        host: str = "127.0.0.1",
        port: int = 0,
    ) -> DashboardServerHandle:
        view = self._ensure_dashboard_view(snapshot) or DashboardView(snapshot)
        server = DashboardHTTPServer((host, port), view)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        return DashboardServerHandle(server, thread)


def _normalize_turns(
    raw: AgentTurn | Sequence[AgentTurn] | None,
) -> list[AgentTurn]:
    if raw is None:
        return []
    if isinstance(raw, AgentTurn):
        return [raw]
    return list(raw)


def _resolve_pack(manifest: Mapping[str, Any]) -> Pack:
    pack_field = manifest.get("pack")
    if isinstance(pack_field, Mapping):
        pack_id = pack_field.get("id")
    elif isinstance(pack_field, str):
        pack_id = pack_field
    else:
        pack_id = None
    if not isinstance(pack_id, str) or not pack_id:
        raise EpisodeRuntimeError(
            "manifest must declare a pack via 'pack.id' or 'pack' (string)",
        )
    try:
        return PACKS.resolve(pack_id)
    except Exception as exc:
        raise EpisodeRuntimeError(f"unknown pack {pack_id!r}") from exc


def _resolve_pack_from_snapshot(snapshot: Snapshot) -> Pack:
    pack_id = snapshot.lineage.get("pack")
    if not isinstance(pack_id, str) or not pack_id:
        raise EpisodeRuntimeError(
            f"snapshot {snapshot.snapshot_id!r} lineage missing 'pack' id",
        )
    try:
        return PACKS.resolve(pack_id)
    except Exception as exc:
        raise EpisodeRuntimeError(f"unknown pack {pack_id!r}") from exc
