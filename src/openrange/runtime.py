"""User-facing convenience wrapper around admit + EpisodeService + dashboard."""

from __future__ import annotations

import threading
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openrange.agent_backend import AgentBackend
from openrange.core.admit import AdmissionFailure, Snapshot, admit
from openrange.core.episode import EpisodeService
from openrange.core.errors import EpisodeRuntimeError
from openrange.core.pack import PACKS, Pack
from openrange.dashboard import (
    DashboardArtifactLog,
    DashboardHTTPServer,
    DashboardView,
)

__all__ = [
    "DashboardServerHandle",
    "EpisodeRuntimeError",
    "OpenRangeRun",
    "RunConfig",
]


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
