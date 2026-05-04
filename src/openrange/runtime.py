"""User-facing runtime convenience layer.

``OpenRangeRun`` ties build + episode + dashboard together for example
scripts and the CLI. The episode primitives (subprocess spawning, log
parsing, file materialization, final-state assembly) live in
``openrange.core.runtime_helpers``; this module is the wrapper around
them, not their owner.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from openrange.agent_backend import AgentBackend
from openrange.core import (
    Manifest,
    Snapshot,
)
from openrange.core import (
    build as core_build,
)
from openrange.core.episode import EpisodeService
from openrange.core.runtime_helpers import EpisodeRuntimeError
from openrange.dashboard import (
    DashboardArtifactLog,
    DashboardHTTPServer,
    DashboardView,
)
from openrange.llm import LLMBackend

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
    # Backend handed to NPCs with ``requires_llm = True``. Unset →
    # those NPCs mark themselves broken with "no backend configured".
    npc_agent_backend: AgentBackend | None = None
    # Convenience shorthand — auto-promoted to
    # ``StrandsAgentBackend(model=npc_llm_model)``. Mutually exclusive
    # with ``npc_agent_backend``.
    npc_llm_model: str | None = None


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
    """Convenience wrapper: build + episode + optional dashboard."""

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
        manifest: str | Path | Mapping[str, object] | Manifest,
        *,
        prompt: str = "",
        llm: LLMBackend | None = None,
        max_repairs: int = 3,
    ) -> Snapshot:
        snapshot = core_build(
            manifest,
            prompt=prompt,
            llm=llm,
            max_repairs=max_repairs,
            event_sink=(
                None if self._dashboard is None else self._dashboard.record_builder_step
            ),
        )
        if self._dashboard is not None:
            self._dashboard.record_builder_step(
                "builder_finished",
                {"snapshot_id": snapshot.id, "task_count": len(snapshot.tasks)},
            )
        return snapshot

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
        view = self._ensure_dashboard_view(snapshot)
        return EpisodeService(
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
