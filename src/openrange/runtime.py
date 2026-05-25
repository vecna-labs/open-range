"""User-facing runtime convenience layer.

``OpenRangeRun`` ties admit + episode + dashboard together for example
scripts and the CLI. The admission seam is now
:func:`openrange.core.admit_loop.admit`; the episode seam is the
``EpisodeService(pack, run_root, ...)`` constructor whose first
positional arg is the resolved Pack. This module is the wrapper around
both, not their owner.

The LLM seam moved into ``TaskFamily.generate()`` under the new pack
shape — there is no top-level ``prompt`` / ``llm`` kwarg on the build
call anymore. A pack that wants LLM-enriched task instructions reaches
its backend through its TaskFamily configuration instead.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from openrange.agent_backend import AgentBackend
from openrange.core._registry import iter_entry_points
from openrange.core.admit_loop import AdmissionFailure, Snapshot, admit
from openrange.core.contracts import Pack
from openrange.core.episode import EpisodeService
from openrange.core.errors import EpisodeRuntimeError, PackError
from openrange.core.pack import PACK_ENTRY_POINT_GROUP, PACKS
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
    """Convenience wrapper: admit + episode + optional dashboard."""

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
        """Admit ``manifest`` into a frozen :class:`Snapshot`.

        Resolves the pack from ``manifest["pack"]["id"]`` (or
        ``manifest["pack"]`` as a string fallback) through the global
        :data:`PACKS` registry, then dispatches to
        :func:`openrange.core.admit_loop.admit`. Raises
        :class:`EpisodeRuntimeError` if the manifest doesn't name a
        pack or the admission loop fails within the repair budget.
        """
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
        """Construct an :class:`EpisodeService` bound to ``snapshot``'s pack.

        The pack is resolved from ``snapshot.lineage["pack"]`` so a run
        can serve replayed snapshots it didn't build. The lineage key
        is set by :func:`admit` at freeze time; missing or unknown ids
        raise :class:`EpisodeRuntimeError`.
        """
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
    """Resolve a Pack from a manifest's ``pack`` field.

    Accepts the typical dict shape ``{"pack": {"id": "..."}}`` and the
    string shorthand ``{"pack": "..."}``. Raises
    :class:`EpisodeRuntimeError` if the manifest doesn't name a pack
    or the id is not registered.
    """
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
    return _resolve_pack_by_id(pack_id)


def _resolve_pack_from_snapshot(snapshot: Snapshot) -> Pack:
    """Resolve a Pack from ``snapshot.lineage["pack"]``."""
    pack_id = snapshot.lineage.get("pack")
    if not isinstance(pack_id, str) or not pack_id:
        raise EpisodeRuntimeError(
            f"snapshot {snapshot.snapshot_id!r} lineage missing 'pack' id",
        )
    return _resolve_pack_by_id(pack_id)


def _resolve_pack_by_id(pack_id: str) -> Pack:
    """Resolve a registered pack by id, falling back to entry-point load.

    Goes through :data:`PACKS` first because that's the canonical
    registry. The legacy ``PackRegistry.resolve`` rejects new-shape
    Pack instances on its ``isinstance(pack, core.pack.Pack)`` check,
    though, which is a known parallel-paths inconsistency Phase 4
    cleans up. As a fallback we walk
    :data:`PACK_ENTRY_POINT_GROUP` directly and instantiate the
    registered factory ourselves — packs ship via the same entry
    point either way, so this stays a single-source-of-truth lookup.
    """
    try:
        return cast(Pack, PACKS.resolve(pack_id))
    except PackError:
        pass
    except Exception as exc:
        raise EpisodeRuntimeError(f"unknown pack {pack_id!r}") from exc
    for name, value in iter_entry_points(
        PACK_ENTRY_POINT_GROUP,
        error_cls=PackError,
        kind="pack",
    ):
        if name != pack_id:
            continue
        pack = value() if callable(value) else value
        return cast(Pack, pack)
    raise EpisodeRuntimeError(f"unknown pack {pack_id!r}")
