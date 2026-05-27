"""Tensorboard-style discovery of run directories under `runs_dir`."""

from __future__ import annotations

import threading
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from openrange.dashboard.view import DashboardView

DASHBOARD_EVENTS = "dashboard.events.jsonl"
DASHBOARD_STATE = "dashboard.json"


@dataclass(frozen=True, slots=True)
class RunRecord:
    """Light metadata for one discovered run directory."""

    id: str
    path: Path
    modified: float

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "path": str(self.path),
            "modified": self.modified,
        }


def discover_runs(runs_dir: Path) -> list[RunRecord]:
    # Missing or non-directory ``runs_dir`` returns []; caller surfaces that.
    if not runs_dir.exists() or not runs_dir.is_dir():
        return []
    records: list[RunRecord] = []
    for entry in runs_dir.iterdir():
        if not entry.is_dir():
            continue
        events = entry / DASHBOARD_EVENTS
        state = entry / DASHBOARD_STATE
        if not (events.exists() and state.exists()):
            continue
        records.append(
            RunRecord(id=entry.name, path=entry, modified=events.stat().st_mtime),
        )
    records.sort(key=lambda r: r.modified, reverse=True)
    return records


class RunsRegistry:
    """Cached views use ``tail=True`` so they poll the on-disk event log;
    otherwise the dashboard freezes on what was on disk at view-creation
    time."""

    def __init__(self, runs_dir: Path) -> None:
        self.runs_dir = Path(runs_dir)
        self._views: dict[str, DashboardView] = {}
        self._lock = threading.Lock()

    def list_runs(self) -> list[RunRecord]:
        return discover_runs(self.runs_dir)

    def view_for(self, run_id: str) -> DashboardView | None:
        run_path = self._safe_run_path(run_id)
        if run_path is None:
            return None
        events = run_path / DASHBOARD_EVENTS
        state = run_path / DASHBOARD_STATE
        if not (events.exists() and state.exists()):
            return None
        with self._lock:
            view = self._views.get(run_id)
            if view is not None:
                return view
            view = DashboardView(
                event_log_path=events,
                state_path=state,
                reset_artifacts=False,
                tail=True,
            )
            self._views[run_id] = view
            return view

    def _safe_run_path(self, run_id: str) -> Path | None:
        """Resolve ``run_id`` to a path constrained to ``runs_dir``.

        Guards against ``?run=../../etc`` style traversal: rejects ids
        with path separators or that resolve outside the runs root.
        """
        if not run_id or "/" in run_id or "\\" in run_id or run_id in {".", ".."}:
            return None
        candidate = (self.runs_dir / run_id).resolve()
        try:
            candidate.relative_to(self.runs_dir.resolve())
        except ValueError:
            return None
        return candidate

    def default_run_id(self) -> str | None:
        runs = self.list_runs()
        return runs[0].id if runs else None

    def close(self) -> None:
        with self._lock:
            views = list(self._views.values())
            self._views.clear()
        for view in views:
            view.close()

    def cached_view_ids(self) -> Iterable[str]:
        with self._lock:
            return tuple(self._views.keys())
