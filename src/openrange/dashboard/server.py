"""Dashboard server: serves the SPA + JSON/SSE routes."""

from __future__ import annotations

import json
from collections.abc import Mapping
from http import HTTPStatus  # ALLOWED_DOMAIN_LEAK: stdlib transport
from http.server import (  # ALLOWED_DOMAIN_LEAK: stdlib transport
    BaseHTTPRequestHandler,
    ThreadingHTTPServer,
)
from pathlib import Path
from typing import cast
from urllib.parse import parse_qs, urlsplit

from openrange.dashboard.runs import RunsRegistry
from openrange.dashboard.view import DashboardView

STATIC_ROOT = Path(__file__).parent / "static"

STATIC_CONTENT_TYPES: Mapping[str, str] = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
}


class DashboardHTTPServer(ThreadingHTTPServer):
    """Serves the dashboard SPA + JSON / SSE endpoints.

    Multi-run mode: pass ``runs`` (a ``RunsRegistry``).
    Single-run mode: pass ``view`` (a ``DashboardView``); the SPA still
    works but ``/api/runs`` returns one synthetic entry for it.
    """

    def __init__(
        self,
        server_address: tuple[str, int],
        view: DashboardView | None = None,
        *,
        runs: RunsRegistry | None = None,
    ) -> None:
        if view is None and runs is None:
            raise ValueError("DashboardHTTPServer needs `view` or `runs`")
        self.view = view
        self.runs = runs
        super().__init__(server_address, DashboardRequestHandler)

    def view_for(self, run_id: str | None) -> DashboardView | None:
        if self.runs is None:
            return self.view
        if run_id is None:
            run_id = self.runs.default_run_id()
        if run_id is None:
            return None
        return self.runs.view_for(run_id)


class DashboardRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        parsed = urlsplit(self.path)
        path = parsed.path
        if path == "/":
            self._write_static("index.html")
            return
        if path.startswith("/static/"):
            self._write_static(path[len("/static/") :])
            return
        if path == "/api/runs":
            self._write_json(self._runs_payload())
            return
        view = self._resolve_view(parsed.query)
        if view is None:
            self._write_json(
                {"error": "no runs available; runs-dir is empty"},
                HTTPStatus.NOT_FOUND,
            )
            return
        if path == "/api/events/stream":
            self._stream_events(view)
            return
        if path == "/api/narrate/stream":
            self._stream_narration(view)
            return
        routes = {
            "/api/briefing": view.briefing,
            "/api/actors": view.actors,
            "/api/topology": view.topology,
            "/api/lineage": view.lineage,
            "/api/state": view.state,
            "/api/inspect": view.inspect,
            "/api/narrate": view.narration,
        }
        route = routes.get(path)
        if route is None:
            self._write_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            return
        self._write_json(route())

    def do_POST(self) -> None:
        parsed = urlsplit(self.path)
        view = self._resolve_view(parsed.query)
        if view is None:
            self._write_json(
                {"error": "no runs available; runs-dir is empty"},
                HTTPStatus.NOT_FOUND,
            )
            return
        routes = {
            "/api/episode/reset": view.reset,
            "/api/episode/play": view.play,
            "/api/episode/pause": view.pause,
        }
        route = routes.get(parsed.path)
        if route is None:
            self._write_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            return
        self._write_json(route())

    @property
    def dashboard_server(self) -> DashboardHTTPServer:
        return cast(DashboardHTTPServer, self.server)

    def _resolve_view(self, query: str) -> DashboardView | None:
        params = parse_qs(query)
        values = params.get("run") or []
        run_id = values[0] if values else None
        return self.dashboard_server.view_for(run_id)

    def _runs_payload(self) -> dict[str, object]:
        server = self.dashboard_server
        registry = server.runs
        if registry is None:
            view = server.view
            if view is None:
                return {"runs": [], "default": None}
            # Single-run mode synthesizes a non-null id so the SPA
            # picks a default run and opens its SSE stream — without
            # a default the runs picker stalls in "no runs found".
            run_id: str
            if view.snapshot is not None:
                run_id = view.snapshot.snapshot_id
            else:
                stored = view._stored_section("topology")  # noqa: SLF001
                stored_id = stored.get("snapshot_id")
                run_id = stored_id if isinstance(stored_id, str) else "single"
            return {
                "runs": [
                    {"id": run_id, "path": "<embedded>", "modified": 0.0},
                ],
                "default": run_id,
            }
        runs = [record.as_dict() for record in registry.list_runs()]
        return {"runs": runs, "default": registry.default_run_id()}

    def _write_static(self, relative: str) -> None:
        target = (STATIC_ROOT / relative).resolve()
        if STATIC_ROOT.resolve() not in target.parents or not target.is_file():
            self._write_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            return
        content_type = STATIC_CONTENT_TYPES.get(
            target.suffix,
            "application/octet-stream",
        )
        body = target.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _write_json(
        self,
        data: object,
        status: HTTPStatus = HTTPStatus.OK,
    ) -> None:
        body = json.dumps(data, sort_keys=True).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _stream_events(self, view: DashboardView) -> None:
        self._write_sse_headers()
        for event in view.bridge.subscribe_sync():
            self._write_sse(event.as_dict(), event=event.type, event_id=event.id)

    def _stream_narration(self, view: DashboardView) -> None:
        self._write_sse_headers()
        for event in view.bridge.subscribe_sync():
            self._write_sse(
                view.narration(),
                event="narration",
                event_id=event.id,
            )

    def _write_sse_headers(self) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()

    def _write_sse(
        self,
        data: Mapping[str, object],
        *,
        event: str,
        event_id: str,
    ) -> None:
        body = (
            f"id: {event_id}\n"
            f"event: {event}\n"
            f"data: {json.dumps(data, sort_keys=True)}\n\n"
        ).encode()
        try:
            self.wfile.write(body)
            self.wfile.flush()
        except BrokenPipeError:  # pragma: no cover - depends on client timing.
            return

    def log_message(self, format: str, *args: object) -> None:
        return
