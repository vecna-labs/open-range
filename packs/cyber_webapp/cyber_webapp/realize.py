"""WebappRuntime. Only ``Backing.PROCESS`` is wired."""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast
from urllib.error import URLError
from urllib.request import urlopen

from graphschema import WorldGraph
from openrange_pack_sdk import (
    Backing,
    OpenRangeError,
    SubprocessRuntime,
)

from cyber_webapp.codegen import _realize_graph
from cyber_webapp.codegen.entrypoint import (
    APP_FILE_NAME,
    REQUEST_LOG_NAME,
    RESULT_FILE_NAME,
)


class WebappRuntimeError(OpenRangeError):
    pass


class WebappRuntime(SubprocessRuntime):
    """SubprocessRuntime for the cyber webapp pack.

    Spawns the rendered ``app.py`` as a subprocess that serves HTTP, parses
    its startup line to get the bound ``host:port``, surfaces an
    ``http_get`` closure to the agent, and reads the request log on every
    poll / collect.
    """

    RESULT_FILE = RESULT_FILE_NAME

    def __init__(self, graph: WorldGraph, backing: Backing) -> None:
        if backing is not Backing.PROCESS:
            raise NotImplementedError(
                f"WebappRuntime does not yet support backing={backing!r}; "
                "only Backing.PROCESS is wired",
            )
        super().__init__(graph)
        # Render eagerly so a graph that breaks codegen fails at construction
        # (admission can re-raise) rather than inside an episode.
        self._files: dict[str, str] = _realize_graph(graph)
        self._base_url: str | None = None
        self._log_offset: int = 0

    def prepare_env_files(self, graph: WorldGraph) -> Mapping[str, str]:
        del graph
        return {**self._files, REQUEST_LOG_NAME: ""}

    def subprocess_command(
        self,
        env_root: Path,
        solver_root: Path,
    ) -> list[str]:
        del solver_root
        app_path = env_root / "pack" / APP_FILE_NAME
        if not app_path.exists():
            raise WebappRuntimeError(
                f"runtime artifact is missing: {app_path.name}",
            )
        return [
            sys.executable,
            str(app_path),
            "--host",
            "127.0.0.1",
            "--port",
            "0",
            "--log",
            str(env_root / "pack" / REQUEST_LOG_NAME),
        ]

    def parse_startup(self, stdout_line: str) -> Mapping[str, Any]:
        try:
            data = json.loads(stdout_line)
        except json.JSONDecodeError as exc:
            raise WebappRuntimeError(
                f"runtime reported invalid listening address: {stdout_line!r}",
            ) from exc
        if not isinstance(data, dict) or "host" not in data or "port" not in data:
            raise WebappRuntimeError(
                f"runtime reported invalid listening address: {data!r}",
            )
        self._base_url = f"http://{data['host']}:{data['port']}"
        self._log_offset = 0
        return {"base_url": self._base_url}

    def surface_extras(self) -> Mapping[str, Any]:
        base_url = self._base_url
        if base_url is None:
            return {}

        def http_get(path: object) -> bytes:
            return cast(bytes, urlopen(base_url + str(path), timeout=5).read())

        def http_get_json(path: object) -> object:
            return json.loads(http_get(path).decode())

        return {"http_get": http_get, "http_get_json": http_get_json}

    def poll_events(self) -> tuple[Mapping[str, Any], ...]:
        log = self._request_log_path()
        if log is None or not log.exists():
            return ()
        try:
            raw = log.read_bytes()
        except OSError:
            return ()
        new_bytes = raw[self._log_offset :]
        if not new_bytes:
            return ()
        # Only consume complete lines; a racy partial line gets picked up
        # on the next poll.
        last_newline = new_bytes.rfind(b"\n")
        if last_newline == -1:
            return ()
        consumed = last_newline + 1
        chunk = new_bytes[:consumed].decode("utf-8", errors="replace")
        self._log_offset += consumed
        events: list[Mapping[str, Any]] = []
        for line in chunk.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(data, Mapping):
                events.append(dict(data))
        return tuple(events)

    def collect_extras(self) -> Mapping[str, Any]:
        result = self._read_result()
        flag = ""
        if isinstance(result.get("flag"), str):
            flag = str(result["flag"])
        elif isinstance(result.get("flag_from_response"), str):
            # Agents write `flag`; families read `flag_from_response`.
            flag = str(result["flag_from_response"])
        requests = self._all_requests()
        requests_made = [str(row.get("path", "")) for row in requests if row]
        return {
            "flag_from_response": flag or None,
            "requests_made": requests_made,
            "endpoint_serves_200": self._probe_root_200(),
        }

    def checkpoint(self) -> Any:
        state = super().checkpoint()
        return {**state, "log_offset": self._log_offset}

    def restore(self, state: Any) -> None:
        if not isinstance(state, Mapping):
            raise WebappRuntimeError(
                f"restore() expected mapping, got {type(state).__name__}",
            )
        log_offset = state.get("log_offset", 0)
        if not isinstance(log_offset, int):
            raise WebappRuntimeError(
                "restore() payload is missing 'log_offset' (int)",
            )
        # The webapp's runtime state (HTTP server, in-memory DB) cannot
        # be checkpointed — restore re-runs reset() to bring up a fresh
        # process, then re-materializes the agent's workspace from the
        # snapshot.
        self.reset()
        super().restore(state)
        self._log_offset = log_offset

    def _request_log_path(self) -> Path | None:
        if self.pack_root is None:
            return None
        return self.pack_root / REQUEST_LOG_NAME

    def _all_requests(self) -> list[Mapping[str, Any]]:
        log = self._request_log_path()
        if log is None or not log.exists():
            return []
        rows: list[Mapping[str, Any]] = []
        try:
            raw = log.read_text(encoding="utf-8")
        except OSError:
            return []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(data, Mapping):
                rows.append(dict(data))
        return rows

    def _probe_root_200(self) -> bool:
        if self._base_url is None:
            return False
        try:
            with urlopen(self._base_url + "/", timeout=2) as resp:
                return bool(getattr(resp, "status", 0) == 200)
        except URLError, TimeoutError, OSError:
            return False
