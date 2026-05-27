"""WebappRuntimeHandle. Only `Backing.PROCESS` is wired."""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast
from urllib.error import URLError
from urllib.request import urlopen

from graphschema import WorldGraph

from cyber_webapp.codegen import _realize_graph
from cyber_webapp.codegen.entrypoint import (
    APP_FILE_NAME,
    REQUEST_LOG_NAME,
    RESULT_FILE_NAME,
)
from openrange.core.errors import OpenRangeError
from openrange.core.pack import Backing


class WebappRuntimeError(OpenRangeError):
    pass


class WebappRuntimeHandle:
    def __init__(self, graph: WorldGraph, backing: Backing) -> None:
        if backing is not Backing.PROCESS:
            raise NotImplementedError(
                f"WebappRuntimeHandle does not yet support backing={backing!r}; "
                "only Backing.PROCESS is wired",
            )
        self._graph = graph
        self._backing = backing
        # Render eagerly so a graph that breaks codegen fails at
        # construction (admission can re-raise) rather than inside an episode.
        self._files: dict[str, str] = _realize_graph(graph)
        self._env_root: Path | None = None
        self._agent_root: Path | None = None
        self._request_log: Path | None = None
        self._process: subprocess.Popen[str] | None = None
        self._base_url: str | None = None
        self._log_offset: int = 0
        self._checkpoint_dirs: list[Path] = []

    def reset(self) -> None:
        if self._process is not None:
            self.stop()
        env_root = Path(tempfile.mkdtemp(prefix="cyber-webapp-env-"))
        agent_root = env_root / "agent"
        agent_root.mkdir(parents=True, exist_ok=True)
        pack_root = env_root / "pack"
        pack_root.mkdir(parents=True, exist_ok=True)
        # Secret must never land on disk — app unlinks seed.json at startup.
        for relative_path, content in self._files.items():
            target = pack_root / relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        request_log = env_root / REQUEST_LOG_NAME
        # Touch so poll_events() before the first request returns ()
        # instead of raising on a missing file.
        request_log.write_text("", encoding="utf-8")

        app_path = pack_root / APP_FILE_NAME
        process = self._start_process(app_path, request_log)
        base_url = self._read_base_url(process)

        self._env_root = env_root
        self._agent_root = agent_root
        self._request_log = request_log
        self._process = process
        self._base_url = base_url
        self._log_offset = 0

    def stop(self) -> None:
        """SIGTERM the process group, SIGKILL after 2s. Idempotent.
        Cleans up the tempdir hierarchy (env_root + every checkpoint
        snapshot) so a long-running harness doesn't accumulate trees in
        ``/tmp``."""
        if self._process is not None:
            _stop_process(self._process)
            self._process = None
        for snapshot_dir in self._checkpoint_dirs:
            shutil.rmtree(snapshot_dir, ignore_errors=True)
        self._checkpoint_dirs = []
        if self._env_root is not None:
            shutil.rmtree(self._env_root, ignore_errors=True)
        self._env_root = None
        self._agent_root = None
        self._request_log = None
        self._base_url = None
        self._log_offset = 0

    def surface(self) -> Mapping[str, Any]:
        """Keys: `base_url`, `http_get`, `http_get_json`, `agent_root`."""
        if self._base_url is None or self._agent_root is None:
            raise WebappRuntimeError(
                "surface() called before reset() — no running webapp",
            )
        base_url = self._base_url

        def http_get(path: object) -> bytes:
            return cast(bytes, urlopen(base_url + str(path), timeout=5).read())

        def http_get_json(path: object) -> object:
            return json.loads(http_get(path).decode())

        return {
            "base_url": base_url,
            "http_get": http_get,
            "http_get_json": http_get_json,
            "agent_root": str(self._agent_root),
        }

    def poll_events(self) -> tuple[Mapping[str, Any], ...]:
        if self._request_log is None or not self._request_log.exists():
            return ()
        try:
            raw = self._request_log.read_bytes()
        except OSError:
            return ()
        new_bytes = raw[self._log_offset :]
        if not new_bytes:
            return ()
        # Only consume complete lines; a racy partial line gets picked
        # up on the next poll.
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

    def terminal(self) -> tuple[bool, str | None]:
        if self._agent_root is None:
            return (False, None)
        if (self._agent_root / RESULT_FILE_NAME).exists():
            return (True, "agent wrote result")
        return (False, None)

    def checkpoint(self) -> Any:
        """Snapshot the agent_root tree. The subprocess is not snapshotted;
        restore re-spawns it."""
        if self._agent_root is None:
            raise WebappRuntimeError("checkpoint() called before reset()")
        snapshot_dir = Path(
            tempfile.mkdtemp(prefix="cyber-webapp-ckpt-"),
        )
        shutil.copytree(self._agent_root, snapshot_dir / "agent", dirs_exist_ok=True)
        self._checkpoint_dirs.append(snapshot_dir)
        return {
            "log_offset": self._log_offset,
            "agent_root_snapshot": str(snapshot_dir),
        }

    def restore(self, state: Any) -> None:
        """Re-materialize agent_root from snapshot, rewind the log offset,
        restart the subprocess. Filesystem state is preserved; server
        state is not."""
        if not isinstance(state, Mapping):
            raise WebappRuntimeError(
                f"restore() expected a mapping payload, got {type(state).__name__}",
            )
        snapshot_path = state.get("agent_root_snapshot")
        log_offset = state.get("log_offset", 0)
        if not isinstance(snapshot_path, str) or not isinstance(log_offset, int):
            raise WebappRuntimeError(
                "restore() payload is missing 'agent_root_snapshot' (str) "
                "or 'log_offset' (int)",
            )
        self.reset()
        if self._agent_root is None:
            raise WebappRuntimeError(
                "restore() failed: reset() did not produce an agent_root",
            )
        snapshot_dir = Path(snapshot_path)
        agent_snapshot = snapshot_dir / "agent"
        if agent_snapshot.exists():
            for child in self._agent_root.iterdir():
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()
            shutil.copytree(agent_snapshot, self._agent_root, dirs_exist_ok=True)
        self._log_offset = log_offset

    def collect(self) -> Mapping[str, Any]:
        if self._agent_root is None:
            return {}
        result = self._read_result()
        flag = ""
        if isinstance(result.get("flag"), str):
            flag = str(result["flag"])
        elif isinstance(result.get("flag_from_response"), str):
            # Agents write `flag`; families read `flag_from_response`.
            flag = str(result["flag_from_response"])
        requests = self._all_requests()
        requests_made = [str(row.get("path", "")) for row in requests if row]
        endpoint_serves_200 = self._probe_root_200()
        return {
            "flag_from_response": flag or None,
            "requests_made": requests_made,
            "endpoint_serves_200": endpoint_serves_200,
            "agent_root": str(self._agent_root),
            "result": dict(result),
        }

    def _start_process(
        self,
        app_path: Path,
        request_log: Path,
    ) -> subprocess.Popen[str]:
        # `start_new_session=True` so a process-group SIGTERM (and
        # harness Ctrl+C) doesn't race-clean the runtime via the shared
        # foreground group.
        if not app_path.exists():
            raise WebappRuntimeError(
                f"runtime artifact is missing: {app_path.name}",
            )
        return subprocess.Popen(
            [
                sys.executable,
                str(app_path),
                "--host",
                "127.0.0.1",
                "--port",
                "0",
                "--log",
                str(request_log),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )

    def _read_base_url(self, process: subprocess.Popen[str]) -> str:
        if process.stdout is None:
            raise WebappRuntimeError("runtime stdout is not available")
        line = process.stdout.readline()
        if not line:
            _stop_process(process)
            raise WebappRuntimeError(
                "runtime did not report a listening address",
            )
        try:
            data = json.loads(line)
        except json.JSONDecodeError as exc:
            _stop_process(process)
            raise WebappRuntimeError(
                f"runtime reported invalid listening address: {line!r}",
            ) from exc
        if not isinstance(data, dict) or "host" not in data or "port" not in data:
            _stop_process(process)
            raise WebappRuntimeError(
                f"runtime reported invalid listening address: {data!r}",
            )
        return f"http://{data['host']}:{data['port']}"

    def _read_result(self) -> Mapping[str, Any]:
        if self._agent_root is None:
            return {}
        result_path = self._agent_root / RESULT_FILE_NAME
        if not result_path.exists():
            return {}
        try:
            data = json.loads(result_path.read_text(encoding="utf-8"))
        except OSError, json.JSONDecodeError:
            return {}
        if not isinstance(data, Mapping):
            return {}
        return dict(data)

    def _all_requests(self) -> list[Mapping[str, Any]]:
        if self._request_log is None or not self._request_log.exists():
            return []
        rows: list[Mapping[str, Any]] = []
        try:
            raw = self._request_log.read_text(encoding="utf-8")
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


def _stop_process(process: subprocess.Popen[str] | None) -> None:
    """Group-kill if the process owns its session; else `terminate()`
    (group-killing the caller's group would suicide the harness).
    SIGKILL after 2s."""
    if process is None or process.poll() is not None:
        return
    own_group = False
    pgid: int | None = None
    try:
        pgid = os.getpgid(process.pid)
        own_group = pgid != os.getpgid(0)
    except ProcessLookupError, OSError:
        pgid = None
    try:
        if own_group and pgid is not None:
            os.killpg(pgid, signal.SIGTERM)
        else:
            process.terminate()
    except ProcessLookupError, OSError:
        return
    try:
        process.wait(timeout=2)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        if own_group and pgid is not None:
            os.killpg(pgid, signal.SIGKILL)
        else:
            process.kill()
    except ProcessLookupError, OSError:
        return
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        return
