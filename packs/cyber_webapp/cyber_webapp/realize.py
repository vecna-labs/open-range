"""WebappRuntimeHandle — the cyber webapp pack's realizer entry point.

Replaces the pre-refactor split between core's ``HTTPBacking`` (process
lifecycle) and ``codegen.realize_graph`` (artifact bundle generation).
One class now owns the whole story: generate the Flask source from a
``WorldGraph``, materialize it to disk on ``reset()``, run the
subprocess, expose the agent-facing IO surface, drain side-effect
events, decide when the agent has finished, and collect the structured
final state the task families read.

The class implements the eight-method ``RuntimeHandle`` Protocol
declared in ``openrange.core.pack``:

  reset()        : materialize app source + start subprocess + open log
  surface()      : ``{base_url, http_get, http_get_json, agent_root}``
                   — the dict NPCs and the agent both consume
  poll_events()  : new request-log lines appended since last poll
  terminal()     : ``result.json`` written by the agent → terminal
  checkpoint()   : ``{log_offset, agent_root_snapshot}`` opaque payload
  restore(state) : reverse the checkpoint
  collect()      : structured final state
                   (``flag_from_response``, ``requests_made``,
                   ``endpoint_serves_200``)
  stop()         : SIGTERM the process group (SIGKILL after 2s)

Backings: only ``Backing.PROCESS`` is supported today. ``CONTAINER`` /
``SIMULATOR`` / ``HYBRID`` raise ``NotImplementedError`` — the
docker-compose isolation path and the per-pack simulator path are
deferred to a later phase.
"""

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

from cyber_webapp.codegen import _realize_graph
from cyber_webapp.codegen.entrypoint import (
    APP_FILE_NAME,
    REQUEST_LOG_NAME,
    RESULT_FILE_NAME,
)
from openrange.core.errors import OpenRangeError
from openrange.core.pack import Backing
from openrange.world_ir import WorldGraph


class WebappRuntimeError(OpenRangeError):
    """Raised when the webapp runtime cannot proceed."""


class WebappRuntimeHandle:
    """A running realized webapp world.

    Construction stages the rendered ``app.py`` + ``seed.json`` source
    in memory (deterministic from ``graph``); ``reset()`` writes them
    to disk and spawns the subprocess. The two-phase split lets the
    pack hand back a handle that names a backing choice without
    paying any I/O cost until the episode loop actually starts.

    Today only ``Backing.PROCESS`` is supported. Other backings raise
    ``NotImplementedError`` — adding container / simulator backings
    means swapping ``_start_process`` / ``_stop_process`` for their
    docker-compose / simulator equivalents while keeping the same
    eight-method surface.
    """

    def __init__(self, graph: WorldGraph, backing: Backing) -> None:
        if backing is not Backing.PROCESS:
            raise NotImplementedError(
                f"WebappRuntimeHandle does not yet support backing={backing!r}; "
                "only Backing.PROCESS is wired in this phase",
            )
        self._graph = graph
        self._backing = backing
        # Render eagerly so a graph that breaks codegen surfaces the
        # PackError at construction (admission can re-raise it as a
        # build-time failure) rather than at reset() inside an episode.
        self._files: dict[str, str] = _realize_graph(graph)
        self._env_root: Path | None = None
        self._agent_root: Path | None = None
        self._request_log: Path | None = None
        self._process: subprocess.Popen[str] | None = None
        self._base_url: str | None = None
        # poll_events offset (bytes into the request log) and last
        # checkpointable position. Kept distinct so a future restore()
        # can replay events the agent has already consumed if the
        # caller wants to.
        self._log_offset: int = 0

    # -- lifecycle ----------------------------------------------------------

    def reset(self) -> None:
        """Materialize app source and start the HTTP subprocess.

        Idempotent against repeated calls only insofar as the second
        call tears down whatever the first started — but the underlying
        filesystem layout is re-created from scratch, so any state the
        agent wrote between calls is lost. A caller wanting checkpoint
        semantics should use ``checkpoint()`` / ``restore()`` instead.
        """
        if self._process is not None:
            self.stop()
        env_root = Path(tempfile.mkdtemp(prefix="cyber-webapp-env-"))
        agent_root = env_root / "agent"
        agent_root.mkdir(parents=True, exist_ok=True)
        pack_root = env_root / "pack"
        pack_root.mkdir(parents=True, exist_ok=True)
        # Materialize the rendered files into the pack root. The
        # generated app reads ``seed.json`` from its own directory at
        # startup, then unlinks the file before serving — so the agent
        # never sees the secret on disk even though pack_root sits in
        # the same tempdir.
        for relative_path, content in self._files.items():
            target = pack_root / relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        request_log = env_root / REQUEST_LOG_NAME
        # Touch the log up front so poll_events() before the first HTTP
        # request returns () instead of raising on a missing file.
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
        """Terminate the runtime subprocess.

        Group-killing if the process owns its own session (it always
        does — ``start_new_session=True``); SIGTERM then SIGKILL after
        a 2s wait. Idempotent against an already-stopped process.
        """
        if self._process is not None:
            _stop_process(self._process)
            self._process = None

    # -- surface / observation ---------------------------------------------

    def surface(self) -> Mapping[str, Any]:
        """The agent-facing IO surface.

        Keys preserved verbatim from the pre-refactor ``HTTPBacking``
        interface so existing NPCs (which bind tools over
        ``base_url`` / ``http_get`` / ``http_get_json``) keep working
        unchanged. ``agent_root`` is the directory the agent writes
        ``result.json`` to — surfaced so the agent or harness can
        locate it without a side-channel.
        """
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
        """Drain new request-log lines since the last call.

        Tracks a byte offset so a partially-written final line in the
        log file (race against the running server) doesn't get
        consumed half-formed; the rest comes in next poll. Malformed
        JSON lines are skipped silently.
        """
        if self._request_log is None or not self._request_log.exists():
            return ()
        try:
            raw = self._request_log.read_bytes()
        except OSError:
            return ()
        new_bytes = raw[self._log_offset :]
        if not new_bytes:
            return ()
        # Only advance the offset by the bytes covering complete lines;
        # if the tail is a partial line (no terminating \n) keep it for
        # the next poll.
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
        """Has the agent written its ``result.json``?

        The cyber convention: the agent signals completion by writing
        a JSON file at ``agent_root/result.json``. Until that file
        exists the episode loop keeps polling.
        """
        if self._agent_root is None:
            return (False, None)
        if (self._agent_root / RESULT_FILE_NAME).exists():
            return (True, "agent wrote result")
        return (False, None)

    # -- checkpoint / restore ----------------------------------------------

    def checkpoint(self) -> Any:
        """Capture an opaque payload for counterfactual replay.

        Snapshots the agent_root tree to a sibling tempdir so a
        subsequent ``restore()`` can rewind it. The process itself is
        not snapshotted — restore re-spawns it. This is the
        cheap-checkpoint path the design ref names; a future
        stateful-backing checkpoint would pickle the process state.
        """
        if self._agent_root is None:
            raise WebappRuntimeError("checkpoint() called before reset()")
        snapshot_dir = Path(
            tempfile.mkdtemp(prefix="cyber-webapp-ckpt-"),
        )
        # Copy contents (dirs_exist_ok so the snapshot dir already
        # existing as a tempdir doesn't trip up copytree).
        shutil.copytree(self._agent_root, snapshot_dir / "agent", dirs_exist_ok=True)
        return {
            "log_offset": self._log_offset,
            "agent_root_snapshot": str(snapshot_dir),
        }

    def restore(self, state: Any) -> None:
        """Reverse a ``checkpoint()`` payload.

        Re-materializes the agent_root tree from the snapshot and
        rewinds the log offset. The HTTP subprocess is restarted —
        cheap-checkpoint semantics: the agent's filesystem state is
        preserved, server state is not.
        """
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
        # Re-spawn so the subprocess + log file start fresh.
        self.reset()
        if self._agent_root is None:
            raise WebappRuntimeError(
                "restore() failed: reset() did not produce an agent_root",
            )
        snapshot_dir = Path(snapshot_path)
        agent_snapshot = snapshot_dir / "agent"
        if agent_snapshot.exists():
            # Wipe the freshly-created agent_root and re-populate from
            # the snapshot. Using copytree(dirs_exist_ok=True) over the
            # existing empty dir is the simplest reliable path.
            for child in self._agent_root.iterdir():
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()
            shutil.copytree(agent_snapshot, self._agent_root, dirs_exist_ok=True)
        self._log_offset = log_offset

    # -- collection --------------------------------------------------------

    def collect(self) -> Mapping[str, Any]:
        """Assemble the final-state dict the task families read.

        The keys are pack convention; the families pick the ones they
        care about (pentest reads ``flag_from_response`` and
        ``requests_made``; build reads ``endpoint_serves_200``).
        """
        if self._agent_root is None:
            return {}
        result = self._read_result()
        flag = ""
        if isinstance(result.get("flag"), str):
            flag = str(result["flag"])
        elif isinstance(result.get("flag_from_response"), str):
            # Tolerate either key name — the agent's contract is the
            # result schema embedded in the task file (``flag``), but
            # families read ``flag_from_response``; accept both.
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

    # -- internals ----------------------------------------------------------

    def _start_process(
        self,
        app_path: Path,
        request_log: Path,
    ) -> subprocess.Popen[str]:
        """Spawn the generated app as a subprocess.

        Owns its own session via ``start_new_session=True`` so
        ``_stop_process`` can SIGTERM the whole group without
        affecting the harness, and so a Ctrl+C on the harness
        terminal doesn't race-clean the runtime via the shared
        foreground group.
        """
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
        """Parse the ``{"host", "port"}`` line the runtime prints at boot."""
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
        """Synchronously probe ``/`` to decide the build smoke-test signal.

        Used by the build family's ``check_success``. Returns False on
        any timeout / network error / non-200 — the strict signal is
        the point.
        """
        if self._base_url is None:
            return False
        try:
            with urlopen(self._base_url + "/", timeout=2) as resp:
                return bool(getattr(resp, "status", 0) == 200)
        except URLError, TimeoutError, OSError:
            return False


def _stop_process(process: subprocess.Popen[str] | None) -> None:
    """Terminate a runtime subprocess; group-kill if it owns its session.

    Falls back to ``process.terminate()`` for bare ``Popen`` instances
    that share the caller's pgid — group-killing those would terminate
    the caller. SIGKILL after 2s if still alive.

    Inlined here (rather than imported from
    ``openrange.core.runtime_helpers``) so the handle stays
    self-sufficient when Phase 4 deletes the runtime-helpers module.
    """
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
