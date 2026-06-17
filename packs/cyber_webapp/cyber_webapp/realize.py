"""Webapp world runtimes: PROCESS (a local subprocess) and CONTAINER (docker)."""

from __future__ import annotations

import atexit
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast
from urllib.error import URLError
from urllib.request import urlopen

from graphschema import WorldGraph
from openrange_pack_sdk import (
    OpenRangeError,
    SubprocessRuntime,
)

from cyber_webapp.codegen import _realize_graph
from cyber_webapp.codegen.entrypoint import (
    APP_FILE_NAME,
    REQUEST_LOG_NAME,
    RESULT_FILE_NAME,
)
from cyber_webapp.container import (
    ServiceImage,
    hardening_run_args,
    has_write_exec_surface,
    image_files,
    realize_services,
)

_CONTAINER_LOG_PATH = "/app/requests.jsonl"
_CONTAINER_PORT = "8000"


class WebappRuntimeError(OpenRangeError):
    pass


class _WebappRuntime(SubprocessRuntime):
    """Shared webapp-runtime logic over a JSON request log.

    Handles the HTTP surface, log-driven events, the graded collect, and
    checkpoint/restore. Subclasses supply the world (``subprocess_command`` /
    ``parse_startup`` / ``prepare_env_files``) and where its log lives
    (``_read_log_bytes``, defaulting to the local request-log file).
    """

    RESULT_FILE = RESULT_FILE_NAME

    def __init__(self, graph: WorldGraph) -> None:
        super().__init__(graph)
        self._base_url: str | None = None
        self._log_offset: int = 0

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
        raw = self._read_log_bytes()
        if raw is None:
            return ()
        new_bytes = raw[self._log_offset :]
        if not new_bytes:
            return ()
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
            flag = str(result["flag_from_response"])
        requests = self._all_requests()
        requests_made = [str(row.get("path", "")) for row in requests if row]
        leaked: set[str] = set()
        for row in requests:
            values = row.get("leaked")
            if isinstance(values, list):
                leaked.update(str(v) for v in values)
        return {
            "flag_from_response": flag or None,
            "requests_made": requests_made,
            "leaked_secret_ids": sorted(leaked),
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
        self.reset()
        super().restore(state)
        self._log_offset = log_offset

    def poolable(self) -> bool:
        return not has_write_exec_surface(self._graph)

    def reset_episode(self) -> None:
        self._clear_request_logs()
        self._log_offset = 0
        if self._solver_root is not None:
            (self._solver_root / self.RESULT_FILE).unlink(missing_ok=True)

    def _clear_request_logs(self) -> None:
        log = self._request_log_path()
        if log is not None and log.exists():
            log.write_bytes(b"")

    def _request_log_path(self) -> Path | None:
        if self.pack_root is None:
            return None
        return self.pack_root / REQUEST_LOG_NAME

    def _read_log_bytes(self) -> bytes | None:
        log = self._request_log_path()
        if log is None or not log.exists():
            return None
        try:
            return log.read_bytes()
        except OSError:
            return None

    def _all_requests(self) -> list[Mapping[str, Any]]:
        raw_bytes = self._read_log_bytes()
        if raw_bytes is None:
            return []
        rows: list[Mapping[str, Any]] = []
        raw = raw_bytes.decode("utf-8", errors="replace")
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


class WebappRuntime(_WebappRuntime):
    """PROCESS backing: the rendered ``app.py`` runs as a local subprocess."""

    def __init__(self, graph: WorldGraph) -> None:
        super().__init__(graph)
        self._files: dict[str, str] = _realize_graph(graph)

    def prepare_env_files(self, graph: WorldGraph) -> Mapping[str, str]:
        del graph
        return {**self._files, REQUEST_LOG_NAME: ""}

    def subprocess_command(self, env_root: Path, solver_root: Path) -> list[str]:
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


_IMAGE_LABEL = "openrange.cyber.image=1"
_built_tags: set[str] = set()
_sweep_registered = False


def _content_tag(
    build_files: Mapping[str, str], *, prefix: str = "openrange-cyber"
) -> str:
    digest = hashlib.sha256(
        json.dumps(build_files, sort_keys=True).encode()
    ).hexdigest()
    return f"{prefix}:{digest[:16]}"


def _image_present(tag: str) -> bool:
    probe = subprocess.run(["docker", "image", "inspect", tag], capture_output=True)
    return probe.returncode == 0


def _ensure_image(tag: str, context: str) -> None:
    if _image_present(tag):
        return
    subprocess.run(
        ["docker", "build", "-q", "--label", _IMAGE_LABEL, "-t", tag, context],
        check=True,
        capture_output=True,
        timeout=600,
    )
    _built_tags.add(tag)
    global _sweep_registered
    if not _sweep_registered:
        atexit.register(_sweep_built_images)
        _sweep_registered = True


def _sweep_built_images() -> None:
    for tag in _built_tags:
        subprocess.run(["docker", "rmi", "-f", tag], capture_output=True)


_WORLD_LABEL = "openrange.cyber.world=1"
_world_containers: set[str] = set()
_world_networks: set[str] = set()
_world_sweep_registered = False


def _register_world_sweep() -> None:
    global _world_sweep_registered
    if not _world_sweep_registered:
        atexit.register(_sweep_world_resources)
        _world_sweep_registered = True


def _track_world_container(name: str) -> None:
    _world_containers.add(name)
    _register_world_sweep()


def _track_world_network(name: str) -> None:
    _world_networks.add(name)
    _register_world_sweep()


def _sweep_world_resources() -> None:
    # A SIGKILL leaks past atexit; the openrange.cyber.world label lets an
    # external prune reclaim what this leaves behind.
    for name in list(_world_containers):
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)
    for net in list(_world_networks):
        subprocess.run(["docker", "network", "rm", net], capture_output=True)


def _truncate_container_log(cname: str | None) -> None:
    if cname is None:
        return
    done = subprocess.run(
        ["docker", "exec", cname, "sh", "-c", f": > {_CONTAINER_LOG_PATH}"],
        capture_output=True,
        timeout=10,
        check=False,
    )
    if done.returncode != 0:
        raise WebappRuntimeError(f"warm world container {cname} is not reusable")


class ContainerWebappRuntime(_WebappRuntime):
    """CONTAINER backing: the world runs as a real docker container.

    ``docker run`` (foreground) is the supervised child — the container prints the
    same startup line a subprocess would; the published host port is read with
    ``docker port`` and the request log out of the running container. The image sets
    ``OPENRANGE_REALFS`` so the file and shell surfaces are real.
    """

    def __init__(self, graph: WorldGraph) -> None:
        super().__init__(graph)
        self._build_files = image_files(graph)
        self._tag = _content_tag(self._build_files)
        self._cname: str | None = None

    def prepare_env_files(self, graph: WorldGraph) -> Mapping[str, str]:
        del graph
        return dict(self._build_files)

    def subprocess_command(self, env_root: Path, solver_root: Path) -> list[str]:
        del solver_root
        _ensure_image(self._tag, str(env_root / "pack"))
        self._cname = f"openrange-cyber-{uuid.uuid4().hex[:12]}"
        _track_world_container(self._cname)
        return [
            "docker",
            "run",
            "--rm",
            "--name",
            self._cname,
            "--label",
            _WORLD_LABEL,
            "-p",
            f"127.0.0.1:0:{_CONTAINER_PORT}",
            *hardening_run_args(),
            self._tag,
        ]

    def parse_startup(self, stdout_line: str) -> Mapping[str, Any]:
        del stdout_line
        mapping = subprocess.run(
            ["docker", "port", str(self._cname), _CONTAINER_PORT],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        host_port = mapping.splitlines()[0].rsplit(":", 1)[-1]
        self._base_url = f"http://127.0.0.1:{host_port}"
        self._log_offset = 0
        return {
            "base_url": self._base_url,
            "target_container": self._cname,
            "target_port": _CONTAINER_PORT,
        }

    def _read_log_bytes(self) -> bytes | None:
        if self._cname is None:
            return None
        try:
            done = subprocess.run(
                ["docker", "exec", self._cname, "cat", _CONTAINER_LOG_PATH],
                capture_output=True,
                timeout=10,
                check=False,
            )
        except Exception:  # noqa: BLE001  # pragma: no cover
            return None
        return done.stdout if done.returncode == 0 else b""

    def _clear_request_logs(self) -> None:
        _truncate_container_log(self._cname)

    def stop(self) -> None:
        super().stop()
        if self._cname is not None:
            subprocess.run(["docker", "rm", "-f", self._cname], capture_output=True)
            _world_containers.discard(self._cname)


class NetworkedContainerWebappRuntime(ContainerWebappRuntime):
    """CONTAINER backing for a *networked* world: one container per service on a real
    docker network. The public service is the foreground child; internal services run
    detached, reachable only by name — so a flag in an internal service is reached only
    by pivoting over the network, not by a path lookup on one server.
    """

    def __init__(self, graph: WorldGraph) -> None:
        super().__init__(graph)
        self._services = realize_services(graph)
        publics = [s for s in self._services if s.exposure == "public"]
        self._public: ServiceImage = publics[0] if publics else self._services[0]
        self._internals = [s for s in self._services if s is not self._public]
        self._build_files = self._public.build_files
        self._tag = _content_tag(self._build_files)
        self._network = f"openrange-net-{uuid.uuid4().hex[:12]}"
        self._network_created = False
        self._internal_runs: list[tuple[str, str]] = []

    def reset(self) -> None:
        self._create_network()
        self._start_internals()
        super().reset()

    def subprocess_command(self, env_root: Path, solver_root: Path) -> list[str]:
        cmd = super().subprocess_command(env_root, solver_root)
        insert = cmd.index("run") + 1
        cmd[insert:insert] = [
            "--network",
            self._network,
            "--network-alias",
            self._public.name,
            "-e",
            "OPENRANGE_NETWORKED=1",
        ]
        return cmd

    def _create_network(self) -> None:
        if self._network_created:  # pragma: no cover - idempotent across resets
            return
        subprocess.run(
            ["docker", "network", "create", "--label", _WORLD_LABEL, self._network],
            check=True,
            capture_output=True,
            timeout=30,
        )
        self._network_created = True
        _track_world_network(self._network)

    def _start_internals(self) -> None:
        if self._internal_runs:  # pragma: no cover - idempotent across resets
            return
        for service in self._internals:
            tag = _content_tag(
                service.build_files, prefix=f"openrange-cyber-{service.name}"
            )
            cname = f"{self._network}-{service.name}"
            self._build_service_image(tag, service.build_files)
            _track_world_container(cname)
            subprocess.run(
                [
                    "docker",
                    "run",
                    "-d",
                    "--name",
                    cname,
                    "--label",
                    _WORLD_LABEL,
                    "--network",
                    self._network,
                    "--network-alias",
                    service.name,
                    *hardening_run_args(),
                    tag,
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=60,
            )
            self._internal_runs.append((cname, tag))
            self._wait_ready_in_container(cname)

    @staticmethod
    def _build_service_image(tag: str, build_files: dict[str, str]) -> None:
        context = Path(tempfile.mkdtemp(prefix="openrange-svc-"))
        try:
            for name, content in build_files.items():
                (context / name).write_text(content, encoding="utf-8")
            _ensure_image(tag, str(context))
        finally:
            shutil.rmtree(context, ignore_errors=True)

    @staticmethod
    def _wait_ready_in_container(cname: str) -> None:
        probe = (
            "import urllib.request as u; u.urlopen('http://localhost:8000/', timeout=2)"
        )
        for _ in range(40):
            done = subprocess.run(
                ["docker", "exec", cname, "python", "-c", probe],
                capture_output=True,
                timeout=10,
                check=False,
            )
            if done.returncode == 0:
                return
        raise WebappRuntimeError(  # pragma: no cover - an internal that never starts
            f"internal service {cname} did not become ready"
        )

    def _read_log_bytes(self) -> bytes | None:
        chunks: list[bytes] = []
        public = super()._read_log_bytes()
        if public:
            chunks.append(public)
        for cname, _tag in self._internal_runs:
            done = subprocess.run(
                ["docker", "exec", cname, "cat", _CONTAINER_LOG_PATH],
                capture_output=True,
                timeout=10,
                check=False,
            )
            if done.returncode == 0 and done.stdout:
                chunks.append(done.stdout)
        if chunks:
            return b"".join(chunks)
        return b"" if self._cname is not None else None  # pragma: no cover

    def poll_events(self) -> tuple[Mapping[str, Any], ...]:
        return ()

    def _clear_request_logs(self) -> None:
        super()._clear_request_logs()
        for cname, _tag in self._internal_runs:
            _truncate_container_log(cname)

    def stop(self) -> None:
        super().stop()
        for cname, _tag in self._internal_runs:
            subprocess.run(["docker", "rm", "-f", cname], capture_output=True)
            _world_containers.discard(cname)
        self._internal_runs.clear()
        if self._network_created:
            subprocess.run(
                ["docker", "network", "rm", self._network], capture_output=True
            )
            self._network_created = False
            _world_networks.discard(self._network)
