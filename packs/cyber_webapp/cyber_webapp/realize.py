"""WebappRuntime (PROCESS backing) and ContainerWebappRuntime (CONTAINER backing)."""

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
from cyber_webapp.container import (
    ServiceImage,
    hardening_run_args,
    image_files,
    realize_services,
)

# Where the container's app writes its request log (the image CMD's --log path).
_CONTAINER_LOG_PATH = "/app/requests.jsonl"
_CONTAINER_PORT = "8000"


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
        raw = self._read_log_bytes()
        if raw is None:
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

    def _read_log_bytes(self) -> bytes | None:
        # The request log as raw bytes, or None if it isn't there yet. The seam the
        # CONTAINER backing overrides to read the log out of the running container.
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


# Container images are content-addressed by their build files, so episodes of the same
# world reuse one image instead of each rebuilding + deleting an identical one. Built
# images are swept at exit — a running container pins its image, so docker skips those.
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
    # Check docker each time, not a "built" memo: the image can be swept or pruned
    # between episodes, and a stale memo would `docker run` a missing one.
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


class ContainerWebappRuntime(WebappRuntime):
    """WebappRuntime that runs the world as a real Docker container.

    ``docker run`` (foreground) is the supervised child: the container's app prints the
    same startup line a local subprocess would, so the SubprocessRuntime handshake still
    works; the published host port is resolved with ``docker port`` (the app only sees
    its in-container port). The request log is read out of the running container, and
    the image sets ``OPENRANGE_REALFS`` so the file and shell surfaces are real.
    """

    def __init__(self, graph: WorldGraph, backing: Backing) -> None:
        if backing is not Backing.CONTAINER:
            raise NotImplementedError(
                f"ContainerWebappRuntime is the CONTAINER backing, got {backing!r}",
            )
        # WebappRuntime.__init__ guards PROCESS-only; the container runtime shares its
        # log/surface/collect logic but its own lifecycle, so init the subprocess base.
        SubprocessRuntime.__init__(self, graph)
        self._files: dict[str, str] = {}
        self._base_url: str | None = None
        self._log_offset = 0
        self._build_files = image_files(graph)
        self._tag = _content_tag(self._build_files)
        self._cname: str | None = None

    def prepare_env_files(self, graph: WorldGraph) -> Mapping[str, str]:
        del graph
        # The image carries app.py + seed.json; pack_root is just the build context.
        return dict(self._build_files)

    def subprocess_command(self, env_root: Path, solver_root: Path) -> list[str]:
        del solver_root
        _ensure_image(self._tag, str(env_root / "pack"))
        self._cname = f"openrange-cyber-{uuid.uuid4().hex[:12]}"
        return [
            "docker",
            "run",
            "--rm",
            "--name",
            self._cname,
            "-p",
            f"127.0.0.1:0:{_CONTAINER_PORT}",
            *hardening_run_args(),
            self._tag,
        ]

    def parse_startup(self, stdout_line: str) -> Mapping[str, Any]:
        # A startup line means the app is up (the readiness signal); it only knows its
        # in-container port, so resolve the published host port for the agent URL.
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
        # Also expose the world's container + in-container port so a harness can put an
        # agent sandbox on the same docker network and let it reach the target by alias
        # (over the wire, not via the host). Agent-invisible — the brief renders only
        # base_url. See the openrange-trl AgentSandbox integration.
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
        except Exception:  # noqa: BLE001  # pragma: no cover - container gone mid-poll
            return None
        # A non-zero rc before the first request just means the log isn't written yet.
        return done.stdout if done.returncode == 0 else b""

    def stop(self) -> None:
        super().stop()  # kills the docker-run child; --rm removes the container
        if self._cname is not None:
            subprocess.run(["docker", "rm", "-f", self._cname], capture_output=True)
        # The image is content-addressed and reused across episodes — not deleted here.


class NetworkedContainerWebappRuntime(ContainerWebappRuntime):
    """The CONTAINER backing for a *networked* world: one container per service on a
    real docker network. The public service is the foreground child (reused from
    ContainerWebappRuntime — it gives the agent's ``base_url``); the internal services
    run detached on the same network, reachable only by name and never published. So a
    flag in an internal service is reachable only by pivoting from the public service
    over the network — genuine network position, not a path lookup on one server.
    """

    def __init__(self, graph: WorldGraph, backing: Backing) -> None:
        super().__init__(graph, backing)
        self._services = realize_services(graph)
        publics = [s for s in self._services if s.exposure == "public"]
        self._public: ServiceImage = publics[0] if publics else self._services[0]
        self._internals = [s for s in self._services if s is not self._public]
        # The foreground child is the public service, not the whole-world image.
        self._build_files = self._public.build_files
        self._tag = _content_tag(self._build_files)
        self._network = f"openrange-net-{uuid.uuid4().hex[:12]}"
        self._network_created = False
        self._internal_runs: list[tuple[str, str]] = []  # (container name, image tag)

    def reset(self) -> None:
        # Network + internal services first, then the public service as the child.
        self._create_network()
        self._start_internals()
        super().reset()

    def subprocess_command(self, env_root: Path, solver_root: Path) -> list[str]:
        cmd = super().subprocess_command(env_root, solver_root)
        # Put the public container on the network so it can reach the internals by name,
        # and switch its SSRF handler to a real cross-network fetch (the public service
        # holds no flag — the secret can only come from the internal host).
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
            ["docker", "network", "create", self._network],
            check=True,
            capture_output=True,
            timeout=30,
        )
        self._network_created = True

    def _start_internals(self) -> None:
        if self._internal_runs:  # pragma: no cover - idempotent across resets
            return
        for service in self._internals:
            tag = _content_tag(
                service.build_files, prefix=f"openrange-cyber-{service.name}"
            )
            cname = f"{self._network}-{service.name}"
            self._build_service_image(tag, service.build_files)
            subprocess.run(
                [
                    "docker",
                    "run",
                    "-d",
                    "--name",
                    cname,
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
        # The internal service publishes no port, so probe it from inside the container
        # (the docker-exec latency itself paces the retries).
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
        # Aggregate the public child's log with every internal service's log, so the
        # verdict sees a leak wherever it happened (the internal service detects it
        # serving its own flag). collect() reads this fully; poll_events is disabled —
        # one offset can't track concatenated, independently-growing logs.
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
        # In practice every internal service has at least its readiness-probe request
        # logged, so this empty fallback is only the pre-reset / no-container state.
        return b"" if self._cname is not None else None  # pragma: no cover

    def poll_events(self) -> tuple[Mapping[str, Any], ...]:
        return ()  # verdict comes from collect()'s full aggregated read, not offsets

    def stop(self) -> None:
        super().stop()  # tears down the public child (its image is kept for reuse)
        for cname, _tag in self._internal_runs:
            subprocess.run(["docker", "rm", "-f", cname], capture_output=True)
        self._internal_runs.clear()
        if self._network_created:
            subprocess.run(
                ["docker", "network", "rm", self._network], capture_output=True
            )
            self._network_created = False
