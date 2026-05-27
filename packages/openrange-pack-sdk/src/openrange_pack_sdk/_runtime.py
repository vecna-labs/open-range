"""Reusable RuntimeHandle base classes for common pack-author patterns.

These are optional. Packs can implement the ``RuntimeHandle`` Protocol
directly. Use one of these when your runtime fits the pattern.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import tempfile
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from contextlib import suppress
from pathlib import Path
from typing import Any

from graphschema import WorldGraph

from openrange_pack_sdk._errors import OpenRangeError


class SubprocessRuntimeHandle(ABC):
    """RuntimeHandle scaffold for packs whose realized world is a child
    subprocess the agent interacts with.

    Domains this fits naturally: a webapp serving HTTP, a trading
    simulator exposing a broker API, an in-pack mock service. Common
    structure: spawn a process, optionally exchange a small startup
    descriptor (URL, port, fd), let the agent act, capture results from
    the agent's filesystem at the end.

    The base owns:

    * Tempdir lifecycle — ``env_root`` (entire scratch), ``agent_root``
      (the agent's workspace; the agent's signal of termination is
      writing ``result.json`` here), ``pack_root`` (where rendered files
      go).
    * Subprocess spawn with ``start_new_session=True`` so process-group
      signals reach the child without affecting the harness.
    * SIGTERM → grace period → SIGKILL on ``stop()``.
    * Filesystem checkpoint / restore of ``agent_root``.
    * Default ``terminal()`` = "agent wrote result.json".
    * Default ``collect()`` = ``{agent_root, result, ...collect_extras}``.

    Packs override (minimum):

    * ``prepare_env_files(graph)`` → ``{relative_path: contents}`` for
      ``pack_root`` (e.g., the codegen-rendered app source).
    * ``subprocess_command(env_root, agent_root)`` → the command to spawn.

    Packs override (as needed):

    * ``parse_startup(stdout_line)`` — extract a surface descriptor from
      the subprocess's first stdout line (e.g., ``{"base_url": ...}``).
    * ``subprocess_env()`` — environment variables for the child.
    * ``surface_extras()`` — extra keys the agent reads (callables, URLs).
    * ``poll_events()`` — per-tick event drain (default = no events).
    * ``collect_extras()`` — per-pack final-state keys.

    The subprocess's stdout is captured; nothing else is consumed beyond
    the startup line. Packs that need request logs or other side-channel
    state typically write to a file under ``env_root`` and read it in
    ``poll_events`` / ``collect_extras``.

    Contract: the spawned subprocess MUST emit at least one newline on
    stdout before the agent acts. ``reset()`` blocks on ``readline()`` to
    capture optional startup info; a process that never writes will hang
    the harness. Packs with no startup info to advertise should print a
    single ``\\n`` immediately.
    """

    RESULT_FILE = "result.json"
    GRACE_SECONDS = 2.0
    STARTUP_TIMEOUT_SECONDS: float = 30.0

    def __init__(self, graph: WorldGraph) -> None:
        self._graph = graph
        self._env_root: Path | None = None
        self._agent_root: Path | None = None
        self._pack_root: Path | None = None
        self._process: subprocess.Popen[str] | None = None
        self._startup_info: dict[str, Any] = {}
        self._checkpoint_dirs: list[Path] = []

    @property
    def env_root(self) -> Path | None:
        """The scratch directory for this episode; ``None`` before ``reset()``."""
        return self._env_root

    @property
    def agent_root(self) -> Path | None:
        """Where the agent writes its artifacts (incl. ``result.json``).
        ``None`` before ``reset()``."""
        return self._agent_root

    @property
    def pack_root(self) -> Path | None:
        """Where ``prepare_env_files`` is materialized. ``None`` before ``reset()``."""
        return self._pack_root

    @property
    def process(self) -> subprocess.Popen[str] | None:
        """The spawned subprocess; ``None`` before ``reset()`` / after ``stop()``."""
        return self._process

    @abstractmethod
    def prepare_env_files(self, graph: WorldGraph) -> Mapping[str, str]:
        """Return ``{relative_path: file_contents}`` written under ``pack_root``
        before the subprocess starts."""

    @abstractmethod
    def subprocess_command(
        self,
        env_root: Path,
        agent_root: Path,
    ) -> Sequence[str]:
        """The argv to ``subprocess.Popen``."""

    def subprocess_env(self) -> Mapping[str, str] | None:
        """Override to set the child's env. Default: inherit parent."""
        return None

    def parse_startup(self, stdout_line: str) -> Mapping[str, Any]:
        """Parse the subprocess's first stdout line into surface keys.

        Default: no startup exchange (returns ``{}``). Common override
        parses JSON: ``{"host": "...", "port": 12345}`` → ``{"base_url":
        f"http://{host}:{port}"}``.
        """
        del stdout_line
        return {}

    def surface_extras(self) -> Mapping[str, Any]:
        """Override to add keys to ``surface()`` (e.g., ``http_get``
        callables that close over ``base_url``)."""
        return {}

    def collect_extras(self) -> Mapping[str, Any]:
        """Override to add keys to ``collect()`` (e.g., parsed request
        logs, computed metrics)."""
        return {}

    def reset(self) -> None:
        self._teardown_process_and_env()
        env_root = Path(tempfile.mkdtemp(prefix=f"{self._tempdir_prefix()}-"))
        agent_root = env_root / "agent"
        agent_root.mkdir(parents=True, exist_ok=True)
        pack_root = env_root / "pack"
        pack_root.mkdir(parents=True, exist_ok=True)
        self._env_root = env_root
        self._agent_root = agent_root
        self._pack_root = pack_root
        for rel, content in self.prepare_env_files(self._graph).items():
            target = pack_root / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        self._process = self._spawn(env_root, agent_root)
        # _spawn always sets stdout=PIPE, so .stdout is non-None.
        assert self._process.stdout is not None
        first_line = _readline_with_timeout(self._process, self.STARTUP_TIMEOUT_SECONDS)
        if first_line:
            self._startup_info = dict(self.parse_startup(first_line))

    def stop(self) -> None:
        """Fully tear down: kill process, wipe env_root, drop all checkpoints.

        Use ``reset()`` between episodes if the caller still holds
        checkpoint references they may want to ``restore()`` — ``reset()``
        preserves them.
        """
        self._teardown_process_and_env()
        for ckpt in self._checkpoint_dirs:
            shutil.rmtree(ckpt, ignore_errors=True)
        self._checkpoint_dirs.clear()

    def _teardown_process_and_env(self) -> None:
        if self._process is not None:
            _terminate_process_group(self._process, self.GRACE_SECONDS)
            self._process = None
        if self._env_root is not None and self._env_root.exists():
            shutil.rmtree(self._env_root, ignore_errors=True)
        self._env_root = None
        self._agent_root = None
        self._pack_root = None
        self._startup_info = {}

    def surface(self) -> Mapping[str, Any]:
        if self._agent_root is None:
            raise OpenRangeError("surface() called before reset()")
        return {
            "agent_root": str(self._agent_root),
            **self._startup_info,
            **self.surface_extras(),
        }

    def poll_events(self) -> tuple[Mapping[str, Any], ...]:
        return ()

    def terminal(self) -> tuple[bool, str | None]:
        if self._agent_root is None:
            return False, None
        if (self._agent_root / self.RESULT_FILE).exists():
            return True, "agent wrote result"
        return False, None

    def checkpoint(self) -> Any:
        if self._agent_root is None:
            raise OpenRangeError("checkpoint() called before reset()")
        snap = Path(tempfile.mkdtemp(prefix=f"{self._tempdir_prefix()}-ckpt-"))
        shutil.copytree(self._agent_root, snap / "agent", dirs_exist_ok=True)
        self._checkpoint_dirs.append(snap)
        return {"agent_root_snapshot": str(snap)}

    def restore(self, state: Any) -> None:
        if not isinstance(state, Mapping):
            raise OpenRangeError(
                f"restore() expects a mapping, got {type(state).__name__}"
            )
        snap_path = state.get("agent_root_snapshot")
        if not isinstance(snap_path, str):
            raise OpenRangeError(
                "restore() payload missing 'agent_root_snapshot' (str)"
            )
        agent_snap = Path(snap_path) / "agent"
        if not agent_snap.exists():
            raise OpenRangeError(f"restore() snapshot missing: {agent_snap}")
        if self._agent_root is None:
            raise OpenRangeError("restore() called before reset()")
        for child in self._agent_root.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
        shutil.copytree(agent_snap, self._agent_root, dirs_exist_ok=True)

    def collect(self) -> Mapping[str, Any]:
        if self._agent_root is None:
            return {}
        result = self._read_result()
        return {
            "agent_root": str(self._agent_root),
            "result": dict(result),
            **self.collect_extras(),
        }

    def _read_result(self) -> Mapping[str, Any]:
        # Caller (`collect`) already gated on agent_root.
        assert self._agent_root is not None
        result_path = self._agent_root / self.RESULT_FILE
        if not result_path.exists():
            return {}
        try:
            data = json.loads(result_path.read_text(encoding="utf-8"))
        except OSError, json.JSONDecodeError:
            return {}
        return dict(data) if isinstance(data, Mapping) else {}

    def _spawn(
        self,
        env_root: Path,
        agent_root: Path,
    ) -> subprocess.Popen[str]:
        cmd = list(self.subprocess_command(env_root, agent_root))
        kwargs: dict[str, Any] = {
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "text": True,
            "start_new_session": True,
        }
        env = self.subprocess_env()
        if env is not None:
            kwargs["env"] = dict(env)
        return subprocess.Popen(cmd, **kwargs)

    def _tempdir_prefix(self) -> str:
        return type(self).__name__.lower()


def _terminate_process_group(
    process: subprocess.Popen[str],
    grace_seconds: float,
) -> None:
    """SIGTERM the process group; SIGKILL after the grace period.

    Relies on ``start_new_session=True`` in ``_spawn`` — that makes the
    child a session/process-group leader, so ``pgid == child.pid``.
    """
    if process.poll() is not None:
        return
    with suppress(ProcessLookupError, PermissionError):
        os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=grace_seconds)
        return
    except subprocess.TimeoutExpired:
        pass
    with suppress(ProcessLookupError, PermissionError):
        os.killpg(process.pid, signal.SIGKILL)
    with suppress(subprocess.TimeoutExpired):
        process.wait(timeout=grace_seconds)


def _readline_with_timeout(
    process: subprocess.Popen[str],
    timeout_seconds: float,
) -> str:
    """Read one line from ``process.stdout``, waiting up to ``timeout_seconds``.

    Returns ``""`` if the child exits without writing (EOF). Raises
    ``OpenRangeError`` if the child neither writes nor exits within the
    budget — that case means the pack's subprocess violated the
    "emit at least one newline before reset returns" contract.
    """
    import select

    assert process.stdout is not None
    fd = process.stdout.fileno()
    ready, _, _ = select.select([fd], [], [], timeout_seconds)
    if not ready:
        raise OpenRangeError(
            f"subprocess did not write a startup line within "
            f"{timeout_seconds:.1f}s; pack must emit a newline before "
            "reset() can return (see SubprocessRuntimeHandle docstring)"
        )
    return process.stdout.readline()
