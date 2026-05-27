"""Reusable runtime base classes for common pack-author patterns.

These are optional. Packs can implement the ``RuntimeHandle`` Protocol
directly. Use one of these when your runtime fits the pattern.

Two siblings share a filesystem-lifecycle base (``env_root`` /
``agent_root`` / ``pack_root``, file-snapshot checkpoint/restore,
``result.json`` terminal signal):

* :class:`SubprocessRuntime` — spawns and supervises a
  long-running child the agent interacts with (e.g. a webapp, a
  simulator). Owns the SIGTERM→SIGKILL grace period and the startup
  handshake.

* :class:`OnDemandRuntime` — no persistent child. Agent acts on
  files; the pack exposes on-demand callables (e.g. ``run_tests``) via
  ``surface_extras``. Fits SWE-style packs where the world is a code
  workspace.
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
from openrange_pack_sdk._helpers import write_tree


class _FilesystemRuntime(ABC):
    """Shared lifecycle base for filesystem-backed RuntimeHandles.

    Owns the tempdir trio (``env_root``, ``agent_root``, ``pack_root``;
    each ``None`` before the first ``reset()`` and after ``stop()``),
    file-snapshot ``checkpoint`` / ``restore`` of the agent's workspace,
    ``terminal()`` via ``result.json``, and the default ``collect()``
    shape. Packs subclass one of the public siblings, not this directly.
    """

    RESULT_FILE = "result.json"

    def __init__(self, graph: WorldGraph) -> None:
        self._graph = graph
        self._env_root: Path | None = None
        self._agent_root: Path | None = None
        self._pack_root: Path | None = None
        self._checkpoint_dirs: list[Path] = []

    @property
    def env_root(self) -> Path | None:
        return self._env_root

    @property
    def agent_root(self) -> Path | None:
        return self._agent_root

    @property
    def pack_root(self) -> Path | None:
        return self._pack_root

    @abstractmethod
    def prepare_env_files(self, graph: WorldGraph) -> Mapping[str, str]:
        """Return ``{relative_path: file_contents}`` written under ``pack_root``
        on each ``reset()``."""

    def surface_extras(self) -> Mapping[str, Any]:
        """Override to add keys to ``surface()`` (callables, URLs, etc.)."""
        return {}

    def collect_extras(self) -> Mapping[str, Any]:
        """Override to add keys to ``collect()`` (parsed logs, metrics)."""
        return {}

    def poll_events(self) -> tuple[Mapping[str, Any], ...]:
        return ()

    @abstractmethod
    def reset(self) -> None:
        """Subclasses do their own setup, calling :meth:`_init_env` to
        prepare the tempdir trio."""

    @abstractmethod
    def stop(self) -> None:
        """Fully tear down. Subclasses must call :meth:`_teardown_env`
        and drop ``_checkpoint_dirs``."""

    def surface(self) -> Mapping[str, Any]:
        if self._agent_root is None:
            raise OpenRangeError("surface() called before reset()")
        return {
            "agent_root": str(self._agent_root),
            **self.surface_extras(),
        }

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

    def _init_env(self) -> None:
        env_root = Path(tempfile.mkdtemp(prefix=f"{self._tempdir_prefix()}-"))
        agent_root = env_root / "agent"
        agent_root.mkdir(parents=True, exist_ok=True)
        pack_root = env_root / "pack"
        pack_root.mkdir(parents=True, exist_ok=True)
        self._env_root = env_root
        self._agent_root = agent_root
        self._pack_root = pack_root
        write_tree(pack_root, self.prepare_env_files(self._graph))

    def _teardown_env(self) -> None:
        if self._env_root is not None and self._env_root.exists():
            shutil.rmtree(self._env_root, ignore_errors=True)
        self._env_root = None
        self._agent_root = None
        self._pack_root = None

    def _drop_checkpoints(self) -> None:
        for ckpt in self._checkpoint_dirs:
            shutil.rmtree(ckpt, ignore_errors=True)
        self._checkpoint_dirs.clear()

    def _read_result(self) -> Mapping[str, Any]:
        assert self._agent_root is not None
        result_path = self._agent_root / self.RESULT_FILE
        if not result_path.exists():
            return {}
        try:
            data = json.loads(result_path.read_text(encoding="utf-8"))
        except OSError, json.JSONDecodeError:
            return {}
        return dict(data) if isinstance(data, Mapping) else {}

    def _tempdir_prefix(self) -> str:
        return type(self).__name__.lower()


class OnDemandRuntime(_FilesystemRuntime):
    """RuntimeHandle for packs with no persistent subprocess.

    Pattern: the agent acts on files under ``agent_root``; the pack
    exposes on-demand callables (e.g. ``run_tests(name)``,
    ``run_cmd(argv)``) via ``surface_extras`` that shell out per call.

    Fits SWE-style packs where the world is a code workspace and the
    interesting actions are file edits + occasional command invocations.
    No startup-line contract, no subprocess supervision.

    Packs override (minimum):

    * ``prepare_env_files(graph)`` → initial files under ``pack_root``.

    Packs override (as needed):

    * ``surface_extras()`` — add callables the agent can invoke.
    * ``collect_extras()`` — add computed-from-workspace keys to
      ``collect()``.
    * ``poll_events()`` — per-tick events (default = no events).
    """

    def reset(self) -> None:
        self._teardown_env()
        self._init_env()

    def stop(self) -> None:
        """Fully tear down: wipe env_root + drop all checkpoints.

        ``reset()`` between episodes preserves checkpoints; use ``stop()``
        only for final teardown.
        """
        self._teardown_env()
        self._drop_checkpoints()


class SubprocessRuntime(_FilesystemRuntime):
    """RuntimeHandle scaffold for packs whose realized world is a child
    subprocess the agent interacts with.

    Domains this fits naturally: a webapp serving HTTP, a simulator
    exposing a broker API, an in-pack mock service. Common structure:
    spawn a process, optionally exchange a small startup descriptor
    (URL, port, fd), let the agent act, capture results from the
    agent's filesystem at the end.

    The class owns:

    * The shared filesystem lifecycle (``env_root`` / ``agent_root`` /
      ``pack_root``, ``checkpoint`` / ``restore``, terminal-via-result.json).
    * Subprocess spawn with ``start_new_session=True`` so process-group
      signals reach the child without affecting the harness.
    * SIGTERM → ``GRACE_SECONDS`` → SIGKILL on ``stop()``.
    * A startup-line handshake bounded by ``STARTUP_TIMEOUT_SECONDS`` so
      a misbehaving child can't hang the harness.

    Packs override (minimum):

    * ``prepare_env_files(graph)`` → ``{relative_path: contents}`` for
      ``pack_root`` (e.g., the codegen-rendered app source).
    * ``subprocess_command(env_root, agent_root)`` → the command to spawn.

    Packs override (as needed):

    * ``parse_startup(stdout_line)`` — extract a surface descriptor from
      the subprocess's first stdout line (e.g., ``{"base_url": ...}``).
    * ``subprocess_env()`` — environment variables for the child.
    * ``subprocess_popen_kwargs()`` — extra ``Popen`` kwargs (e.g.
      ``stdin=subprocess.PIPE`` for two-way comms). Additive; don't
      override ``stdout``/``stderr``/``start_new_session`` — the SDK
      relies on those.
    * ``surface_extras()`` — extra keys the agent reads (callables, URLs).
    * ``poll_events()`` — per-tick event drain (default = no events).
    * ``collect_extras()`` — per-pack final-state keys.

    The subprocess's stdout is captured; nothing else is consumed beyond
    the startup line. Packs that need request logs or other side-channel
    state typically write to a file under ``env_root`` and read it in
    ``poll_events`` / ``collect_extras``.

    Contract: the spawned subprocess MUST emit at least one newline on
    stdout before the agent acts. ``reset()`` blocks on ``readline()``
    (bounded by ``STARTUP_TIMEOUT_SECONDS``) to capture optional startup
    info. Packs with no startup info to advertise should print a single
    ``\\n`` immediately.
    """

    GRACE_SECONDS = 2.0
    STARTUP_TIMEOUT_SECONDS: float = 30.0

    def __init__(self, graph: WorldGraph) -> None:
        super().__init__(graph)
        self._process: subprocess.Popen[str] | None = None
        self._startup_info: dict[str, Any] = {}

    @property
    def process(self) -> subprocess.Popen[str] | None:
        """The spawned subprocess; ``None`` before ``reset()`` / after ``stop()``."""
        return self._process

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

    def subprocess_popen_kwargs(self) -> Mapping[str, Any]:
        """Override to add extra ``Popen`` kwargs (e.g.
        ``{"stdin": subprocess.PIPE}`` for two-way comms).

        Additive: the SDK always sets ``stdout=PIPE``, ``stderr=PIPE``,
        ``text=True``, ``start_new_session=True``. Don't override those
        — the startup-line readline, process-group kill, and stderr
        capture depend on them.
        """
        return {}

    def parse_startup(self, stdout_line: str) -> Mapping[str, Any]:
        """Parse the subprocess's first stdout line into surface keys.

        Default: no startup exchange (returns ``{}``). Common override
        parses JSON: ``{"host": "...", "port": 12345}`` → ``{"base_url":
        f"http://{host}:{port}"}``.
        """
        del stdout_line
        return {}

    def reset(self) -> None:
        self._teardown_subprocess()
        self._teardown_env()
        self._init_env()
        assert self._env_root is not None and self._agent_root is not None
        self._process = self._spawn(self._env_root, self._agent_root)
        assert self._process.stdout is not None
        first_line = _readline_with_timeout(self._process, self.STARTUP_TIMEOUT_SECONDS)
        if first_line:
            self._startup_info = dict(self.parse_startup(first_line))

    def stop(self) -> None:
        """Fully tear down: kill process, wipe env_root, drop all checkpoints.

        ``reset()`` between episodes preserves checkpoints; use ``stop()``
        only for final teardown.
        """
        self._teardown_subprocess()
        self._teardown_env()
        self._drop_checkpoints()

    def surface(self) -> Mapping[str, Any]:
        base = super().surface()
        return {**base, **self._startup_info, **self.surface_extras()}

    def _teardown_subprocess(self) -> None:
        if self._process is not None:
            _terminate_process_group(self._process, self.GRACE_SECONDS)
            self._process = None
        self._startup_info = {}

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
            **dict(self.subprocess_popen_kwargs()),
        }
        env = self.subprocess_env()
        if env is not None:
            kwargs["env"] = dict(env)
        return subprocess.Popen(cmd, **kwargs)


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
            "reset() can return (see SubprocessRuntime docstring)"
        )
    return process.stdout.readline()
