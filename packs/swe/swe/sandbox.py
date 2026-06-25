"""Run an argv under the strongest isolation the host actually supports.

Trust model — read before deploying. The SWE grader executes an agent's patched
code plus arbitrary repo tests: that *is* arbitrary code execution. This module
is the one chokepoint where every such run is wrapped. It picks a backend by
probing the host, and reports which one actually ran in ``SandboxResult.isolation``
so callers never have to *assume* a guarantee the host couldn't give.

What is always enforced
- **Wall-clock timeout.** A hard ceiling; on expiry the whole process group (or
  the bwrap pid namespace) is killed, so runaway test subprocesses die too.
- **POSIX rlimits** (best-effort): ``RLIMIT_CPU`` tied to the timeout and a
  generous ``RLIMIT_FSIZE``. Each is set with ``suppress`` — silently skipped
  where the cap is unavailable (e.g. macOS ``RLIMIT_AS``). We deliberately do
  *not* cap ``RLIMIT_NPROC`` (per-user, would starve the host) or ``RLIMIT_AS``
  by default (real test runs / installs legitimately need the memory).

What is enforced only where the host supports it (Linux + ``bwrap``)
- **Filesystem confinement:** the host is bound read-only, the workspace tree
  read-write, ``/tmp`` a fresh tmpfs. The submission cannot mutate the host.
- **Network isolation:** ``--unshare-net`` when ``network=False`` (the grading
  run). Installs pass ``network=True`` and keep egress.
- **PID/IPC/UTS isolation:** ``--unshare-pid`` et al. plus ``--die-with-parent``.

What is NOT enforced anywhere yet
- **Syscall filtering (seccomp)** and full **container** isolation. Those are the
  prerequisite for *adversarial*, public-facing eval traffic — see ``DESIGN.md``.

So on a Linux host with ``bwrap`` this is safe for untrusted code on a disposable
machine; on macOS (no namespaces) it degrades to a bare subprocess + rlimits +
wall-clock, safe only for *trusted* submissions. The ``isolation`` field makes
that degradation observable rather than silent. Force a backend with
``OPENRANGE_SWE_SANDBOX=auto|none|bwrap`` (default ``auto``).
"""

from __future__ import annotations

import contextlib
import os
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

__all__ = ["SandboxResult", "run_sandboxed"]

_ENV_BACKEND = "OPENRANGE_SWE_SANDBOX"
_FSIZE_LIMIT = 2 * 1024 * 1024 * 1024  # 2 GiB — guard a test that fills the disk.
_CPU_HEADROOM = 60  # CPU-seconds of slack over the wall-clock budget.
_KILL_GRACE = 5.0


def _is_linux() -> bool:
    # Indirected so mypy doesn't constant-fold ``sys.platform`` on a non-Linux
    # checkout and mark the bwrap path unreachable under ``warn_unreachable``.
    return sys.platform == "linux"


@dataclass(frozen=True, slots=True)
class SandboxResult:
    """Outcome of one sandboxed run.

    ``isolation`` names the backend that actually ran (``"bwrap+netns"``,
    ``"bwrap"``, ``"subprocess"``) so a caller can be honest about the guarantee
    it got rather than the one it asked for. ``timed_out`` is the hard signal the
    grader keys on; ``returncode``/``stdout``/``stderr`` drive the agent-facing
    test-runner tool.
    """

    returncode: int
    stdout: str
    stderr: str
    timed_out: bool
    isolation: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out


def run_sandboxed(
    args: Sequence[str],
    *,
    root: Path,
    timeout: float,
    network: bool = False,
) -> SandboxResult:
    """Run ``[python, *args]`` in ``root`` under the best available isolation.

    ``args`` are the interpreter arguments (``["-m", "pytest", ...]``); this
    interpreter (``sys.executable``) is prepended. ``network`` is the egress
    knob: the grading run leaves it ``False`` (isolated where possible); an
    editable install passes ``True``. The call never raises for a non-zero exit
    or a timeout — those are reported in the result, because a failed test run is
    data, not an error.
    """
    inner = [sys.executable, *args]
    env = _child_env(root)
    backend = _select_backend(network)
    if backend == "bwrap":
        cmd = _bwrap_wrap(inner, root=root, network=network)
        isolation = "bwrap" if network else "bwrap+netns"
    else:
        cmd = inner
        isolation = "subprocess"
    return _exec(cmd, cwd=root, timeout=timeout, env=env, isolation=isolation)


def _select_backend(network: bool) -> str:
    """Pick ``"bwrap"`` or ``"none"`` from the env override and a live probe."""
    override = os.environ.get(_ENV_BACKEND, "auto").strip().lower()
    if override in {"none", "subprocess"}:
        return "none"
    if override == "bwrap":
        return "bwrap" if _bwrap_usable() else "none"
    # auto
    return "bwrap" if _bwrap_usable() else "none"


@lru_cache(maxsize=1)
def _bwrap_usable() -> bool:
    """True iff we are on Linux and a trivial ``bwrap`` sandbox actually runs.

    Probed once (not just ``which``) because many Linux hosts ship ``bwrap`` but
    disable unprivileged user namespaces, so the binary exists yet every sandbox
    fails at setup. The probe runs in the *grading* mode (``network=False`` →
    ``--unshare-net``, the strictest combination we use), so a host that allows
    user namespaces but blocks network namespaces fails the probe and degrades to
    a visible ``subprocess`` rather than passing here and then failing every real
    grade at setup. A host that clears this probe also clears the looser
    ``network=True`` install path. On non-Linux this short-circuits — the probe
    never runs.
    """
    if not _is_linux() or shutil.which("bwrap") is None:
        return False
    probe = _bwrap_wrap(
        [sys.executable, "-c", "pass"], root=Path("/tmp"), network=False
    )
    try:
        proc = subprocess.run(
            probe,
            capture_output=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return proc.returncode == 0


def _bwrap_wrap(inner: Sequence[str], *, root: Path, network: bool) -> list[str]:
    """Wrap ``inner`` in a bubblewrap argv: host read-only, workspace writable."""
    cmd = [
        "bwrap",
        "--ro-bind",
        "/",
        "/",  # whole host visible read-only (python, stdlib…)
        "--dev",
        "/dev",
        "--proc",
        "/proc",
        "--tmpfs",
        "/tmp",
        "--bind",
        str(root),
        str(root),  # re-mount the workspace writable on top
        "--chdir",
        str(root),
        "--unshare-pid",
        "--unshare-ipc",
        "--unshare-uts",
        "--die-with-parent",
    ]
    if not network:
        cmd.append("--unshare-net")
    cmd.append("--")
    cmd.extend(inner)
    return cmd


def _child_env(root: Path) -> dict[str, str]:
    """Inherit the host env, then pin determinism and prepend the workspace.

    Prepending ``root`` (and ``root/src``) to ``PYTHONPATH`` makes both a flat
    package layout and a ``src/`` layout importable *before* any editable
    install, so the no-build-file repos (the calc fixture) still resolve.
    """
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    parts = [str(root), str(root / "src")]
    existing = env.get("PYTHONPATH")
    if existing:
        parts.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(parts)
    return env


def _exec(
    cmd: Sequence[str],
    *,
    cwd: Path,
    timeout: float,
    env: Mapping[str, str],
    isolation: str,
) -> SandboxResult:
    proc = subprocess.Popen(  # argv is built here, never a shell string.
        list(cmd),
        cwd=str(cwd),
        env=dict(env),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        preexec_fn=_set_rlimits(timeout) if os.name == "posix" else None,
        start_new_session=os.name == "posix",
    )
    try:
        out, err = proc.communicate(timeout=timeout)
        timed_out = False
    except subprocess.TimeoutExpired:
        _terminate(proc)
        out, err = proc.communicate()
        timed_out = True
    return SandboxResult(
        returncode=proc.returncode,
        stdout=out.decode("utf-8", "replace"),
        stderr=err.decode("utf-8", "replace"),
        timed_out=timed_out,
        isolation=isolation,
    )


def _terminate(proc: subprocess.Popen[bytes]) -> None:
    """Kill the whole process group so test-spawned children die too."""
    if os.name == "posix":
        with contextlib.suppress(ProcessLookupError, OSError):
            os.killpg(os.getpgid(proc.pid), 9)
            return
    with contextlib.suppress(ProcessLookupError, OSError):
        proc.kill()


def _set_rlimits(timeout: float):  # type: ignore[no-untyped-def]
    """Build a POSIX ``preexec_fn`` that caps CPU and file size in the child.

    Set between fork and exec, the limits are inherited across the exec *and*
    across bwrap's namespace setup, so they apply to the real test process under
    either backend. Each cap is best-effort: unsupported ones are skipped.
    """
    cpu_seconds = int(timeout) + _CPU_HEADROOM

    def apply() -> None:
        import resource

        for name, limit in (
            ("RLIMIT_CPU", cpu_seconds),
            ("RLIMIT_FSIZE", _FSIZE_LIMIT),
        ):
            with contextlib.suppress(ValueError, OSError, AttributeError):
                resource.setrlimit(getattr(resource, name), (limit, limit))

    return apply
