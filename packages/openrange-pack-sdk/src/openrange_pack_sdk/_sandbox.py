"""Isolated execution of agent-submitted code.

Trust model — read before deploying.

Runs an untrusted *submission* (agent-written Python source) together with a
trusted *driver* (a pack-supplied module-level function that exercises the
submission) in a single subprocess, fed one JSON envelope over stdin. The
result is written to a private file the submission has no handle to — not
stdout — so a submission that scribbles on stdout or exits early cannot forge a
result. Each call is its own subprocess.

What IS enforced
- Wall-clock timeout (parent ``subprocess.run(..., timeout=...)``). Hard.
- Subprocess isolation: the submission cannot mutate parent process state.
- ``RLIMIT_AS`` / ``RLIMIT_CPU`` — applied in the child; effective on Linux,
  silently skipped where the interpreter already exceeds the cap (e.g. macOS).
- ``PYTHONDONTWRITEBYTECODE=1`` — no ``__pycache__`` writes.

What is NOT enforced
- Filesystem isolation. The submission can read/write anything the host UID can.
- Network egress. The submission can open sockets.
- Syscall surface. The submission can shell out.

So this is safe for *trusted* submissions in a research loop on a disposable
host where the model is yours and exfiltration is not the threat. It is NOT
safe for adversarial code on a host you care about: hardening it for that
(firejail / bwrap / seccomp / container) is a prerequisite for public-facing
eval traffic.
"""

from __future__ import annotations

import inspect
import json
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_DEFAULT_WALL_TIMEOUT = 5.0
_DEFAULT_MEMORY_BYTES = 256 * 1024 * 1024
_DEFAULT_CPU_SECONDS = 5

_HARNESS = Path(__file__).with_name("_harness.py")

# The child file-loads the driver's module, so it must be a module-level
# function in an import-light module (no heavy pack __init__).
Driver = Callable[[Callable[..., Any], Any], Mapping[str, Any]]


@dataclass(frozen=True, slots=True)
class SandboxResult:
    """Outcome of one sandbox run. ``ok`` ⇒ ``result`` holds the driver's
    payload; otherwise ``error`` explains the failure."""

    ok: bool
    result: dict[str, object]
    error: str


def run_submission(
    source: str,
    *,
    entry: str,
    driver: Driver,
    stdin_obj: object,
    timeout: float = _DEFAULT_WALL_TIMEOUT,
    memory_bytes: int = _DEFAULT_MEMORY_BYTES,
    cpu_seconds: int = _DEFAULT_CPU_SECONDS,
) -> SandboxResult:
    """Run untrusted ``source`` plus the trusted ``driver`` in an isolated
    subprocess, returning a structured result rather than raising — one bad
    submission is a failed case, not a failed run.

    ``driver`` is a module-level function invoked as ``driver(entry, case)`` in
    the child, where ``entry`` is the submission's callable of that name and
    ``case`` is ``stdin_obj`` (JSON round-tripped); it returns a JSON object.
    """
    if not entry.isidentifier():
        raise ValueError(f"entry must be a Python identifier; got {entry!r}")
    with tempfile.TemporaryDirectory(prefix="openrange-sandbox-") as tmp:
        result_path = Path(tmp) / "result.json"
        envelope = json.dumps(
            {
                "source": source,
                "entry": entry,
                "driver_module": driver.__module__,
                "driver_qualname": driver.__qualname__,
                "driver_file": inspect.getfile(driver),
                "case": stdin_obj,
                "memory_bytes": memory_bytes,
                "cpu_seconds": cpu_seconds,
                "result_path": str(result_path),
            }
        )
        try:
            proc = subprocess.run(
                [sys.executable, str(_HARNESS)],
                input=envelope.encode("utf-8"),
                capture_output=True,
                timeout=timeout,
                env={"PYTHONDONTWRITEBYTECODE": "1"},
                cwd=tmp,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return SandboxResult(False, {}, f"timed out after {timeout}s")
        if not result_path.exists():
            detail = proc.stderr.decode("utf-8", "replace").strip()[:200]
            return SandboxResult(
                False,
                {},
                f"submission exited ({proc.returncode}) without a result: {detail}",
            )
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            return SandboxResult(False, {}, f"unreadable sandbox result: {exc}")
    if not isinstance(payload, dict) or not payload.get("ok"):
        error = payload.get("error") if isinstance(payload, dict) else None
        return SandboxResult(False, {}, str(error or "unknown failure"))
    result = payload.get("result")
    if not isinstance(result, dict):
        return SandboxResult(
            False, {}, f"driver payload is {type(result).__name__}, expected object"
        )
    return SandboxResult(True, result, "")
