"""Isolated execution of agent-submitted code.

Trust model — read before deploying.

This runs an untrusted *submission* (agent-written Python source) together
with a trusted *driver* (pack-written Python source that exercises the
submission) in a single subprocess, fed one JSON input over stdin, and
returns whatever JSON ``payload`` the driver produces. Each call is its own
subprocess, so a misbehaving submission (infinite loop, raised exception,
mutated globals) cannot taint the parent or another call.

What IS enforced
- Wall-clock timeout (parent ``subprocess.run(..., timeout=...)``). Hard.
- Subprocess isolation: the submission cannot mutate parent process state.
- ``RLIMIT_AS`` / ``RLIMIT_CPU`` — applied in the child; effective on Linux,
  silently skipped where the interpreter already exceeds the cap (e.g. macOS).
- ``PYTHONDONTWRITEBYTECODE=1`` — no ``__pycache__`` writes.

What is NOT enforced
- Filesystem isolation. The submission can read/write anything the host UID
  can.
- Network egress. The submission can open sockets.
- Syscall surface. The submission can shell out.

So this is safe for *trusted* submissions in a research loop on a disposable
host where the model is yours and exfiltration is not the threat. It is NOT
safe for adversarial code on a host you care about: hardening it for that
(firejail / bwrap / seccomp / container) is a prerequisite for public-facing
eval traffic.

The driver runs *after* the submission loads; it sees the entry callable as
``entry`` and the decoded stdin as ``case``, and must assign a
JSON-serializable ``payload``. The driver is trusted pack code: subprocess
isolation protects the host from the submission, not the driver from it.
"""

from __future__ import annotations

import base64
import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

_DEFAULT_WALL_TIMEOUT = 5.0
_DEFAULT_MEMORY_BYTES = 256 * 1024 * 1024
_DEFAULT_CPU_SECONDS = 5

# A .format() template. Submission and driver are base64'd, so their own braces
# never reach .format(); only the fields below and the doubled literal braces do.
_HARNESS = """
import base64
import io
import json
import resource
import sys

for _name, _limit in (("RLIMIT_AS", {mem}), ("RLIMIT_CPU", {cpu})):
    try:
        resource.setrlimit(getattr(resource, _name), (_limit, _limit))
    except (ValueError, OSError):
        pass


def _emit(obj):
    sys.__stdout__.write(json.dumps(obj))
    sys.exit(0)


_sink = io.StringIO()
_ns = {{}}
sys.stdout = _sink
try:
    exec(base64.b64decode("{source_b64}").decode("utf-8"), _ns)
except BaseException as exc:
    sys.stdout = sys.__stdout__
    _emit({{
        "ok": False,
        "error": f"source did not load: {{type(exc).__name__}}: {{exc}}"[:500],
    }})
finally:
    sys.stdout = sys.__stdout__

entry = _ns.get("{entry}")
if not callable(entry):
    _emit({{"ok": False, "error": "submission defines no callable {entry}"}})

case = json.loads(sys.stdin.read())
sys.stdout = _sink
try:
    _dns = {{"entry": entry, "case": case}}
    exec(base64.b64decode("{driver_b64}").decode("utf-8"), _dns)
    _payload = {{"ok": True, "result": _dns["payload"]}}
except BaseException as exc:
    _payload = {{
        "ok": False,
        "error": f"submission failed: {{type(exc).__name__}}: {{exc}}"[:500],
    }}
finally:
    sys.stdout = sys.__stdout__
_emit(_payload)
"""


@dataclass(frozen=True, slots=True)
class SandboxResult:
    """Outcome of one sandbox run. ``ok`` ⇒ ``result`` holds the driver's
    JSON ``payload``; otherwise ``error`` explains the failure."""

    ok: bool
    result: dict[str, object]
    error: str


def run_submission(
    source: str,
    *,
    entry: str,
    driver: str,
    stdin_obj: object,
    timeout: float = _DEFAULT_WALL_TIMEOUT,
    memory_bytes: int = _DEFAULT_MEMORY_BYTES,
    cpu_seconds: int = _DEFAULT_CPU_SECONDS,
) -> SandboxResult:
    """Run untrusted ``source`` + trusted ``driver`` in an isolated subprocess.

    ``driver`` is Python source that may reference ``entry`` (the submission's
    callable of that name) and ``case`` (``stdin_obj``, JSON round-tripped),
    and must assign a JSON-serializable mapping to ``payload``. Returns a
    structured ``SandboxResult`` rather than raising, so one bad submission is
    a failed case, not a failed run.
    """
    if not entry.isidentifier():
        raise ValueError(f"entry must be a Python identifier; got {entry!r}")
    program = _HARNESS.format(
        mem=memory_bytes,
        cpu=cpu_seconds,
        source_b64=base64.b64encode(source.encode("utf-8")).decode("ascii"),
        driver_b64=base64.b64encode(driver.encode("utf-8")).decode("ascii"),
        entry=entry,
    )
    with tempfile.TemporaryDirectory(prefix="openrange-sandbox-") as tmp:
        prog = Path(tmp) / "prog.py"
        prog.write_text(program, encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, str(prog)],
                input=json.dumps(stdin_obj),
                capture_output=True,
                text=True,
                timeout=timeout,
                env={"PYTHONDONTWRITEBYTECODE": "1"},
                cwd=tmp,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return SandboxResult(False, {}, f"timed out after {timeout}s")

    if proc.returncode != 0 and not proc.stdout:
        return SandboxResult(
            False,
            {},
            f"subprocess exited {proc.returncode}: {proc.stderr[:200].strip()}",
        )
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return SandboxResult(
            False, {}, f"non-JSON harness output: {proc.stdout[:200]!r}"
        )
    if not payload.get("ok"):
        return SandboxResult(False, {}, str(payload.get("error", "unknown failure")))
    result = payload.get("result")
    if not isinstance(result, dict):
        return SandboxResult(
            False, {}, f"driver payload is {type(result).__name__}, expected object"
        )
    return SandboxResult(True, result, "")
