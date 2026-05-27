"""Grader: execute agent-submitted handler source against a contract.

Trust model — read before deploying.

The grader runs untrusted source as a real Python subprocess. Each test
case runs in its own subprocess so a misbehaving handler (infinite loop,
raised exception) cannot taint other cases or the parent.

What IS enforced
- Wall-clock timeout (parent ``subprocess.run(..., timeout=...)``). Hard.
- Subprocess isolation: agent source cannot mutate parent process state.
- ``RLIMIT_CPU`` — applied in the child; effective on Linux; silently
  skipped on macOS where Python's interpreter already exceeds.
- ``PYTHONDONTWRITEBYTECODE=1`` env — no ``__pycache__`` writes.

What is NOT enforced
- Filesystem isolation. Agent source can ``open("/etc/hosts").read()``,
  write under ``$HOME``, list directories, anything the host UID can do.
- Network egress. Agent source can ``socket.connect()`` or bind ports.
- Syscall surface. Agent source can ``import subprocess`` and run shell.
- ``RLIMIT_AS`` on macOS (RLIMIT setrlimit() rejects shrinking below the
  current VM size; Python's interpreter is well above any useful cap).

This means: the grader is safe for trusted agent submissions in a research
loop where the host is disposable, the user owns the model, and exfil is
not a threat. It is NOT safe for adversarial agent code on a host you care
about. Production sandboxing (firejail / bwrap / seccomp / container) is
the ROADMAP follow-up before public-facing eval traffic lands.
"""

from __future__ import annotations

import base64
import json
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from cyber_webapp.families.build.contracts import ContractCase

_MEMORY_LIMIT_BYTES = 256 * 1024 * 1024
_CPU_SECONDS = 5
_DEFAULT_WALL_TIMEOUT = 5.0

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

source = base64.b64decode("{source_b64}").decode("utf-8")

namespace = {{}}
sink = io.StringIO()
sys.stdout = sink
try:
    exec(source, namespace)
except BaseException as exc:
    sys.stdout = sys.__stdout__
    sys.__stdout__.write(json.dumps({{
        "ok": False,
        "error": f"source did not load: {{type(exc).__name__}}: {{exc}}"[:500],
    }}))
    sys.exit(0)
finally:
    sys.stdout = sys.__stdout__

handle = namespace.get("handle")
if not callable(handle):
    sys.__stdout__.write(json.dumps({{
        "ok": False,
        "error": "no callable 'handle' defined",
    }}))
    sys.exit(0)

case = json.loads(sys.stdin.read())
sys.stdout = sink
try:
    result = handle(case["query"], case["state"])
    status, headers, body = result
    if isinstance(body, str):
        body = body.encode("utf-8")
    payload = {{
        "ok": True,
        "status": int(status),
        "headers": dict(headers),
        "body_b64": base64.b64encode(body).decode("ascii"),
    }}
except BaseException as exc:
    payload = {{
        "ok": False,
        "error": f"handler raised: {{type(exc).__name__}}: {{exc}}"[:500],
    }}
finally:
    sys.stdout = sys.__stdout__

sys.__stdout__.write(json.dumps(payload))
"""


@dataclass(frozen=True, slots=True)
class CaseResult:
    description: str
    passed: bool
    reason: str
    status: int | None
    body_preview: str


@dataclass(frozen=True, slots=True)
class ContractReport:
    passed: int
    total: int
    cases: tuple[CaseResult, ...]

    @property
    def all_passed(self) -> bool:
        return self.total > 0 and self.passed == self.total


def grade_source(
    source: str,
    cases: Sequence[ContractCase],
    *,
    timeout: float = _DEFAULT_WALL_TIMEOUT,
) -> ContractReport:
    results: list[CaseResult] = []
    passed = 0
    for case in cases:
        result = _run_case(source, case, timeout=timeout)
        results.append(result)
        if result.passed:
            passed += 1
    return ContractReport(passed=passed, total=len(cases), cases=tuple(results))


def _run_case(source: str, case: ContractCase, *, timeout: float) -> CaseResult:
    program = _HARNESS.format(
        mem=_MEMORY_LIMIT_BYTES,
        cpu=_CPU_SECONDS,
        source_b64=base64.b64encode(source.encode("utf-8")).decode("ascii"),
    )
    case_input = json.dumps({"query": dict(case.query), "state": dict(case.state)})
    with tempfile.TemporaryDirectory(prefix="webapp-build-grade-") as tmp:
        prog_path = Path(tmp) / "prog.py"
        prog_path.write_text(program, encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, str(prog_path)],
                input=case_input,
                capture_output=True,
                text=True,
                timeout=timeout,
                env={"PYTHONDONTWRITEBYTECODE": "1"},
                cwd=tmp,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return CaseResult(
                description=case.description,
                passed=False,
                reason=f"timed out after {timeout}s",
                status=None,
                body_preview="",
            )

    if proc.returncode != 0 and not proc.stdout:
        return CaseResult(
            description=case.description,
            passed=False,
            reason=f"subprocess exited {proc.returncode}: {proc.stderr[:200].strip()}",
            status=None,
            body_preview="",
        )
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return CaseResult(
            description=case.description,
            passed=False,
            reason=f"non-JSON harness output: {proc.stdout[:200]!r}",
            status=None,
            body_preview="",
        )
    if not payload.get("ok"):
        return CaseResult(
            description=case.description,
            passed=False,
            reason=str(payload.get("error", "unknown failure")),
            status=None,
            body_preview="",
        )
    status = int(payload["status"])
    headers = payload["headers"]
    body = base64.b64decode(payload["body_b64"])
    ok, why = case.predicate(status, headers, body)
    return CaseResult(
        description=case.description,
        passed=ok,
        reason=why if not ok else "",
        status=status,
        body_preview=body[:200].decode("utf-8", "replace"),
    )
