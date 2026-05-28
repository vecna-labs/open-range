"""Grader: execute agent-submitted handler source against a contract.

Each case runs through the SDK sandbox (``openrange_pack_sdk.run_submission``)
— see its module docstring for the full trust model and isolation caveats.
The handler is the untrusted submission; ``_drive`` is the trusted glue that
calls it as ``handle(query, state) -> (status, headers, body)`` and serializes
the response for the predicate to check. Each case is its own subprocess.
"""

from __future__ import annotations

import base64
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from openrange_pack_sdk import run_submission

if TYPE_CHECKING:
    from cyber_webapp.families.build.contracts import ContractCase

_DEFAULT_WALL_TIMEOUT = 5.0


def _drive(entry: Callable[..., Any], case: Mapping[str, Any]) -> dict[str, object]:
    status, headers, body = entry(case["query"], case["state"])
    if isinstance(body, str):
        body = body.encode("utf-8")
    # base64 the body — it may be non-UTF-8 bytes that can't go into JSON.
    return {
        "status": int(status),
        "headers": dict(headers),
        "body_b64": base64.b64encode(body).decode("ascii"),
    }


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
    run = run_submission(
        source,
        entry="handle",
        driver=_drive,
        stdin_obj={"query": dict(case.query), "state": dict(case.state)},
        timeout=timeout,
    )
    if not run.ok:
        return CaseResult(case.description, False, run.error, None, "")
    status = run.result["status"]
    headers = run.result["headers"]
    body_b64 = run.result["body_b64"]
    # Narrowing for the type checker; the driver guarantees these shapes.
    assert isinstance(status, int)
    assert isinstance(headers, dict)
    assert isinstance(body_b64, str)
    body = base64.b64decode(body_b64)
    ok, why = case.predicate(status, headers, body)
    return CaseResult(
        description=case.description,
        passed=ok,
        reason=why if not ok else "",
        status=status,
        body_preview=body[:200].decode("utf-8", "replace"),
    )
