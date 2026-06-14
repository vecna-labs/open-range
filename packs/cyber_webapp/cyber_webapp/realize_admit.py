"""Dynamic admission for LLM-realized handlers, pack-side pieces.

Today's admission is *structural* (`check_feasibility`: a graph path exists). An
LLM-realized handler can be wrong, so it is admitted *dynamically*: run the intended
exploit and a benign request against the realized world, and let the consequence
verifier decide — the exploit must leak the flag, the benign request must not. Accept
iff solvable and not trivial.

This module owns the pure verdict (`classify_admission`); the per-shape exploit it
checks comes from `reference_solver.exploit_and_benign`. Driving a live episode runs
`EpisodeService`, which is a *host* concern — packs must not import `openrange` — so
the orchestration lives with the caller, not here.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from graphschema import WorldGraph

from cyber_webapp.consequence import detect_leak
from cyber_webapp.reference_solver import exploit_and_benign


@dataclass(frozen=True)
class AdmissionVerdict:
    accepted: bool
    solvable: bool  # the exploit leaked the flag
    trivial: bool  # a benign request leaked the flag (no exploit needed)
    reason: str
    faithful: bool = (
        True  # the control computed (vacuously true when no control applies)
    )


def classify_admission(
    graph: WorldGraph, exploit_body: str, benign_body: str
) -> AdmissionVerdict:
    """The pure verdict: the exploit must leak, a benign request must not."""
    solvable = detect_leak(graph, [exploit_body]).occurred
    trivial = detect_leak(graph, [benign_body]).occurred
    accepted = solvable and not trivial
    if accepted:
        reason = "accepted: the exploit leaks the flag, a benign request does not"
    elif not solvable:
        reason = "rejected: the exploit did not leak the flag (not solvable)"
    else:
        reason = "rejected: a benign request leaks the flag (trivial)"
    return AdmissionVerdict(accepted, solvable, trivial, reason)


def classify_admission_with_control(
    graph: WorldGraph,
    exploit_body: str,
    benign_body: str,
    control_body: str | None,
    control_expected: str | None,
) -> AdmissionVerdict:
    """`classify_admission` plus a faithfulness control: a real handler must return
    `control_expected` for the control request (it computed/fetched a known non-flag
    value), so a flag-hard-coding handler fails it. No control -> faithful is True."""
    solvable = detect_leak(graph, [exploit_body]).occurred
    trivial = detect_leak(graph, [benign_body]).occurred
    faithful = control_expected is None or (
        control_body is not None and control_expected in control_body
    )
    accepted = solvable and not trivial and faithful
    if accepted:
        reason = "accepted: exploit leaks, benign does not, control computes"
    elif not solvable:
        reason = "rejected: the exploit did not leak the flag (not solvable)"
    elif trivial:
        reason = "rejected: a benign request leaks the flag (trivial)"
    else:
        reason = "rejected: the control did not compute (faked/hard-coded handler)"
    return AdmissionVerdict(accepted, solvable, trivial, reason, faithful)


def classify_service_admission(
    graph: WorldGraph,
    *,
    oracle_exploit_body: str,
    oracle_benign_body: str,
    benign_endpoint_bodies: Mapping[str, str],
    root_ok: bool,
) -> AdmissionVerdict:
    """Admit a whole-service benign realization: the world's oracle still fires (its
    exploit leaks, a benign request does not), NO realized benign endpoint leaks the
    flag, and the world still boots (GET / is 200). Any benign endpoint that serves the
    flag would make the world trivially winnable, so it rejects the realization."""
    solvable = detect_leak(graph, [oracle_exploit_body]).occurred
    trivial = detect_leak(graph, [oracle_benign_body]).occurred
    leaking = sorted(
        path
        for path, body in benign_endpoint_bodies.items()
        if detect_leak(graph, [body]).occurred
    )
    accepted = solvable and not trivial and not leaking and root_ok
    if accepted:
        reason = "accepted: oracle holds, no benign endpoint leaks, world boots"
    elif not solvable:
        reason = "rejected: the exploit did not leak the flag (not solvable)"
    elif trivial:
        reason = "rejected: a benign request leaks the flag (trivial)"
    elif leaking:
        reason = f"rejected: benign endpoint(s) leak the flag: {', '.join(leaking)}"
    else:
        reason = "rejected: the world did not boot (GET / not 200)"
    return AdmissionVerdict(accepted, solvable, trivial, reason)


def cmdi_exploit_and_benign(graph: WorldGraph) -> tuple[str, str]:
    """The canonical command-injection exploit + a benign request, as URL paths."""
    return exploit_and_benign(graph, "command_injection")
