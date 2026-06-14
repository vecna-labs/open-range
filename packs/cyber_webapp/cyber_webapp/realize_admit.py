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


def cmdi_exploit_and_benign(graph: WorldGraph) -> tuple[str, str]:
    """The canonical command-injection exploit + a benign request, as URL paths."""
    return exploit_and_benign(graph, "command_injection")
