"""Dynamic admission for LLM-realized handlers (M0 — DESIGN.md §9), pack-side pieces.

Today's admission is *structural* (`check_feasibility`: a graph path exists). An
LLM-realized handler can be wrong, so it is admitted *dynamically*: run the intended
exploit and a benign request against the realized world, and let the consequence
verifier decide — the exploit must leak the flag, the benign request must not. Accept
iff solvable and not trivial.

This module owns the parts a *pack* may: the pure verdict (`classify_admission`) and
the per-class exploit oracle (`cmdi_exploit_and_benign`). Driving a live episode runs
`EpisodeService`, which is a *host* concern — packs must not import `openrange` — so
the orchestration lives with the caller, not here.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from urllib.parse import quote

from graphschema import WorldGraph
from openrange_pack_sdk import PackError

from cyber_webapp.consequence import detect_leak


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


def _cmdi_payload(params: Mapping[str, object], path: str) -> str:
    ctx = params.get("inj_context", "separator")
    if ctx == "substitution":
        return f"$(cat {path})"
    if ctx == "quoted":
        quote_char = str(params.get("quote", "'"))
        return f"{quote_char}; cat {path}; echo {quote_char}"
    return f"127.0.0.1; cat {path}"


def _flag_record_key(graph: WorldGraph) -> str:
    for record in graph.by_kind("record"):
        for edge in graph.out_edges(record.id, "holds"):
            if graph.nodes[edge.dst].attrs.get("kind") == "flag":
                return str(record.attrs["key"])
    raise PackError("no record holds the flag")


def cmdi_exploit_and_benign(graph: WorldGraph) -> tuple[str, str]:
    """The canonical command-injection exploit + a benign request, as URL paths."""
    vuln = next(
        n
        for n in graph.by_kind("vulnerability")
        if n.attrs.get("kind") == "command_injection"
    )
    params = vuln.attrs["params"]
    if not isinstance(params, Mapping):
        raise PackError("command_injection vuln has no params mapping")
    endpoint_id = next(e.dst for e in graph.out_edges(vuln.id, "affects"))
    public_url = str(graph.nodes[endpoint_id].attrs["public_url"])
    param = str(params["target_param"])
    payload = _cmdi_payload(params, _flag_record_key(graph))
    return (
        f"{public_url}?{param}={quote(payload)}",
        f"{public_url}?{param}={quote('127.0.0.1')}",
    )
