"""The webapp's consequence verdict: does a world's reference breach actually leak?

Structural re-admission only proves a *path* exists on the graph. ``verdict`` drives
the reference breach over HTTP against an already-running world and requires the flag
to leak via the exploit while a benign request does not (the consequence verifier,
[#312](https://github.com/vecna-labs/open-range/issues/312)). ``accepts`` wraps that
as the bool an :data:`~openrange.core.curriculum.EvolutionGate` wants -- core's
``consequence_gate`` realizes the world and calls it. Pack-side on purpose: it touches
no ``openrange`` host code, only the reference solver and ``urllib``.
"""

from __future__ import annotations

import urllib.error
import urllib.request

from graphschema import WorldGraph
from openrange_pack_sdk import Snapshot

from cyber_webapp import _is_networked
from cyber_webapp.realize_admit import AdmissionVerdict, classify_admission
from cyber_webapp.reference_solver import Request, exploit_and_benign, solve_chain


def _fetch(url: str) -> str:
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            raw: bytes = resp.read()
    except urllib.error.HTTPError as exc:
        raw = exc.read()
    except Exception:  # noqa: BLE001 — an unreachable surface just can't leak
        return ""
    return raw.decode("utf-8", "replace")


def perform(base_url: str, request: Request) -> str:
    """Execute a reference-solver ``Request`` against a live world and return its body.
    A GET carries its query in the path; a body-shaped (POST) request sends its payload
    as the body with a content type, so an exploit is delivered in the right shape."""
    data = request.body.encode() if request.body is not None else None
    built = urllib.request.Request(
        base_url + request.path, data=data, method=request.method
    )
    if request.content_type:
        built.add_header("Content-Type", request.content_type)
    try:
        with urllib.request.urlopen(built, timeout=15) as resp:
            raw: bytes = resp.read()
    except urllib.error.HTTPError as exc:
        raw = exc.read()
    except Exception:  # noqa: BLE001 — an unreachable surface just can't leak
        return ""
    return raw.decode("utf-8", "replace")


def verdict(graph: WorldGraph, base_url: str, entry_path: str) -> AdmissionVerdict:
    """Drive the reference breach against an already-running world at ``base_url`` and
    classify whether the exploit leaks while a benign request to ``entry_path`` does
    not."""

    def fetch(path: str) -> str:
        return _fetch(base_url + str(path))

    benign = fetch(entry_path)
    if _is_networked(graph):
        try:
            terminal = solve_chain(graph, fetch).terminal
        except Exception:  # noqa: BLE001 — a chain that won't drive isn't solvable
            return AdmissionVerdict(False, False, False, "reference breach failed")
        return classify_admission(graph, terminal, benign)
    for vuln in graph.by_kind("vulnerability"):
        try:
            exploit_req, benign_req = exploit_and_benign(graph, str(vuln.attrs["kind"]))
        except Exception:  # noqa: BLE001 — no reference exploit for this kind
            continue
        return classify_admission(
            graph, perform(base_url, exploit_req), perform(base_url, benign_req)
        )
    return AdmissionVerdict(False, False, False, "no reference exploit to verify")


def accepts(snapshot: Snapshot, base_url: str) -> bool:
    """True iff the evolved world's reference breach leaks (a benign request does not).
    The shape ``consequence_gate`` wants: it realizes the world, this drives the breach.
    A world with no pentest task or entrypoint can't be assessed, so it passes."""
    task = next(
        (t for t in snapshot.tasks if t.meta.get("family") == "webapp.pentest"),
        None,
    )
    if task is None or not task.entrypoints:
        return True
    entry = str(snapshot.graph.nodes[task.entrypoints[0]].attrs["public_url"])
    return verdict(snapshot.graph, base_url, entry).accepted
