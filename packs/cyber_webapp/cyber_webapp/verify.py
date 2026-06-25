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
from cyber_webapp.realize_admit import (
    AdmissionVerdict,
    classify_admission,
    classify_service_admission,
)
from cyber_webapp.reference_solver import (
    Request,
    exploit_and_benign,
    solve_chain,
    wrap_payload,
)


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


def _benign_sweep(graph: WorldGraph, base_url: str) -> dict[str, str]:
    networked = _is_networked(graph)
    service_of_endpoint = {
        e.dst: e.src for e in graph.edges.values() if e.kind == "exposes"
    }
    services = {n.id: n for n in graph.nodes.values() if n.kind == "service"}
    bodies: dict[str, str] = {}
    for endpoint in graph.by_kind("endpoint"):
        service = services.get(service_of_endpoint.get(endpoint.id, ""))
        # internal services answer only via the pivot; a direct fetch can't reach them
        if (
            networked
            and service is not None
            and service.attrs.get("exposure") != "public"
        ):
            continue
        public_url = str(endpoint.attrs.get("public_url", ""))
        if public_url:
            bodies[public_url] = _fetch(base_url + public_url)
    return bodies


def verdict(graph: WorldGraph, base_url: str, entry_path: str) -> AdmissionVerdict:
    """Drive the reference breach against an already-running world at ``base_url`` and
    classify it whole-world: the exploit must leak, the benign entry must not, and NO
    directly reachable benign endpoint may leak — a sibling endpoint that serves the
    flag would make the world winnable without the intended exploit."""

    def fetch(path: str) -> str:
        return _fetch(base_url + str(path))

    benign = fetch(entry_path)
    reachable = _benign_sweep(graph, base_url)
    if _is_networked(graph):
        try:
            terminal = solve_chain(graph, fetch).terminal
        except Exception:  # noqa: BLE001 — a chain that won't drive isn't solvable
            return AdmissionVerdict(False, False, False, "reference breach failed")
        return classify_service_admission(
            graph,
            oracle_exploit_body=terminal,
            oracle_benign_body=benign,
            benign_endpoint_bodies=reachable,
            root_ok=True,
        )
    for vuln in graph.by_kind("vulnerability"):
        try:
            exploit_req, benign_req = exploit_and_benign(graph, str(vuln.attrs["kind"]))
        except Exception:  # noqa: BLE001 — no reference exploit for this kind
            continue
        return classify_service_admission(
            graph,
            oracle_exploit_body=perform(base_url, exploit_req),
            oracle_benign_body=perform(base_url, benign_req),
            benign_endpoint_bodies=reachable,
            root_ok=True,
        )
    return AdmissionVerdict(False, False, False, "no reference exploit to verify")


def accepts(snapshot: Snapshot, base_url: str) -> bool:
    """True iff the evolved world's reference breach leaks (a benign request does not).
    The shape ``consequence_gate`` wants: it realizes the world, this drives the breach.
    A world with no pentest task or entrypoint can't be assessed, so it is rejected."""
    task = next(
        (t for t in snapshot.tasks if t.meta.get("family") == "webapp.pentest"),
        None,
    )
    if task is None or not task.entrypoints:
        return False
    entry = str(snapshot.graph.nodes[task.entrypoints[0]].attrs["public_url"])
    return verdict(snapshot.graph, base_url, entry).accepted


def verdict_authored(
    graph: WorldGraph, base_url: str, kind: str, exploit: str, benign: str
) -> AdmissionVerdict:
    """Drive an authored (exploit, benign) payload pair against a running world and
    classify it with the same gate the reference solver uses -- source (b) of #317. The
    payloads wrap into the kind's request shape; the flag still leaks (or not) from the
    live world, so a memorized value can't pass a re-seeded world (see ``reseed``)."""
    exploit_req = wrap_payload(graph, kind, exploit)
    benign_req = wrap_payload(graph, kind, benign)
    return classify_admission(
        graph, perform(base_url, exploit_req), perform(base_url, benign_req)
    )
