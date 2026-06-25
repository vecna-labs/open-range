"""A world's difficulty, read off the path the reference solver actually walks.

Difficulty is the cost of executing the winning chain, not the size of the attack
surface. The credential-reuse hop count dominates; the number of internal pivots, the
dmz->internal crossing, blind recon, the breadth of internal hosts to triage, and the
entry exploit's class are smaller corrections; and off-path decoys saturate at a cap so
they can never out-rank a real hop. Every chain term is gated on a reachable public
entry, so a world whose chain no longer has a way in scores like the flat world it has
become.
"""

from __future__ import annotations

from graphschema import Node, WorldGraph

from cyber_webapp.mutation import (
    _affects_target_id,
    _credential_walk,
    _oracle_path_targets,
)
from cyber_webapp.vulnerabilities import CATALOG

_W_HOP = 10
_W_PIVOT = 4
_W_BOUNDARY = 3
_W_BLIND = 5
_W_CLASS = 4
_W_FANOUT = 1.0
_FANOUT_CAP = 6.0
_DECOY_CAP = 3.0
_DECOY_PER = 0.3

_RECON_KIND = "config_disclosure"
_CHAIN_KINDS = ("credential_leak", "credential_gated_relay", "credential_gated_flag")
_OFF_PATH_EXCLUDED = {"ssrf", _RECON_KIND, "metadata_credential_leak", *_CHAIN_KINDS}


def _entry_ssrf(graph: WorldGraph) -> Node | None:
    public_eps = {
        edge.dst
        for svc in graph.by_kind("service")
        if svc.attrs.get("exposure") == "public"
        for edge in graph.out_edges(svc.id, "exposes")
    }
    for vuln in graph.by_kind("vulnerability"):
        if vuln.attrs.get("kind") != "ssrf":
            continue
        if {edge.dst for edge in graph.out_edges(vuln.id, "affects")} & public_eps:
            return vuln
    return None


def _placement_index(vuln: Node) -> tuple[int, str]:
    suffix = vuln.id.rsplit("_", 1)[-1]
    return (int(suffix) if suffix.isdigit() else 1_000_000, vuln.id)


def _oracle_vuln(graph: WorldGraph) -> Node | None:
    oracle_endpoints, _ = _oracle_path_targets(graph)
    on_oracle = [
        vuln
        for vuln in graph.by_kind("vulnerability")
        if _affects_target_id(graph, vuln.id) in oracle_endpoints
        and vuln.attrs.get("kind") != _RECON_KIND
    ]
    for kind in ("credential_gated_flag", "metadata_credential_leak"):
        gated = [vuln for vuln in on_oracle if vuln.attrs.get("kind") == kind]
        if gated:
            return gated[0]
    # The sampler places the flag-reading oracle first (id `vuln_<kind>_0`), so the
    # lowest placement index is the real exploit, not an alphabetically-earlier decoy.
    return min(on_oracle, key=_placement_index) if on_oracle else None


def _class_weight(vuln: Node | None) -> float:
    if vuln is None:
        return 0.0
    kind = str(vuln.attrs.get("kind", ""))
    spec = CATALOG.get(kind)
    return spec.exploit_complexity if spec is not None else 0.5


def world_difficulty(graph: WorldGraph) -> float:
    entry = _entry_ssrf(graph)
    oracle = _oracle_vuln(graph)
    vulns = list(graph.by_kind("vulnerability"))

    if entry is not None:
        walk = _credential_walk(graph)
        chain_hops = len(walk)
        pivots = len({host for host, _ in walk}) if walk else 1
        boundaries = 1
        blind = 0 if any(v.attrs.get("kind") == _RECON_KIND for v in vulns) else 1
        internal = sum(
            1 for s in graph.by_kind("service") if s.attrs.get("exposure") != "public"
        )
        walk_and_target_hosts = chain_hops + 1
        fanout = max(0, internal - walk_and_target_hosts)
    else:
        chain_hops = pivots = boundaries = blind = fanout = 0

    on_path_class = _class_weight(entry) + _class_weight(oracle)

    oracle_id = oracle.id if oracle is not None else None
    off_path = sum(
        1
        for vuln in vulns
        if vuln.id != oracle_id and vuln.attrs.get("kind") not in _OFF_PATH_EXCLUDED
    )

    return round(
        _W_HOP * chain_hops
        + _W_PIVOT * pivots
        + _W_BOUNDARY * boundaries
        + _W_BLIND * blind
        + min(_FANOUT_CAP, _W_FANOUT * fanout)
        + _W_CLASS * on_path_class
        + min(_DECOY_CAP, _DECOY_PER * off_path),
        2,
    )
