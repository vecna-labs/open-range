"""A world's difficulty as one comparable number, read from the graph.

The pivot chain depth — how many credentials must be looted and reused in
sequence to reach the flag — is what makes a world harder, so it dominates; vuln
count only breaks ties between equal-depth worlds. The weight exceeds the
per-world vuln count (single digits) so depth always outranks surface.
"""

from __future__ import annotations

from graphschema import WorldGraph

_CHAIN_WEIGHT = 10


def world_difficulty(graph: WorldGraph) -> int:
    vulns = list(graph.by_kind("vulnerability"))
    chain_depth = sum(
        1
        for vuln in vulns
        if str(vuln.attrs.get("kind", "")).startswith("credential_gated")
    )
    return chain_depth * _CHAIN_WEIGHT + len(vulns)
