"""A world's difficulty as one comparable number, read from the graph.

The chain weight must exceed the per-world vuln count (single digits) so chain
depth always outranks vuln count, which only breaks ties between equal depths.
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
