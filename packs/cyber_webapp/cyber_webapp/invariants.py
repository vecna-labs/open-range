"""Pack-level invariants for the new-shape cyber webapp ontology.

Each invariant is a plain function `(graph) -> list[Issue]` returned by
`WebappPack.invariants()`. They run as the Tier-3 layer of OpenRange's
admission loop — after structural + ontology conformance, before task
feasibility.

Equivalent to the old `NoOrphanNodesConstraint` / `SecretReachableConstraint`
/ `OraclePathExistsConstraint` from `ontology.py` (which targeted the
old WorldSchema/GraphConstraint shape). Same semantics, function shape.
"""

from __future__ import annotations

from openrange.world_ir import Issue, WorldGraph

_ORPHAN_EXEMPT: frozenset[str] = frozenset({"host", "network"})


def no_orphan_nodes(graph: WorldGraph) -> list[Issue]:
    """Every non-exempt node must touch at least one edge.

    Exemptions: `host` and `network` may stand alone as scaffolding
    (a host with no service yet attached; an isolated network segment).
    """
    referenced: set[str] = set()
    for edge in graph.edges.values():
        referenced.add(edge.src)
        referenced.add(edge.dst)
    issues: list[Issue] = []
    for node in graph.nodes.values():
        if node.kind in _ORPHAN_EXEMPT:
            continue
        if node.id not in referenced:
            issues.append(
                Issue(
                    "error",
                    "orphan_node",
                    f"node {node.id!r} of kind {node.kind!r} has no incident edges",
                    node.id,
                )
            )
    return issues


def secret_must_be_held(graph: WorldGraph) -> list[Issue]:
    """Every `secret` must be the destination of a `holds` edge.

    A floating secret is a sampling bug — the realizer wouldn't know
    where to seed the value.
    """
    held: set[str] = set()
    for edge in graph.edges.values():
        if edge.kind == "holds":
            held.add(edge.dst)
    issues: list[Issue] = []
    for node in graph.by_kind("secret"):
        if node.id not in held:
            issues.append(
                Issue(
                    "error",
                    "secret_not_held",
                    f"secret {node.id!r} is not held by any record",
                    node.id,
                )
            )
    return issues


def oracle_path_exists(graph: WorldGraph) -> list[Issue]:
    """At least one flag-kind secret must be reachable via an attack chain.

    Concrete check (deliberately permissive at the pack level):
        - There exists a flag-kind secret S
        - S is held by record R in data_store D
        - Some service V is backed_by D
        - V exposes some endpoint E
        - Some vulnerability affects E (or V directly)

    Runtime feasibility (does the chain *actually* work when the agent
    runs against the realized world?) is checked by the pentest family's
    `check_feasibility` — this invariant is the gross "the world isn't
    shaped impossibly" gate.
    """
    flags = [n for n in graph.by_kind("secret") if n.attrs.get("kind") == "flag"]
    if not flags:
        return [
            Issue(
                "error",
                "no_flag_secret",
                "no flag-kind secret in graph; agents cannot complete a task",
                "graph",
            )
        ]

    holds_by_secret: dict[str, str] = {}
    contains_by_record: dict[str, str] = {}
    backed_by_store: dict[str, list[str]] = {}
    exposes_by_service: dict[str, list[str]] = {}
    vuln_targets: set[str] = set()
    for edge in graph.edges.values():
        if edge.kind == "holds":
            holds_by_secret[edge.dst] = edge.src
        elif edge.kind == "contains":
            contains_by_record[edge.dst] = edge.src
        elif edge.kind == "backed_by":
            backed_by_store.setdefault(edge.dst, []).append(edge.src)
        elif edge.kind == "exposes":
            exposes_by_service.setdefault(edge.src, []).append(edge.dst)
        elif edge.kind == "affects":
            vuln_targets.add(edge.dst)

    issues: list[Issue] = []
    for flag in flags:
        record_id = holds_by_secret.get(flag.id)
        if record_id is None:
            continue  # secret_must_be_held will flag this
        store_id = contains_by_record.get(record_id)
        if store_id is None:
            issues.append(
                Issue(
                    "error",
                    "flag_record_unstored",
                    f"flag {flag.id!r}: holding record {record_id!r} not "
                    f"contained in any data_store",
                    flag.id,
                )
            )
            continue
        services = backed_by_store.get(store_id, [])
        if not services:
            issues.append(
                Issue(
                    "error",
                    "flag_store_unreachable",
                    f"flag {flag.id!r}: data_store {store_id!r} has no service "
                    f"backing it (no attack surface)",
                    flag.id,
                )
            )
            continue
        chain_found = False
        for service_id in services:
            if service_id in vuln_targets:
                chain_found = True
                break
            for endpoint_id in exposes_by_service.get(service_id, []):
                if endpoint_id in vuln_targets:
                    chain_found = True
                    break
            if chain_found:
                break
        if not chain_found:
            issues.append(
                Issue(
                    "error",
                    "no_oracle_chain",
                    f"flag {flag.id!r}: no vulnerability affects any service "
                    f"or endpoint in the path to it",
                    flag.id,
                )
            )
    return issues
