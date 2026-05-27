from __future__ import annotations

from graphschema import Issue, WorldGraph

_ORPHAN_EXEMPT: frozenset[str] = frozenset({"host", "network"})

_VULN_KINDS_REQUIRING_DB: frozenset[str] = frozenset({"sql_injection"})


def no_orphan_nodes(graph: WorldGraph) -> list[Issue]:
    """Every non-exempt node touches at least one edge. Exempt: `_ORPHAN_EXEMPT`."""
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
    """Every `secret` is the destination of a `holds` edge."""
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
    """A flag-kind secret S is reachable via:
    flag ← holds ← record ← contains ← data_store ← backed_by ← service,
    and that service (or one of its endpoints) is targeted by a vulnerability."""
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
            continue
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


def sqli_targets_db_backed_service(graph: WorldGraph) -> list[Issue]:
    """SQL-injection vulns must target endpoints of services with a
    `backed_by` data_store edge (else the handler queries nothing)."""
    db_backed_services: set[str] = {
        e.src for e in graph.edges.values() if e.kind == "backed_by"
    }
    service_of_endpoint: dict[str, str] = {
        e.dst: e.src for e in graph.edges.values() if e.kind == "exposes"
    }
    issues: list[Issue] = []
    for vuln in graph.by_kind("vulnerability"):
        if str(vuln.attrs.get("kind", "")) not in _VULN_KINDS_REQUIRING_DB:
            continue
        for affects in graph.out_edges(vuln.id, "affects"):
            target = graph.nodes.get(affects.dst)
            if target is None:
                continue
            if target.kind == "service":
                service_id = target.id
            elif target.kind == "endpoint":
                service_id = service_of_endpoint.get(target.id, "")
            else:
                continue
            if service_id not in db_backed_services:
                issues.append(
                    Issue(
                        "error",
                        "sqli_without_db_backing",
                        f"vuln {vuln.id!r} (kind={vuln.attrs.get('kind')!r}) "
                        f"affects {target.id!r} on service {service_id!r} which "
                        f"has no backed_by data_store; the realized handler "
                        f"would query a non-existent table",
                        vuln.id,
                    )
                )
    return issues
