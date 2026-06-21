from __future__ import annotations

from collections.abc import Mapping

from graphschema import Issue, WorldGraph

_ORPHAN_EXEMPT: frozenset[str] = frozenset({"host", "network"})

_VULN_KINDS_REQUIRING_DB: frozenset[str] = frozenset({"sql_injection", "idor"})


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


def unique_vuln_per_endpoint(graph: WorldGraph) -> list[Issue]:
    """A vulnerability's (kind, endpoint) is its identity: the codegen renders one
    handler per pair, so a second same-kind vuln on the same endpoint is a dead node
    the graph over-claims. Reject it so generation and evolution stay coherent."""
    target_of: dict[str, str] = {}
    for edge in graph.edges.values():
        if edge.kind == "affects":
            target_of.setdefault(edge.src, edge.dst)
    seen: set[tuple[str, str]] = set()
    issues: list[Issue] = []
    for vuln in graph.by_kind("vulnerability"):
        target = target_of.get(vuln.id)
        if target is None:
            continue
        key = (str(vuln.attrs.get("kind")), target)
        if key in seen:
            issues.append(
                Issue(
                    "error",
                    "duplicate_vuln_on_endpoint",
                    f"a {key[0]} vulnerability already targets {target!r}",
                    vuln.id,
                )
            )
        seen.add(key)
    return issues


_CHAIN_PRODUCER_KINDS: frozenset[str] = frozenset(
    {"credential_leak", "credential_gated_relay"}
)
_CHAIN_GATE_KINDS: frozenset[str] = frozenset(
    {"credential_gated_relay", "credential_gated_flag"}
)


def credential_reuse_binding(graph: WorldGraph) -> list[Issue]:
    """Every `requires_credential` gate must obtain its credential from exactly
    one strictly-earlier hop on the `enables` chain, and the producing and
    gating vulns must keep their credential-chain kinds.

    Binds by credential-node identity + enable ordering; value-consistency is
    `credential_value_binding`'s job and cycle-freedom holds by DAG construction,
    while the handler param-name / response-shape contract stays the verifier's. The
    kind check is what stops a mutation that rewrites a chain vuln in place from
    leaving the binding structurally intact but the world unsolvable. Endpoints
    without a `requires_credential` edge are untouched.
    """
    producers: dict[str, list[str]] = {}
    affects: dict[str, list[str]] = {}
    enables: dict[str, set[str]] = {}
    for edge in graph.edges.values():
        if edge.kind == "produces":
            producers.setdefault(edge.dst, []).append(edge.src)
        elif edge.kind == "affects":
            affects.setdefault(edge.dst, []).append(edge.src)
        elif edge.kind == "enables":
            enables.setdefault(edge.src, set()).add(edge.dst)
    vuln_kind = {
        n.id: str(n.attrs.get("kind", "")) for n in graph.by_kind("vulnerability")
    }

    def reaches(src: str, dst: str) -> bool:
        seen: set[str] = set()
        stack = list(enables.get(src, ()))
        while stack:
            cur = stack.pop()
            if cur == dst:
                return True
            if cur in seen:
                continue
            seen.add(cur)
            stack.extend(enables.get(cur, ()))
        return False

    issues: list[Issue] = []
    for edge in graph.edges.values():
        if edge.kind != "produces":
            continue
        if vuln_kind.get(edge.src) not in _CHAIN_PRODUCER_KINDS:
            issues.append(
                Issue(
                    "error",
                    "credential_binding",
                    f"credential {edge.dst!r} is produced by {edge.src!r} of kind "
                    f"{vuln_kind.get(edge.src)!r}, not a credential-chain hop",
                    edge.src,
                )
            )
    for edge in graph.edges.values():
        if edge.kind != "requires_credential":
            continue
        endpoint_id, cred_id = edge.src, edge.dst
        produced_by = producers.get(cred_id, [])
        if len(produced_by) != 1:
            issues.append(
                Issue(
                    "error",
                    "credential_binding",
                    f"credential {cred_id!r} required by {endpoint_id!r} is "
                    f"produced by {len(produced_by)} hop(s), expected exactly 1",
                    endpoint_id,
                )
            )
            continue
        producer = produced_by[0]
        gates = [
            v
            for v in affects.get(endpoint_id, [])
            if vuln_kind.get(v) in _CHAIN_GATE_KINDS
        ]
        if not gates:
            issues.append(
                Issue(
                    "error",
                    "credential_binding",
                    f"endpoint {endpoint_id!r} requires a credential but no "
                    f"credential-gate vuln affects it",
                    endpoint_id,
                )
            )
            continue
        for gate in gates:
            if producer == gate or not reaches(producer, gate):
                issues.append(
                    Issue(
                        "error",
                        "credential_binding",
                        f"credential {cred_id!r}: producer {producer!r} is not a "
                        f"strictly-earlier hop than gate {gate!r}",
                        endpoint_id,
                    )
                )
    return issues


def credential_value_binding(graph: WorldGraph) -> list[Issue]:
    """The credential node is the single source of truth for its token value: the
    producing hop emits exactly the node's ``value_ref`` and the gating hop validates
    against exactly that same value. This catches drift between the node and the
    handler param-string copies -- a mutation or hand-edit that changes one but not
    the other, leaving the chain structurally bound yet unsolvable. The reference
    solver stays response-driven; this is the admission backstop that lets the graph
    node be authoritative.
    """
    cred_value = {
        n.id: str(n.attrs.get("value_ref", "")) for n in graph.by_kind("credential")
    }
    vuln = {n.id: n for n in graph.by_kind("vulnerability")}
    affects: dict[str, list[str]] = {}
    for edge in graph.edges.values():
        if edge.kind == "affects":
            affects.setdefault(edge.dst, []).append(edge.src)

    def _param(vuln_id: str, key: str) -> str | None:
        node = vuln.get(vuln_id)
        params = node.attrs.get("params") if node is not None else None
        if not isinstance(params, Mapping):
            return None
        value = params.get(key)
        return None if value is None else str(value)

    issues: list[Issue] = []
    for edge in graph.edges.values():
        if edge.kind == "produces":
            kind = str(vuln[edge.src].attrs.get("kind", "")) if edge.src in vuln else ""
            relay = kind == "credential_gated_relay"
            emitted = _param(edge.src, "next_credential" if relay else "credential")
            if emitted is not None and emitted != cred_value.get(edge.dst):
                issues.append(
                    Issue(
                        "error",
                        "credential_value",
                        f"producer {edge.src!r} emits a token that does not match "
                        f"credential node {edge.dst!r}'s value_ref",
                        edge.src,
                    )
                )
        elif edge.kind == "requires_credential":
            expected = cred_value.get(edge.dst)
            for gate in affects.get(edge.src, []):
                if str(vuln[gate].attrs.get("kind", "")) not in _CHAIN_GATE_KINDS:
                    continue
                got = _param(gate, "credential")
                if got is not None and got != expected:
                    issues.append(
                        Issue(
                            "error",
                            "credential_value",
                            f"gate {gate!r} validates a token that does not match "
                            f"credential node {edge.dst!r}'s value_ref",
                            gate,
                        )
                    )
    return issues


def _contains_value(obj: object, needle: str) -> bool:
    if isinstance(obj, str):
        return needle in obj
    if isinstance(obj, Mapping):
        return any(_contains_value(v, needle) for v in obj.values())
    if isinstance(obj, (list, tuple, set)):
        return any(_contains_value(v, needle) for v in obj)
    return False


def flag_confined_to_gate(graph: WorldGraph) -> list[Issue]:
    """In a credential-reuse chain the flag is served only by the terminal gate out of
    the HIDDEN ``secret_flag``, so the real value must appear nowhere else: the loot
    record holds a decoy and no PUBLIC node or vuln param carries it. Otherwise a single
    response-leak on an earlier hop would short-circuit the whole chain. Chain-only --
    a single-service world legitimately keeps the flag in its loot record.
    """
    if not any(
        v.attrs.get("kind") == "credential_gated_flag"
        for v in graph.by_kind("vulnerability")
    ):
        return []
    if "secret_flag" not in graph.nodes:
        return []  # presence is secret_must_be_held's job
    flag = str(graph.nodes["secret_flag"].attrs.get("value_ref", ""))
    if not flag:
        return []
    return [
        Issue(
            "error",
            "flag_short_circuit",
            f"node {nid!r} ({node.kind}) exposes the real flag value outside the "
            f"gated secret -- the chain could be short-circuited",
            nid,
        )
        for nid, node in graph.nodes.items()
        if nid != "secret_flag" and _contains_value(node.attrs, flag)
    ]


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
