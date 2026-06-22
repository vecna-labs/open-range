"""Curriculum-driven mutation proposals for the webapp pack."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence

from graphschema import Edge, GraphPatch, Node, Visibility, WorldGraph
from openrange_pack_sdk import EpisodeReportLike, Mutation, Snapshot

from cyber_webapp.ontology import ONTOLOGY_ID
from cyber_webapp.sampling import _INTERNAL_ONLY_KINDS, _is_networked
from cyber_webapp.vulnerabilities import CATALOG as VULN_CATALOG

# Keep the import alive even though only the validator reads ONTOLOGY_ID.
_ = ONTOLOGY_ID

# Floor so a "soften by removing this kind" pick is always available
# even when the path-hit heuristic detects nothing.
_REMOVE_RELEVANCE_FLOOR = 0.05

# Fixed mid-value: no agent-data signal exists for a kind that isn't in
# the world yet.
_ADD_ABSENT_RELEVANCE = 0.5

# Less drastic than fully removing all instances; rotates which exploit
# the agent has to learn while holding attack-surface count steady.
_SWAP_PRESENT_RELEVANCE = 0.2

# Above the decoy harden (0.5) so deepening the chain is the preferred frontier
# step: it extends the required skill instead of adding an off-path vuln.
_APPEND_HOP_RELEVANCE = 0.9

_GATE_PATH = "/internal/vault"
_TOKEN_PARAMS: tuple[str, ...] = ("token", "api_key", "auth", "session", "key")
_FOOTHOLD_KIND = "ssrf"


def _protected_kinds(graph: WorldGraph) -> frozenset[str]:
    if _is_networked(graph):
        return _INTERNAL_ONLY_KINDS | {_FOOTHOLD_KIND}
    return _INTERNAL_ONLY_KINDS


def coerce_string_list(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list | tuple | frozenset | set):
        return [str(v) for v in value]
    return []


def available_mutations(
    graph: WorldGraph,
    family_id: str,
    reports: Sequence[EpisodeReportLike],
) -> tuple[Mutation, ...]:
    """Deterministic in `(graph, family_id, reports)`."""
    vulns_by_kind = _vulns_by_kind(graph)
    paths_per_vuln = _affected_paths_per_vuln(graph)
    path_hits = _successful_path_hits(reports)

    options: list[Mutation] = []

    hop = _harden_append_hop_mutation(graph, family_id)
    if hop is not None:
        options.append(hop)

    options.extend(
        _harden_add_absent_mutations(graph, family_id, vulns_by_kind),
    )

    protected = _protected_kinds(graph)
    for kind, node_ids in vulns_by_kind.items():
        if kind in protected:
            continue
        score = _exploitation_score(node_ids, paths_per_vuln, path_hits)
        relevance = max(score, _REMOVE_RELEVANCE_FLOOR)
        options.append(
            _soften_remove_kind_mutation(
                graph,
                family_id,
                kind,
                node_ids,
                relevance,
                score,
            ),
        )

    options.extend(
        _diversify_swap_kind_mutations(graph, family_id, vulns_by_kind),
    )

    return tuple(options)


def _harden_add_absent_mutations(
    graph: WorldGraph,
    family_id: str,
    vulns_by_kind: Mapping[str, Sequence[str]],
) -> list[Mutation]:
    endpoints = list(graph.by_kind("endpoint"))
    services = list(graph.by_kind("service"))
    if not endpoints and not services:
        return []
    # Prefer off-oracle surfaces for the decoy: a record-reading vuln on the
    # flag's own surface can become a second path to it (easier). Only a
    # preference — auto_evolve's consequence gate is the actual safeguard.
    oracle_endpoints, oracle_services = _oracle_path_targets(graph)
    endpoints_decoy_first = sorted(
        endpoints,
        key=lambda n: (1 if n.id in oracle_endpoints else 0, n.id),
    )
    services_decoy_first = sorted(
        services,
        key=lambda n: (1 if n.id in oracle_services else 0, n.id),
    )

    existing_kinds_by_target = _existing_kinds_by_target(graph)
    existing_node_ids = set(graph.nodes.keys())

    mutations: list[Mutation] = []
    for kind in sorted(VULN_CATALOG):
        if kind in vulns_by_kind or kind in _INTERNAL_ONLY_KINDS:
            continue
        catalog_entry = VULN_CATALOG[kind]
        target_kinds = catalog_entry.target_kinds
        candidates: Sequence[Node]
        if "endpoint" in target_kinds:
            candidates = endpoints_decoy_first
        elif "service" in target_kinds:
            candidates = services_decoy_first
        else:
            continue
        target = next(
            (t for t in candidates if (kind, t.id) not in existing_kinds_by_target),
            None,
        )
        if target is None:
            continue
        vuln_id = _fresh_vuln_id(kind, existing_node_ids)
        existing_node_ids.add(vuln_id)
        vuln_node = Node(
            id=vuln_id,
            kind="vulnerability",
            attrs={
                "kind": kind,
                "family": catalog_entry.family,
                "params": _default_vuln_params(kind, target.id),
            },
            visibility=Visibility.HIDDEN,
        )
        affects_edge = Edge(
            id=_edge_id(vuln_id, "affects", target.id),
            kind="affects",
            src=vuln_id,
            dst=target.id,
            attrs={"injection_site": str(target.attrs.get("path", "service"))},
        )
        patch = GraphPatch(
            nodes_added=[vuln_node],
            edges_added=[affects_edge],
        )
        mutations.append(
            Mutation(
                patch=patch,
                direction="harden",
                relevance=_ADD_ABSENT_RELEVANCE,
                family=family_id,
                note=f"add {kind} on {target.id}",
            ),
        )
    return mutations


def _soften_remove_kind_mutation(
    graph: WorldGraph,
    family_id: str,
    kind: str,
    vuln_node_ids: Sequence[str],
    relevance: float,
    score: float,
) -> Mutation:
    # ``apply_patch`` drops dangling edges automatically; we enumerate
    # them anyway so the patch reads as a complete diff and so callers
    # inspecting ``edges_removed`` see the full picture.
    edge_ids: list[str] = []
    vuln_id_set = set(vuln_node_ids)
    for edge in graph.edges.values():
        if edge.src in vuln_id_set or edge.dst in vuln_id_set:
            edge_ids.append(edge.id)
    patch = GraphPatch(
        nodes_removed=list(vuln_node_ids),
        edges_removed=edge_ids,
    )
    return Mutation(
        patch=patch,
        direction="soften",
        relevance=relevance,
        family=family_id,
        note=(
            f"remove {kind} ({len(vuln_node_ids)} instance(s); "
            f"exploit score {score:.2f})"
        ),
    )


def _diversify_swap_kind_mutations(
    graph: WorldGraph,
    family_id: str,
    vulns_by_kind: Mapping[str, Sequence[str]],
) -> list[Mutation]:
    # In-place update — affects edge keeps its id since src/kind/dst are unchanged.
    if not vulns_by_kind:
        return []
    networked = _is_networked(graph)
    protected = _protected_kinds(graph)
    existing_kinds_by_target = _existing_kinds_by_target(graph)
    mutations: list[Mutation] = []
    for kind in sorted(vulns_by_kind):
        if kind in protected:
            continue
        node_ids = sorted(vulns_by_kind[kind])
        if not node_ids:
            continue
        vuln_node = graph.nodes.get(node_ids[0])
        if vuln_node is None:
            continue
        target_id = _affects_target_id(graph, vuln_node.id)
        if target_id is None:
            continue
        target = graph.nodes.get(target_id)
        if target is None:
            continue
        alt_kind = _pick_alt_kind(
            current_kind=kind,
            target=target,
            existing_kinds_by_target=existing_kinds_by_target,
        )
        if alt_kind is None:
            continue
        if networked and alt_kind == _FOOTHOLD_KIND:
            continue  # never introduce a second public foothold into a networked world
        alt_entry = VULN_CATALOG[alt_kind]
        updated_node = Node(
            id=vuln_node.id,
            kind="vulnerability",
            attrs={
                "kind": alt_kind,
                "family": alt_entry.family,
                "params": _default_vuln_params(alt_kind, target.id),
            },
            visibility=Visibility.HIDDEN,
        )
        patch = GraphPatch(nodes_updated=[updated_node])
        mutations.append(
            Mutation(
                patch=patch,
                direction="diversify",
                relevance=_SWAP_PRESENT_RELEVANCE,
                family=family_id,
                note=f"swap {vuln_node.id} from {kind} to {alt_kind}",
            ),
        )
    return mutations


def _harden_append_hop_mutation(
    graph: WorldGraph,
    family_id: str,
) -> Mutation | None:
    # The flag's backing store is repointed to the new host so it owns the flag under
    # the per-service scoped seed the CONTAINER backing uses, not only the shared
    # PROCESS seed — keeping the deepened world solvable across backings.
    terminal = _credential_gated_flag(graph)
    if terminal is None:
        return None
    term_ep = _affects_target_id(graph, terminal.id)
    if term_ep is None:
        return None
    store = _flag_store(graph)
    backed_by = next(
        (e for e in graph.edges.values() if e.kind == "backed_by" and e.dst == store),
        None,
    )
    if store is None or backed_by is None:
        return None
    new_host = _spare_internal_service(graph, _chain_service_ids(graph))
    if new_host is None:
        return None
    new_host_name = str(new_host.attrs.get("name", new_host.id))

    term_params = dict(terminal.attrs.get("params", {}))
    next_index = sum(
        1 for n in graph.by_kind("credential") if n.id.startswith("cred_chain_")
    )
    new_cred_id = f"cred_chain_{next_index}"
    seed = f"{terminal.id}:append:{next_index}"
    new_token = hashlib.sha256(seed.encode()).hexdigest()[:24]
    new_param = _TOKEN_PARAMS[_stable_index(seed + ":param", len(_TOKEN_PARAMS))]
    new_ep_id = f"ep_{new_host_name}_vault"
    new_flag_id = _fresh_vuln_id("credential_gated_flag", set(graph.nodes))

    relay = Node(
        id=terminal.id,
        kind="vulnerability",
        attrs={
            "kind": "credential_gated_relay",
            "family": str(terminal.attrs.get("family", "code_web")),
            "params": {
                "credential": term_params.get("credential", ""),
                "token_param": term_params.get("token_param", "token"),
                "next_credential": new_token,
                "next_vault_host": new_host_name,
                "next_vault_path": _GATE_PATH,
                "next_token_param": new_param,
            },
        },
        visibility=Visibility.HIDDEN,
    )
    cred = Node(
        id=new_cred_id,
        kind="credential",
        attrs={"kind": "token", "value_ref": new_token},
    )
    endpoint = Node(
        id=new_ep_id,
        kind="endpoint",
        attrs={
            "path": _GATE_PATH,
            "public_url": f"/svc/{new_host_name}{_GATE_PATH}",
            "method": "GET",
            "auth_required": True,
            "behavior_ref": "credential.gate",
        },
    )
    flag = Node(
        id=new_flag_id,
        kind="vulnerability",
        attrs={
            "kind": "credential_gated_flag",
            "family": "code_web",
            "params": {"credential": new_token, "token_param": new_param},
        },
        visibility=Visibility.HIDDEN,
    )
    patch = GraphPatch(
        nodes_added=[cred, endpoint, flag],
        nodes_updated=[relay],
        edges_removed=[backed_by.id],
        edges_added=[
            _chain_edge(new_host.id, "exposes", new_ep_id),
            _chain_edge(new_flag_id, "affects", new_ep_id, _GATE_PATH),
            _chain_edge(terminal.id, "enables", new_flag_id),
            _chain_edge(new_ep_id, "requires_credential", new_cred_id),
            _chain_edge(terminal.id, "produces", new_cred_id),
            _chain_edge(new_host.id, "backed_by", store),
        ],
    )
    return Mutation(
        patch=patch,
        direction="harden",
        relevance=_APPEND_HOP_RELEVANCE,
        family=family_id,
        note=f"append a credential hop via {new_host_name}",
    )


def _chain_edge(src: str, kind: str, dst: str, injection_site: str = "") -> Edge:
    attrs = {"injection_site": injection_site} if injection_site else {}
    return Edge(id=_edge_id(src, kind, dst), kind=kind, src=src, dst=dst, attrs=attrs)


def _credential_gated_flag(graph: WorldGraph) -> Node | None:
    for node in graph.by_kind("vulnerability"):
        if node.attrs.get("kind") == "credential_gated_flag":
            return node
    return None


def _service_of_endpoint(graph: WorldGraph, endpoint_id: str) -> Node | None:
    for edge in graph.in_edges(endpoint_id, "exposes"):
        return graph.nodes.get(edge.src)
    return None


def _chain_service_ids(graph: WorldGraph) -> set[str]:
    used: set[str] = set()
    for vuln in graph.by_kind("vulnerability"):
        if not str(vuln.attrs.get("kind", "")).startswith("credential"):
            continue
        endpoint_id = _affects_target_id(graph, vuln.id)
        service = _service_of_endpoint(graph, endpoint_id) if endpoint_id else None
        if service is not None:
            used.add(service.id)
    return used


def _spare_internal_service(graph: WorldGraph, used: set[str]) -> Node | None:
    spares = [
        node
        for node in graph.by_kind("service")
        if node.attrs.get("exposure") != "public" and node.id not in used
    ]
    return min(spares, key=lambda n: n.id, default=None)


def _flag_store(graph: WorldGraph) -> str | None:
    flag = next(
        (n for n in graph.by_kind("secret") if n.attrs.get("kind") == "flag"), None
    )
    if flag is None:
        return None
    record = next(
        (e.src for e in graph.edges.values() if e.kind == "holds" and e.dst == flag.id),
        None,
    )
    if record is None:
        return None
    return next(
        (
            e.src
            for e in graph.edges.values()
            if e.kind == "contains" and e.dst == record
        ),
        None,
    )


def _credential_walk(graph: WorldGraph) -> list[tuple[str, str]]:
    enables: dict[str, list[str]] = {}
    for edge in graph.edges.values():
        if edge.kind == "enables":
            enables.setdefault(edge.src, []).append(edge.dst)
    current = next(
        (
            n
            for n in graph.by_kind("vulnerability")
            if n.attrs.get("kind") == "credential_leak"
        ),
        None,
    )
    walk: list[tuple[str, str]] = []
    seen: set[str] = set()
    while current is not None and current.id not in seen:
        seen.add(current.id)
        kind = str(current.attrs.get("kind", ""))
        if kind in ("credential_gated_relay", "credential_gated_flag"):
            endpoint_id = _affects_target_id(graph, current.id)
            service = _service_of_endpoint(graph, endpoint_id) if endpoint_id else None
            host = str(service.attrs.get("name", service.id)) if service else ""
            cred = str(dict(current.attrs.get("params", {})).get("credential", ""))
            walk.append((host, cred))
            if kind == "credential_gated_flag":
                break
        following = sorted(enables.get(current.id, []))
        current = next(
            (
                graph.nodes[nid]
                for nid in following
                if nid in graph.nodes
                and str(graph.nodes[nid].attrs.get("kind", "")).startswith(
                    "credential_gated"
                )
            ),
            None,
        )
    return walk


def monotone_chain_gate(parent: Snapshot) -> Callable[[Snapshot, Mutation], bool]:
    """Admit a child only if it extends the parent's credential chain by exactly
    one hop: the parent's solve walk must be a literal prefix of the child's. A
    parent with no chain to extend is rejected.
    """
    parent_walk = _credential_walk(parent.graph)

    def gate(evolved: Snapshot, mutation: Mutation) -> bool:
        del mutation
        if not parent_walk:
            return False
        child_walk = _credential_walk(evolved.graph)
        return (
            len(child_walk) == len(parent_walk) + 1
            and child_walk[: len(parent_walk)] == parent_walk
        )

    return gate


def _vulns_by_kind(graph: WorldGraph) -> dict[str, list[str]]:
    by_kind: dict[str, list[str]] = {}
    for node in graph.by_kind("vulnerability"):
        attr_kind = str(node.attrs.get("kind", ""))
        if attr_kind:
            by_kind.setdefault(attr_kind, []).append(node.id)
    return {k: sorted(v) for k, v in by_kind.items()}


def _affected_paths_per_vuln(graph: WorldGraph) -> dict[str, set[str]]:
    paths: dict[str, set[str]] = {}
    for edge in graph.edges.values():
        if edge.kind != "affects":
            continue
        vuln = graph.nodes.get(edge.src)
        target = graph.nodes.get(edge.dst)
        if vuln is None or vuln.kind != "vulnerability" or target is None:
            continue
        path = str(target.attrs.get("path", ""))
        if path:
            paths.setdefault(edge.src, set()).add(path)
    return paths


def _existing_kinds_by_target(graph: WorldGraph) -> set[tuple[str, str]]:
    out: set[tuple[str, str]] = set()
    for edge in graph.edges.values():
        if edge.kind != "affects":
            continue
        source_node = graph.nodes.get(edge.src)
        if source_node is None or source_node.kind != "vulnerability":
            continue
        vuln_kind = str(source_node.attrs.get("kind", ""))
        if not vuln_kind:
            continue
        out.add((vuln_kind, edge.dst))
    return out


def _affects_target_id(graph: WorldGraph, vuln_id: str) -> str | None:
    for edge in graph.out_edges(vuln_id, "affects"):
        return edge.dst
    return None


def _oracle_path_targets(graph: WorldGraph) -> tuple[set[str], set[str]]:
    """Returns `(endpoint_ids, service_ids)` on the path from a flag
    secret back to an exposed surface. Both empty when no flag exists."""
    flag_secret_ids = {
        n.id for n in graph.by_kind("secret") if n.attrs.get("kind") == "flag"
    }
    if not flag_secret_ids:
        return set(), set()
    holding_record_ids: set[str] = set()
    for e in graph.edges.values():
        if e.kind == "holds" and e.dst in flag_secret_ids:
            holding_record_ids.add(e.src)
    holding_store_ids: set[str] = set()
    for e in graph.edges.values():
        if e.kind == "contains" and e.dst in holding_record_ids:
            holding_store_ids.add(e.src)
    backing_service_ids: set[str] = set()
    for e in graph.edges.values():
        if e.kind == "backed_by" and e.dst in holding_store_ids:
            backing_service_ids.add(e.src)
    oracle_endpoint_ids: set[str] = set()
    for e in graph.edges.values():
        if e.kind == "exposes" and e.src in backing_service_ids:
            target = graph.nodes.get(e.dst)
            if target is not None and target.kind == "endpoint":
                oracle_endpoint_ids.add(e.dst)
    return oracle_endpoint_ids, backing_service_ids


def _successful_path_hits(
    reports: Sequence[EpisodeReportLike],
) -> dict[str, int]:
    # Status filtering already happened upstream — `requests_made` is the kept set.
    counts: dict[str, int] = {}
    for report in reports:
        requests_value = report.final_state.get("requests_made")
        if not isinstance(requests_value, list | tuple):
            continue
        for row in requests_value:
            if not isinstance(row, str):
                continue
            path = row.strip()
            if path:
                counts[path] = counts.get(path, 0) + 1
    return counts


def _exploitation_score(
    vuln_node_ids: Sequence[str],
    paths_per_vuln: Mapping[str, set[str]],
    path_hits: Mapping[str, int],
) -> float:
    if not path_hits:
        return 0.0
    affected: set[str] = set()
    for node_id in vuln_node_ids:
        affected.update(paths_per_vuln.get(node_id, ()))
    hits = sum(path_hits.get(p, 0) for p in affected)
    total = sum(path_hits.values())
    return min(1.0, hits / max(1, total))


def _fresh_vuln_id(kind: str, existing_ids: set[str]) -> str:
    index = 0
    while f"vuln_{kind}_{index}" in existing_ids:
        index += 1
    return f"vuln_{kind}_{index}"


def _edge_id(src: str, kind: str, dst: str) -> str:
    # Synthesizing from the triple keeps the same semantic edge stable
    # across patches and avoids id collisions when several proposals
    # are inspected side-by-side.
    return f"{src}--{kind}-->{dst}"


def _pick_alt_kind(
    current_kind: str,
    target: Node,
    existing_kinds_by_target: set[tuple[str, str]],
) -> str | None:
    target_node_kind = target.kind
    for alt in sorted(VULN_CATALOG):
        if alt == current_kind or alt in _INTERNAL_ONLY_KINDS:
            continue
        if (alt, target.id) in existing_kinds_by_target:
            continue
        if target_node_kind not in VULN_CATALOG[alt].target_kinds:
            continue
        return alt
    return None


# Vuln-parameter pools mirror sampling.py. Each pool is deterministically
# indexed by the target id's hash so two calls with the same (kind,
# target_id) yield the same params.
_SQLI_PARAMS: tuple[str, ...] = ("q", "query", "search", "term", "filter", "ref")
_SQLI_TABLES: tuple[str, ...] = (
    "records",
    "rows",
    "items",
    "data",
    "entries",
    "documents",
)
_SQLI_COLUMNS: tuple[str, ...] = (
    "value",
    "secret",
    "data",
    "content",
    "payload",
    "body",
)
_BROKEN_AUTHZ_HEADERS: tuple[str, ...] = (
    "X-User-Role",
    "X-Auth-Role",
    "X-Internal-Role",
    "X-Privilege",
    "X-Service-Role",
)
_BROKEN_AUTHZ_VALUES: tuple[str, ...] = (
    "admin",
    "root",
    "superuser",
    "operator",
    "service-admin",
)
_BROKEN_AUTHZ_FIELDS: tuple[str, ...] = (
    "value",
    "data",
    "secret",
    "content",
    "result",
)
_SSRF_PARAMS: tuple[str, ...] = (
    "url",
    "target",
    "endpoint",
    "callback",
    "redirect",
    "ref",
)
_SSRF_PATTERNS: tuple[str, ...] = (
    r"^https?://internal\.",
    r"^https?://int\.",
    r"^https?://private\.",
    r"^https?://corp\.",
    r"^https?://intranet\.",
)


def _stable_index(seed: str, modulo: int) -> int:
    if modulo <= 0:
        return 0
    digest = hashlib.sha256(seed.encode()).digest()
    return int.from_bytes(digest[:8], "big") % modulo


def _default_vuln_params(kind: str, target_id: str) -> dict[str, object]:
    # Keyed by a hash of (kind, target_id) instead of an rng so
    # ``available_mutations`` stays a pure function. Strings come from
    # the same pools sampling.py uses so a curriculum-introduced vuln
    # is indistinguishable from a sampler-introduced one at the
    # template-rendering layer.
    seed = f"{kind}:{target_id}"
    if kind == "sql_injection":
        return {
            "target_param": _SQLI_PARAMS[
                _stable_index(seed + ":param", len(_SQLI_PARAMS))
            ],
            "table": _SQLI_TABLES[_stable_index(seed + ":table", len(_SQLI_TABLES))],
            "leak_column": _SQLI_COLUMNS[
                _stable_index(seed + ":col", len(_SQLI_COLUMNS))
            ],
        }
    if kind == "ssrf":
        return {
            "target_param": _SSRF_PARAMS[
                _stable_index(seed + ":param", len(_SSRF_PARAMS))
            ],
            "allowlist_pattern": _SSRF_PATTERNS[
                _stable_index(seed + ":pat", len(_SSRF_PATTERNS))
            ],
        }
    if kind == "broken_authz":
        return {
            "trust_header": _BROKEN_AUTHZ_HEADERS[
                _stable_index(seed + ":hdr", len(_BROKEN_AUTHZ_HEADERS))
            ],
            "expected_value": _BROKEN_AUTHZ_VALUES[
                _stable_index(seed + ":val", len(_BROKEN_AUTHZ_VALUES))
            ],
            "leak_field": _BROKEN_AUTHZ_FIELDS[
                _stable_index(seed + ":fld", len(_BROKEN_AUTHZ_FIELDS))
            ],
        }
    return {}
