"""Curriculum-driven mutation proposals for the webapp pack."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence

from graphschema import Edge, GraphPatch, Node, Visibility, WorldGraph
from openrange_pack_sdk import EpisodeReportLike, Mutation

from cyber_webapp.ontology import ONTOLOGY_ID
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

    options.extend(
        _harden_add_absent_mutations(graph, family_id, vulns_by_kind),
    )

    for kind, node_ids in vulns_by_kind.items():
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
    oracle_endpoints, oracle_services = _oracle_path_targets(graph)
    endpoints_oracle_first = sorted(
        endpoints,
        key=lambda n: (0 if n.id in oracle_endpoints else 1, n.id),
    )
    services_oracle_first = sorted(
        services,
        key=lambda n: (0 if n.id in oracle_services else 1, n.id),
    )

    existing_kinds_by_target = _existing_kinds_by_target(graph)
    existing_node_ids = set(graph.nodes.keys())

    mutations: list[Mutation] = []
    for kind in sorted(VULN_CATALOG):
        if kind in vulns_by_kind:
            continue
        catalog_entry = VULN_CATALOG[kind]
        target_kinds = catalog_entry.target_kinds
        candidates: Sequence[Node]
        if "endpoint" in target_kinds:
            candidates = endpoints_oracle_first
        elif "service" in target_kinds:
            candidates = services_oracle_first
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
    existing_kinds_by_target = _existing_kinds_by_target(graph)
    mutations: list[Mutation] = []
    for kind in sorted(vulns_by_kind):
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
        if alt == current_kind:
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
