"""Curriculum-driven mutation proposals for the webapp pack (NEW shape).

These functions enumerate candidate `Mutation`s a TaskFamily may apply
to evolve a snapshot. Each candidate carries a `GraphPatch` (the
universal diff type from `openrange.world_ir`), a direction tag
(`harden` / `soften` / `diversify`), a relevance score (0..1), and the
requesting family's id. Core's `Builder.evolve(snapshot, mutation)`
returns the patch verbatim by default; `apply_patch` then applies it.

The shape change vs the v1 module:
  - No more `apply_curriculum(graph, dict_directive, rng)` mutating
    in place. Mutations carry a `GraphPatch`; core applies it.
  - No more `directive: Mapping[str, object]` on the Mutation. The
    `GraphPatch` is the directive.
  - `available_mutations` is no longer pack-level; the per-family
    `WebappBuild.available_mutations` / `WebappPentest.available_mutations`
    methods delegate here with their own `family_id` so each family's
    proposals carry the right `family` tag.

The semantic content is preserved:
  - For each catalog kind ABSENT from the world: propose ADDING a new
    vulnerability (a "harden" move under the new direction vocabulary —
    the world gains a defensive surface to test).
  - For each kind PRESENT in the world: propose REMOVING all vulns of
    that kind (a "soften" move — fewer paths for an exploit to land).
    Relevance is scored by how much successful agent traffic landed on
    those vulns' endpoints; a no-signal floor keeps the move available.
  - For each kind PRESENT in the world: propose SWAPPING a vuln of that
    kind to a different catalog kind (a "diversify" move — `nodes_updated`
    replaces the vuln in place; affected edges retain the same src/dst).

Edge ids are synthesized deterministically from `src/kind/dst` so two
proposals against the same graph state produce the same patch object.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

from cyber_webapp.ontology_v2 import ONTOLOGY_ID
from cyber_webapp.vulnerabilities import CATALOG as VULN_CATALOG
from openrange.core.contracts import EpisodeReportLike, Mutation
from openrange.world_ir import Edge, GraphPatch, Node, Visibility, WorldGraph

# Avoid noise about the ontology import: it's used as a sanity reference
# even though we don't read fields off it at runtime (the validator does).
_ = ONTOLOGY_ID

# Tiny baseline so a "soften by removing this kind" pick is always
# available even when the agent passed without our path-hit heuristic
# detecting the exploit on those endpoints.
_REMOVE_RELEVANCE_FLOOR = 0.05

# Static relevance for "introduce a new kind that is absent from the world."
# No agent-data signal possible for a kind that doesn't exist yet, so the
# score is a fixed mid-value.
_ADD_ABSENT_RELEVANCE = 0.5

# Static relevance for "swap a present kind to a different catalog kind."
# Less drastic than fully removing all instances; gives the curriculum a
# way to keep the attack-surface count steady while rotating which exploit
# the agent has to learn.
_SWAP_PRESENT_RELEVANCE = 0.2


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


def coerce_string_list(value: object) -> list[str]:
    """Normalize a string-list field (string, list, set, tuple).

    Retained from the v1 module as a small general-purpose helper. The
    curriculum-directive paths that depended on it are gone in the new
    shape (patches are typed dataclasses, not free-form mappings), but
    callers that hand-author manifest overrides may still want it.
    """
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
    """Procedural enumeration of webapp-pack mutation candidates.

    Returns a deterministic tuple of `Mutation` candidates the curriculum
    may pick from. The same `(graph, family_id, reports)` triple always
    yields the same tuple in the same order.

    `family_id` tags each emitted `Mutation` so the curriculum knows
    which family proposed it. The set of proposals does not currently
    depend on `family_id` — both `WebappBuild` and `WebappPentest`
    benefit from the same vuln-set perturbations — but the parameter is
    plumbed so per-family proposal sets can diverge later without a
    signature change.
    """
    vulns_by_kind = _vulns_by_kind(graph)
    paths_per_vuln = _affected_paths_per_vuln(graph)
    path_hits = _successful_path_hits(reports)

    options: list[Mutation] = []

    # harden: add an absent vuln kind (new defensive surface for the agent
    # to discover and exploit).
    options.extend(
        _harden_add_absent_mutations(graph, family_id, vulns_by_kind),
    )

    # soften: remove all instances of a present vuln kind. Score by how
    # much successful agent traffic landed on those instances' endpoints.
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

    # diversify: swap one vuln of a present kind to a different catalog
    # kind. The vuln's node id is reused (the patch updates in place); the
    # affects edge id is preserved so downstream consumers don't see edge
    # churn for a logical kind swap.
    options.extend(
        _diversify_swap_kind_mutations(graph, family_id, vulns_by_kind),
    )

    return tuple(options)


# ---------------------------------------------------------------------------
# Patch builders — one helper per direction
# ---------------------------------------------------------------------------


def _harden_add_absent_mutations(
    graph: WorldGraph,
    family_id: str,
    vulns_by_kind: Mapping[str, Sequence[str]],
) -> list[Mutation]:
    """Emit one ADD-vuln Mutation per catalog kind absent from the world."""
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

    # which (kind, target_id) pairs already exist — don't propose a vuln of
    # the same kind on a node that already carries one.
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
        existing_node_ids.add(vuln_id)  # keep subsequent picks distinct
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
    """Emit one REMOVE-vulns-of-kind Mutation for a present kind.

    The patch removes the vuln nodes themselves; `apply_patch` then drops
    any dangling edges automatically (see `world_ir.apply_patch`), so we
    don't need to enumerate `affects` / `enables` edges here. We DO drop
    them explicitly anyway so the patch reads as a complete diff and so
    callers inspecting `edges_removed` see the full picture.
    """
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
    """Emit one SWAP-kind Mutation per present vuln kind.

    Picks the first vuln node of each present kind (sorted by id for
    determinism) and proposes updating it in place to a different
    catalog kind whose target_kinds overlap the current target's kind.
    The affects edge keeps its id (deterministic from src/kind/dst); the
    edge's attrs are also kept since the injection site doesn't change.
    """
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
        # Pick a different catalog kind whose target_kinds include this
        # target's kind, and that isn't already on this target.
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


# ---------------------------------------------------------------------------
# Graph-walking helpers
# ---------------------------------------------------------------------------


def _vulns_by_kind(graph: WorldGraph) -> dict[str, list[str]]:
    by_kind: dict[str, list[str]] = {}
    for node in graph.by_kind("vulnerability"):
        attr_kind = str(node.attrs.get("kind", ""))
        if attr_kind:
            by_kind.setdefault(attr_kind, []).append(node.id)
    # stable iteration order for downstream determinism
    return {k: sorted(v) for k, v in by_kind.items()}


def _affected_paths_per_vuln(graph: WorldGraph) -> dict[str, set[str]]:
    """Map each vuln node id to the set of HTTP paths of endpoints it affects."""
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
    """Set of (vuln_kind, target_node_id) pairs already wired in the graph."""
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
    """Return the first node id this vuln points at via an `affects` edge."""
    for edge in graph.out_edges(vuln_id, "affects"):
        return edge.dst
    return None


def _oracle_path_targets(graph: WorldGraph) -> tuple[set[str], set[str]]:
    """Walk the flag → service → endpoint chain so `add` proposals land on it.

    Returns `(oracle_endpoint_ids, oracle_service_ids)`: every endpoint
    and service on the path from a `flag` secret back to an exposed
    surface. Empty sets when the graph has no flag (callers fall through
    to plain ordering).
    """
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


# ---------------------------------------------------------------------------
# Relevance scoring
# ---------------------------------------------------------------------------


@runtime_checkable
class _ReportWithFinalState(Protocol):
    """Local widening of `EpisodeReportLike` for path-hit scoring.

    The core protocol only declares `passed`. The relevance heuristic
    here needs `final_state["requests"]`. Reports lacking the attribute
    just contribute no signal — `_successful_path_hits` skips them
    silently and the floor relevance still keeps every candidate
    available.
    """

    @property
    def final_state(self) -> Mapping[str, Any]: ...


def _successful_path_hits(
    reports: Sequence[EpisodeReportLike],
) -> dict[str, int]:
    """Count non-error path hits across reports.

    Filters 4xx/5xx — we want paths the agent successfully interacted
    with, not paths it probed and got rejected on. Reports without a
    `final_state` attribute contribute zero (the heuristic degrades
    gracefully rather than refusing to score).
    """
    counts: dict[str, int] = {}
    for report in reports:
        if not isinstance(report, _ReportWithFinalState):
            continue
        requests_value = report.final_state.get("requests")
        if not isinstance(requests_value, list | tuple):
            continue
        for row in requests_value:
            if not isinstance(row, Mapping):
                continue
            try:
                status = int(row.get("status", 0))
            except TypeError, ValueError:
                continue
            if status >= 400:
                continue
            path = str(row.get("path", ""))
            if path:
                counts[path] = counts.get(path, 0) + 1
    return counts


def _exploitation_score(
    vuln_node_ids: Sequence[str],
    paths_per_vuln: Mapping[str, set[str]],
    path_hits: Mapping[str, int],
) -> float:
    """Fraction of successful agent requests that hit endpoints carrying
    a vuln of the given kind. 0..1; 0 if no signal."""
    if not path_hits:
        return 0.0
    affected: set[str] = set()
    for node_id in vuln_node_ids:
        affected.update(paths_per_vuln.get(node_id, ()))
    hits = sum(path_hits.get(p, 0) for p in affected)
    total = sum(path_hits.values())
    return min(1.0, hits / max(1, total))


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------


def _fresh_vuln_id(kind: str, existing_ids: set[str]) -> str:
    """Mint a deterministic, collision-free vuln node id for the given kind."""
    index = 0
    while f"vuln_{kind}_{index}" in existing_ids:
        index += 1
    return f"vuln_{kind}_{index}"


def _edge_id(src: str, kind: str, dst: str) -> str:
    """Deterministic edge id synthesized from `src/kind/dst`.

    Edge ids in the meta-model are just opaque strings, but admission
    needs them unique. Synthesizing from the triple keeps the same
    semantic edge stable across patches and avoids id collisions when
    several proposals are inspected side-by-side.
    """
    return f"{src}--{kind}-->{dst}"


def _pick_alt_kind(
    current_kind: str,
    target: Node,
    existing_kinds_by_target: set[tuple[str, str]],
) -> str | None:
    """Pick a different catalog kind compatible with `target`'s kind.

    Deterministic — iterates the catalog in sorted order and returns
    the first kind that (a) isn't `current_kind`, (b) targets the same
    kind as `target`, and (c) isn't already wired to this target.
    """
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


# Vuln-parameter pools, mirrored from sampling.py so a freshly minted vuln
# carries plausible parameters without depending on sampling.py's types.
# Each pool is deterministically indexed by the target id's hash so two
# calls with the same (kind, target_id) yield the same params.
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
    """Deterministic 0..modulo-1 index derived from `seed`.

    Avoids pulling `random.Random` into a function that must be a pure
    map from `(graph, family_id, reports)` to mutations.
    """
    if modulo <= 0:
        return 0
    digest = hashlib.sha256(seed.encode()).digest()
    return int.from_bytes(digest[:8], "big") % modulo


def _default_vuln_params(kind: str, target_id: str) -> dict[str, object]:
    """Deterministic per-target params for a vuln of `kind`.

    Mirrors the structure of `cyber_webapp.sampling.default_vuln_params`
    but is keyed by a hash of `(kind, target_id)` instead of an rng so
    `available_mutations` stays a pure function. The exact strings come
    from the same pools sampling.py uses, so a curriculum-introduced
    vuln is indistinguishable from a sampler-introduced one at the
    template-rendering layer.
    """
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


__all__ = [
    "available_mutations",
    "coerce_string_list",
]
