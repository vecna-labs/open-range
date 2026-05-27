"""Typed property-graph meta-model.

The fixed typed-property-graph layer this package is built on. It owns
`Node`, `Edge`, `WorldGraph`, the declarative `Ontology` schema, and the
three-tier `validate()` (structural, ontology conformance, caller
invariants).

This module is deliberately ontology-agnostic — it knows nothing about
any specific domain, nor any specific cognitive ontology. A graph's
domain meaning lives entirely in its `Ontology`, declared as data; one
generic validator checks any graph against any ontology.

The four primitives in this file are:

  * `Node` / `Edge`         — the data of a typed property graph.
  * `WorldGraph`            — a graph plus an ontology id and opaque
                              meta; supports `content_hash()` for
                              deterministic content-addressed identity.
  * `Ontology`, `NodeKind`, `EdgeKind`, `AttrSpec` — the declarative
                              schema; pure data, no runtime.
  * `validate()`            — three-tier validator returning structured
                              `Issue` values rather than raising.
  * `GraphPatch` + `apply_patch()` — universal diff type for mutating a
                              graph in place.

The JSON shape implied by `content_hash()` is intended to be the
on-the-wire contract: keys sorted, no whitespace, empty optional fields
omitted, `meta`/`runtime` excluded so two graphs that differ only in
provenance noise share identity.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class AttrType(StrEnum):
    """Allowed attribute primitive types in an ontology.

    String-valued so the enum *is* its wire form — `AttrType.STRING.value ==
    "string"` matches the lowercase wire tokens.
    """

    STRING = "string"
    INT = "int"
    FLOAT = "float"
    BOOL = "bool"
    ENUM = "enum"
    REF = "ref"
    JSON = "json"


class Visibility(StrEnum):
    """Whether a node is part of the surface an observer can see directly.

    `PUBLIC` is the default and is omitted on the wire. `HIDDEN` nodes still
    exist in the graph (an undisclosed asset, a private piece of state),
    but they are not part of the directly-observable surface — discovering
    hidden state may be the point.
    """

    PUBLIC = "public"
    HIDDEN = "hidden"


class Role(StrEnum):
    """The fixed, world-absolute role vocabulary the generic machinery reads.

    Roles are deliberately a small closed set so generic code can branch on
    them without knowing any domain. World-absolute means a role is true
    regardless of what task is being run; task-relative facts live on the
    caller's task spec, never here.
    """

    ACTOR = "actor"
    NPC = "npc"
    EXTERNAL = "external"


@dataclass
class AttrSpec:
    """One attribute slot declared by an ontology.

    `type` chooses the primitive (or `REF` for an id pointing at another node,
    or `JSON` for an opaque blob). `enum` is required when `type` is `ENUM` and
    lists the legal values; `ref_kinds`, when set on a `REF`, restricts the
    target node's kind. `default` is informational — the validator does not
    apply it; the caller decides whether to.
    """

    type: AttrType
    required: bool = False
    enum: list[str] | None = None
    ref_kinds: list[str] | None = None
    default: Any = None
    description: str = ""


@dataclass
class NodeKind:
    """A node-kind declaration in an ontology.

    `parent` names another kind whose attrs this kind inherits — the child's
    `attrs` override the parent's by key, and required attrs from the parent
    are still enforced. `attrs` is the *local* attr map (parent attrs are
    composed in by the validator).
    """

    id: str
    parent: str | None = None
    attrs: dict[str, AttrSpec] = field(default_factory=dict)
    description: str = ""


@dataclass
class EdgeKind:
    """An edge-kind declaration in an ontology.

    `endpoints` is the list of allowed `(src_kind, dst_kind)` pairs. `src_max`
    and `dst_max` are degree caps per node of the given side, both `None` for
    unbounded.
    """

    id: str
    endpoints: list[tuple[str, str]] = field(default_factory=list)
    src_max: int | None = None
    dst_max: int | None = None
    attrs: dict[str, AttrSpec] = field(default_factory=dict)
    description: str = ""


@dataclass
class Ontology:
    """A declared schema for one graph: node kinds and edge kinds.

    The schema is itself plain data so a single generic validator can check
    any graph against any ontology. Nothing hard-codes the meaning of a
    specific kind.

    Graph-wide invariants beyond the declared node/edge shape are passed in
    as callables to `validate()`, not stored on the Ontology — they are
    functions, not data, and they depend on caller-specific reasoning that
    an Ontology deliberately cannot express.
    """

    id: str
    node_kinds: dict[str, NodeKind] = field(default_factory=dict)
    edge_kinds: dict[str, EdgeKind] = field(default_factory=dict)


@dataclass
class Node:
    """A node in a typed property graph.

    `kind` is the ontology-defined vocabulary. `attrs` is the property bag,
    type-checked against the ontology. `roles` is a subset of the fixed `Role`
    enum. `runtime` and `meta` are opaque to the meta-model — the core never
    reads them, so callers can stash whatever they like there. `meta` and
    `runtime` are also excluded from `content_hash()`.
    """

    id: str
    kind: str
    attrs: dict[str, Any] = field(default_factory=dict)
    roles: set[Role] = field(default_factory=set)
    visibility: Visibility = Visibility.PUBLIC
    runtime: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Edge:
    """A directed typed edge from `src` to `dst`.

    Both `src` and `dst` must reference nodes in the same graph. `attrs` is the
    property bag, type-checked against the ontology's `EdgeKind.attrs`.
    """

    id: str
    kind: str
    src: str
    dst: str
    attrs: dict[str, Any] = field(default_factory=dict)


@dataclass
class Issue:
    """One validation finding.

    `severity` is `"error"` or `"warning"`. `code` is a stable machine-readable
    tag so callers can group / suppress findings without parsing `message`.
    `where` is the offending id (node id, edge id) or pseudo-path.
    """

    severity: str
    code: str
    message: str
    where: str


@dataclass
class WorldGraph:
    """A typed property graph: nodes, edges, an ontology id, opaque meta.

    `ontology` is the id string of the schema the graph is supposed to conform
    to. The graph does not carry the schema itself — `validate()` takes the
    ontology as a separate argument so the same graph can be checked against
    multiple revisions of its schema.

    `meta` is excluded from `content_hash()`: two graphs that differ only in
    provenance / manifest noise share the same content-addressed identity.
    """

    ontology: str
    nodes: dict[str, Node] = field(default_factory=dict)
    edges: dict[str, Edge] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def add_node(self, node: Node) -> None:
        """Insert a node. Raises `KeyError` on id collision."""
        if node.id in self.nodes:
            raise KeyError(f"duplicate node id: {node.id!r}")
        self.nodes[node.id] = node

    def add_edge(self, edge: Edge) -> None:
        """Insert an edge. Raises `KeyError` on id collision."""
        if edge.id in self.edges:
            raise KeyError(f"duplicate edge id: {edge.id!r}")
        self.edges[edge.id] = edge

    def in_edges(self, node_id: str, kind: str | None = None) -> list[Edge]:
        """All edges whose `dst` is `node_id`, optionally filtered by kind."""
        return [
            e
            for e in self.edges.values()
            if e.dst == node_id and (kind is None or e.kind == kind)
        ]

    def out_edges(self, node_id: str, kind: str | None = None) -> list[Edge]:
        """All edges whose `src` is `node_id`, optionally filtered by kind."""
        return [
            e
            for e in self.edges.values()
            if e.src == node_id and (kind is None or e.kind == kind)
        ]

    def by_kind(self, kind: str) -> list[Node]:
        """All nodes of a given kind. Order is insertion order."""
        return [n for n in self.nodes.values() if n.kind == kind]

    def content_hash(self) -> str:
        """`sha256:<hex>` over `ontology + nodes + edges` only.

        Excludes `meta`, plus each node's `runtime` and `meta`, so two graphs
        with the same logical content share one identity regardless of which
        producer made them. The serialization is deterministic: keys sorted,
        no whitespace, empty optional fields omitted.
        """
        nodes_data = [
            _node_data(n) for n in sorted(self.nodes.values(), key=lambda n: n.id)
        ]
        edges_data = [
            _edge_data(e) for e in sorted(self.edges.values(), key=lambda e: e.id)
        ]
        data = {
            "ontology": self.ontology,
            "nodes": nodes_data,
            "edges": edges_data,
        }
        encoded = json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
        return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _node_data(node: Node) -> dict[str, Any]:
    out: dict[str, Any] = {
        "id": node.id,
        "kind": node.kind,
        "attrs": dict(sorted(node.attrs.items())),
    }
    if node.roles:
        out["roles"] = sorted(r.value for r in node.roles)
    if node.visibility is not Visibility.PUBLIC:
        out["visibility"] = node.visibility.value
    return out


def _edge_data(edge: Edge) -> dict[str, Any]:
    out: dict[str, Any] = {
        "id": edge.id,
        "kind": edge.kind,
        "src": edge.src,
        "dst": edge.dst,
    }
    if edge.attrs:
        out["attrs"] = dict(sorted(edge.attrs.items()))
    return out


@dataclass
class GraphPatch:
    """A diff applied to a `WorldGraph` via `apply_patch`.

    Updates fully replace the existing node/edge (no per-attr merge) — the
    caller decides what the new shape is, and the patch carries the full
    replacement. Removals are ids only.
    """

    nodes_added: list[Node] = field(default_factory=list)
    nodes_updated: list[Node] = field(default_factory=list)
    nodes_removed: list[str] = field(default_factory=list)
    edges_added: list[Edge] = field(default_factory=list)
    edges_updated: list[Edge] = field(default_factory=list)
    edges_removed: list[str] = field(default_factory=list)


def apply_patch(graph: WorldGraph, patch: GraphPatch) -> WorldGraph:
    """Apply a `GraphPatch` to `graph` IN PLACE and return it.

    Mutates in place because a `WorldGraph` is already mutable and copying
    would be wasted work for the common builder-repair / accreting-memory
    update use case. The return value is the same object — returned only so
    callers can chain. Order: removals, then updates, then additions, so an
    update that happens to share an id with a removal in the same patch keeps
    the new value and an addition cannot collide with something the patch
    itself just removed.
    """
    for nid in patch.nodes_removed:
        graph.nodes.pop(nid, None)
        # an edge dangling on a removed node would silently violate the
        # structural rule, so drop them here. The patch can still add edges
        # back in the additions pass.
        for eid in [e.id for e in graph.edges.values() if e.src == nid or e.dst == nid]:
            graph.edges.pop(eid, None)
    for eid in patch.edges_removed:
        graph.edges.pop(eid, None)
    for node in patch.nodes_updated:
        graph.nodes[node.id] = node
    for edge in patch.edges_updated:
        graph.edges[edge.id] = edge
    for node in patch.nodes_added:
        if node.id in graph.nodes:
            raise KeyError(f"patch adds duplicate node id: {node.id!r}")
        graph.nodes[node.id] = node
    for edge in patch.edges_added:
        if edge.id in graph.edges:
            raise KeyError(f"patch adds duplicate edge id: {edge.id!r}")
        graph.edges[edge.id] = edge
    return graph


def validate(
    graph: WorldGraph,
    ontology: Ontology,
    invariants: list[Callable[[WorldGraph], list[Issue]]] | None = None,
) -> list[Issue]:
    """Three-tier validation: structural, ontology conformance, caller invariants.

    Tier 1 (structural) checks the shape of the graph regardless of any
    schema: required fields are present and stringly typed, ids are unique,
    and edge endpoints point at nodes that actually exist.

    Tier 2 (conformance) checks the graph against `ontology`: every node /
    edge kind is declared, every required attr is present, attr value types
    match their `AttrSpec`, enum values are in range, REF attrs resolve to a
    real node of an allowed kind, edge endpoints match a declared
    `(src_kind, dst_kind)` pair, and degree caps are respected. Parent NodeKind
    chains contribute required attrs.

    Tier 3 (invariants) runs each caller-supplied callable on the graph and
    concatenates its findings. Callers use these for the domain-specific
    invariants the generic schema cannot express (cross-cutting structural
    rules unique to a particular world-family).
    """
    issues: list[Issue] = []
    issues.extend(_validate_structural(graph))
    issues.extend(_validate_conformance(graph, ontology))
    for inv in invariants or []:
        issues.extend(inv(graph))
    return issues


def _validate_structural(graph: WorldGraph) -> list[Issue]:
    issues: list[Issue] = []
    for nid, node in graph.nodes.items():
        # dict-key vs node.id desync would only happen if a caller mutated
        # `graph.nodes` directly with a wrong key; catch it here so callers
        # don't get cryptic conformance errors later.
        if nid != node.id:
            issues.append(
                Issue(
                    "error",
                    "node_id_mismatch",
                    f"node stored under key {nid!r} reports id {node.id!r}",
                    nid,
                )
            )
        if not isinstance(node.id, str) or not node.id:
            issues.append(
                Issue(
                    "error",
                    "node_missing_id",
                    "node is missing a non-empty string id",
                    nid,
                )
            )
        if not isinstance(node.kind, str) or not node.kind:
            issues.append(
                Issue(
                    "error",
                    "node_missing_kind",
                    f"node {node.id!r} is missing a non-empty string kind",
                    node.id,
                )
            )
    for eid, edge in graph.edges.items():
        if eid != edge.id:
            issues.append(
                Issue(
                    "error",
                    "edge_id_mismatch",
                    f"edge stored under key {eid!r} reports id {edge.id!r}",
                    eid,
                )
            )
        if not isinstance(edge.id, str) or not edge.id:
            issues.append(
                Issue(
                    "error",
                    "edge_missing_id",
                    "edge is missing a non-empty string id",
                    eid,
                )
            )
        if not isinstance(edge.kind, str) or not edge.kind:
            issues.append(
                Issue(
                    "error",
                    "edge_missing_kind",
                    f"edge {edge.id!r} is missing a non-empty string kind",
                    edge.id,
                )
            )
        if edge.src not in graph.nodes:
            issues.append(
                Issue(
                    "error",
                    "edge_dangling_src",
                    f"edge {edge.id!r} src {edge.src!r} is not a node in the graph",
                    edge.id,
                )
            )
        if edge.dst not in graph.nodes:
            issues.append(
                Issue(
                    "error",
                    "edge_dangling_dst",
                    f"edge {edge.id!r} dst {edge.dst!r} is not a node in the graph",
                    edge.id,
                )
            )
    return issues


def _validate_conformance(graph: WorldGraph, ontology: Ontology) -> list[Issue]:
    issues: list[Issue] = []
    for node in graph.nodes.values():
        node_kind = ontology.node_kinds.get(node.kind)
        if node_kind is None:
            issues.append(
                Issue(
                    "error",
                    "unknown_node_kind",
                    f"node {node.id!r} has kind {node.kind!r} not declared in "
                    f"ontology {ontology.id!r}",
                    node.id,
                )
            )
            continue
        attrs = _compose_node_attrs(node_kind, ontology)
        issues.extend(_check_attrs(node.id, node.attrs, attrs, graph, ontology))

    # degree caps are aggregate, so collect once per (node, kind, side).
    out_counts: dict[tuple[str, str], int] = {}
    in_counts: dict[tuple[str, str], int] = {}
    for edge in graph.edges.values():
        edge_kind = ontology.edge_kinds.get(edge.kind)
        if edge_kind is None:
            issues.append(
                Issue(
                    "error",
                    "unknown_edge_kind",
                    f"edge {edge.id!r} has kind {edge.kind!r} not declared in "
                    f"ontology {ontology.id!r}",
                    edge.id,
                )
            )
            continue
        issues.extend(
            _check_attrs(edge.id, edge.attrs, edge_kind.attrs, graph, ontology)
        )
        src_node = graph.nodes.get(edge.src)
        dst_node = graph.nodes.get(edge.dst)
        # dangling endpoints were already flagged structurally; skip the
        # edge-kind pair check rather than emit a redundant cascade.
        if src_node is None or dst_node is None:
            continue
        pair = (src_node.kind, dst_node.kind)
        if edge_kind.endpoints and pair not in [tuple(p) for p in edge_kind.endpoints]:
            allowed = ", ".join(f"({s}, {d})" for s, d in edge_kind.endpoints)
            issues.append(
                Issue(
                    "error",
                    "edge_endpoint_mismatch",
                    f"edge {edge.id!r} of kind {edge.kind!r} has endpoints "
                    f"({src_node.kind}, {dst_node.kind}); allowed: {allowed}",
                    edge.id,
                )
            )
        out_counts[(edge.src, edge.kind)] = out_counts.get((edge.src, edge.kind), 0) + 1
        in_counts[(edge.dst, edge.kind)] = in_counts.get((edge.dst, edge.kind), 0) + 1

    for (src_id, edge_kind_id), count in out_counts.items():
        edge_kind = ontology.edge_kinds[edge_kind_id]
        if edge_kind.src_max is not None and count > edge_kind.src_max:
            issues.append(
                Issue(
                    "error",
                    "edge_src_degree_exceeded",
                    f"node {src_id!r} has {count} out-edges of kind {edge_kind_id!r}; "
                    f"src_max={edge_kind.src_max}",
                    src_id,
                )
            )
    for (dst_id, edge_kind_id), count in in_counts.items():
        edge_kind = ontology.edge_kinds[edge_kind_id]
        if edge_kind.dst_max is not None and count > edge_kind.dst_max:
            issues.append(
                Issue(
                    "error",
                    "edge_dst_degree_exceeded",
                    f"node {dst_id!r} has {count} in-edges of kind {edge_kind_id!r}; "
                    f"dst_max={edge_kind.dst_max}",
                    dst_id,
                )
            )
    return issues


def _compose_node_attrs(kind: NodeKind, ontology: Ontology) -> dict[str, AttrSpec]:
    """Walk the parent chain and merge attrs; child overrides parent by key."""
    chain: list[NodeKind] = []
    seen: set[str] = set()
    current: NodeKind | None = kind
    while current is not None:
        if current.id in seen:
            # malformed ontology — break the cycle silently. Cycle detection
            # belongs to a separate ontology-validator pass; here we just
            # avoid spinning.
            break
        seen.add(current.id)
        chain.append(current)
        current = ontology.node_kinds.get(current.parent) if current.parent else None
    composed: dict[str, AttrSpec] = {}
    for k in reversed(chain):
        composed.update(k.attrs)
    return composed


def _check_attrs(
    where: str,
    values: dict[str, Any],
    specs: dict[str, AttrSpec],
    graph: WorldGraph,
    ontology: Ontology,
) -> list[Issue]:
    issues: list[Issue] = []
    for name, spec in specs.items():
        if name not in values:
            if spec.required:
                issues.append(
                    Issue(
                        "error",
                        "missing_required_attr",
                        f"{where!r} is missing required attr {name!r}",
                        where,
                    )
                )
            continue
        issues.extend(
            _check_attr_value(where, name, values[name], spec, graph, ontology)
        )
    return issues


_PRIMITIVE_CHECK: dict[AttrType, Callable[[Any], bool]] = {
    AttrType.STRING: lambda v: isinstance(v, str),
    AttrType.INT: lambda v: isinstance(v, int) and not isinstance(v, bool),
    AttrType.FLOAT: lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    AttrType.BOOL: lambda v: isinstance(v, bool),
    AttrType.JSON: lambda v: True,
}


def _check_attr_value(
    where: str,
    name: str,
    value: Any,
    spec: AttrSpec,
    graph: WorldGraph,
    ontology: Ontology,
) -> list[Issue]:
    if spec.type is AttrType.ENUM:
        if not spec.enum or value not in spec.enum:
            allowed = ", ".join(repr(v) for v in (spec.enum or []))
            return [
                Issue(
                    "error",
                    "enum_value_invalid",
                    f"{where!r} attr {name!r} value {value!r} not in enum: [{allowed}]",
                    where,
                )
            ]
        return []
    if spec.type is AttrType.REF:
        if not isinstance(value, str):
            return [
                Issue(
                    "error",
                    "ref_not_string",
                    f"{where!r} attr {name!r} REF value {value!r} is not a string id",
                    where,
                )
            ]
        target = graph.nodes.get(value)
        if target is None:
            return [
                Issue(
                    "error",
                    "ref_dangling",
                    f"{where!r} attr {name!r} REF {value!r} does not resolve to a node",
                    where,
                )
            ]
        if spec.ref_kinds and target.kind not in spec.ref_kinds:
            allowed = ", ".join(repr(k) for k in spec.ref_kinds)
            return [
                Issue(
                    "error",
                    "ref_kind_disallowed",
                    f"{where!r} attr {name!r} REF {value!r} points at a "
                    f"{target.kind!r}; allowed ref_kinds: [{allowed}]",
                    where,
                )
            ]
        return []
    check = _PRIMITIVE_CHECK[spec.type]
    if not check(value):
        return [
            Issue(
                "error",
                "attr_type_mismatch",
                f"{where!r} attr {name!r} expected {spec.type.value}, got "
                f"{type(value).__name__} ({value!r})",
                where,
            )
        ]
    return []
