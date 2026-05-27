"""Tests for the typed-property-graph meta-model.

Covers:
- WorldGraph mutation (add_node, add_edge, in/out_edges, by_kind).
- content_hash determinism: same content -> same hash; meta/runtime excluded.
- AttrSpec / NodeKind / EdgeKind / Ontology declaration shape.
- Three-tier validate(): structural, conformance, caller invariants.
- GraphPatch + apply_patch ordering and cascading edge removal.
- Role and Visibility serialization in content_hash payloads.
"""

from __future__ import annotations

import json

import pytest

from graphschema import (
    AttrSpec,
    AttrType,
    Edge,
    EdgeKind,
    GraphPatch,
    Issue,
    Node,
    NodeKind,
    Ontology,
    Role,
    Visibility,
    WorldGraph,
    apply_patch,
    validate,
)


def test_add_node_inserts_into_dict() -> None:
    g = WorldGraph(ontology="x@1")
    g.add_node(Node("a", "thing"))
    assert "a" in g.nodes
    assert g.nodes["a"].kind == "thing"


def test_add_node_rejects_duplicate_id() -> None:
    g = WorldGraph(ontology="x@1")
    g.add_node(Node("a", "thing"))
    with pytest.raises(KeyError, match="duplicate node id"):
        g.add_node(Node("a", "thing"))


def test_add_edge_inserts_into_dict() -> None:
    g = WorldGraph(ontology="x@1")
    g.add_node(Node("a", "thing"))
    g.add_node(Node("b", "thing"))
    g.add_edge(Edge("e1", "rel", "a", "b"))
    assert "e1" in g.edges


def test_add_edge_rejects_duplicate_id() -> None:
    g = WorldGraph(ontology="x@1")
    g.add_node(Node("a", "thing"))
    g.add_node(Node("b", "thing"))
    g.add_edge(Edge("e1", "rel", "a", "b"))
    with pytest.raises(KeyError):
        g.add_edge(Edge("e1", "rel", "a", "b"))


def test_in_and_out_edges_filter_by_kind() -> None:
    g = WorldGraph(ontology="x@1")
    for nid in ("a", "b", "c"):
        g.add_node(Node(nid, "thing"))
    g.add_edge(Edge("e1", "traversed", "a", "b"))
    g.add_edge(Edge("e2", "part_of", "a", "c"))
    g.add_edge(Edge("e3", "traversed", "b", "a"))
    assert {e.id for e in g.out_edges("a")} == {"e1", "e2"}
    assert {e.id for e in g.out_edges("a", "traversed")} == {"e1"}
    assert {e.id for e in g.in_edges("a")} == {"e3"}
    assert {e.id for e in g.in_edges("a", "part_of")} == set()


def test_by_kind_returns_insertion_order() -> None:
    g = WorldGraph(ontology="x@1")
    for nid in ("c", "a", "b"):
        g.add_node(Node(nid, "thing"))
    g.add_node(Node("z", "thought"))
    things = g.by_kind("thing")
    assert [n.id for n in things] == ["c", "a", "b"]
    assert [n.id for n in g.by_kind("thought")] == ["z"]


def _two_node_graph() -> WorldGraph:
    g = WorldGraph(ontology="x@1")
    g.add_node(Node("a", "thing", attrs={"label": "A"}))
    g.add_node(Node("b", "thing", attrs={"label": "B"}))
    g.add_edge(Edge("e1", "rel", "a", "b", attrs={"weight": 1}))
    return g


def test_content_hash_is_sha256_prefixed() -> None:
    h = _two_node_graph().content_hash()
    assert h.startswith("sha256:")
    assert len(h) == len("sha256:") + 64


def test_content_hash_deterministic_for_identical_graphs() -> None:
    assert _two_node_graph().content_hash() == _two_node_graph().content_hash()


def test_content_hash_changes_with_ontology_id() -> None:
    g1 = _two_node_graph()
    g2 = _two_node_graph()
    g2.ontology = "y@1"
    assert g1.content_hash() != g2.content_hash()


def test_content_hash_changes_with_attrs() -> None:
    g1 = _two_node_graph()
    g2 = _two_node_graph()
    g2.nodes["a"].attrs["label"] = "DIFFERENT"
    assert g1.content_hash() != g2.content_hash()


def test_content_hash_excludes_meta() -> None:
    g1 = _two_node_graph()
    g2 = _two_node_graph()
    g2.meta = {"manifest": "different", "producer_version": "2.0"}
    assert g1.content_hash() == g2.content_hash()


def test_content_hash_excludes_node_meta_and_runtime() -> None:
    g1 = _two_node_graph()
    g2 = _two_node_graph()
    g2.nodes["a"].meta = {"provenance_blob": "noise"}
    g2.nodes["a"].runtime = {"port": 8080}
    assert g1.content_hash() == g2.content_hash()


def test_content_hash_changes_with_visibility() -> None:
    g1 = _two_node_graph()
    g2 = _two_node_graph()
    g2.nodes["a"].visibility = Visibility.HIDDEN
    assert g1.content_hash() != g2.content_hash()


def test_content_hash_changes_with_roles() -> None:
    g1 = _two_node_graph()
    g2 = _two_node_graph()
    g2.nodes["a"].roles = {Role.ACTOR}
    assert g1.content_hash() != g2.content_hash()


def test_content_hash_independent_of_insertion_order() -> None:
    g1 = WorldGraph(ontology="x@1")
    g1.add_node(Node("a", "thing"))
    g1.add_node(Node("b", "thing"))
    g2 = WorldGraph(ontology="x@1")
    g2.add_node(Node("b", "thing"))
    g2.add_node(Node("a", "thing"))
    assert g1.content_hash() == g2.content_hash()


def test_node_payload_omits_public_visibility_and_empty_roles() -> None:
    g = WorldGraph(ontology="x@1")
    g.add_node(Node("a", "thing", attrs={"label": "A"}))
    h = g.content_hash()
    # If visibility=PUBLIC were emitted on the wire, this hash would differ
    # from one where we explicitly set visibility=PUBLIC again (no-op).
    g.nodes["a"].visibility = Visibility.PUBLIC
    g.nodes["a"].roles = set()
    assert g.content_hash() == h


def _simple_ontology() -> Ontology:
    """A minimal ontology used across the validation tests below."""
    return Ontology(
        id="x@1",
        node_kinds={
            "thing": NodeKind(
                "thing",
                attrs={
                    "label": AttrSpec(AttrType.STRING, required=True),
                    "weight": AttrSpec(AttrType.FLOAT),
                    "tier": AttrSpec(AttrType.ENUM, enum=["a", "b", "c"]),
                    "ref_to": AttrSpec(AttrType.REF, ref_kinds=["thing"]),
                },
            ),
            "thought": NodeKind(
                "thought",
                attrs={
                    "claim": AttrSpec(AttrType.STRING, required=True),
                },
            ),
        },
        edge_kinds={
            "rel": EdgeKind(
                "rel",
                endpoints=[("thing", "thing")],
                attrs={
                    "weight": AttrSpec(AttrType.INT, default=1),
                },
            ),
            "about": EdgeKind("about", endpoints=[("thought", "thing")]),
        },
    )


def test_structural_dangling_edge_src() -> None:
    g = WorldGraph(ontology="x@1")
    g.add_node(Node("a", "thing", attrs={"label": "A"}))
    g.edges["e1"] = Edge("e1", "rel", "MISSING", "a")
    issues = validate(g, _simple_ontology())
    codes = {i.code for i in issues}
    assert "edge_dangling_src" in codes


def test_structural_dangling_edge_dst() -> None:
    g = WorldGraph(ontology="x@1")
    g.add_node(Node("a", "thing", attrs={"label": "A"}))
    g.edges["e1"] = Edge("e1", "rel", "a", "MISSING")
    issues = validate(g, _simple_ontology())
    assert "edge_dangling_dst" in {i.code for i in issues}


def test_structural_dict_key_id_mismatch() -> None:
    g = WorldGraph(ontology="x@1")
    # Direct dict access — bypasses add_node guard.
    g.nodes["wrong_key"] = Node("a", "thing", attrs={"label": "A"})
    issues = validate(g, _simple_ontology())
    assert "node_id_mismatch" in {i.code for i in issues}


def test_structural_missing_node_kind() -> None:
    g = WorldGraph(ontology="x@1")
    g.nodes["a"] = Node("a", "", attrs={"label": "A"})  # empty kind
    issues = validate(g, _simple_ontology())
    assert "node_missing_kind" in {i.code for i in issues}


def test_conformance_unknown_node_kind() -> None:
    g = WorldGraph(ontology="x@1")
    g.add_node(Node("a", "mysterious", attrs={"label": "A"}))
    issues = validate(g, _simple_ontology())
    codes = {i.code for i in issues}
    assert "unknown_node_kind" in codes


def test_conformance_missing_required_attr() -> None:
    g = WorldGraph(ontology="x@1")
    g.add_node(Node("a", "thing", attrs={}))  # missing required `label`
    issues = validate(g, _simple_ontology())
    assert "missing_required_attr" in {i.code for i in issues}


def test_conformance_attr_type_mismatch() -> None:
    g = WorldGraph(ontology="x@1")
    g.add_node(Node("a", "thing", attrs={"label": "A", "weight": "not-a-float"}))
    issues = validate(g, _simple_ontology())
    assert "attr_type_mismatch" in {i.code for i in issues}


def test_conformance_bool_not_int() -> None:
    """bool is a subclass of int but should NOT pass as INT in our validator."""
    g = WorldGraph(ontology="x@1")
    g.add_node(Node("a", "thing", attrs={"label": "A"}))
    g.add_node(Node("b", "thing", attrs={"label": "B"}))
    g.add_edge(Edge("e1", "rel", "a", "b", attrs={"weight": True}))
    issues = validate(g, _simple_ontology())
    assert "attr_type_mismatch" in {i.code for i in issues}


def test_conformance_int_accepted_for_float() -> None:
    g = WorldGraph(ontology="x@1")
    g.add_node(Node("a", "thing", attrs={"label": "A", "weight": 5}))
    issues = validate(g, _simple_ontology())
    errors = [i for i in issues if i.severity == "error"]
    assert errors == []


def test_conformance_enum_value_invalid() -> None:
    g = WorldGraph(ontology="x@1")
    g.add_node(Node("a", "thing", attrs={"label": "A", "tier": "z"}))
    issues = validate(g, _simple_ontology())
    assert "enum_value_invalid" in {i.code for i in issues}


def test_conformance_ref_dangling() -> None:
    g = WorldGraph(ontology="x@1")
    g.add_node(Node("a", "thing", attrs={"label": "A", "ref_to": "MISSING"}))
    issues = validate(g, _simple_ontology())
    assert "ref_dangling" in {i.code for i in issues}


def test_conformance_ref_kind_disallowed() -> None:
    g = WorldGraph(ontology="x@1")
    g.add_node(Node("a", "thing", attrs={"label": "A", "ref_to": "b"}))
    g.add_node(Node("b", "thought", attrs={"claim": "..."}))
    issues = validate(g, _simple_ontology())
    assert "ref_kind_disallowed" in {i.code for i in issues}


def test_conformance_edge_endpoint_mismatch() -> None:
    g = WorldGraph(ontology="x@1")
    g.add_node(Node("a", "thing", attrs={"label": "A"}))
    g.add_node(Node("b", "thought", attrs={"claim": "..."}))
    g.add_edge(Edge("e1", "rel", "a", "b"))  # rel expects (thing, thing)
    issues = validate(g, _simple_ontology())
    assert "edge_endpoint_mismatch" in {i.code for i in issues}


def test_conformance_unknown_edge_kind() -> None:
    g = WorldGraph(ontology="x@1")
    g.add_node(Node("a", "thing", attrs={"label": "A"}))
    g.add_node(Node("b", "thing", attrs={"label": "B"}))
    g.add_edge(Edge("e1", "unknown_relation", "a", "b"))
    issues = validate(g, _simple_ontology())
    assert "unknown_edge_kind" in {i.code for i in issues}


def test_conformance_degree_caps() -> None:
    onto = Ontology(
        id="x@1",
        node_kinds={
            "thing": NodeKind(
                "thing",
                attrs={
                    "label": AttrSpec(AttrType.STRING, required=True),
                },
            ),
        },
        edge_kinds={
            "rel": EdgeKind("rel", endpoints=[("thing", "thing")], src_max=1),
        },
    )
    g = WorldGraph(ontology="x@1")
    g.add_node(Node("a", "thing", attrs={"label": "A"}))
    g.add_node(Node("b", "thing", attrs={"label": "B"}))
    g.add_node(Node("c", "thing", attrs={"label": "C"}))
    g.add_edge(Edge("e1", "rel", "a", "b"))
    g.add_edge(Edge("e2", "rel", "a", "c"))  # exceeds src_max=1
    issues = validate(g, onto)
    assert "edge_src_degree_exceeded" in {i.code for i in issues}


def test_conformance_parent_attrs_inherited() -> None:
    onto = Ontology(
        id="x@1",
        node_kinds={
            "component": NodeKind(
                "component",
                attrs={
                    "name": AttrSpec(AttrType.STRING, required=True),
                },
            ),
            "service": NodeKind(
                "service",
                parent="component",
                attrs={
                    "port": AttrSpec(AttrType.INT),
                },
            ),
        },
        edge_kinds={},
    )
    g = WorldGraph(ontology="x@1")
    g.add_node(Node("a", "service", attrs={"port": 8080}))
    issues = validate(g, onto)
    assert "missing_required_attr" in {i.code for i in issues}


def test_invariants_are_invoked_and_concatenated() -> None:
    def at_least_one_thought(g: WorldGraph) -> list[Issue]:
        if not g.by_kind("thought"):
            return [Issue("error", "no_thought", "graph must have a thought", "graph")]
        return []

    g = WorldGraph(ontology="x@1")
    g.add_node(Node("a", "thing", attrs={"label": "A"}))
    issues = validate(g, _simple_ontology(), invariants=[at_least_one_thought])
    assert "no_thought" in {i.code for i in issues}


def test_apply_patch_adds_nodes_and_edges() -> None:
    g = WorldGraph(ontology="x@1")
    g.add_node(Node("a", "thing"))
    patch = GraphPatch(
        nodes_added=[Node("b", "thing")],
        edges_added=[Edge("e1", "rel", "a", "b")],
    )
    apply_patch(g, patch)
    assert "b" in g.nodes
    assert "e1" in g.edges


def test_apply_patch_updates_replace_entirely() -> None:
    g = WorldGraph(ontology="x@1")
    g.add_node(
        Node("a", "thing", attrs={"label": "OLD", "other": "kept-by-caller-or-not"})
    )
    patch = GraphPatch(nodes_updated=[Node("a", "thing", attrs={"label": "NEW"})])
    apply_patch(g, patch)
    # update REPLACES; if a caller wanted to keep `other` they'd include it.
    assert g.nodes["a"].attrs == {"label": "NEW"}


def test_apply_patch_removes_node_and_cascades_edges() -> None:
    g = WorldGraph(ontology="x@1")
    g.add_node(Node("a", "thing"))
    g.add_node(Node("b", "thing"))
    g.add_node(Node("c", "thing"))
    g.add_edge(Edge("e1", "rel", "a", "b"))
    g.add_edge(Edge("e2", "rel", "b", "c"))
    g.add_edge(Edge("e3", "rel", "a", "c"))
    apply_patch(g, GraphPatch(nodes_removed=["b"]))
    assert "b" not in g.nodes
    assert "e1" not in g.edges
    assert "e2" not in g.edges
    assert "e3" in g.edges


def test_apply_patch_order_removals_before_additions() -> None:
    """An addition with the same id as a removal in the same patch should land."""
    g = WorldGraph(ontology="x@1")
    g.add_node(Node("a", "thing", attrs={"label": "OLD"}))
    patch = GraphPatch(
        nodes_removed=["a"],
        nodes_added=[Node("a", "thing", attrs={"label": "REINTRODUCED"})],
    )
    apply_patch(g, patch)
    assert g.nodes["a"].attrs["label"] == "REINTRODUCED"


def test_apply_patch_duplicate_add_raises() -> None:
    g = WorldGraph(ontology="x@1")
    g.add_node(Node("a", "thing"))
    with pytest.raises(KeyError, match="patch adds duplicate node id"):
        apply_patch(g, GraphPatch(nodes_added=[Node("a", "thing")]))


def test_hash_payload_is_canonical_json() -> None:
    """The hash payload sorts keys and omits empty optional fields — that
    means a json.loads of the hashed bytes is a stable, comparable shape."""
    g = WorldGraph(ontology="x@1")
    g.add_node(
        Node(
            "a",
            "thing",
            attrs={"z": 1, "a": 2},
            roles={Role.ACTOR},
            visibility=Visibility.HIDDEN,
        )
    )
    g.add_node(Node("b", "thing", attrs={"label": "B"}))
    g.add_edge(Edge("e1", "rel", "a", "b", attrs={"k": "v"}))
    from graphschema._ir import _edge_data, _node_data

    data = {
        "ontology": g.ontology,
        "nodes": [_node_data(n) for n in sorted(g.nodes.values(), key=lambda n: n.id)],
        "edges": [_edge_data(e) for e in sorted(g.edges.values(), key=lambda e: e.id)],
    }
    encoded = json.dumps(data, sort_keys=True, separators=(",", ":"))
    decoded = json.loads(encoded)
    assert decoded["ontology"] == "x@1"
    assert decoded["nodes"][0]["id"] == "a"
    assert decoded["nodes"][0]["roles"] == ["actor"]
    assert decoded["nodes"][0]["visibility"] == "hidden"
    # public visibility AND empty roles are NOT serialized
    assert "visibility" not in decoded["nodes"][1]
    assert "roles" not in decoded["nodes"][1]
