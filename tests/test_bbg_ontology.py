"""Tests for the `bbg@0.1.0` ontology shipped in `openrange.ontologies.bbg`.

The ontology is a value, not a runtime — these tests check:

- The ontology id is stable (`bbg@0.1.0`) — any breaking change to the
  schema must bump it.
- Required node kinds (`thing`, `thought`) and edge kinds (`traversed`,
  `part_of`, `anchored_to`, `revises`) are present with the documented
  endpoints.
- Required attrs and their enum values match the wire format.
- A minimal valid BBG-shaped graph passes `validate()` against this
  ontology end-to-end.
"""

from __future__ import annotations

from openrange.ontologies.bbg import BBG_ONTOLOGY_ID, wayfinder_ontology
from openrange.world_ir import (
    AttrType,
    Edge,
    Node,
    WorldGraph,
    validate,
)

# ---------------------------------------------------------------------------
# Shape: id + structure
# ---------------------------------------------------------------------------


def test_ontology_id_is_bbg_0_1_0() -> None:
    """The id is a wire-format version handle; changing it is a deliberate
    breaking change that consumers must explicitly notice."""
    assert BBG_ONTOLOGY_ID == "bbg.wayfinder@0.1.0"
    assert wayfinder_ontology().id == "bbg.wayfinder@0.1.0"


def test_ontology_returns_a_fresh_instance() -> None:
    a = wayfinder_ontology()
    b = wayfinder_ontology()
    # not the same object — mutating one must not leak to the other
    assert a is not b
    a.node_kinds["thing"].attrs.pop("label", None)
    assert "label" in b.node_kinds["thing"].attrs


def test_node_kinds_thing_and_thought_only() -> None:
    onto = wayfinder_ontology()
    assert set(onto.node_kinds) == {"thing", "thought"}


def test_edge_kinds_are_the_four() -> None:
    onto = wayfinder_ontology()
    assert set(onto.edge_kinds) == {
        "traversed",
        "part_of",
        "anchored_to",
        "revises",
    }


# ---------------------------------------------------------------------------
# Edge endpoints
# ---------------------------------------------------------------------------


def test_traversed_endpoints_are_thing_to_thing() -> None:
    onto = wayfinder_ontology()
    assert onto.edge_kinds["traversed"].endpoints == [("thing", "thing")]


def test_part_of_endpoints_are_thing_to_thing() -> None:
    onto = wayfinder_ontology()
    assert onto.edge_kinds["part_of"].endpoints == [("thing", "thing")]


def test_anchored_to_endpoints_are_thought_to_thing() -> None:
    onto = wayfinder_ontology()
    assert onto.edge_kinds["anchored_to"].endpoints == [("thought", "thing")]


def test_revises_endpoints_are_thought_to_thought() -> None:
    onto = wayfinder_ontology()
    assert onto.edge_kinds["revises"].endpoints == [("thought", "thought")]


# ---------------------------------------------------------------------------
# Required attrs + enums
# ---------------------------------------------------------------------------


def test_thing_required_attrs() -> None:
    onto = wayfinder_ontology()
    thing = onto.node_kinds["thing"]
    assert thing.attrs["label"].required
    assert thing.attrs["status"].required
    assert thing.attrs["provenance"].required


def test_thought_required_attrs() -> None:
    onto = wayfinder_ontology()
    thought = onto.node_kinds["thought"]
    assert thought.attrs["claim"].required
    assert thought.attrs["status"].required
    assert thought.attrs["provenance"].required


def test_provenance_enum_is_three_values() -> None:
    onto = wayfinder_ontology()
    prov = onto.node_kinds["thing"].attrs["provenance"]
    assert prov.type is AttrType.ENUM
    assert sorted(prov.enum or []) == sorted(["trajectory", "referenced", "inferred"])


def test_thing_status_enum() -> None:
    onto = wayfinder_ontology()
    status = onto.node_kinds["thing"].attrs["status"]
    assert sorted(status.enum or []) == sorted(["incidental", "salient"])


def test_thought_status_enum() -> None:
    onto = wayfinder_ontology()
    status = onto.node_kinds["thought"].attrs["status"]
    assert sorted(status.enum or []) == sorted(["open", "confirmed", "refuted"])


def test_traversed_outcome_enum() -> None:
    onto = wayfinder_ontology()
    outcome = onto.edge_kinds["traversed"].attrs["outcome"]
    assert sorted(outcome.enum or []) == sorted(["productive", "dead_end", "neutral"])


# ---------------------------------------------------------------------------
# End-to-end: a small BBG-shaped graph passes validation against this ontology
# ---------------------------------------------------------------------------


def _tiny_bbg() -> WorldGraph:
    """A minimal valid BBG: two things, one thought anchored to one of them,
    one traversal edge between the things, one revises chain."""
    g = WorldGraph(ontology=BBG_ONTOLOGY_ID)
    g.add_node(
        Node(
            "thing.a",
            "thing",
            attrs={
                "label": "place A",
                "provenance": "trajectory",
                "status": "incidental",
                "category": "place",
                "first_seen": 0,
                "visits": 1,
                "explored": True,
            },
        )
    )
    g.add_node(
        Node(
            "thing.b",
            "thing",
            attrs={
                "label": "place B",
                "provenance": "trajectory",
                "status": "salient",
                "category": "place",
                "first_seen": 1,
                "visits": 2,
                "explored": True,
            },
        )
    )
    g.add_node(
        Node(
            "thought.0",
            "thought",
            attrs={
                "claim": "B is the way in",
                "provenance": "inferred",
                "status": "open",
                "confidence": 0.7,
                "formed_at": 1,
            },
        )
    )
    g.add_node(
        Node(
            "thought.1",
            "thought",
            attrs={
                "claim": "B is a dead end",
                "provenance": "inferred",
                "status": "open",
                "confidence": 0.6,
                "formed_at": 3,
            },
        )
    )
    g.add_edge(
        Edge(
            "e.a-b",
            "traversed",
            "thing.a",
            "thing.b",
            attrs={"outcome": "productive", "count": 1},
        )
    )
    g.add_edge(Edge("e.0->b", "anchored_to", "thought.0", "thing.b"))
    g.add_edge(Edge("e.0~1", "revises", "thought.1", "thought.0"))
    return g


def test_tiny_bbg_passes_validation() -> None:
    onto = wayfinder_ontology()
    issues = validate(_tiny_bbg(), onto)
    errors = [i for i in issues if i.severity == "error"]
    assert errors == [], "valid BBG should have no errors; got: " + repr(errors)


def test_invalid_outcome_value_is_rejected() -> None:
    onto = wayfinder_ontology()
    g = _tiny_bbg()
    g.edges["e.a-b"].attrs["outcome"] = "weird"
    issues = validate(g, onto)
    assert "enum_value_invalid" in {i.code for i in issues}


def test_anchored_to_must_be_thought_to_thing() -> None:
    onto = wayfinder_ontology()
    g = _tiny_bbg()
    # anchor a thought-to-thought via anchored_to — endpoints disallow it
    g.add_edge(Edge("e.bad", "anchored_to", "thought.0", "thought.1"))
    issues = validate(g, onto)
    assert "edge_endpoint_mismatch" in {i.code for i in issues}
