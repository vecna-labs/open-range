"""Typed property-graph meta-model.

The top-level `graphschema` namespace re-exports the meta-model from
`graphschema._ir`. Downstream packages declare their own ontologies on
top of these primitives.

Typical use:

    from graphschema import (
        AttrSpec, AttrType, Edge, EdgeKind, Node, NodeKind,
        Ontology, WorldGraph, validate,
    )

    onto = Ontology(
        id="rooms@0.1.0",
        node_kinds={"room": NodeKind("room", attrs={
            "label": AttrSpec(AttrType.STRING, required=True),
        })},
        edge_kinds={"leads_to": EdgeKind("leads_to", endpoints=[("room", "room")])},
    )
    g = WorldGraph(ontology=onto.id)
    g.add_node(Node("a", "room", attrs={"label": "lobby"}))
    issues = validate(g, onto)
"""

from __future__ import annotations

from graphschema._ir import (
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

__all__ = [
    "AttrSpec",
    "AttrType",
    "Edge",
    "EdgeKind",
    "GraphPatch",
    "Issue",
    "Node",
    "NodeKind",
    "Ontology",
    "Role",
    "Visibility",
    "WorldGraph",
    "apply_patch",
    "validate",
]
