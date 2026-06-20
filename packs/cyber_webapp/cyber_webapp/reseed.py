"""Re-roll the flag on an admitted world, changing nothing else.

The flag value lives in exactly two nodes -- the loot record's ``fields.value`` and the
HIDDEN ``secret_flag``'s ``value_ref`` -- so a copy of the world with both rewritten is
byte-identical except for the secret. This backs the #317 re-seed integrity check: a
genuine exploit (one that reads the flag out of the live world) recovers the fresh
value, while a memorized one (a hard-coded old value) does not. It is a post-build
transform, so it never changes how worlds are originally generated.
"""

from __future__ import annotations

import dataclasses

from graphschema import Edge, Node, WorldGraph
from openrange_pack_sdk import PackError, Snapshot


def replant_flag(snapshot: Snapshot, new_value: str) -> Snapshot:
    """Return a copy of ``snapshot`` whose flag is ``new_value`` and whose structure is
    otherwise identical (a fresh ``snapshot_id``). Raises if the world has no flag."""
    graph = snapshot.graph
    if "secret_flag" not in graph.nodes:
        raise PackError("world has no flag to replant")
    old_value = str(graph.nodes["secret_flag"].attrs["value_ref"])

    clone = WorldGraph(ontology=graph.ontology, meta=dict(graph.meta))
    for nid, node in graph.nodes.items():
        attrs = dict(node.attrs)
        if nid == "secret_flag":
            attrs["value_ref"] = new_value
        elif (
            node.kind == "record"
            and isinstance(attrs.get("fields"), dict)
            and attrs["fields"].get("value") == old_value
        ):
            attrs["fields"] = {**attrs["fields"], "value": new_value}
        clone.nodes[nid] = Node(
            id=node.id,
            kind=node.kind,
            attrs=attrs,
            roles=set(node.roles),
            visibility=node.visibility,
            runtime=dict(node.runtime),
            meta=dict(node.meta),
        )
    for eid, edge in graph.edges.items():
        clone.edges[eid] = Edge(
            id=edge.id,
            kind=edge.kind,
            src=edge.src,
            dst=edge.dst,
            attrs=dict(edge.attrs),
        )

    return dataclasses.replace(snapshot, snapshot_id=clone.content_hash(), graph=clone)
