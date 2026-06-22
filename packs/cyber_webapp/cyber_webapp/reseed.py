"""Re-roll the flag (and, for a chain world, its credentials) on an admitted world,
changing nothing else.

The flag value lives in exactly two nodes -- the loot record's ``fields.value`` and the
HIDDEN ``secret_flag``'s ``value_ref`` -- so a copy of the world with both rewritten is
byte-identical except for the secret. This backs the #317 re-seed integrity check: a
genuine exploit (one that reads the flag out of the live world) recovers the fresh
value, while a memorized one (a hard-coded old value) does not. ``reseed_chain`` extends
this to the credential-reuse chain, re-rolling the flag and every per-hop token so a
genuine response-driven breach loots the live values while a memorized walk fails. Both
are post-build transforms, so they never change how worlds are originally generated.
"""

from __future__ import annotations

import dataclasses
import random

from graphschema import Edge, Node, WorldGraph
from openrange_pack_sdk import PackError, Snapshot

from cyber_webapp.sampling import _b62, generate_flag


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


def reseed_chain(snapshot: Snapshot, rng: random.Random) -> Snapshot:
    """Return a copy of ``snapshot`` with a fresh flag and fresh chain tokens, otherwise
    identical (a new ``snapshot_id``). Each value's consistent copies are rewritten
    together -- the flag's two nodes, and each token's credential node plus the
    producer/gate params that mirror it (see ``credential_value_binding``) -- so the
    re-seeded world still admits. A genuine breach loots the live values and recovers
    the fresh flag; a memorized one does not. Raises if the world has no flag.
    """
    graph = snapshot.graph
    if "secret_flag" not in graph.nodes:
        raise PackError("world has no flag to reseed")
    subst: dict[str, str] = {
        str(graph.nodes["secret_flag"].attrs["value_ref"]): generate_flag(rng)
    }
    for node in graph.by_kind("credential"):
        if node.attrs.get("kind") == "token":
            subst[str(node.attrs["value_ref"])] = _b62(rng, 24)

    def _sub(value: object) -> object:
        return subst.get(value, value) if isinstance(value, str) else value

    clone = WorldGraph(ontology=graph.ontology, meta=dict(graph.meta))
    for nid, node in graph.nodes.items():
        attrs = dict(node.attrs)
        if "value_ref" in attrs:
            attrs["value_ref"] = _sub(attrs["value_ref"])
        if node.kind == "record" and isinstance(attrs.get("fields"), dict):
            attrs["fields"] = {k: _sub(v) for k, v in attrs["fields"].items()}
        if node.kind == "vulnerability" and isinstance(attrs.get("params"), dict):
            attrs["params"] = {k: _sub(v) for k, v in attrs["params"].items()}
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
