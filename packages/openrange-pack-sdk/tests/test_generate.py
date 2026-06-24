"""realize_verified keeps only verifier-accepted authoring, and re-freezes.

No mocks: a real WorldAuthor over a real WorldGraph drives the loop end to end.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from graphschema import Node, WorldGraph
from openrange_pack_sdk import Snapshot, realize_verified


@dataclass
class _Author:
    accept: set[str]

    def authorable(self, snapshot: Snapshot) -> Sequence[str]:
        return ("kept", "rejected", "skipped")

    def apply(self, snapshot: Snapshot, key: str) -> bool:
        if key == "skipped":
            return False
        snapshot.graph.nodes["n"].attrs[key] = "authored"
        return True

    def verify(self, snapshot: Snapshot, key: str) -> bool:
        return key in self.accept

    def revert(self, snapshot: Snapshot, key: str) -> None:
        del snapshot.graph.nodes["n"].attrs[key]


def _snapshot() -> Snapshot:
    graph = WorldGraph(ontology="t@1", nodes={"n": Node(id="n", kind="thing")})
    return Snapshot(
        snapshot_id=graph.content_hash(),
        ontology_id="t@1",
        graph=graph,
        tasks=(),
        lineage={"seed": 1},
        history=(),
    )


def test_keeps_only_verifier_accepted_authoring() -> None:
    snap = _snapshot()
    out = realize_verified(snap, _Author(accept={"kept"}))

    assert out.lineage["realized"] == ("kept",)
    attrs = out.graph.nodes["n"].attrs
    assert attrs.get("kept") == "authored"
    assert "rejected" not in attrs
    assert "skipped" not in attrs
    assert out.lineage["seed"] == 1
    assert out.snapshot_id == out.graph.content_hash() != snap.snapshot_id


def test_nothing_accepted_realizes_nothing() -> None:
    out = realize_verified(_snapshot(), _Author(accept=set()))

    assert out.lineage["realized"] == ()
    assert out.graph.nodes["n"].attrs == {}
