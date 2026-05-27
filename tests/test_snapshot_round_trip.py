"""Snapshot round-trip tests.

:func:`snapshot_to_dict` is the wire-shape projection declared in
``CONTRACTS.md`` Section 5. These tests confirm that a real
:class:`WebappPack` :func:`admit` produces a :class:`Snapshot` whose
dict projection matches the section-5 keys, carries the content-
addressed id, records ``history`` beside the timeless graph, carries
``lineage`` with pack id+version, serializes HIDDEN visibility, and
round-trips through JSON.
"""

from __future__ import annotations

import json
from typing import Any

from cyber_webapp import WebappPack
from graphschema import Visibility
from openrange_pack_sdk import Snapshot

from openrange.core.admit import admit, snapshot_to_dict


def _admit_snapshot() -> Snapshot:
    snap = admit(WebappPack(), manifest={"seed": 0})
    assert isinstance(snap, Snapshot), f"expected Snapshot, got {type(snap).__name__}"
    return snap


def test_snapshot_dict_has_section_5_top_level_keys() -> None:
    snap = _admit_snapshot()
    payload = snapshot_to_dict(snap)
    assert set(payload.keys()) == {
        "snapshot_id",
        "ontology_id",
        "graph",
        "tasks",
        "lineage",
        "history",
    }


def test_snapshot_dict_graph_block_shape() -> None:
    """The `graph` sub-dict carries an ontology id plus node + edge lists."""
    snap = _admit_snapshot()
    graph = snapshot_to_dict(snap)["graph"]
    assert {"ontology", "nodes", "edges"} <= set(graph.keys())
    assert set(graph.keys()) <= {"ontology", "nodes", "edges", "meta"}
    assert graph["ontology"] == "cyber.webapp@v2"
    assert isinstance(graph["nodes"], list)
    assert isinstance(graph["edges"], list)
    # Every node carries id + kind + attrs at minimum.
    for n in graph["nodes"]:
        assert {"id", "kind", "attrs"} <= set(n.keys())
    # Every edge carries id + kind + src + dst at minimum.
    for e in graph["edges"]:
        assert {"id", "kind", "src", "dst"} <= set(e.keys())


def test_snapshot_dict_is_json_serializable() -> None:
    """The dict round-trips through `json.dumps` with no custom encoder.

    Catches MappingProxy / tuple / dataclass leaks that would break
    storage and any read API that ships snapshots over the wire.
    """
    snap = _admit_snapshot()
    payload = snapshot_to_dict(snap)
    encoded = json.dumps(payload, sort_keys=True)
    decoded = json.loads(encoded)
    assert decoded["snapshot_id"] == payload["snapshot_id"]
    assert decoded["graph"]["ontology"] == payload["graph"]["ontology"]


def test_snapshot_id_equals_graph_content_hash() -> None:
    """The only way to derive a snapshot id."""
    snap = _admit_snapshot()
    payload = snapshot_to_dict(snap)
    assert snap.snapshot_id == snap.graph.content_hash()
    assert payload["snapshot_id"] == snap.graph.content_hash()
    assert payload["snapshot_id"].startswith("sha256:")


def test_snapshot_history_records_all_phases_in_order() -> None:
    """admit() emits `build`, `validate`, `feasibility`, `freeze` — in order.

    The first successful pass produces exactly those four phases; repair
    attempts insert extra `validate`/`feasibility`/`repair` events
    BEFORE the final `freeze`. Either way, `build` is first, `freeze`
    is last, and `validate` + `feasibility` both appear.
    """
    snap = _admit_snapshot()
    payload = snapshot_to_dict(snap)
    phases = [e["phase"] for e in payload["history"]]
    assert phases[0] == "build"
    assert phases[-1] == "freeze"
    assert "validate" in phases
    assert "feasibility" in phases
    # `seq` is monotonically increasing — the in-order requirement.
    seqs = [e["seq"] for e in payload["history"]]
    assert seqs == list(range(len(seqs)))


def test_snapshot_history_carries_build_refs() -> None:
    """The opening `build` event records the generated task ids."""
    snap = _admit_snapshot()
    payload = snapshot_to_dict(snap)
    build_event = payload["history"][0]
    assert build_event["phase"] == "build"
    assert set(build_event["refs"]) == {t.id for t in snap.tasks}


def test_snapshot_lineage_carries_pack_provenance() -> None:
    """lineage holds the manifest plus pack id+version (per admit.py)."""
    snap = _admit_snapshot()
    payload = snapshot_to_dict(snap)
    lineage = payload["lineage"]
    assert lineage["pack"] == "webapp"
    assert lineage["pack_version"] == "v2"
    assert lineage["manifest"] == {"seed": 0}
    assert "attempts" in lineage


def test_hidden_nodes_serialize_with_visibility_tag() -> None:
    """Per `_node_dict`, HIDDEN nodes must carry a `visibility` key in
    the wire payload; PUBLIC nodes omit it (the encoding's compactness
    rule)."""
    snap = _admit_snapshot()
    payload = snapshot_to_dict(snap)

    # Pull the node-id sets from the live Snapshot for each visibility.
    hidden_ids = {
        n.id for n in snap.graph.nodes.values() if n.visibility is Visibility.HIDDEN
    }
    public_ids = {
        n.id for n in snap.graph.nodes.values() if n.visibility is Visibility.PUBLIC
    }
    # The webapp pack always plants at least one HIDDEN secret + vuln.
    assert hidden_ids, "expected at least one HIDDEN node in the cyber world"

    serialized: dict[str, dict[str, Any]] = {
        n["id"]: n for n in payload["graph"]["nodes"]
    }

    for nid in hidden_ids:
        assert serialized[nid].get("visibility") == "hidden", (
            f"HIDDEN node {nid!r} did not serialize its visibility tag"
        )
    for nid in public_ids:
        assert "visibility" not in serialized[nid], (
            f"PUBLIC node {nid!r} unexpectedly carried a visibility tag "
            f"(should be omitted)"
        )


def test_snapshot_tasks_serialize_with_taskspec_shape() -> None:
    """Each task dict carries the TaskSpec fields per CONTRACTS Section 5."""
    snap = _admit_snapshot()
    payload = snapshot_to_dict(snap)
    assert len(payload["tasks"]) == len(snap.tasks)
    for task_dict, task in zip(payload["tasks"], snap.tasks, strict=True):
        assert task_dict["id"] == task.id
        assert task_dict["instruction"] == task.instruction
        assert task_dict["entrypoints"] == list(task.entrypoints)
        assert task_dict["goal_nodes"] == list(task.goal_nodes)
        assert task_dict["feasibility_check"] == task.feasibility_check
        assert task_dict["success_check"] == task.success_check
