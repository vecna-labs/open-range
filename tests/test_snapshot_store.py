"""Tests for :class:`SnapshotStore` and :func:`snapshot_from_dict`.

Exercises save / load round-trip integrity (including HIDDEN nodes,
:class:`TaskSpec`, and history ordering) and rejection of mismatched
ids, missing files, and malformed JSON.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
from graphschema import (
    AttrSpec,
    AttrType,
    Edge,
    EdgeKind,
    Node,
    NodeKind,
    Ontology,
    Role,
    Visibility,
    WorldGraph,
)
from openrange_pack_sdk import (
    Backing,
    Builder,
    BuildEvent,
    BuildResult,
    EpisodeResult,
    FeasibilityVerdict,
    Manifest,
    Pack,
    PackPrior,
    RuntimeHandle,
    Snapshot,
    TaskFamily,
    TaskSpec,
)

from openrange.core.admit import (
    admit,
    snapshot_to_dict,
)
from openrange.core.errors import StoreError
from openrange.core.store import SnapshotStore, snapshot_from_dict

_TEST_ONTOLOGY = Ontology(
    id="test@1",
    node_kinds={
        "repo": NodeKind(
            "repo",
            attrs={"name": AttrSpec(AttrType.STRING, required=True)},
        ),
        "endpoint": NodeKind(
            "endpoint",
            attrs={"path": AttrSpec(AttrType.STRING, required=True)},
        ),
        "secret": NodeKind(
            "secret",
            attrs={"kind": AttrSpec(AttrType.STRING, required=True)},
        ),
    },
    edge_kinds={
        "exposes": EdgeKind(
            "exposes",
            endpoints=[("repo", "endpoint")],
            attrs={"hits": AttrSpec(AttrType.INT)},
        ),
        "holds": EdgeKind("holds", endpoints=[("endpoint", "secret")]),
    },
)


def _build_test_graph() -> WorldGraph:
    """A small world: repo -> endpoint -> secret. The secret is HIDDEN.

    The repo carries an ACTOR role to exercise role serialization. The
    `exposes` edge carries an `attrs` mapping to exercise edge-attrs
    serialization.
    """
    g = WorldGraph(ontology="test@1")
    g.add_node(
        Node(
            "repo.a",
            "repo",
            attrs={"name": "alpha"},
            roles={Role.ACTOR},
        )
    )
    g.add_node(Node("ep.login", "endpoint", attrs={"path": "/login"}))
    g.add_node(
        Node(
            "sec.flag",
            "secret",
            attrs={"kind": "flag"},
            visibility=Visibility.HIDDEN,
        )
    )
    g.add_edge(Edge("e1", "exposes", "repo.a", "ep.login", attrs={"hits": 3}))
    g.add_edge(Edge("e2", "holds", "ep.login", "sec.flag"))
    return g


class _NoopHandle:
    """A RuntimeHandle stub — admit() never realizes, so it stays untouched."""

    def reset(self) -> None: ...

    def surface(self) -> Mapping[str, Any]:
        return {}

    def poll_events(self) -> tuple[Mapping[str, Any], ...]:
        return ()

    def terminal(self) -> tuple[bool, str | None]:
        return False, None

    def checkpoint(self) -> Any:
        return None

    def restore(self, state: Any) -> None:
        del state

    def collect(self) -> Mapping[str, Any]:
        return {}

    def stop(self) -> None: ...


class _PentestFamily(TaskFamily):
    id = "test.pentest"
    pack_id = "test"

    def generate(
        self,
        graph: WorldGraph,
        manifest: Manifest,
        prior: PackPrior | None,
    ) -> list[TaskSpec]:
        del manifest, prior
        endpoint = next(iter(graph.by_kind("endpoint")), None)
        secret = next(iter(graph.by_kind("secret")), None)
        if endpoint is None or secret is None:
            return []
        return [
            TaskSpec(
                id="test.pentest.0",
                instruction="Recover the hidden flag.",
                entrypoints=(endpoint.id,),
                goal_nodes=(secret.id,),
                feasibility_check="test.pentest",
                success_check="test.pentest",
                meta={"family": "test.pentest", "difficulty": 0.7},
            )
        ]

    def check_feasibility(
        self,
        graph: WorldGraph,
        task: TaskSpec,
    ) -> FeasibilityVerdict:
        for e in graph.out_edges(task.entrypoints[0], "holds"):
            if e.dst in task.goal_nodes:
                return FeasibilityVerdict(True)
        return FeasibilityVerdict(False, "no holds chain to goal")

    def check_success(
        self,
        graph: WorldGraph,
        task: TaskSpec,
        final_state: Mapping[str, Any],
    ) -> EpisodeResult:
        del graph, task
        return EpisodeResult(success=bool(final_state.get("flag_found")))


class _StaticBuilder(Builder):
    """A builder that returns a fixed BuildResult — no repair."""

    def __init__(self, result: BuildResult) -> None:
        self._result = result

    def build(self, manifest: Manifest) -> BuildResult:
        del manifest
        return self._result


class _TestPack(Pack):
    id = "test"
    version = "0.1.0"

    def __init__(self, builder: Builder) -> None:
        self._builder = builder

    def ontology(self) -> Ontology:
        return _TEST_ONTOLOGY

    def make_builder(self, prior: PackPrior | None) -> Builder:
        del prior
        return self._builder

    def realize(self, graph: WorldGraph, backing: Backing) -> RuntimeHandle:
        del graph, backing
        return _NoopHandle()

    def task_families(self) -> list[TaskFamily]:
        return [_PentestFamily()]


def _admit_one(manifest: Mapping[str, Any] | None = None) -> Snapshot:
    """Drive the stub pack through admit() and assert success.

    The cast lifts `admit()`'s `Snapshot | AdmissionFailure` return type;
    a failure here would mean the stub pack is broken (a separate
    concern from store integrity).
    """
    g = _build_test_graph()
    tasks = _PentestFamily().generate(g, {}, None)
    build_result = BuildResult(graph=g, tasks=tasks, admission_meta={"seed": 42})
    pack = _TestPack(_StaticBuilder(build_result))
    snap = admit(pack, manifest=manifest or {"goal": "test"})
    assert isinstance(snap, Snapshot), f"expected Snapshot, got {type(snap).__name__}"
    return snap


def test_snapshot_from_dict_round_trip_preserves_identity() -> None:
    """The reconstructed Snapshot has the same content-addressed id."""
    snap = _admit_one()
    payload = snapshot_to_dict(snap)
    restored = snapshot_from_dict(payload)
    assert restored.snapshot_id == snap.snapshot_id
    assert restored.graph.content_hash() == snap.graph.content_hash()


def test_snapshot_from_dict_preserves_graph_shape() -> None:
    """Nodes + edges round-trip with id / kind / src / dst / attrs intact."""
    snap = _admit_one()
    restored = snapshot_from_dict(snapshot_to_dict(snap))
    assert set(restored.graph.nodes.keys()) == set(snap.graph.nodes.keys())
    assert set(restored.graph.edges.keys()) == set(snap.graph.edges.keys())
    for nid, node in snap.graph.nodes.items():
        rn = restored.graph.nodes[nid]
        assert rn.kind == node.kind
        assert rn.attrs == node.attrs
        assert rn.roles == node.roles
        assert rn.visibility == node.visibility
    for eid, edge in snap.graph.edges.items():
        re = restored.graph.edges[eid]
        assert (re.kind, re.src, re.dst) == (edge.kind, edge.src, edge.dst)
        assert re.attrs == edge.attrs


def test_snapshot_from_dict_preserves_tasks() -> None:
    """Tasks survive the round trip as `TaskSpec` instances, not dicts."""
    snap = _admit_one()
    restored = snapshot_from_dict(snapshot_to_dict(snap))
    assert len(restored.tasks) == len(snap.tasks)
    for rt, t in zip(restored.tasks, snap.tasks, strict=True):
        assert isinstance(rt, TaskSpec)
        assert rt.id == t.id
        assert rt.instruction == t.instruction
        assert rt.entrypoints == t.entrypoints
        assert rt.goal_nodes == t.goal_nodes
        assert rt.feasibility_check == t.feasibility_check
        assert rt.success_check == t.success_check
        assert dict(rt.meta) == dict(t.meta)


def test_snapshot_from_dict_preserves_history_in_order() -> None:
    """BuildEvents survive in order, with seq / phase / detail / refs."""
    snap = _admit_one()
    restored = snapshot_from_dict(snapshot_to_dict(snap))
    assert len(restored.history) == len(snap.history)
    for re, ev in zip(restored.history, snap.history, strict=True):
        assert isinstance(re, BuildEvent)
        assert re.seq == ev.seq
        assert re.phase == ev.phase
        assert re.detail == ev.detail
        assert re.refs == ev.refs


def test_snapshot_from_dict_preserves_lineage() -> None:
    """lineage round-trips as a plain mapping (manifest + pack provenance)."""
    snap = _admit_one(manifest={"goal": "demo", "seed": 7})
    restored = snapshot_from_dict(snapshot_to_dict(snap))
    assert dict(restored.lineage) == dict(snap.lineage)


def test_snapshot_from_dict_preserves_hidden_visibility() -> None:
    """HIDDEN nodes survive the round trip with their visibility tag."""
    snap = _admit_one()
    hidden_ids = {
        n.id for n in snap.graph.nodes.values() if n.visibility is Visibility.HIDDEN
    }
    assert hidden_ids, "expected at least one HIDDEN node in the stub world"
    restored = snapshot_from_dict(snapshot_to_dict(snap))
    for nid in hidden_ids:
        assert restored.graph.nodes[nid].visibility is Visibility.HIDDEN


def test_store_save_writes_file_named_by_snapshot_id(tmp_path: Path) -> None:
    """save() returns the path it wrote to, named `<snapshot_id>.json`."""
    snap = _admit_one()
    store = SnapshotStore(tmp_path)
    path = store.save(snap)
    assert path == tmp_path / f"{snap.snapshot_id}.json"
    assert path.exists()


def test_store_save_creates_root_directory_if_missing(tmp_path: Path) -> None:
    """save() should mkdir its root so callers don't have to."""
    nested_root = tmp_path / "nested" / "snapshots"
    assert not nested_root.exists()
    store = SnapshotStore(nested_root)
    snap = _admit_one()
    store.save(snap)
    assert nested_root.is_dir()


def test_store_load_round_trip_reconstructs_snapshot(tmp_path: Path) -> None:
    """save() then load() reconstructs an equivalent Snapshot."""
    snap = _admit_one()
    store = SnapshotStore(tmp_path)
    store.save(snap)
    restored = store.load(snap.snapshot_id)
    # Identity preserved.
    assert restored.snapshot_id == snap.snapshot_id
    assert restored.ontology_id == snap.ontology_id
    # Graph hash matches — the strongest "same graph" check.
    assert restored.graph.content_hash() == snap.graph.content_hash()
    # Tasks + history preserved.
    assert len(restored.tasks) == len(snap.tasks)
    assert restored.history == snap.history


def test_store_save_writes_canonical_json(tmp_path: Path) -> None:
    """The on-disk file is JSON with sorted keys + indent, ending in newline.

    Determinism of the on-disk shape matters for diffing snapshots
    across builds and CI artifacts.
    """
    snap = _admit_one()
    store = SnapshotStore(tmp_path)
    path = store.save(snap)
    raw = path.read_text(encoding="utf-8")
    assert raw.endswith("\n")
    parsed = json.loads(raw)
    # Top-level keys present and JSON-compatible.
    assert parsed["snapshot_id"] == snap.snapshot_id
    assert set(parsed.keys()) == {
        "snapshot_id",
        "ontology_id",
        "graph",
        "tasks",
        "lineage",
        "history",
    }


def test_store_load_missing_file_raises_store_error(tmp_path: Path) -> None:
    store = SnapshotStore(tmp_path)
    with pytest.raises(StoreError, match="not found"):
        store.load("sha256:does-not-exist")


def test_store_load_invalid_json_raises_store_error(tmp_path: Path) -> None:
    store = SnapshotStore(tmp_path)
    bad = tmp_path / "sha256:bogus.json"
    bad.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(StoreError, match="not valid JSON"):
        store.load("sha256:bogus")


def test_store_load_non_mapping_raises_store_error(tmp_path: Path) -> None:
    store = SnapshotStore(tmp_path)
    bad = tmp_path / "sha256:listy.json"
    bad.write_text("[1, 2, 3]\n", encoding="utf-8")
    with pytest.raises(StoreError, match="must be a mapping"):
        store.load("sha256:listy")


def test_store_load_mismatched_id_raises_store_error(tmp_path: Path) -> None:
    """A file whose stored snapshot_id != its filename is rejected.

    Catches accidental rename / tampering — the content hash must agree
    with the storage key.
    """
    snap = _admit_one()
    store = SnapshotStore(tmp_path)
    store.save(snap)
    # Rename the file to a different id; loading by that id should fail.
    src = tmp_path / f"{snap.snapshot_id}.json"
    dst = tmp_path / "sha256:wrong-id.json"
    src.rename(dst)
    with pytest.raises(StoreError, match="id mismatch"):
        store.load("sha256:wrong-id")


def test_snapshot_from_dict_rejects_missing_required_field() -> None:
    """Drops `snapshot_id` and confirms StoreError fires."""
    snap = _admit_one()
    payload = snapshot_to_dict(snap)
    del payload["snapshot_id"]
    with pytest.raises(StoreError, match="snapshot_id"):
        snapshot_from_dict(payload)


def test_snapshot_from_dict_rejects_bad_field_type() -> None:
    """`graph` must be a mapping; a string is rejected."""
    snap = _admit_one()
    payload = snapshot_to_dict(snap)
    payload["graph"] = "not a mapping"
    with pytest.raises(StoreError, match="graph"):
        snapshot_from_dict(payload)


def test_snapshot_from_dict_rejects_unknown_role() -> None:
    """Unknown role values trip the Role enum at reconstruction time."""
    snap = _admit_one()
    payload = snapshot_to_dict(snap)
    # Find a node and corrupt its roles list.
    payload["graph"]["nodes"][0]["roles"] = ["not-a-role"]
    with pytest.raises(StoreError, match="role"):
        snapshot_from_dict(payload)


def test_snapshot_from_dict_rejects_unknown_visibility() -> None:
    """Unknown visibility values are rejected."""
    snap = _admit_one()
    payload = snapshot_to_dict(snap)
    payload["graph"]["nodes"][0]["visibility"] = "translucent"
    with pytest.raises(StoreError, match="visibility"):
        snapshot_from_dict(payload)


def test_snapshot_from_dict_preserves_graph_meta() -> None:
    """Non-empty `graph.meta` survives the round trip via dict and JSON."""
    snap = _admit_one()
    snap.graph.meta["discovery_title"] = "Ops Portal API"
    snap.graph.meta["build_seed"] = 42
    payload = snapshot_to_dict(snap)
    assert payload["graph"].get("meta") == {
        "discovery_title": "Ops Portal API",
        "build_seed": 42,
    }
    restored = snapshot_from_dict(payload)
    assert restored.graph.meta == snap.graph.meta
    encoded = json.dumps(payload, sort_keys=True)
    decoded = json.loads(encoded)
    restored_via_json = snapshot_from_dict(decoded)
    assert restored_via_json.graph.meta == snap.graph.meta


def test_snapshot_to_dict_omits_empty_graph_meta() -> None:
    """An empty `graph.meta` does not emit the key (canonical-JSON omission)."""
    snap = _admit_one()
    assert dict(snap.graph.meta) == {}
    payload = snapshot_to_dict(snap)
    assert "meta" not in payload["graph"]


def test_snapshot_from_dict_is_inverse_of_snapshot_to_dict_via_json() -> None:
    """The full pipe `Snapshot -> dict -> JSON -> dict -> Snapshot` round-trips.

    Catches MappingProxy / tuple / dataclass leaks that would survive
    `snapshot_to_dict` in memory but break once a JSON encode/decode
    round trip sits in the middle.
    """
    snap = _admit_one()
    payload = snapshot_to_dict(snap)
    encoded = json.dumps(payload, sort_keys=True)
    decoded = json.loads(encoded)
    restored = snapshot_from_dict(decoded)
    assert restored.snapshot_id == snap.snapshot_id
    assert restored.graph.content_hash() == snap.graph.content_hash()
    assert restored.history == snap.history
