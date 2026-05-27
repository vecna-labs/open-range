"""Snapshot persistence."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from graphschema import Edge, Node, Role, Visibility, WorldGraph
from openrange_pack_sdk import BuildEvent, Snapshot, TaskSpec

from openrange.core.admit import snapshot_to_dict
from openrange.core.errors import StoreError


class SnapshotStore:
    """JSON file per Snapshot at `<root>/<snapshot_id>.json`."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def save(self, snapshot: Snapshot) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / f"{snapshot.snapshot_id}.json"
        data = snapshot_to_dict(snapshot)
        path.write_text(
            json.dumps(data, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return path

    def load(self, snapshot_id: str) -> Snapshot:
        """Raises `StoreError` if the file is missing, malformed, or its
        stored `snapshot_id` does not match `<root>/<snapshot_id>.json`."""
        path = self.root / f"{snapshot_id}.json"
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise StoreError(f"snapshot {snapshot_id!r} not found") from exc
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise StoreError(f"snapshot {snapshot_id!r} is not valid JSON") from exc
        if not isinstance(data, Mapping):
            raise StoreError("stored snapshot must be a mapping")
        snap = snapshot_from_dict(cast(Mapping[str, Any], data))
        if snap.snapshot_id != snapshot_id:
            raise StoreError(
                f"snapshot id mismatch: file {snapshot_id!r} carries id "
                f"{snap.snapshot_id!r}"
            )
        return snap


def snapshot_from_dict(data: Mapping[str, Any]) -> Snapshot:
    """Exact inverse of `snapshot_to_dict`. Raises `StoreError` on any
    structural problem."""
    return Snapshot(
        snapshot_id=_require_str(data, "snapshot_id"),
        ontology_id=_require_str(data, "ontology_id"),
        graph=_graph_from_dict(_require_mapping(data, "graph")),
        tasks=tuple(_task_from_dict(t) for t in _require_sequence(data, "tasks")),
        lineage=dict(_require_mapping(data, "lineage")),
        history=tuple(_event_from_dict(e) for e in _require_sequence(data, "history")),
    )


def _graph_from_dict(data: Mapping[str, Any]) -> WorldGraph:
    graph = WorldGraph(ontology=_require_str(data, "ontology"))
    for n in _require_sequence(data, "nodes"):
        graph.add_node(_node_from_dict(n))
    for e in _require_sequence(data, "edges"):
        graph.add_edge(_edge_from_dict(e))
    meta_raw = data.get("meta")
    if meta_raw is not None:
        if not isinstance(meta_raw, Mapping):
            raise StoreError(
                f"graph 'meta' must be a mapping, got {type(meta_raw).__name__}",
            )
        graph.meta.update(dict(meta_raw))
    return graph


def _node_from_dict(data: Mapping[str, Any]) -> Node:
    roles_raw = data.get("roles", [])
    if not isinstance(roles_raw, list):
        raise StoreError(f"node {data.get('id')!r} 'roles' must be a list")
    try:
        roles = {Role(r) for r in roles_raw}
    except ValueError as exc:
        raise StoreError(
            f"node {data.get('id')!r} carries unknown role: {exc}"
        ) from exc
    visibility_raw = data.get("visibility", Visibility.PUBLIC.value)
    if not isinstance(visibility_raw, str):
        raise StoreError(f"node {data.get('id')!r} 'visibility' must be a string")
    try:
        visibility = Visibility(visibility_raw)
    except ValueError as exc:
        raise StoreError(
            f"node {data.get('id')!r} carries unknown visibility {visibility_raw!r}"
        ) from exc
    attrs_raw = data.get("attrs", {})
    if not isinstance(attrs_raw, Mapping):
        raise StoreError(f"node {data.get('id')!r} 'attrs' must be a mapping")
    return Node(
        id=_require_str(data, "id"),
        kind=_require_str(data, "kind"),
        attrs=dict(attrs_raw),
        roles=roles,
        visibility=visibility,
    )


def _edge_from_dict(data: Mapping[str, Any]) -> Edge:
    attrs_raw = data.get("attrs", {})
    if not isinstance(attrs_raw, Mapping):
        raise StoreError(f"edge {data.get('id')!r} 'attrs' must be a mapping")
    return Edge(
        id=_require_str(data, "id"),
        kind=_require_str(data, "kind"),
        src=_require_str(data, "src"),
        dst=_require_str(data, "dst"),
        attrs=dict(attrs_raw),
    )


def _task_from_dict(data: Mapping[str, Any]) -> TaskSpec:
    entrypoints_raw = data.get("entrypoints", [])
    goal_nodes_raw = data.get("goal_nodes", [])
    if not isinstance(entrypoints_raw, list) or not all(
        isinstance(s, str) for s in entrypoints_raw
    ):
        raise StoreError(
            f"task {data.get('id')!r} 'entrypoints' must be a list of strings"
        )
    if not isinstance(goal_nodes_raw, list) or not all(
        isinstance(s, str) for s in goal_nodes_raw
    ):
        raise StoreError(
            f"task {data.get('id')!r} 'goal_nodes' must be a list of strings"
        )
    meta_raw = data.get("meta", {})
    if not isinstance(meta_raw, Mapping):
        raise StoreError(f"task {data.get('id')!r} 'meta' must be a mapping")
    return TaskSpec(
        id=_require_str(data, "id"),
        instruction=_require_str(data, "instruction"),
        entrypoints=tuple(entrypoints_raw),
        goal_nodes=tuple(goal_nodes_raw),
        feasibility_check=_require_str(data, "feasibility_check"),
        success_check=_require_str(data, "success_check"),
        meta=dict(meta_raw),
    )


def _event_from_dict(data: Mapping[str, Any]) -> BuildEvent:
    refs_raw = data.get("refs", [])
    if not isinstance(refs_raw, list) or not all(isinstance(s, str) for s in refs_raw):
        raise StoreError(
            f"history event seq={data.get('seq')!r}: 'refs' must be a list of strings"
        )
    seq_raw = data.get("seq")
    if not isinstance(seq_raw, int) or isinstance(seq_raw, bool):
        raise StoreError(
            f"history event 'seq' must be an int, got {type(seq_raw).__name__}"
        )
    return BuildEvent(
        seq=seq_raw,
        phase=_require_str(data, "phase"),
        detail=_require_str(data, "detail"),
        refs=tuple(refs_raw),
    )


def _require_str(data: Mapping[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str):
        raise StoreError(
            f"required field {key!r} must be a string, got {type(value).__name__}"
        )
    return value


def _require_mapping(data: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = data.get(key)
    if not isinstance(value, Mapping):
        raise StoreError(
            f"required field {key!r} must be a mapping, got {type(value).__name__}"
        )
    return cast(Mapping[str, Any], value)


def _require_sequence(data: Mapping[str, Any], key: str) -> list[Mapping[str, Any]]:
    value = data.get(key)
    if not isinstance(value, list):
        raise StoreError(
            f"required field {key!r} must be a list, got {type(value).__name__}"
        )
    for item in value:
        if not isinstance(item, Mapping):
            raise StoreError(
                f"every item in {key!r} must be a mapping, got {type(item).__name__}"
            )
    return [cast(Mapping[str, Any], item) for item in value]
