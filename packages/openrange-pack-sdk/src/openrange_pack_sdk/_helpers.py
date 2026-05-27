"""Pack-author ergonomics: graph helpers + manifest field accessors +
filesystem helpers.

Thin wrappers around ``graphschema``'s primitives that codify the
conventions every pack would otherwise reinvent (deterministic edge ids,
attrs default to ``{}`` not ``None``, return the constructed node so
callers can chain) plus typed readers for the free-form Manifest mapping
and a tree-write helper realizers reuse.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from graphschema import Edge, Node, Role, Visibility, WorldGraph

from openrange_pack_sdk._types import Manifest


def edge_id(src: str, kind: str, dst: str) -> str:
    """Standard edge id: ``"{src}__{kind}__{dst}"``.

    Deterministic from the triple, so two builds that emit the same edge
    set content-address to the same snapshot id.
    """
    return f"{src}__{kind}__{dst}"


def add_node(
    graph: WorldGraph,
    *,
    kind: str,
    id: str,
    attrs: Mapping[str, Any] | None = None,
    roles: set[Role] | None = None,
    visibility: Visibility = Visibility.PUBLIC,
) -> Node:
    """Construct + insert a Node. Returns the Node so callers can chain."""
    node = Node(
        id=id,
        kind=kind,
        attrs=dict(attrs or {}),
        roles=roles or set(),
        visibility=visibility,
    )
    graph.add_node(node)
    return node


def add_edge(
    graph: WorldGraph,
    *,
    kind: str,
    src: str,
    dst: str,
    attrs: Mapping[str, Any] | None = None,
) -> Edge:
    """Construct + insert an Edge with ``edge_id(src, kind, dst)``."""
    edge = Edge(
        id=edge_id(src, kind, dst),
        kind=kind,
        src=src,
        dst=dst,
        attrs=dict(attrs or {}),
    )
    graph.add_edge(edge)
    return edge


def manifest_int(manifest: Manifest, key: str, *, default: int = 0) -> int:
    """Read ``manifest[key]`` as int. ``bool`` is rejected explicitly so
    ``True`` does not silently become ``1``."""
    raw = manifest.get(key, default)
    if isinstance(raw, bool):
        return default
    return raw if isinstance(raw, int) else default


def manifest_str(manifest: Manifest, key: str, *, default: str = "") -> str:
    """Read ``manifest[key]`` as str, returning ``default`` otherwise."""
    raw = manifest.get(key, default)
    return raw if isinstance(raw, str) else default


def manifest_bool(manifest: Manifest, key: str, *, default: bool = False) -> bool:
    """Read ``manifest[key]`` as bool, returning ``default`` otherwise."""
    raw = manifest.get(key, default)
    return raw if isinstance(raw, bool) else default


def manifest_float(manifest: Manifest, key: str, *, default: float = 0.0) -> float:
    """Read ``manifest[key]`` as float. Promotes int → float; rejects bool
    so ``True`` does not silently become ``1.0``."""
    raw = manifest.get(key, default)
    if isinstance(raw, bool):
        return default
    if isinstance(raw, int | float):
        return float(raw)
    return default


def manifest_list(
    manifest: Manifest,
    key: str,
    *,
    default: list[Any] | None = None,
) -> list[Any]:
    """Read ``manifest[key]`` as a list, returning a *copy* so callers
    can mutate without surprising the manifest's owner. Returns
    ``default`` (or ``[]``) when the key is absent or non-list."""
    raw = manifest.get(key)
    if isinstance(raw, list):
        return list(raw)
    return list(default) if default is not None else []


def write_tree(root: Path, files: Mapping[str, str]) -> None:
    """Write ``{relative_path: contents}`` under ``root``, creating any
    needed parent directories.

    Used by the filesystem-based RuntimeHandle bases to materialize
    ``prepare_env_files`` output; exposed as a standalone helper so
    packs that implement ``RuntimeHandle`` directly can use the same
    convention without copy-pasting the loop.
    """
    for rel, content in files.items():
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
