"""Staged, constraint-propagating generation (packs/cyber_webapp/DESIGN.md).

The loot shape chosen first *bounds* the oracle's exploit shape, so a world is
solvable by construction. These drive the real pipeline end to end (no mocks):
a file-loot world admits, realizes, and is solved by a genuine path-traversal
HTTP exploit that recovers the flag from the in-memory file store.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping
from pathlib import Path

from cyber_webapp import WebappPack
from cyber_webapp.codegen import _realize_graph
from cyber_webapp.vulnerabilities import CATALOG
from graphschema import WorldGraph
from openrange_pack_sdk import Snapshot

from openrange.core.admit import admit
from openrange.core.episode import EpisodeService


def _manifest(loot: str, seed: int = 7, **extra: object) -> dict[str, object]:
    return {
        "pack": {"id": "webapp"},
        "runtime": {"tick": {"mode": "off"}},
        "npc": [],
        "seed": seed,
        "loot_shapes": {loot: 1, "db" if loot == "file" else "file": 0},
        **extra,
    }


def _admit(loot: str, seed: int = 7, **extra: object) -> Snapshot:
    snap = admit(WebappPack(), manifest=_manifest(loot, seed, **extra), max_repairs=3)
    assert isinstance(snap, Snapshot), snap
    return snap


def _store_kinds(graph: WorldGraph) -> set[str]:
    return {str(n.attrs.get("kind")) for n in graph.by_kind("data_store")}


def _oracle_shapes(graph: WorldGraph) -> set[str]:
    shapes: set[str] = set()
    for vuln in graph.by_kind("vulnerability"):
        kind = str(vuln.attrs.get("kind", ""))
        if kind in CATALOG:
            shapes.add(CATALOG[kind].shape)
    return shapes


def test_file_loot_admits_and_forces_file_read_oracle() -> None:
    snap = _admit("file")
    assert "file" in _store_kinds(snap.graph)
    assert "file_read" in _oracle_shapes(snap.graph)


def test_db_loot_admits_and_forces_response_leak_oracle() -> None:
    snap = _admit("db")
    assert "kv" in _store_kinds(snap.graph)
    assert "file" not in _store_kinds(snap.graph)
    assert "file_read" not in _oracle_shapes(snap.graph)


def test_loot_shape_is_manifest_selectable() -> None:
    assert _store_kinds(_admit("file").graph) == {"file"}
    assert _store_kinds(_admit("db").graph) == {"kv"}


def test_file_loot_keeps_flag_out_of_db_and_secrets() -> None:
    # Shape purity: the flag lives only in the in-memory file map, so a stray
    # response-leak vuln can't shortcut the file-read challenge.
    snap = _admit("file")
    seed = json.loads(_realize_graph(snap.graph)["seed.json"])
    flag = str(snap.graph.nodes["secret_flag"].attrs["value_ref"])
    assert flag in seed["files"].values()
    assert not any(flag in str(row) for row in seed["records"].values())
    assert flag not in seed["secrets"].values()


def test_manifest_knobs_ignore_non_mapping_values() -> None:
    # A bad loot_shapes / vuln_kinds value is dropped, not crashed on.
    snap = admit(
        WebappPack(),
        manifest={
            "pack": {"id": "webapp"},
            "seed": 7,
            "runtime": {"tick": {"mode": "off"}},
            "npc": [],
            "loot_shapes": "not-a-mapping",
            "vuln_kinds": 5,
        },
        max_repairs=3,
    )
    assert isinstance(snap, Snapshot), snap


def test_degenerate_loot_weights_fall_back_to_db() -> None:
    # All-zero and non-int weights leave an empty pool, which resolves to db.
    for weights in ({"db": 0, "file": 0}, {"db": "lots", "file": True}):
        snap = admit(
            WebappPack(),
            manifest={
                "pack": {"id": "webapp"},
                "seed": 7,
                "runtime": {"tick": {"mode": "off"}},
                "npc": [],
                "loot_shapes": weights,
            },
            max_repairs=3,
        )
        assert isinstance(snap, Snapshot), snap
        assert _store_kinds(snap.graph) == {"kv"}


def test_file_loot_is_deterministic() -> None:
    assert _admit("file", seed=3).snapshot_id == _admit("file", seed=3).snapshot_id
    assert _admit("file", seed=3).snapshot_id != _admit("file", seed=4).snapshot_id


def _path_traversal_target(graph: WorldGraph) -> tuple[str, str, str]:
    """(endpoint public_url, query param, absolute flag file path)."""
    vuln = next(
        n
        for n in graph.by_kind("vulnerability")
        if n.attrs.get("kind") == "path_traversal"
    )
    params = vuln.attrs["params"]
    assert isinstance(params, Mapping)
    endpoint_id = next(e.dst for e in graph.out_edges(vuln.id, "affects"))
    flag_path = next(
        str(n.attrs["key"])
        for n in graph.by_kind("record")
        if str(n.attrs.get("key", "")).startswith("/")
    )
    return (
        str(graph.nodes[endpoint_id].attrs["public_url"]),
        str(params["target_param"]),
        flag_path,
    )


def test_path_traversal_episode_recovers_flag(tmp_path: Path) -> None:
    snap = _admit("file")
    url_path, param, flag_path = _path_traversal_target(snap.graph)
    expected = str(snap.graph.nodes["secret_flag"].attrs["value_ref"])
    pentest = next(t for t in snap.tasks if t.meta.get("family") == "webapp.pentest")

    svc = EpisodeService(WebappPack(), tmp_path)
    try:
        handle = svc.start_episode(snap, pentest.id)
        base = svc.surface(handle)["base_url"]
        # A real exploit: an absolute path escapes the handler's base dir.
        exploit = f"{base}{url_path}?{param}={urllib.parse.quote(flag_path)}"
        recovered = urllib.request.urlopen(exploit, timeout=10).read().decode().strip()
        assert recovered == expected, recovered
        (svc.solver_root(handle) / "result.json").write_text(
            json.dumps({"flag": recovered})
        )
        report = svc.stop_episode(handle)
    finally:
        svc.close()
    assert report.passed, report.episode_result.reason


def test_command_injection_episode_recovers_flag(tmp_path: Path) -> None:
    # File loot also serves the code-exec shape: a forced command_injection
    # oracle, solved by a real shell-metacharacter injection.
    snap = _admit("file", vuln_kinds={"command_injection": 1})
    vuln = next(
        n
        for n in snap.graph.by_kind("vulnerability")
        if n.attrs.get("kind") == "command_injection"
    )
    params = vuln.attrs["params"]
    assert isinstance(params, Mapping)
    endpoint_id = next(e.dst for e in snap.graph.out_edges(vuln.id, "affects"))
    url_path = str(snap.graph.nodes[endpoint_id].attrs["public_url"])
    flag_path = next(
        str(n.attrs["key"])
        for n in snap.graph.by_kind("record")
        if str(n.attrs.get("key", "")).startswith("/")
    )
    expected = str(snap.graph.nodes["secret_flag"].attrs["value_ref"])
    pentest = next(t for t in snap.tasks if t.meta.get("family") == "webapp.pentest")

    svc = EpisodeService(WebappPack(), tmp_path)
    try:
        handle = svc.start_episode(snap, pentest.id)
        base = svc.surface(handle)["base_url"]
        injection = urllib.parse.quote(f"127.0.0.1; cat {flag_path}")
        exploit = f"{base}{url_path}?{params['target_param']}={injection}"
        recovered = urllib.request.urlopen(exploit, timeout=10).read().decode().strip()
        assert recovered == expected, recovered
        (svc.solver_root(handle) / "result.json").write_text(
            json.dumps({"flag": recovered})
        )
        report = svc.stop_episode(handle)
    finally:
        svc.close()
    assert report.passed, report.episode_result.reason


def test_path_traversal_wrong_path_recovers_nothing(tmp_path: Path) -> None:
    snap = _admit("file")
    url_path, param, _ = _path_traversal_target(snap.graph)
    pentest = next(t for t in snap.tasks if t.meta.get("family") == "webapp.pentest")

    svc = EpisodeService(WebappPack(), tmp_path)
    status = None
    try:
        handle = svc.start_episode(snap, pentest.id)
        base = svc.surface(handle)["base_url"]
        miss = f"{base}{url_path}?{param}={urllib.parse.quote('/no/such/file')}"
        try:
            urllib.request.urlopen(miss, timeout=10)
        except urllib.error.HTTPError as exc:
            status = exc.code
        # The handler 404s on a non-existent path; the solver submits nothing.
        report = svc.stop_episode(handle)
    finally:
        svc.close()
    assert status == 404
    assert not report.passed
