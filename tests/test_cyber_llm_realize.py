"""LLM handler realization across classes (DESIGN.md §9, #260).

Two halves, both free of a live LLM: the per-class realization *request* is a pure
function of the world (tested directly), and the dynamic admission gate is exercised by
injecting hand-written handlers — a faithful one is admitted, a trivial one rejected —
proving the gate generalizes past command-injection to the response-leak and file-read
families via the reference solver.
"""

from __future__ import annotations

import urllib.error
import urllib.request
from pathlib import Path

import pytest
from cyber_webapp import WebappPack
from cyber_webapp.llm_realize import (
    handler_from_result,
    realization_request,
)
from cyber_webapp.ontology import ONTOLOGY_ID
from cyber_webapp.realize_admit import AdmissionVerdict, classify_admission
from cyber_webapp.reference_solver import _vuln_of_kind, exploit_and_benign
from graphschema import Node, WorldGraph
from openrange_pack_sdk import PackError, Snapshot

from openrange.core.admit import admit
from openrange.core.episode import EpisodeService


def _admit(loot: str, kind: str, **pin: object) -> Snapshot:
    snap = admit(
        WebappPack(),
        manifest={
            "pack": {"id": "webapp"},
            "runtime": {"tick": {"mode": "off"}},
            "npc": [],
            "seed": 7,
            "loot_shapes": {loot: 1, "db" if loot == "file" else "file": 0},
            "vuln_kinds": {kind: 1},
        },
        max_repairs=3,
    )
    assert isinstance(snap, Snapshot), snap
    if pin:
        params = _vuln_of_kind(snap.graph, kind).attrs["params"]
        assert isinstance(params, dict)
        params.update(pin)  # pin the context so the exploit + handler agree
    return snap


def _fetch(url: str) -> str:
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            return str(resp.read().decode())
    except urllib.error.HTTPError as exc:
        return exc.read().decode()


def _gate(snap: Snapshot, kind: str, handler: str, workdir: Path) -> AdmissionVerdict:
    graph = snap.graph
    _vuln_of_kind(graph, kind).attrs["realized_handler"] = handler
    exploit_path, benign_path = exploit_and_benign(graph, kind)
    svc = EpisodeService(WebappPack(), workdir)
    try:
        task = next(t for t in snap.tasks if t.meta.get("family") == "webapp.pentest")
        handle = svc.start_episode(snap, task.id)
        base = str(svc.surface(handle)["base_url"])
        exploit_body, benign_body = (
            _fetch(base + exploit_path),
            _fetch(base + benign_path),
        )
    finally:
        svc.close()
    return classify_admission(graph, exploit_body, benign_body)


def _faithful_sqli(graph: WorldGraph) -> str:
    p = _vuln_of_kind(graph, "sql_injection").attrs["params"]
    param, table, leak = str(p["target_param"]), str(p["table"]), str(p["leak_column"])
    return "\n".join(
        [
            "def handle(query, state):",
            "    import json",
            f"    val = (query.get({param!r}, ['']) or [''])[0]",
            "    kc = state['schema']['key_column']",
            f'    sql = "SELECT " + kc + ", {leak} FROM {table} WHERE " + kc'
            ' + " = \'" + val + "\'"',
            "    try:",
            "        rows = [dict(r) for r in state['db'].execute(sql).fetchall()]",
            "    except Exception as e:",
            "        return 500, {}, json.dumps({'error': str(e)}).encode()",
            "    return 200, {}, json.dumps({'rows': rows}).encode()",
        ]
    )


def _trivial_sqli(graph: WorldGraph) -> str:
    p = _vuln_of_kind(graph, "sql_injection").attrs["params"]
    table, leak = str(p["table"]), str(p["leak_column"])
    return "\n".join(
        [
            "def handle(query, state):",
            "    import json",
            f"    rows = [dict(r) for r in"
            f" state['db'].execute('SELECT key, {leak} FROM {table}').fetchall()]",
            "    return 200, {}, json.dumps({'rows': rows}).encode()",
        ]
    )


def _faithful_path(graph: WorldGraph) -> str:
    p = _vuln_of_kind(graph, "path_traversal").attrs["params"]
    param, base = str(p["target_param"]), str(p["base_dir"])
    return "\n".join(
        [
            "def handle(query, state):",
            "    import posixpath",
            f"    req = (query.get({param!r}, ['']) or [''])[0]",
            "    while '../' in req:",
            "        req = req.replace('../', '')",
            f"    resolved = posixpath.normpath(posixpath.join({base!r}, req))",
            "    files = state.get('files', {})",
            "    if resolved in files:",
            "        return 200, {}, str(files[resolved]).encode()",
            "    return 404, {}, b'nope'",
        ]
    )


_CONTEXTS = [
    ("command_injection", "file", "inj_context", "separator"),
    ("command_injection", "file", "inj_context", "substitution"),
    ("command_injection", "file", "inj_context", "quoted"),
    ("sql_injection", "db", "context", "single"),
    ("sql_injection", "db", "context", "numeric"),
    ("sql_injection", "db", "context", "double"),
    ("path_traversal", "file", "confinement", "absolute_only"),
    ("path_traversal", "file", "confinement", "relative"),
    ("path_traversal", "file", "confinement", "dotdot_filter"),
]


@pytest.mark.parametrize(("kind", "loot", "key", "ctx"), _CONTEXTS)
def test_realization_request_per_context(
    kind: str, loot: str, key: str, ctx: str
) -> None:
    # Every class × sampled-context pair yields a handler-authoring prompt that names
    # the flag, so the realized handler matches the exploit the solver will run.
    req = realization_request(_admit(loot, kind, **{key: ctx}).graph, kind)
    assert "def handle" in req.prompt
    assert "flag" in req.prompt.lower()
    assert req.json_schema is not None


def test_realization_request_sqli_names_its_table() -> None:
    graph = _admit("db", "sql_injection").graph
    table = str(_vuln_of_kind(graph, "sql_injection").attrs["params"]["table"])
    assert table in realization_request(graph, "sql_injection").prompt


def test_realization_request_rejects_unrealized_kind() -> None:
    # ssti has a vuln in the graph but no realization prompt yet.
    with pytest.raises(PackError):
        realization_request(_admit("file", "ssti").graph, "ssti")


def test_realization_request_rejects_non_mapping_params() -> None:
    graph = WorldGraph(ontology=ONTOLOGY_ID)
    graph.add_node(
        Node(
            id="v",
            kind="vulnerability",
            attrs={"kind": "command_injection", "params": "not-a-map"},
        )
    )
    with pytest.raises(PackError):
        realization_request(graph, "command_injection")


def test_handler_from_result() -> None:
    assert handler_from_result({"handler": "def handle(): ..."}) == "def handle(): ..."
    assert handler_from_result({}) == ""
    assert handler_from_result(None) == ""
    assert handler_from_result({"handler": 123}) == ""


def test_gate_admits_a_realized_sqli_handler(tmp_path: Path) -> None:
    snap = _admit("db", "sql_injection", context="single")
    accepted = _gate(snap, "sql_injection", _faithful_sqli(snap.graph), tmp_path / "ok")
    assert accepted.accepted, accepted.reason
    rejected = _gate(
        snap, "sql_injection", _trivial_sqli(snap.graph), tmp_path / "triv"
    )
    assert not rejected.accepted and rejected.trivial  # benign also leaks


def test_gate_admits_a_realized_path_handler(tmp_path: Path) -> None:
    snap = _admit("file", "path_traversal", confinement="absolute_only")
    accepted = _gate(
        snap, "path_traversal", _faithful_path(snap.graph), tmp_path / "ok"
    )
    assert accepted.accepted, accepted.reason
