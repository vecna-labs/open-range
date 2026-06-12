"""The dynamic admission gate for LLM-realized handlers.

The gate renders + runs a world, exploits it, and lets the consequence verifier
decide: the exploit must leak the flag, a benign request must not. These drive the
real pipeline (no mocks) — the verdict logic on synthetic bodies, the codegen hook on
a rendered graph, and the gate end-to-end on a live command-injection episode. The
pack owns the pure verdict + exploit oracle; the episode orchestration is the host's,
so it lives here in the test, not in the pack.
"""

from __future__ import annotations

import urllib.request
from collections.abc import Callable
from pathlib import Path

from cyber_webapp import WebappPack
from cyber_webapp.codegen import _realize_graph
from cyber_webapp.ontology import ONTOLOGY_ID
from cyber_webapp.realize_admit import (
    AdmissionVerdict,
    classify_admission,
    cmdi_exploit_and_benign,
)
from graphschema import Edge, Node, Visibility, WorldGraph
from openrange_pack_sdk import Snapshot

from openrange.core.admit import admit

_FLAG = "ORANGE_LONG_FLAG_VALUE_0123"


def _flag_graph() -> WorldGraph:
    graph = WorldGraph(ontology=ONTOLOGY_ID)
    graph.add_node(
        Node(
            id="secret_flag",
            kind="secret",
            attrs={"value_ref": _FLAG},
            visibility=Visibility.HIDDEN,
        )
    )
    return graph


def _admit_cmdi() -> Snapshot:
    snap = admit(
        WebappPack(),
        manifest={
            "pack": {"id": "webapp"},
            "runtime": {"tick": {"mode": "off"}},
            "npc": [],
            "seed": 7,
            "loot_shapes": {"file": 1, "db": 0},
            "vuln_kinds": {"command_injection": 1},
        },
        max_repairs=3,
    )
    assert isinstance(snap, Snapshot), snap
    return snap


def _run_and_classify(snap: Snapshot, tmp_path: Path) -> AdmissionVerdict:
    # The host orchestrates the episode (a pack must not import openrange); the pack
    # supplies the exploit oracle and the verdict.
    from openrange.core.episode import EpisodeService

    graph = snap.graph
    exploit_path, benign_path = cmdi_exploit_and_benign(graph)
    service = EpisodeService(WebappPack(), tmp_path)
    try:
        task = next(t for t in snap.tasks if t.meta.get("family") == "webapp.pentest")
        handle = service.start_episode(snap, task.id)
        base = str(service.surface(handle)["base_url"])
        exploit_body = (
            urllib.request.urlopen(base + exploit_path, timeout=10).read().decode()
        )
        benign_body = (
            urllib.request.urlopen(base + benign_path, timeout=10).read().decode()
        )
    finally:
        service.close()
    return classify_admission(graph, exploit_body, benign_body)


def test_classify_admission_accepts_solvable_and_not_trivial() -> None:
    verdict = classify_admission(
        _flag_graph(), f"leaked {_FLAG} here", "clean response"
    )
    assert verdict.accepted and verdict.solvable and not verdict.trivial


def test_classify_admission_rejects_trivial() -> None:
    verdict = classify_admission(
        _flag_graph(), f"x {_FLAG}", f"benign also has {_FLAG}"
    )
    assert not verdict.accepted and verdict.trivial
    assert "trivial" in verdict.reason


def test_classify_admission_rejects_not_solvable() -> None:
    verdict = classify_admission(_flag_graph(), "exploit leaked nothing", "clean")
    assert not verdict.accepted and not verdict.solvable
    assert "not solvable" in verdict.reason


def test_codegen_uses_a_realized_handler_over_the_template() -> None:
    graph = _admit_cmdi().graph
    vuln = next(
        n
        for n in graph.by_kind("vulnerability")
        if n.attrs.get("kind") == "command_injection"
    )
    vuln.attrs["realized_handler"] = (
        "def handle(query, state):\n    return 200, {}, b'REALIZED_MARKER_XYZ'\n"
    )
    assert "REALIZED_MARKER_XYZ" in _realize_graph(graph)["app.py"]


def test_admission_gate_accepts_a_faithful_cmdi_world(tmp_path: Path) -> None:
    verdict = _run_and_classify(_admit_cmdi(), tmp_path)
    assert verdict.accepted, verdict.reason
    assert verdict.solvable and not verdict.trivial


def _cmdi_with_realized(make_handler: Callable[[str, str], str]) -> Snapshot:
    # A command-injection world whose vuln carries a stand-in "realized" handler, as
    # an LLM would write. The injection context is pinned to "separator" so the
    # exploit is a plain `;cat <path>`, which the faithful handler below also speaks.
    from cyber_webapp.realize_admit import _flag_record_key

    snap = _admit_cmdi()
    graph = snap.graph
    vuln = next(
        n
        for n in graph.by_kind("vulnerability")
        if n.attrs.get("kind") == "command_injection"
    )
    params = vuln.attrs["params"]
    assert isinstance(params, dict)
    params["inj_context"] = "separator"
    param = str(params["target_param"])
    vuln.attrs["realized_handler"] = make_handler(param, _flag_record_key(graph))
    return snap


def _faithful_handler(param: str, flag_path: str) -> str:
    # A different-but-real command injection: split on ';', run `cat <path>` against
    # the file store — so the flag is reachable only by injecting the command.
    del flag_path
    return (
        "def handle(query, state):\n"
        "    import shlex\n"
        f"    raw = (query.get({param!r}) or [''])[0]\n"
        "    files = state.get('files', {})\n"
        "    if ';' not in raw:\n"
        "        return 200, {}, b'pong'\n"
        "    parts = shlex.split(raw.split(';', 1)[1])\n"
        "    if len(parts) == 2 and parts[0] == 'cat':\n"
        "        return 200, {}, files.get(parts[1], '').encode()\n"
        "    return 200, {}, b'rejected'\n"
    )


def _trivial_handler(param: str, flag_path: str) -> str:
    # Returns the flag on any request — solvable without the injection at all.
    del param
    return (
        "def handle(query, state):\n"
        f"    return 200, {{}}, state['files'][{flag_path!r}].encode()\n"
    )


def _broken_handler(param: str, flag_path: str) -> str:
    del param, flag_path
    return "def handle(query, state):\n    return 200, {}, b'nothing here'\n"


def test_admission_gate_accepts_a_faithful_realized_handler(tmp_path: Path) -> None:
    verdict = _run_and_classify(_cmdi_with_realized(_faithful_handler), tmp_path)
    assert verdict.accepted, verdict.reason


def test_admission_gate_rejects_a_trivial_realized_handler(tmp_path: Path) -> None:
    verdict = _run_and_classify(_cmdi_with_realized(_trivial_handler), tmp_path)
    assert not verdict.accepted and verdict.trivial


def test_admission_gate_rejects_a_broken_realized_handler(tmp_path: Path) -> None:
    verdict = _run_and_classify(_cmdi_with_realized(_broken_handler), tmp_path)
    assert not verdict.accepted and not verdict.solvable


def test_internal_helpers_cover_defensive_branches() -> None:
    import pytest
    from cyber_webapp.realize_admit import _cmdi_payload, _flag_record_key
    from openrange_pack_sdk import PackError

    assert _cmdi_payload({"inj_context": "substitution"}, "/f") == "$(cat /f)"
    assert _cmdi_payload({"inj_context": "quoted", "quote": '"'}, "/f") == (
        '"; cat /f; echo "'
    )
    assert _cmdi_payload({"inj_context": "separator"}, "/f").endswith("cat /f")

    # A record that holds only a non-flag secret is walked past; a world with no flag
    # record at all is malformed (the gate's caller guarantees one).
    no_flag = WorldGraph(ontology=ONTOLOGY_ID)
    no_flag.add_node(Node(id="rec", kind="record", attrs={"key": "/data"}))
    no_flag.add_node(Node(id="sec", kind="secret", attrs={"kind": "password"}))
    no_flag.add_edge(Edge(id="h", kind="holds", src="rec", dst="sec", attrs={}))
    with pytest.raises(PackError):
        _flag_record_key(no_flag)

    # A command_injection vuln whose params aren't a mapping is rejected.
    malformed = WorldGraph(ontology=ONTOLOGY_ID)
    malformed.add_node(
        Node(
            id="v",
            kind="vulnerability",
            attrs={"kind": "command_injection", "params": "not-a-map"},
        )
    )
    with pytest.raises(PackError):
        cmdi_exploit_and_benign(malformed)
