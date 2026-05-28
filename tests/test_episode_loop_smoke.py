"""End-to-end smoke: the real episode loop closes without an LLM.

Drives the actual pipeline that the unit tests stub out —
``admit`` → ``start_episode`` → scripted agent writes ``result.json`` →
``stop_episode`` (runtime ``collect`` → ``check_success``) → real
``EpisodeReport`` → ``auto_evolve`` → next snapshot. A scripted agent
(the kind's reference handler for build; the graph's flag for pentest)
stands in for the LLM, so the integration seams are exercised for real
rather than mocked.
"""

from __future__ import annotations

import json
from pathlib import Path

from cyber_webapp import WebappPack
from cyber_webapp.families.build.reference import api_list_reference
from openrange_pack_sdk import Snapshot, TaskSpec

from openrange.core import auto_evolve
from openrange.core.admit import admit
from openrange.core.episode import EpisodeService

MANIFEST = {
    "world": {"goal": "episode loop smoke"},
    "pack": {"id": "webapp"},
    "runtime": {"tick": {"mode": "off"}},
    "npc": [],
}


def _admit() -> Snapshot:
    snap = admit(WebappPack(), MANIFEST)
    assert isinstance(snap, Snapshot), snap
    return snap


def _only_task(snap: Snapshot, family: str) -> TaskSpec:
    tasks = [t for t in snap.tasks if t.meta.get("family") == family]
    assert len(tasks) == 1, f"expected exactly one {family} task, got {tasks}"
    return tasks[0]


def test_build_episode_grades_submitted_handler(tmp_path: Path) -> None:
    snap = _admit()
    task = _only_task(snap, "webapp.build")
    svc = EpisodeService(WebappPack(), tmp_path)
    try:
        handle = svc.start_episode(snap, task.id)
        agent_root = svc.agent_root(handle)
        (agent_root / "result.json").write_text(
            json.dumps({"endpoint_impl": api_list_reference(1)}),
            encoding="utf-8",
        )
        report = svc.stop_episode(handle)
    finally:
        svc.close()
    assert report.passed, report.episode_result.reason


def test_pentest_episode_then_evolve(tmp_path: Path) -> None:
    snap = _admit()
    task = _only_task(snap, "webapp.pentest")
    flag_value = snap.graph.nodes[task.goal_nodes[0]].attrs["value_ref"]

    svc = EpisodeService(WebappPack(), tmp_path)
    try:
        handle = svc.start_episode(snap, task.id)
        agent_root = svc.agent_root(handle)
        (agent_root / "result.json").write_text(
            json.dumps({"flag": flag_value}),
            encoding="utf-8",
        )
        report = svc.stop_episode(handle)
    finally:
        svc.close()
    assert report.passed, report.episode_result.reason

    # One pass → "harden" direction; pentest has harden mutations to apply.
    evolved = auto_evolve(snap, report, pack=WebappPack())
    assert evolved is not None
    assert evolved.snapshot_id != snap.snapshot_id
    assert any(event.phase == "evolve" for event in evolved.history)
