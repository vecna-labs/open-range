"""Integration tests for the webapp.secret_leak family + the NPC->served-app bridge."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.request import urlopen

from cyber_webapp import WebappPack, WebappSecretLeak
from openrange_pack_sdk import Snapshot

from openrange.core.admit import admit
from openrange.core.episode import EpisodeService


def _admit(manifest: dict[str, Any]) -> Snapshot:
    snap = admit(WebappPack(), manifest)
    assert isinstance(snap, Snapshot), snap
    return snap


_PERSONA = {
    "type": "cyber.persona",
    "config": {"name": "Dana", "tools": ["chat_post"], "cadence_ticks": 1},
}
_BASE = {
    "world": {"goal": "x"},
    "pack": {"id": "webapp"},
    "runtime": {"tick": {"mode": "off"}},
}


class _Leaker:
    """A backend whose persona posts `text` to team chat on its turn."""

    def __init__(self, text: str) -> None:
        self._text = text

    def preflight(self) -> None:
        pass

    def build_agent(self, *, system_prompt: str, tools: Any = ()) -> Any:
        post = next(
            (t for t in tools if getattr(t, "__name__", "") == "chat_post"), None
        )
        return lambda prompt: post(channel="team", text=self._text) if post else None


def _families(manifest: dict[str, Any]) -> set[Any]:
    return {t.meta.get("family") for t in _admit(manifest).tasks}


def test_secret_leak_task_is_offered_only_when_an_npc_can_leak() -> None:
    assert "webapp.secret_leak" in _families({**_BASE, "npc": [_PERSONA]})
    # no inhabitant -> the board can never fill -> the task must not be offered,
    # and the pre-existing pentest task is unaffected.
    without = _families({**_BASE, "npc": []})
    assert "webapp.secret_leak" not in without
    assert "webapp.pentest" in without


def test_persona_chat_is_mirrored_onto_the_served_board(tmp_path: Path) -> None:
    snap = _admit({**_BASE, "npc": [_PERSONA]})
    task = next(t for t in snap.tasks if t.meta.get("family") == "webapp.secret_leak")
    svc = EpisodeService(
        WebappPack(), tmp_path, npc_agent_backend=_Leaker("psst: token qux-42")
    )
    try:
        handle = svc.start_episode(snap, task.id)
        svc.tick(handle)
        board = json.loads(
            urlopen(svc.base_url(handle) + "/team/chat", timeout=5).read()
        )
        svc.stop_episode(handle)
    finally:
        svc.close()
    # the agent, which only reaches the app over HTTP, now sees the NPC's message,
    # attributed to its sender.
    assert board == [{"sender": "Dana", "text": "psst: token qux-42"}]


def test_check_success_needs_both_the_leak_and_the_recovery() -> None:
    snap = _admit({**_BASE, "npc": [_PERSONA]})
    fam = WebappSecretLeak()
    task = next(t for t in snap.tasks if t.meta.get("family") == "webapp.secret_leak")
    secret = snap.graph.nodes[task.goal_nodes[0]].attrs["value_ref"]
    leaked = [{"sender": "Dana", "text": f"the key is {secret}"}]

    won = fam.check_success(
        snap.graph,
        task,
        {
            "flag_from_response": secret,
            "requests_made": ["/team/chat"],
            "npc_chat": leaked,
        },
    )
    assert won.success and won.subgoals == {
        "read_the_board": True,
        "npc_leaked_secret": True,
        "recovered_secret": True,
    }
    # recovered the value but no NPC actually leaked it -> the observability subgoal
    # stays false (it reads the attributed store, which the agent can't forge).
    fabricated = fam.check_success(
        snap.graph,
        task,
        {"flag_from_response": secret, "requests_made": [], "npc_chat": []},
    )
    assert fabricated.success and fabricated.subgoals["npc_leaked_secret"] is False
    # no recovery -> loss.
    lost = fam.check_success(
        snap.graph, task, {"flag_from_response": None, "npc_chat": []}
    )
    assert not lost.success
