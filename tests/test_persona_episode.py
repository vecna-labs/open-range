"""Integration tests for PersonaAgent through the REAL episode/runtime seams.

Covers the runtime-level robustness fixes that unit tests with fake interfaces
can't reach: actor_id dedup in `_start_npcs`, per-slot resolve resilience, and
the auto-tick stop path not nulling a live daemon's Event.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import pytest
from cyber_webapp import WebappPack
from openrange_pack_sdk import NPCError, Snapshot, TaskSpec

from openrange.core.admit import admit
from openrange.core.episode import EpisodeService
from openrange.npc import resolve_manifest_npcs


class _RecBackend:
    """A recording NPC backend (no real model)."""

    def preflight(self) -> None:
        pass

    def build_agent(self, *, system_prompt: str, tools: Any = ()) -> Any:
        return lambda prompt: {"message": "ok"}


def _admit(npc: list[dict[str, Any]], tick: str = "off") -> tuple[Snapshot, TaskSpec]:
    manifest = {
        "world": {"goal": "persona episode"},
        "pack": {"id": "webapp"},
        "runtime": {"tick": {"mode": tick, "rate_hz": 50}},
        "npc": npc,
    }
    snap = admit(WebappPack(), manifest)
    assert isinstance(snap, Snapshot), snap
    task = next(t for t in snap.tasks if t.meta.get("family") == "webapp.pentest")
    return snap, task


def test_duplicate_named_personas_are_deduped(tmp_path: Path) -> None:
    # two count=1 entries with the same name would share an actor_id (and thus
    # any id-scoped state); _start_npcs must seat only one.
    entry = {
        "type": "cyber.persona",
        "config": {"name": "clerk", "tools": ["http_get"]},
    }
    snap, task = _admit([dict(entry), dict(entry)])
    svc = EpisodeService(WebappPack(), tmp_path, npc_agent_backend=_RecBackend())
    try:
        handle = svc.start_episode(snap, task.id)
        seated = svc._episodes[handle.id].npcs
        assert len([n for n in seated if n.actor_id == "clerk"]) == 1
        svc.stop_episode(handle)
    finally:
        svc.close()


def test_resolve_skips_a_bad_slot_but_raises_on_unknown_type() -> None:
    # a per-slot construction error (cadence_ticks:0) must not kill the run...
    npcs = resolve_manifest_npcs(
        (
            {
                "type": "cyber.persona",
                "config": {"name": "Good", "tools": ["http_get"]},
            },
            {"type": "cyber.persona", "config": {"name": "Bad", "cadence_ticks": 0}},
        )
    )
    assert [n.actor_id for n in npcs] == ["Good"]
    # ...but an unknown type is an authoring error and stays loud.
    with pytest.raises(NPCError):
        resolve_manifest_npcs(({"type": "no.such.npc", "config": {}},))


def test_stop_auto_tick_keeps_event_while_daemon_still_alive(tmp_path: Path) -> None:
    # regression for the auto-tick stop crash: if a tick outruns the join window,
    # the Event must NOT be nulled out from under the live loop.
    svc = EpisodeService(WebappPack(), tmp_path)
    try:
        running: Any = type("_R", (), {})()
        running.tick_stop = threading.Event()
        up = threading.Event()

        def _slow() -> None:
            up.set()
            time.sleep(6)  # still running when the 5 s join times out

        running.tick_thread = threading.Thread(target=_slow, daemon=True)
        running.tick_thread.start()
        up.wait()
        svc._stop_auto_tick(running)
        assert running.tick_stop is not None  # not nulled under a live daemon
        assert running.tick_thread is not None
    finally:
        svc.close()


class _ToolCallingBackend:
    """An NPC backend whose agent actually invokes the bound comms tools, so the
    persona's _emit_speak fires (a text-only fake would record nothing)."""

    def preflight(self) -> None:
        pass

    def build_agent(self, *, system_prompt: str, tools: Any = ()) -> Any:
        by_name = {getattr(t, "__name__", ""): t for t in tools}

        def agent(prompt: str) -> object:
            if "chat_post" in by_name:
                by_name["chat_post"](channel="ops", text="standup at 10")
            if "mail_send" in by_name:
                by_name["mail_send"](to="sam", subject="q3", body="need the numbers")
            return {"message": "ok"}

        return agent


def test_persona_seats_and_speaks_on_the_dashboard(tmp_path: Path) -> None:
    # front-back regression: a persona emits present (seat) on start and a speak
    # event when it uses a comms tool, both attributed to its actor_id — asserted
    # through the REAL DashboardView bridge, not the backend's belief.
    from openrange.dashboard import DashboardView

    entry = {
        "type": "cyber.persona",
        "config": {
            "name": "Dana",
            "role": "accountant",
            "tools": ["chat_post", "mail_send"],
            "cadence_ticks": 1,
        },
    }
    snap, task = _admit([entry])
    run_root = tmp_path / "ep"
    run_root.mkdir()
    dash = DashboardView(
        snap,
        event_log_path=run_root / "dashboard.events.jsonl",
        state_path=run_root / "dashboard.json",
        reset_artifacts=True,
    )
    svc = EpisodeService(
        WebappPack(), run_root, dashboard=dash, npc_agent_backend=_ToolCallingBackend()
    )
    try:
        handle = svc.start_episode(snap, task.id)
        svc.tick(handle)  # persona acts -> uses comms tools -> speaks
        svc.stop_episode(handle)
    finally:
        svc.close()

    actions: list[dict[str, Any]] = []
    for event in dash.bridge.snapshot_buffer():
        e = event.as_dict()
        data = e.get("data")
        if e.get("actor") == "Dana" and isinstance(data, dict):
            action = data.get("action")
            if data.get("actor_kind") == "npc" and isinstance(action, dict):
                actions.append(action)
    assert actions, "no npc events attributed to the seated persona 'Dana'"
    # the persona seats (present) and then speaks a callout, both as 'Dana'
    assert any(a.get("present") is True for a in actions)
    speaks = [a for a in actions if a.get("speak")]
    assert any(a["speak"] == "standup at 10" for a in speaks)
    assert all(a["display_name"] == "Dana" for a in speaks)
    # mail derives the 'email' callout class via kind (P2 render fix)
    assert any(a.get("kind") == "mail" for a in actions)
