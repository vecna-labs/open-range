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
