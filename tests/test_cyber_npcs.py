"""Behavioral tests for cyber NPCs.

The NPCs receive an ``interface`` mapping matching the shape produced
by ``WebappRuntime.surface()``:
``{base_url, http_get, http_get_json, solver_root}``. Tests use a fake
interface that records GET calls so we can assert on cadence,
rotation, and graceful error handling — no real subprocess needed.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from cyber_webapp.npcs.admin_audit import (
    AdminAudit,
)
from cyber_webapp.npcs.admin_audit import (
    factory as admin_audit_factory,
)
from cyber_webapp.npcs.browsing_user import (
    BrowsingUser,
)
from cyber_webapp.npcs.browsing_user import (
    factory as browsing_user_factory,
)


class _RuntimeSurface(dict[str, Any]):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[str] = []

        def http_get(path: object) -> bytes:
            self.calls.append(str(path))
            return b""

        def http_get_json(path: object) -> object:
            return json.loads(http_get(path).decode() or "null")

        self["base_url"] = "http://test.local"
        self["http_get"] = http_get
        self["http_get_json"] = http_get_json
        self["solver_root"] = "/tmp/fake-agent-root"


def test_browsing_user_acts_on_first_step_then_obeys_cadence() -> None:
    npc = BrowsingUser(cadence_ticks=3, paths=("/a",))
    iface = _RuntimeSurface()

    # First tick: act.
    npc.step(iface)
    assert iface.calls == ["/a"]

    # Next two ticks: silent (cooldown).
    npc.step(iface)
    npc.step(iface)
    assert iface.calls == ["/a"]

    # Cooldown elapsed: act again.
    npc.step(iface)
    assert iface.calls == ["/a", "/a"]


def test_browsing_user_rotates_through_paths() -> None:
    npc = BrowsingUser(cadence_ticks=1, paths=("/a", "/b", "/c"))
    iface = _RuntimeSurface()
    for _ in range(7):
        npc.step(iface)
    # cadence_ticks=1: every step acts → 7 calls cycling a→b→c→a→b→c→a
    assert iface.calls == ["/a", "/b", "/c", "/a", "/b", "/c", "/a"]


def test_browsing_user_swallows_http_errors() -> None:
    npc = BrowsingUser(cadence_ticks=1, paths=("/boom",))

    def boom(path: object) -> bytes:
        raise RuntimeError("HTTP exploded")

    iface: dict[str, Any] = {"base_url": "http://x", "http_get": boom}
    # Should not raise — NPC failures must not sink the episode.
    npc.step(iface)
    npc.step(iface)


def test_browsing_user_no_op_when_interface_missing_http_get() -> None:
    npc = BrowsingUser(cadence_ticks=1, paths=("/a",))
    npc.step({})  # missing http_get → silent no-op


def test_browsing_user_rejects_invalid_construction() -> None:
    with pytest.raises(ValueError, match="cadence_ticks"):
        BrowsingUser(cadence_ticks=0)
    with pytest.raises(ValueError, match="paths"):
        BrowsingUser(paths=())


def test_browsing_user_factory_reads_config() -> None:
    npc = browsing_user_factory(
        {
            "cadence_ticks": 4,
            "paths": ["/x", "/y"],
            "timeout_seconds": 0.5,
        },
    )
    assert isinstance(npc, BrowsingUser)
    iface = _RuntimeSurface()
    # First tick acts; next 3 silent; 5th acts again.
    for _ in range(5):
        npc.step(iface)
    assert iface.calls == ["/x", "/y"]


def test_browsing_user_factory_rejects_bad_config() -> None:
    with pytest.raises(ValueError, match="paths must be a list"):
        browsing_user_factory({"paths": "not-a-list"})
    with pytest.raises(ValueError, match="cadence_ticks must be an int"):
        browsing_user_factory({"cadence_ticks": "fast"})


def test_admin_audit_polls_audit_path_at_cadence() -> None:
    npc = AdminAudit(cadence_ticks=2, audit_path="/openapi.json")
    iface = _RuntimeSurface()
    npc.step(iface)
    npc.step(iface)
    npc.step(iface)
    npc.step(iface)
    # cadence=2: ticks 0, 2 → 2 calls.
    assert iface.calls == ["/openapi.json", "/openapi.json"]


def test_admin_audit_factory_defaults_to_openapi_json() -> None:
    npc = admin_audit_factory({})
    assert isinstance(npc, AdminAudit)
    iface = _RuntimeSurface()
    npc.step(iface)
    assert iface.calls == ["/openapi.json"]


def test_admin_audit_factory_honors_audit_path() -> None:
    npc = admin_audit_factory({"cadence_ticks": 1, "audit_path": "/internal/health"})
    iface = _RuntimeSurface()
    npc.step(iface)
    assert iface.calls == ["/internal/health"]


def test_admin_audit_swallows_errors() -> None:
    npc = AdminAudit(cadence_ticks=1)

    def boom(path: object) -> bytes:
        raise RuntimeError("denied")

    iface: dict[str, Any] = {"http_get": boom}
    npc.step(iface)  # must not raise


def test_admin_audit_rejects_empty_audit_path() -> None:
    with pytest.raises(ValueError, match="audit_path"):
        AdminAudit(audit_path="")


def test_admin_audit_factory_rejects_bad_path() -> None:
    with pytest.raises(ValueError, match="audit_path must be a string"):
        admin_audit_factory({"audit_path": 42})


def test_curious_employee_factory_constructs_with_defaults() -> None:
    from cyber_webapp.npcs.curious_employee import CuriousEmployee
    from cyber_webapp.npcs.curious_employee import factory as ce_factory

    npc = ce_factory({})
    assert isinstance(npc, CuriousEmployee)
    assert npc.requires_llm is True
    assert npc._cadence_ticks == 5
    # No per-NPC backend by default; runtime supplies one via context.
    assert npc._backend_override is None
    assert "internal employee" in npc._system_prompt


def test_curious_employee_factory_defaults_no_backend_override() -> None:
    """The pack stays free of openrange runtime imports; per-NPC backend
    overrides are configured at the harness level via RunConfig."""
    from cyber_webapp.npcs.curious_employee import CuriousEmployee
    from cyber_webapp.npcs.curious_employee import factory as ce_factory

    npc = ce_factory(
        {
            "cadence_ticks": 2,
            "system_prompt": "You are a tester.",
        },
    )
    assert isinstance(npc, CuriousEmployee)
    assert npc._cadence_ticks == 2
    assert npc._backend_override is None
    assert npc._system_prompt == "You are a tester."


def test_curious_employee_factory_rejects_bad_config() -> None:
    from cyber_webapp.npcs.curious_employee import factory as ce_factory

    with pytest.raises(ValueError, match="cadence_ticks"):
        ce_factory({"cadence_ticks": "fast"})
    with pytest.raises(ValueError, match="system_prompt"):
        ce_factory({"system_prompt": ""})


def test_curious_employee_registered_via_entry_point() -> None:
    """The pack's pyproject.toml registers cyber.curious_employee."""
    from openrange.npc import NPCS

    assert "cyber.curious_employee" in NPCS.ids()


def test_office_chatter_factory_constructs_with_required_name() -> None:
    from cyber_webapp.npcs.office_chatter import OfficeChatter
    from cyber_webapp.npcs.office_chatter import factory as oc_factory

    npc = oc_factory({"name": "Alice"})
    assert isinstance(npc, OfficeChatter)
    assert npc.actor_id == "Alice"
    assert npc.requires_llm is False


def test_office_chatter_factory_rejects_bad_config() -> None:
    from cyber_webapp.npcs.office_chatter import factory as oc_factory

    with pytest.raises(ValueError, match="name"):
        oc_factory({})
    with pytest.raises(ValueError, match="cadence_ticks"):
        oc_factory({"name": "x", "cadence_ticks": "fast"})
    with pytest.raises(ValueError, match="walk_probability"):
        oc_factory({"name": "x", "walk_probability": "high"})
    with pytest.raises(ValueError, match="colleagues"):
        oc_factory({"name": "x", "colleagues": "Bob"})


def test_office_chatter_factory_reads_colleagues() -> None:
    from cyber_webapp.npcs.office_chatter import OfficeChatter
    from cyber_webapp.npcs.office_chatter import factory as oc_factory

    npc = oc_factory({"name": "Alice", "colleagues": ["Bob", "Carol"]})
    assert isinstance(npc, OfficeChatter)
    assert npc._colleagues == ("Bob", "Carol")


def test_office_chatter_emits_walks_via_record_action() -> None:
    """Each cadence tick emits a move event carrying an opener and reply."""
    from cyber_webapp.npcs.office_chatter import OfficeChatter

    npc = OfficeChatter(
        name="Alice",
        colleagues=("Bob", "Carol"),
        cadence_ticks=1,
        walk_probability=1.0,
        seed=1,
    )
    recorded: list[dict[str, Any]] = []

    def record(
        action: dict[str, Any],
        *,
        target: str | None = None,
        observation: object = None,
    ) -> None:
        recorded.append({"action": dict(action), "target": target})

    npc.start({"record_action": record})
    presence = recorded[0]["action"]
    assert presence["present"] is True
    assert isinstance(presence["home_index"], int)
    npc.step({})
    npc.step({})
    walks = [e for e in recorded if e["action"].get("move")]
    assert len(walks) == 2
    for entry in walks:
        assert entry["action"]["move"] == "wandering"
        assert isinstance(entry["action"]["home_index"], int)
        assert entry["action"]["target_name"] in {"Bob", "Carol"}
        assert isinstance(entry["action"]["opener"], str)
        assert isinstance(entry["action"]["reply"], str)
        assert entry["action"]["opener"]
        assert entry["action"]["reply"]


def test_office_chatter_omits_target_name_when_no_colleagues() -> None:
    """Without colleagues, walks still emit but carry no target_name."""
    from cyber_webapp.npcs.office_chatter import OfficeChatter

    npc = OfficeChatter(name="Solo", cadence_ticks=1, walk_probability=1.0, seed=1)
    recorded: list[dict[str, Any]] = []

    def record(
        action: dict[str, Any],
        *,
        target: str | None = None,
        observation: object = None,
    ) -> None:
        recorded.append(dict(action))

    npc.start({"record_action": record})
    npc.step({})
    walks = [a for a in recorded if a.get("move")]
    assert walks
    assert "target_name" not in walks[0]


def test_office_chatter_home_index_is_stable_per_name() -> None:
    """Same name → same home_index across processes (SHA1, not built-in hash)."""
    from cyber_webapp.npcs.office_chatter import OfficeChatter

    a = OfficeChatter(name="Alice")
    b = OfficeChatter(name="Alice")
    c = OfficeChatter(name="Bob")
    assert a._home_index == b._home_index
    assert a._home_index != c._home_index


def test_office_chatter_passes_home_as_target() -> None:
    from cyber_webapp.npcs.office_chatter import OfficeChatter

    npc = OfficeChatter(
        name="Bob",
        cadence_ticks=1,
        home="svc-web",
        walk_probability=1.0,
        seed=1,
    )
    recorded: list[dict[str, Any]] = []

    def record(
        action: dict[str, Any],
        *,
        target: str | None = None,
        observation: object = None,
    ) -> None:
        recorded.append({"action": dict(action), "target": target})

    npc.start({"record_action": record})
    npc.step({})
    walks = [e for e in recorded if e["action"].get("move")]
    assert walks[0]["target"] == "svc-web"


def test_office_chatter_obeys_cadence() -> None:
    from cyber_webapp.npcs.office_chatter import OfficeChatter

    npc = OfficeChatter(
        name="Carol",
        cadence_ticks=3,
        walk_probability=1.0,
        seed=1,
    )
    calls: list[dict[str, Any]] = []

    def record(
        action: dict[str, Any],
        *,
        target: str | None = None,
        observation: object = None,
    ) -> None:
        calls.append(dict(action))

    npc.start({"record_action": record})
    # Drop the start-time presence event; cadence-driven walks
    # follow. Initial cooldown is staggered so we just count.
    calls.clear()
    for _ in range(30):
        npc.step({})
    assert len(calls) == 10
    assert all(c.get("move") == "wandering" for c in calls)


def test_office_chatter_walk_probability_zero_emits_nothing() -> None:
    """walk_probability=0 means cadence ticks pass silently."""
    from cyber_webapp.npcs.office_chatter import OfficeChatter

    npc = OfficeChatter(name="Eve", cadence_ticks=1, walk_probability=0.0, seed=1)
    calls: list[dict[str, Any]] = []

    def record(
        action: dict[str, Any],
        *,
        target: str | None = None,
        observation: object = None,
    ) -> None:
        calls.append(dict(action))

    npc.start({"record_action": record})
    pre = len(calls)  # presence event
    for _ in range(10):
        npc.step({})
    assert len(calls) == pre  # no further events


def test_office_chatter_silent_without_record_action() -> None:
    """No record_action in context → NPC is a no-op, doesn't raise."""
    from cyber_webapp.npcs.office_chatter import OfficeChatter

    npc = OfficeChatter(name="Dave", cadence_ticks=1, seed=1)
    npc.start({})  # no record_action key
    npc.step({})  # must not raise
    npc.step({})


def test_office_chatter_registered_via_entry_point() -> None:
    from openrange.npc import NPCS

    assert "cyber.office_chatter" in NPCS.ids()
