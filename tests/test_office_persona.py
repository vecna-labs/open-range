"""Behavioral tests for the LLM-backed ``cyber.office_persona`` NPC.

The persona is a single-shot LLM-backed NPC: each cadence tick the
configured backend returns a JSON ``{speak, visit}`` payload; the NPC
does the HTTP visit itself via the runtime interface.

Two backend shapes get exercised:

* ``_PermissiveBackend`` — accepts any ``tools`` argument.
* ``_CodexLikeBackend`` — rejects non-empty ``tools``; confirms the
  persona works against :class:`CodexAgentBackend`.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import pytest
from cyber_webapp.npcs.office_persona import OfficePersona, _stable_home_index
from cyber_webapp.npcs.office_persona import factory as op_factory
from openrange_pack_sdk import AgentBackendError


class _StubSession:
    """Callable that returns a canned reply and records prompts."""

    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[str] = []

    def __call__(self, prompt: str) -> object:
        self.calls.append(prompt)
        return self.response


class _PermissiveBackend:
    """AgentBackend test double that accepts any ``tools`` argument."""

    def __init__(self, *, response: str) -> None:
        self.response = response
        self.preflight_calls = 0
        self.builds: list[tuple[str, list[Callable[..., Any]]]] = []
        self._session = _StubSession(response)

    def preflight(self) -> None:
        self.preflight_calls += 1

    def build_agent(
        self,
        *,
        system_prompt: str,
        tools: Sequence[Callable[..., Any]] = (),
    ) -> Any:
        self.builds.append((system_prompt, list(tools)))
        return self._session

    @property
    def session(self) -> _StubSession:
        return self._session


class _CodexLikeBackend:
    """Rejects non-empty ``tools`` — mimics real ``CodexAgentBackend``.

    The OfficePersona must build_agent with empty tools; if it doesn't,
    this fixture surfaces the same AgentBackendError that broke
    codex_eval in production.
    """

    def __init__(self, *, response: str) -> None:
        self.response = response
        self.preflight_calls = 0
        self.builds: list[tuple[str, list[Callable[..., Any]]]] = []
        self._session = _StubSession(response)

    def preflight(self) -> None:
        self.preflight_calls += 1

    def build_agent(
        self,
        *,
        system_prompt: str,
        tools: Sequence[Callable[..., Any]] = (),
    ) -> Any:
        if tools:
            raise AgentBackendError(
                "CodexAgentBackend does not support tool injection.",
            )
        self.builds.append((system_prompt, []))
        return self._session

    @property
    def session(self) -> _StubSession:
        return self._session


def _record(actions: list[dict[str, Any]]) -> Callable[..., None]:
    def record(
        action: Mapping[str, object],
        *,
        target: str | None = None,
        observation: object = None,
    ) -> None:
        actions.append({"action": dict(action), "target": target})

    return record


def _interface(http_calls: list[str]) -> dict[str, Any]:

    def http_get(path: object) -> bytes:
        http_calls.append(str(path))
        return b'{"ok": true, "page": "demo"}'

    def http_get_json(path: object) -> object:
        return json.loads(http_get(path).decode())

    return {
        "base_url": "http://test.local",
        "http_get": http_get,
        "http_get_json": http_get_json,
        "solver_root": "/tmp/fake-agent-root",
    }


def test_factory_constructs_with_required_name() -> None:
    npc = op_factory({"name": "Alice", "role": "engineer"})
    assert isinstance(npc, OfficePersona)
    assert npc.actor_id == "Alice"
    assert npc._role == "engineer"
    assert npc.requires_llm is True


def test_factory_appends_replication_suffix_to_actor_id() -> None:
    """`_replication_suffix` (set by `resolve_manifest_npcs` when count > 1)
    is appended to ``name`` so the NPC's actor_id matches the dashboard
    row id (``f"{name}-{index+1}"``)."""
    npc = op_factory(
        {
            "name": "Alice",
            "role": "engineer",
            "_replication_suffix": "-2",
        },
    )
    assert isinstance(npc, OfficePersona)
    assert npc.actor_id == "Alice-2"


def test_resolve_manifest_aligns_actor_ids_with_dashboard_row_ids() -> None:
    """For count>1 the resolved NPCs and `personas_from_manifest` rows
    agree on ids — the bug was the NPC kept the bare name while rows
    were suffixed."""
    from openrange.dashboard.topology import personas_from_manifest
    from openrange.npc import resolve_manifest_npcs

    entries = (
        {
            "type": "cyber.office_persona",
            "count": 2,
            "config": {"name": "Alice", "role": "engineer"},
        },
    )
    npcs = resolve_manifest_npcs(entries)
    rows = personas_from_manifest([dict(e) for e in entries])
    assert [n.actor_id for n in npcs] == [r["id"] for r in rows]
    assert [n.actor_id for n in npcs] == ["Alice-1", "Alice-2"]


def test_resolve_manifest_leaves_actor_id_bare_when_count_is_one() -> None:
    """Single-spawn entries get no suffix."""
    from openrange.dashboard.topology import personas_from_manifest
    from openrange.npc import resolve_manifest_npcs

    entries = (
        {
            "type": "cyber.office_persona",
            "config": {"name": "Solo", "role": "ops"},
        },
    )
    npcs = resolve_manifest_npcs(entries)
    rows = personas_from_manifest([dict(e) for e in entries])
    assert [n.actor_id for n in npcs] == ["Solo"]
    assert [r["id"] for r in rows] == ["Solo"]


def test_factory_defaults_no_backend_override() -> None:
    """The pack stays free of openrange runtime imports; per-NPC backend
    overrides are configured at the harness level via RunConfig."""
    npc = op_factory(
        {
            "name": "Carol",
            "role": "it_admin",
            "title": "Sec Eng",
            "tone": "calm",
            "colleagues": ["Dave"],
        },
    )
    assert isinstance(npc, OfficePersona)
    assert npc._backend_override is None
    assert npc._title == "Sec Eng"
    assert npc._colleagues == ("Dave",)


def test_factory_rejects_bad_config() -> None:
    with pytest.raises(ValueError, match="name"):
        op_factory({})
    with pytest.raises(ValueError, match="role"):
        op_factory({"name": "x", "role": ""})
    with pytest.raises(ValueError, match="cadence_ticks"):
        op_factory({"name": "x", "cadence_ticks": "fast"})
    with pytest.raises(ValueError, match="colleagues"):
        op_factory({"name": "x", "colleagues": "Bob"})
    with pytest.raises(ValueError, match="title"):
        op_factory({"name": "x", "title": 42})
    with pytest.raises(ValueError, match="seed"):
        op_factory({"name": "x", "seed": "high"})


def test_persona_emits_presence_event_with_role_and_home_index() -> None:
    backend = _PermissiveBackend(response='{"speak": "morning", "visit": "/"}')
    npc = OfficePersona(
        name="Alice",
        role="engineer",
        title="Backend Engineer",
        tone="dry, precise",
        agent_backend=backend,
        cadence_ticks=1,
        seed=1,
    )
    actions: list[dict[str, Any]] = []
    npc.start({"record_action": _record(actions)})
    assert npc.broken_reason is None
    assert actions, "presence event should fire from start()"
    presence = actions[0]["action"]
    assert presence["present"] is True
    assert presence["actor_kind"] == "npc"
    assert presence["display_name"] == "Alice"
    assert presence["role"] == "engineer"
    assert presence["title"] == "Backend Engineer"
    assert presence["tone"] == "dry, precise"
    assert isinstance(presence["home_index"], int)
    assert presence["home_index"] == _stable_home_index("Alice")


@pytest.mark.parametrize(
    "backend_factory",
    [
        lambda r: _PermissiveBackend(response=r),
        lambda r: _CodexLikeBackend(response=r),
    ],
    ids=["permissive_backend", "codex_like_backend"],
)
def test_persona_records_speech_and_visit_via_single_shot(
    backend_factory: Callable[[str], Any],
) -> None:
    """Each tick parses the JSON reply, calls http_get, records speech.

    The Codex-shape backend rejects non-empty tools — if the persona ever
    regresses to using ``visit_url`` tool dispatch this parametrize will
    surface the AgentBackendError immediately.
    """
    backend = backend_factory(
        '{"speak": "Reviewing the perf report.", "visit": "/api/users"}'
    )
    npc = OfficePersona(
        name="Bob",
        role="engineer",
        colleagues=("Alice",),
        agent_backend=backend,
        cadence_ticks=1,
        seed=7,
    )
    actions: list[dict[str, Any]] = []
    http_calls: list[str] = []
    npc.start({"record_action": _record(actions)})
    actions.clear()
    npc.step(_interface(http_calls))

    # build_agent must be called with EMPTY tools — that's what makes
    # the persona compatible with CodexAgentBackend.
    assert backend.builds, "build_agent should fire on first step"
    _system_prompt, tools = backend.builds[0]
    assert tools == [], "single-shot persona must not inject tools"

    # The NPC drove the HTTP visit itself, no tool callback needed.
    assert http_calls == ["/api/users"]
    speak_actions = [a for a in actions if a["action"].get("speak")]
    assert speak_actions, "agent reply should record a speak action"
    speak = speak_actions[0]["action"]
    assert speak["display_name"] == "Bob"
    assert speak["role"] == "engineer"
    assert speak["actor_kind"] == "npc"
    assert speak["speak"] == "Reviewing the perf report."
    assert speak["visit"] == "/api/users"


def test_persona_ignores_codex_like_backend_when_episode_runs() -> None:
    """End-to-end smoke: codex-shape backend → no broken_reason set.

    Mirrors what happens when ``RunConfig.npc_agent_backend`` is a
    ``CodexAgentBackend`` and the runtime invokes ``start()`` then
    ``step()`` repeatedly. Was failing before the AgentNPC → NPC
    refactor with: "CodexAgentBackend does not support tool injection".
    """
    backend = _CodexLikeBackend(response='{"speak": "ok", "visit": "/"}')
    npc = OfficePersona(
        name="Erin",
        role="engineer",
        agent_backend=backend,
        cadence_ticks=1,
    )
    actions: list[dict[str, Any]] = []
    http_calls: list[str] = []
    npc.start({"record_action": _record(actions)})
    assert npc.broken_reason is None
    npc.step(_interface(http_calls))
    npc.step(_interface(http_calls))
    npc.step(_interface(http_calls))
    assert npc.broken_reason is None
    assert len(http_calls) == 3, "every cadence tick should drive HTTP traffic"


def test_persona_falls_back_to_first_sentence_when_json_parse_fails() -> None:
    backend = _PermissiveBackend(response="Quick lunch break, back in 20.")
    npc = OfficePersona(
        name="Eve",
        role="ops",
        agent_backend=backend,
        cadence_ticks=1,
    )
    actions: list[dict[str, Any]] = []
    http_calls: list[str] = []
    npc.start({"record_action": _record(actions)})
    actions.clear()
    npc.step(_interface(http_calls))

    speak = next(a["action"] for a in actions if a["action"].get("speak"))
    assert speak["speak"] == "Quick lunch break, back in 20."


def test_persona_obeys_cadence() -> None:
    backend = _PermissiveBackend(response='{"speak": "ok", "visit": "/"}')
    npc = OfficePersona(
        name="Frank",
        role="finance",
        agent_backend=backend,
        cadence_ticks=3,
    )
    actions: list[dict[str, Any]] = []
    http_calls: list[str] = []
    npc.start({"record_action": _record(actions)})
    actions.clear()
    # Tick 0: act. Ticks 1+2: cooldown. Tick 3: act again.
    npc.step(_interface(http_calls))
    npc.step(_interface(http_calls))
    npc.step(_interface(http_calls))
    npc.step(_interface(http_calls))
    speak_count = sum(1 for a in actions if a["action"].get("speak"))
    assert speak_count == 2
    assert len(http_calls) == 2


def test_persona_marks_broken_without_backend() -> None:
    npc = OfficePersona(
        name="Solo",
        role="engineer",
        cadence_ticks=1,
    )
    actions: list[dict[str, Any]] = []
    npc.start({"record_action": _record(actions), "agent_backend": None})
    assert npc.broken_reason is not None
    # Presence still fires before the broken check, so the dashboard
    # sees the persona at least once even without a backend.
    assert actions and actions[0]["action"]["present"] is True


def test_persona_records_visit_target_from_home_config() -> None:
    backend = _PermissiveBackend(
        response='{"speak": "auth done", "visit": "/login"}',
    )
    npc = OfficePersona(
        name="Grace",
        role="it_admin",
        home="svc-web",
        agent_backend=backend,
        cadence_ticks=1,
    )
    actions: list[dict[str, Any]] = []
    http_calls: list[str] = []
    npc.start({"record_action": _record(actions)})
    actions.clear()
    npc.step(_interface(http_calls))
    speak_entry = next(a for a in actions if a["action"].get("speak"))
    assert speak_entry["target"] == "svc-web"


def test_persona_includes_walk_when_colleagues_present() -> None:
    """High-probability walk picks should land on a configured colleague.

    Run multiple ticks because a single tick can roll below the 0.45
    walk threshold; we just need at least one `move` event in the run.
    """
    backend = _PermissiveBackend(response='{"speak": "ok", "visit": "/"}')
    npc = OfficePersona(
        name="Iris",
        role="hr",
        colleagues=("Jules", "Karim"),
        agent_backend=backend,
        cadence_ticks=1,
        seed=0,
    )
    actions: list[dict[str, Any]] = []
    http_calls: list[str] = []
    npc.start({"record_action": _record(actions)})
    actions.clear()
    for _ in range(20):
        npc.step(_interface(http_calls))
    moves = [a["action"] for a in actions if a["action"].get("move") == "wandering"]
    assert moves, "20 ticks at p=0.45 should produce at least one walk"
    for entry in moves:
        assert entry["target_name"] in {"Jules", "Karim"}


def test_persona_swallows_runtime_errors_without_breaking() -> None:
    """A throwing session does not sink the episode."""

    class _Throws:
        def __call__(self, prompt: str) -> object:
            raise RuntimeError("model went poof")

    class _ThrowingBackend(_PermissiveBackend):
        def build_agent(self, **kw: Any) -> Any:
            return _Throws()

    backend = _ThrowingBackend(response="")
    npc = OfficePersona(
        name="Mia",
        role="legal",
        agent_backend=backend,
        cadence_ticks=1,
    )
    npc.start({"record_action": _record([])})
    # Repeated invocation must not raise or mark broken.
    npc.step(_interface([]))
    npc.step(_interface([]))
    assert npc.broken_reason is None


def test_persona_marks_broken_on_build_failure() -> None:
    """Backend rejecting build_agent (other than empty-tools) → broken."""

    class _BuildFailBackend(_PermissiveBackend):
        def build_agent(self, **_kw: Any) -> Any:
            raise AgentBackendError("preflight ok, build broken")

    backend = _BuildFailBackend(response="")
    npc = OfficePersona(
        name="Nina",
        role="engineer",
        agent_backend=backend,
        cadence_ticks=1,
    )
    npc.start({"record_action": _record([])})
    npc.step(_interface([]))
    reason = npc.broken_reason
    assert reason is not None
    assert "failed to construct agent" in reason
    # Subsequent steps short-circuit; no further build attempts.
    npc.step(_interface([]))
    npc.step(_interface([]))


def test_persona_registered_via_entry_point() -> None:
    from openrange.npc import NPCS

    assert "cyber.office_persona" in NPCS.ids()


def test_stable_home_index_is_per_name() -> None:
    a = _stable_home_index("Alice")
    b = _stable_home_index("Alice")
    c = _stable_home_index("Bob")
    assert a == b
    assert a != c


def test_constructor_backend_preflight_failure_marks_broken() -> None:
    class _BadBackend(_PermissiveBackend):
        def preflight(self) -> None:
            raise AgentBackendError("missing dep")

    backend = _BadBackend(response="")
    npc = OfficePersona(name="Pat", role="ops", agent_backend=backend)
    assert npc.broken_reason is not None
    assert "preflight failed" in npc.broken_reason


def test_runtime_backend_captured_from_context() -> None:
    """If no constructor backend, persona uses ``context['agent_backend']``."""
    backend = _CodexLikeBackend(response='{"speak": "hey", "visit": "/api"}')
    npc = OfficePersona(name="Quinn", role="sales", cadence_ticks=1)
    actions: list[dict[str, Any]] = []
    http_calls: list[str] = []
    npc.start({"record_action": _record(actions), "agent_backend": backend})
    assert npc.broken_reason is None
    actions.clear()
    npc.step(_interface(http_calls))
    assert http_calls == ["/api"]
    speak = next(a["action"] for a in actions if a["action"].get("speak"))
    assert speak["speak"] == "hey"
