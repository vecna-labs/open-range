"""Tests for the generic PersonaAgent, comms channels, and scoped memory.

Covered: persona rendering; data-driven tool binding over an arbitrary
pack-surfaced interface; the standard comms vocabulary bound to the persona's
own identity through a SHARED surface (the real runtime seam); through-the-world
message passing with a read cursor; per-persona/per-run memory isolation;
signature-faithful multi-arg tools; config permutations; generalization across
pack shapes; and a scale smoke test.

No Strands agent loop, no model, no network — a recording backend stands in for
the LLM so behavior is asserted deterministically. Strands itself IS installed,
so tool decoration exercises the real ``@tool``.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Sequence
from typing import Any

import pytest
from openrange_pack_sdk import (
    AgentBackendError,
    ChatStore,
    DictMemory,
    MailboxStore,
    PersonaAgent,
    render_persona,
    surface_chat,
    surface_mailbox,
)
from openrange_pack_sdk.npcs.persona_agent import (
    _as_tool,
    _comms_adapter,
    _wrap_action,
    factory,
)

# --------------------------------------------------------------------------- #
# Test doubles (mirror the recording backend used across the OpenRange suite). #
# --------------------------------------------------------------------------- #


class _RecordingAgent:
    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.tools: Sequence[Callable[..., Any]] = ()

    def __call__(self, prompt: str) -> object:
        self.prompts.append(prompt)
        return {"message": "ok"}


class _RecordingBackend:
    def __init__(self) -> None:
        self.preflight_calls = 0
        self.agent = _RecordingAgent()
        self.built_prompt: str | None = None

    def preflight(self) -> None:
        self.preflight_calls += 1

    def build_agent(
        self,
        *,
        system_prompt: str,
        tools: Sequence[Callable[..., Any]] = (),
    ) -> Any:
        self.built_prompt = system_prompt
        self.agent.tools = tuple(tools)
        return self.agent


def _run(npc: PersonaAgent, *, run_id: str = "run-1") -> _RecordingBackend:
    """Start an NPC with a fresh recording backend, ready to act on the next
    step. (start() phase-staggers the cooldown; these tests exercise the acting
    path, so we clear it — the stagger has its own dedicated test.)"""
    backend = _RecordingBackend()
    npc._backend_override = backend
    npc.start({"episode_id": run_id, "agent_backend": backend})
    npc._cooldown = 0
    return backend


def _tool_named(agent: _RecordingAgent, name: str) -> Callable[..., Any]:
    for t in agent.tools:
        if getattr(t, "__name__", "") == name:
            return t
    have = [getattr(t, "__name__", "?") for t in agent.tools]
    raise AssertionError(f"tool {name!r} not built; have {have}")


def _world_surface(mbox: MailboxStore, chat: ChatStore) -> dict[str, Any]:
    """One shared, identity-neutral surface — exactly what a pack's
    surface_extras() produces and the runtime hands to every NPC."""
    return {**surface_mailbox(mbox), **surface_chat(chat)}


# --------------------------------------------------------------------------- #
# render_persona                                                              #
# --------------------------------------------------------------------------- #


def test_render_persona_has_identity_goal_and_constraints() -> None:
    prompt = render_persona(
        {"name": "Dana", "role": "accountant", "goal": "reconcile invoices"}
    )
    assert "Dana" in prompt and "accountant" in prompt
    assert "reconcile invoices" in prompt
    assert "CONSTRAINTS" in prompt
    assert "not an AI assistant" in prompt
    assert "YOUR OWN goal" in prompt


def test_render_persona_minimal_config_is_valid() -> None:
    prompt = render_persona({})
    assert prompt
    assert "CONSTRAINTS" in prompt


def test_render_persona_folds_in_style_axes() -> None:
    prompt = render_persona(
        {
            "name": "Sam",
            "role": "sysadmin",
            "tone": "brusque",
            "behavior_axes": {"terse": True, "skeptical": True},
        }
    )
    assert "brusque" in prompt
    assert "short and clipped" in prompt
    assert "question requests" in prompt


# --------------------------------------------------------------------------- #
# Tool binding is data-driven over the pack-provided interface                #
# --------------------------------------------------------------------------- #


def test_binds_only_declared_and_provided_tools() -> None:
    npc = PersonaAgent(config={"name": "U", "tools": ["http_get", "shell"]})
    backend = _run(npc)
    npc.step({"http_get": lambda p: f"body:{p}"})
    names = {getattr(t, "__name__", "") for t in backend.agent.tools}
    assert names == {"http_get"}


def test_unknown_surface_key_is_generic_action_tool() -> None:
    npc = PersonaAgent(config={"name": "U", "tools": ["frobnicate"]})
    backend = _run(npc)
    npc.step({"frobnicate": lambda arg: f"did:{arg}"})
    assert _tool_named(backend.agent, "frobnicate")("x") == "did:x"


def test_generic_action_tool_truncates_and_survives_errors() -> None:
    npc = PersonaAgent(config={"name": "U", "tools": ["big", "boom"]})
    backend = _run(npc)

    def boom(_: str) -> str:
        raise RuntimeError("kaboom")

    npc.step({"big": lambda _: "x" * 5000, "boom": boom})
    assert len(_tool_named(backend.agent, "big")("q")) == 1500
    assert "boom failed: kaboom" in _tool_named(backend.agent, "boom")("q")


def test_generic_action_tool_handles_noarg_callable() -> None:
    npc = PersonaAgent(config={"name": "U", "tools": ["ping"]})
    backend = _run(npc)
    npc.step({"ping": lambda: "pong"})
    assert _tool_named(backend.agent, "ping")() == "pong"


def test_action_tool_returning_none() -> None:
    npc = PersonaAgent(config={"name": "U", "tools": ["fire"]})
    backend = _run(npc)
    npc.step({"fire": lambda x: None})
    assert _tool_named(backend.agent, "fire")("go") == "(no output)"


# --------------------------------------------------------------------------- #
# Signature-faithful, single-call tool wrapping (fixes double-fire + multi-arg) #
# --------------------------------------------------------------------------- #


def test_multi_arg_affordance_signature_is_preserved() -> None:
    def sql(table: str, where: str = "") -> str:
        return f"{table}:{where}"

    wrapped = _wrap_action("sql", sql)
    params = list(inspect.signature(wrapped).parameters)
    assert params == ["table", "where"]
    assert wrapped("users", "id=1") == "users:id=1"


def test_action_tool_invokes_exactly_once() -> None:
    calls: list[tuple[Any, ...]] = []

    def eff(arg: str) -> str:
        calls.append((arg,))
        raise TypeError("internal type error, NOT an arity mismatch")

    wrapped = _wrap_action("eff", eff)
    out = wrapped("x")
    assert "eff failed" in out
    assert len(calls) == 1  # never retried -> a side-effecting tool can't double-fire


def test_multi_arg_tool_gets_real_strands_schema() -> None:
    def sql(table: str, where: str = "") -> str:
        return "ok"

    tool: Any = _as_tool(_wrap_action("sql", sql), "sql")
    props = tool.tool_spec["inputSchema"]["json"]["properties"]
    assert set(props) == {"table", "where"}


# --------------------------------------------------------------------------- #
# Standard comms vocabulary -> identity-injected typed tools                   #
# --------------------------------------------------------------------------- #


def test_comms_adapter_injects_sender_model_signature_omits_it() -> None:
    mbox = MailboxStore()
    surf = surface_mailbox(mbox)
    adapter = _comms_adapter("mail_send", surf["mail_send"], "alice")
    assert adapter is not None
    assert list(inspect.signature(adapter).parameters) == ["to", "subject", "body"]
    adapter(to="bob", subject="s", body="b")
    assert mbox.all()[0].sender == "alice"  # injected, not model-chosen


def test_comms_write_tools_attribute_to_the_persona() -> None:
    npc = PersonaAgent(
        config={"name": "A", "tools": ["mail_send", "chat_post", "speak"]}
    )
    backend = _run(npc)
    mbox, chat = MailboxStore(), ChatStore()
    said: list[str] = []

    def _speak(text: str) -> str:
        said.append(text)
        return "spoke"

    npc.step({**_world_surface(mbox, chat), "speak": _speak})
    _tool_named(backend.agent, "mail_send")(to="B", subject="hi", body="yo")
    _tool_named(backend.agent, "chat_post")(channel="ops", text="hello")
    _tool_named(backend.agent, "speak")(text="ahem")
    assert mbox.all()[0].sender == "A" and mbox.read("B")[0].body == "yo"
    assert chat.all()[0].sender == "A" and chat.read("ops")[0].body == "hello"
    assert said == ["ahem"]


def test_read_keys_are_not_tools_but_feed_the_prompt() -> None:
    npc = PersonaAgent(config={"name": "B", "tools": ["mail_read"]})
    backend = _run(npc)
    mbox = MailboxStore()
    mbox.send(sender="A", to="B", subject="urgent", body="call me")
    npc.step(surface_mailbox(mbox))
    assert not any(
        getattr(t, "__name__", "") == "mail_read" for t in backend.agent.tools
    )
    assert "urgent" in backend.agent.prompts[-1]
    assert "call me" in backend.agent.prompts[-1]


# --------------------------------------------------------------------------- #
# The core seam-fidelity test: ONE shared surface, many personas, correct       #
# per-sender attribution (this is what a real _step_npcs hands out).           #
# --------------------------------------------------------------------------- #


def test_shared_surface_attributes_each_persona_correctly() -> None:
    """Regression guard for the surface-seam: the runtime hands the SAME frozen
    surface to every NPC, so identity must be injected NPC-side. Two personas
    driven off ONE shared surface must each send as themselves."""
    mbox, chat = MailboxStore(), ChatStore()
    shared = _world_surface(mbox, chat)  # ONE surface, shared by all
    a = PersonaAgent(config={"name": "alice", "tools": ["mail_send"]})
    b = PersonaAgent(config={"name": "bob", "tools": ["mail_send"]})
    ba, bb = _run(a), _run(b)
    a.step(shared)
    b.step(shared)
    _tool_named(ba.agent, "mail_send")(to="carol", body="from a")
    _tool_named(bb.agent, "mail_send")(to="carol", body="from b")
    by_sender = {(m.sender, m.body) for m in mbox.all()}
    assert by_sender == {("alice", "from a"), ("bob", "from b")}
    # a grader subtracts known decoys by sender attribution
    assert {m.sender for m in mbox.all()} == {"alice", "bob"}


def test_message_flows_A_to_B_via_shared_store_next_tick() -> None:
    mbox, chat = MailboxStore(), ChatStore()
    shared = _world_surface(mbox, chat)
    a = PersonaAgent(config={"name": "A", "tools": ["mail_send"], "cadence_ticks": 1})
    b = PersonaAgent(config={"name": "B", "tools": ["mail_read"], "cadence_ticks": 1})
    ba, bb = _run(a), _run(b)
    a.step(shared)
    _tool_named(ba.agent, "mail_send")(to="B", subject="ping", body="from A")
    b.step(shared)
    assert "from A" in bb.agent.prompts[-1]


def test_read_cursor_prevents_reinjecting_old_mail() -> None:
    mbox, chat = MailboxStore(), ChatStore()
    shared = _world_surface(mbox, chat)
    b = PersonaAgent(config={"name": "B", "tools": ["mail_read"], "cadence_ticks": 1})
    bb = _run(b)
    mbox.send(sender="A", to="B", body="first")
    b.step(shared)
    assert "first" in bb.agent.prompts[-1]
    # next tick with no new mail: the stale message is NOT re-injected
    b.step(shared)
    assert "first" not in bb.agent.prompts[-1]
    assert "Since your last turn" not in bb.agent.prompts[-1]
    # a genuinely new message does surface
    mbox.send(sender="A", to="B", body="second")
    b.step(shared)
    assert "second" in bb.agent.prompts[-1]


def test_reads_gated_on_declared_tools() -> None:
    # an NPC that did not declare mail_read must not receive mail context
    mbox, chat = MailboxStore(), ChatStore()
    mbox.send(sender="A", to="B", body="secret")
    npc = PersonaAgent(config={"name": "B", "tools": ["mail_send"], "cadence_ticks": 1})
    backend = _run(npc)
    npc.step(_world_surface(mbox, chat))
    assert "secret" not in backend.agent.prompts[-1]


def test_sender_identity_cannot_be_spoofed_by_the_model() -> None:
    # the model-facing mail_send has no `sender` parameter at all
    mbox = MailboxStore()
    npc = PersonaAgent(config={"name": "alice", "tools": ["mail_send"]})
    backend = _run(npc)
    npc.step(surface_mailbox(mbox))
    send = _tool_named(backend.agent, "mail_send")
    assert "sender" not in inspect.signature(send).parameters
    send(to="bob", body="y")
    assert mbox.all()[0].sender == "alice"


def test_chat_since_cursor_only_returns_new_lines() -> None:
    chat = ChatStore()
    first = chat.post(sender="A", channel="ops", text="one")
    chat.post(sender="B", channel="ops", text="two")
    assert [m.body for m in chat.read("ops", since=first)] == ["two"]


def test_mailbox_broadcast_and_since() -> None:
    mbox = MailboxStore()
    mbox.send(sender="sys", to="", subject="notice", body="all hands")
    assert mbox.read("anyone")[0].body == "all hands"
    mid = mbox.read("anyone")[0].id
    assert mbox.read("anyone", since=mid) == []


# --------------------------------------------------------------------------- #
# Scoped memory isolation                                                     #
# --------------------------------------------------------------------------- #


def test_memory_tools_only_appear_when_enabled() -> None:
    off = PersonaAgent(config={"name": "U", "tools": []})
    backend = _run(off)
    off.step({})
    assert not any(
        getattr(t, "__name__", "") in {"remember", "recall"}
        for t in backend.agent.tools
    )

    on = PersonaAgent(config={"name": "U", "tools": [], "long_term_memory": True})
    backend = _run(on)
    on.step({})
    names = {getattr(t, "__name__", "") for t in backend.agent.tools}
    assert {"remember", "recall"} <= names


def test_memory_is_isolated_per_persona_and_per_run() -> None:
    shared = DictMemory()
    a = PersonaAgent(
        config={"name": "A", "tools": [], "long_term_memory": True}, memory=shared
    )
    b = PersonaAgent(
        config={"name": "B", "tools": [], "long_term_memory": True}, memory=shared
    )
    ba, bb = _run(a, run_id="r1"), _run(b, run_id="r1")
    a.step({})
    b.step({})
    _tool_named(ba.agent, "remember")("A saw a secret token")
    assert "secret token" not in _tool_named(bb.agent, "recall")("secret")
    assert "secret token" in _tool_named(ba.agent, "recall")("secret")

    a2 = PersonaAgent(
        config={"name": "A", "tools": [], "long_term_memory": True}, memory=shared
    )
    ba2 = _run(a2, run_id="r2")
    a2.step({})
    assert _tool_named(ba2.agent, "recall")("secret") == "(nothing on that yet)"


def test_blank_name_still_isolates_memory() -> None:
    # two unnamed personas must not share a scope (actor_id falls back to unique)
    shared = DictMemory()
    a = PersonaAgent(config={"tools": [], "long_term_memory": True}, memory=shared)
    b = PersonaAgent(config={"tools": [], "long_term_memory": True}, memory=shared)
    assert a.actor_id != b.actor_id
    ba, bb = _run(a), _run(b)
    a.step({})
    b.step({})
    _tool_named(ba.agent, "remember")("a-note")
    assert "a-note" not in _tool_named(bb.agent, "recall")("note")


def test_dict_memory_ranks_overlap_then_recency() -> None:
    mem = DictMemory()
    mem.store("s", "the finance portal is at /finance")
    mem.store("s", "coffee machine is broken")
    mem.store("s", "finance portal needs a new password")
    hits = mem.retrieve("s", "finance portal", k=2)
    assert all("finance" in h or "portal" in h for h in hits)
    assert mem.retrieve("s", "zzz")[0] == "finance portal needs a new password"


# --------------------------------------------------------------------------- #
# actor_id / replication / factory                                            #
# --------------------------------------------------------------------------- #


def test_replication_suffix_yields_distinct_actor_ids() -> None:
    a = PersonaAgent(config={"name": "clerk", "_replication_suffix": "-1"})
    b = PersonaAgent(config={"name": "clerk", "_replication_suffix": "-2"})
    assert (a.actor_id, b.actor_id) == ("clerk-1", "clerk-2")


def test_scope_uses_actor_id_including_suffix() -> None:
    a = PersonaAgent(config={"name": "clerk", "_replication_suffix": "-1"})
    _run(a, run_id="ep")
    assert a._scope == "ep:clerk-1"


def test_factory_returns_persona_agent() -> None:
    npc = factory({"name": "Z", "tools": ["http_get"]})
    assert isinstance(npc, PersonaAgent)
    assert npc.actor_id == "Z"


# --------------------------------------------------------------------------- #
# Cadence + broken-state inherited from AgentNPC                              #
# --------------------------------------------------------------------------- #


def test_cadence_gates_action() -> None:
    npc = PersonaAgent(config={"name": "U", "tools": ["http_get"], "cadence_ticks": 3})
    backend = _run(npc)
    iface = {"http_get": lambda p: "ok"}
    npc.step(iface)
    assert len(backend.agent.prompts) == 1
    npc.step(iface)
    npc.step(iface)
    assert len(backend.agent.prompts) == 1
    npc.step(iface)
    assert len(backend.agent.prompts) == 2


def test_missing_backend_marks_broken_not_crash() -> None:
    npc = PersonaAgent(config={"name": "U"})
    npc.start({"episode_id": "r"})
    assert npc.broken_reason is not None
    npc.step({})
    assert npc._agent is None


def test_bad_cadence_rejected() -> None:
    with pytest.raises(ValueError):
        PersonaAgent(config={"name": "U", "cadence_ticks": 0})
    with pytest.raises(ValueError):
        PersonaAgent(config={"name": "U", "cadence_ticks": "five"})
    with pytest.raises(ValueError):
        PersonaAgent(config={"name": "U", "cadence_ticks": True})  # bool is not an int


def test_tool_reject_backend_marks_broken() -> None:
    class _RejectTools(_RecordingBackend):
        def build_agent(
            self, *, system_prompt: str, tools: Sequence[Callable[..., Any]] = ()
        ) -> Any:
            if tools:
                raise AgentBackendError("no tools here")
            return super().build_agent(system_prompt=system_prompt, tools=tools)

    npc = PersonaAgent(config={"name": "U", "tools": ["http_get"]})
    backend = _RejectTools()
    npc._backend_override = backend
    npc.start({"episode_id": "r", "agent_backend": backend})
    npc._cooldown = 0
    npc.step({"http_get": lambda p: "ok"})
    assert npc.broken_reason is not None
    assert "failed to construct agent" in npc.broken_reason


# --------------------------------------------------------------------------- #
# Configuration coverage — the real distinct outcomes, not an inflated product #
# --------------------------------------------------------------------------- #

_TOOLSETS = {
    "empty": [],
    "web": ["http_get"],
    "network": ["shell"],
    "mail": ["mail_send", "mail_read"],
    "chat": ["chat_post", "chat_read"],
    "mixed": ["mail_send", "chat_post", "http_get"],
    "enterprise": ["sql", "admin_lock", "mail_read"],
}


@pytest.mark.parametrize("toolset", sorted(_TOOLSETS))
@pytest.mark.parametrize("ltm", [False, True])
@pytest.mark.parametrize("cadence", [1, 5])
def test_config_permutations_build_and_act(
    toolset: str, ltm: bool, cadence: int
) -> None:
    tools = _TOOLSETS[toolset]
    npc = PersonaAgent(
        config={
            "name": f"{toolset}-npc",
            "role": "worker",
            "goal": "do the job",
            "tools": tools,
            "cadence_ticks": cadence,
            "long_term_memory": ltm,
        }
    )
    backend = _run(npc)
    mbox, chat = MailboxStore(), ChatStore()
    iface: dict[str, Any] = {
        "http_get": lambda p="": "ok",
        "shell": lambda c="": "0",
        "sql": lambda table="", where="": "[]",
        "admin_lock": lambda user="", reason="": "locked",
        **_world_surface(mbox, chat),
    }
    npc.step(iface)
    assert not npc.broken_reason
    assert len(backend.agent.prompts) == 1
    built = {getattr(t, "__name__", "") for t in backend.agent.tools}
    for key in tools:
        if key in {"mail_read", "chat_read"}:
            assert key not in built
        else:
            assert key in built
    if ltm:
        assert {"remember", "recall"} <= built


# --------------------------------------------------------------------------- #
# Generalization: the SAME class across pack shapes                           #
# --------------------------------------------------------------------------- #

_PACK_SHAPES: dict[str, tuple[list[str], dict[str, Any]]] = {
    "web": (["http_get"], {"http_get": lambda p="": "<html/>"}),
    "network": (["shell"], {"shell": lambda c="": "root@host"}),
    "terminal": (["run_tests"], {"run_tests": lambda a="": "3 passed"}),
    "enterprise": (
        ["sql", "admin_lock"],
        {"sql": lambda t="", w="": "[]", "admin_lock": lambda u="", r="": "locked"},
    ),
}


@pytest.mark.parametrize("shape", sorted(_PACK_SHAPES))
def test_same_class_generalizes_across_pack_shapes(shape: str) -> None:
    tools, extra = _PACK_SHAPES[shape]
    npc = PersonaAgent(config={"name": f"{shape}-npc", "role": shape, "tools": tools})
    backend = _run(npc)
    npc.step(extra)
    assert not npc.broken_reason
    built = {getattr(t, "__name__", "") for t in backend.agent.tools}
    assert all(t in built for t in tools)


def test_social_pack_full_stack() -> None:
    """A population of distinct personas emailing + chatting + remembering over
    ONE shared world surface, with correct per-persona attribution."""
    mbox, chat, mem = MailboxStore(), ChatStore(), DictMemory()
    shared = _world_surface(mbox, chat)
    roster: list[dict[str, Any]] = [
        {
            "name": "accountant",
            "tools": ["mail_send", "mail_read", "chat_post", "chat_read"],
            "channels": ["office"],
            "long_term_memory": True,
            "cadence_ticks": 1,
        },
        {
            "name": "itadmin",
            "tools": ["mail_send", "mail_read", "chat_read"],
            "channels": ["office"],
            "long_term_memory": True,
            "cadence_ticks": 1,
        },
        {
            "name": "exec",
            "tools": ["chat_post", "chat_read"],
            "channels": ["office"],
            "cadence_ticks": 1,
        },
    ]
    npcs = [PersonaAgent(config=c, memory=mem) for c in roster]
    backends = [_run(n, run_id="social") for n in npcs]

    for n in npcs:
        n.step(shared)
    _tool_named(backends[0].agent, "mail_send")(
        to="itadmin", subject="access", body="need vault access"
    )
    _tool_named(backends[2].agent, "chat_post")(channel="office", text="standup at 10")

    for n in npcs:
        n.step(shared)
    assert "need vault access" in backends[1].agent.prompts[-1]
    assert "standup at 10" in backends[0].agent.prompts[-1]
    assert {m.sender for m in mbox.all()} == {"accountant"}
    assert {m.sender for m in chat.all()} == {"exec"}


# --------------------------------------------------------------------------- #
# Integration through the REAL registry seam (entry point + replication)      #
# --------------------------------------------------------------------------- #


def test_resolve_manifest_npcs_through_real_registry() -> None:
    from openrange.npc import resolve_manifest_npcs

    npcs = resolve_manifest_npcs(
        (
            {
                "type": "cyber.persona",
                "count": 3,
                "config": {
                    "name": "clerk",
                    "role": "accountant",
                    "tools": ["http_get"],
                },
            },
        )
    )
    assert len(npcs) == 3
    assert all(isinstance(n, PersonaAgent) for n in npcs)
    assert {n.actor_id for n in npcs} == {"clerk-1", "clerk-2", "clerk-3"}


# --------------------------------------------------------------------------- #
# Scale smoke test                                                            #
# --------------------------------------------------------------------------- #


def test_ten_thousand_personas_construct_lazily() -> None:
    """10k personas construct instantly and hold NO agent/model state until they
    act — the property that lets a population idle cheaply. (Throughput under
    load is a separate, inference-bound axis, not tested here.)"""
    roster = [
        PersonaAgent(
            config={
                "name": f"user{i}",
                "role": ["accountant", "sysadmin", "exec"][i % 3],
                "tools": ["mail_send", "mail_read"],
                "cadence_ticks": (i % 9) + 1,
            }
        )
        for i in range(10_000)
    ]
    assert len({n.actor_id for n in roster}) == 10_000
    assert all(n._agent is None for n in roster)


def test_cadence_stagger_spreads_actions_no_thundering_herd() -> None:
    """A population on the same cadence must NOT act in lockstep. The
    deterministic phase-stagger spreads first actions across the whole cadence
    window, so no single tick sees the entire population act."""
    shared = _CountingBackend()
    npcs = []
    for i in range(900):
        n = PersonaAgent(config={"name": f"u{i}", "tools": [], "cadence_ticks": 9})
        n._backend_override = shared
        n.start({"episode_id": "scale", "agent_backend": shared})
        npcs.append(n)
    per_tick = []
    for _ in range(9):  # one full cadence window
        before = shared.invocations
        for n in npcs:
            n.step({})
        per_tick.append(shared.invocations - before)
    assert sum(per_tick) == 900  # everyone acts exactly once over the window
    assert max(per_tick) < 250  # but spread out — never the whole herd at once
    assert all(c > 0 for c in per_tick)  # every tick does some work


def test_cadence_stagger_is_deterministic() -> None:
    # same run_id + actor -> same phase offset (replayable)
    a = PersonaAgent(config={"name": "x", "tools": [], "cadence_ticks": 7})
    b = PersonaAgent(config={"name": "x", "tools": [], "cadence_ticks": 7})
    _run_no_reset(a, "ep")
    _run_no_reset(b, "ep")
    assert a._cooldown == b._cooldown


def _run_no_reset(npc: PersonaAgent, run_id: str) -> None:
    backend = _RecordingBackend()
    npc._backend_override = backend
    npc.start({"episode_id": run_id, "agent_backend": backend})


class _CountingBackend:
    def __init__(self) -> None:
        self.invocations = 0

    def preflight(self) -> None:
        pass

    def build_agent(
        self, *, system_prompt: str, tools: Sequence[Callable[..., Any]] = ()
    ) -> Any:
        backend = self

        def agent(prompt: str) -> object:
            backend.invocations += 1
            return {"message": "ok"}

        return agent


# --------------------------------------------------------------------------- #
# Second-pass refinements — regression guards                                 #
# --------------------------------------------------------------------------- #


class _FlakyBackend:
    """Records every prompt; its agent raises on the first N calls."""

    def __init__(self, fail_ticks: int) -> None:
        self.calls = 0
        self.fail_ticks = fail_ticks
        self.prompts: list[str] = []

    def preflight(self) -> None:
        pass

    def build_agent(
        self, *, system_prompt: str, tools: Sequence[Callable[..., Any]] = ()
    ) -> Any:
        return self

    def __call__(self, prompt: str) -> object:
        self.prompts.append(prompt)
        self.calls += 1
        if self.calls <= self.fail_ticks:
            raise RuntimeError("model hiccup")
        return {"message": "ok"}


def test_pending_comms_survives_a_failed_tick() -> None:
    # a message read on a tick whose LLM call fails must be re-shown, not lost
    mbox, chat = MailboxStore(), ChatStore()
    shared = _world_surface(mbox, chat)
    b = _FlakyBackend(fail_ticks=1)
    npc = PersonaAgent(config={"name": "B", "tools": ["mail_read"], "cadence_ticks": 1})
    npc._backend_override = b
    npc.start({"episode_id": "ep", "agent_backend": b})
    npc._cooldown = 0
    mbox.send(sender="A", to="B", body="urgent thing")
    npc.step(shared)  # reads it, builds prompt, agent RAISES -> buffer kept
    npc.step(shared)  # no new mail (cursor advanced) but buffer re-shows it; succeeds
    assert "urgent thing" in b.prompts[1]


def test_persona_does_not_read_its_own_mail() -> None:
    mbox = MailboxStore()
    npc = PersonaAgent(
        config={"name": "Dana", "tools": ["mail_send", "mail_read"], "cadence_ticks": 1}
    )
    b = _run(npc)
    npc.step(surface_mailbox(mbox))
    mbox.send(sender="Dana", to="Dana", body="note to self")
    mbox.send(sender="Sam", to="Dana", body="from sam")
    npc.step(surface_mailbox(mbox))
    prompt = b.agent.prompts[-1]
    assert "from sam" in prompt
    assert "note to self" not in prompt  # own mail is skipped


def test_bad_message_id_does_not_break_the_reader() -> None:
    npc = PersonaAgent(config={"name": "B", "tools": ["mail_read"], "cadence_ticks": 1})
    _run(npc)

    def bad_reader(box: str, since: int = 0) -> list[dict[str, object]]:
        return [{"id": None, "sender": "A", "body": "x"}]

    npc.step({"mail_read": bad_reader})  # must not raise
    assert not npc.broken_reason


def test_memory_matches_across_adjacent_punctuation() -> None:
    mem = DictMemory()
    mem.store("s", "reconciled the finance, portal.")
    mem.store("s", "unrelated note about the coffee machine")
    hits = mem.retrieve("s", "finance")
    assert hits[0] == "reconciled the finance, portal."  # overlap, not recency


def test_persona_renders_social_grounding() -> None:
    prompt = render_persona(
        {
            "name": "Dana",
            "role": "accountant",
            "contacts": ["Sam", "exec"],
            "channels": ["finance"],
            "example_line": "ugh not again",
        }
    )
    assert "People you deal with here: Sam, exec" in prompt
    assert "Chat channels you use: finance" in prompt
    assert "ugh not again" in prompt


def test_persona_grammar_is_clean() -> None:
    prompt = render_persona(
        {"name": "Ana", "role": "accountant", "traits": {"blunt": True}}
    )
    assert "an accountant" in prompt  # article agreement
    assert "You come across as blunt" in prompt
    assert "You You" not in prompt


def test_user_prompt_is_diegetic() -> None:
    npc = PersonaAgent(
        config={
            "name": "U",
            "tools": ["http_get"],
            "goal": "file expenses",
            "cadence_ticks": 1,
        }
    )
    b = _run(npc)
    npc.step({"http_get": lambda p: "ok"})
    prompt = b.agent.prompts[-1]
    assert "It's your turn" not in prompt  # no fourth-wall / game-speak
    assert "using a tool" not in prompt
    assert "file expenses" in prompt


def test_reserved_param_names_still_build_a_usable_tool() -> None:
    from openrange_pack_sdk.npcs.persona_agent import _wrap_action

    def notify(self: str, msg: str) -> str:  # 'self' is a strands-reserved name
        return f"{self}:{msg}"

    tool: Any = _as_tool(_wrap_action("notify", notify), "notify")
    props = tool.tool_spec["inputSchema"]["json"]["properties"]
    assert "msg" in props  # msg survives; self is sanitized to arg0
    raw = _wrap_action("notify", notify)
    assert raw(arg0="A", msg="hi") == "A:hi"


def test_chat_read_without_channels_warns(caplog: Any) -> None:
    import logging

    with caplog.at_level(logging.WARNING):
        npc = PersonaAgent(config={"name": "U", "tools": ["chat_read"]})
        _run(npc)
    assert any("chat_read but no channels" in r.message for r in caplog.records)
