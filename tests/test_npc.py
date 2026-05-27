"""Core NPC contract + registry tests."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import pytest
from openrange_pack_sdk import (
    NPC,
    AgentBackendError,
    AgentNPC,
    NPCError,
)

from openrange.npc import (
    NPCRegistry,
    resolve_manifest_npcs,
)


class _RecordingNPC(NPC):
    def __init__(self, label: str = "rec") -> None:
        self.label = label
        self.started_with: Mapping[str, Any] | None = None
        self.steps: list[Mapping[str, Any]] = []
        self.stopped: bool = False

    def start(self, context: Mapping[str, Any]) -> None:
        self.started_with = dict(context)

    def step(self, interface: Mapping[str, Any]) -> None:
        self.steps.append(dict(interface))

    def stop(self) -> None:
        self.stopped = True


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registry_resolves_registered_factory() -> None:
    reg = NPCRegistry()
    reg.register("test.rec", lambda config: _RecordingNPC(str(config.get("label", ""))))
    npc = reg.resolve("test.rec", {"label": "alpha"})
    assert isinstance(npc, _RecordingNPC)
    assert npc.label == "alpha"


def test_registry_unknown_npc_raises() -> None:
    reg = NPCRegistry()
    with pytest.raises(NPCError, match="unknown NPC"):
        reg.resolve("does.not.exist", {})


def test_registry_factory_returning_non_npc_raises() -> None:
    reg = NPCRegistry()
    reg.register("bad.factory", lambda config: "not an npc")  # type: ignore[arg-type,return-value]
    with pytest.raises(NPCError, match="did not return an NPC"):
        reg.resolve("bad.factory", {})


def test_registry_default_does_not_autodiscover() -> None:
    """Test instances start empty so tests get a clean slate."""
    reg = NPCRegistry()
    assert reg.ids() == ()


def test_registry_global_autodiscovers_cyber_npcs() -> None:
    """The autodiscover global picks up the entry-point-registered cyber NPCs."""
    from openrange.npc import NPCS

    ids = NPCS.ids()
    assert "cyber.browsing_user" in ids
    assert "cyber.admin_audit" in ids


# ---------------------------------------------------------------------------
# Manifest resolution
# ---------------------------------------------------------------------------


def test_resolve_manifest_npcs_spawns_count_instances() -> None:
    reg = NPCRegistry()
    reg.register("test.rec", lambda config: _RecordingNPC())
    npcs = resolve_manifest_npcs(
        ({"type": "test.rec", "count": 3},),
        registry=reg,
    )
    assert len(npcs) == 3
    assert all(isinstance(npc, _RecordingNPC) for npc in npcs)


def test_resolve_manifest_npcs_default_count_is_one() -> None:
    reg = NPCRegistry()
    reg.register("test.rec", lambda config: _RecordingNPC())
    npcs = resolve_manifest_npcs(
        ({"type": "test.rec"},),
        registry=reg,
    )
    assert len(npcs) == 1


def test_resolve_manifest_npcs_passes_config() -> None:
    reg = NPCRegistry()
    captured: dict[str, object] = {}

    def factory(config: Mapping[str, object]) -> NPC:
        captured.update(config)
        return _RecordingNPC()

    reg.register("test.rec", factory)
    resolve_manifest_npcs(
        ({"type": "test.rec", "config": {"foo": "bar", "n": 7}},),
        registry=reg,
    )
    assert captured == {"foo": "bar", "n": 7}


def test_resolve_manifest_npcs_zero_count_skips() -> None:
    reg = NPCRegistry()
    reg.register("test.rec", lambda config: _RecordingNPC())
    npcs = resolve_manifest_npcs(
        ({"type": "test.rec", "count": 0},),
        registry=reg,
    )
    assert npcs == []


def test_resolve_manifest_npcs_rejects_missing_type() -> None:
    reg = NPCRegistry()
    with pytest.raises(NPCError, match="non-empty 'type'"):
        resolve_manifest_npcs(({"count": 1},), registry=reg)


def test_resolve_manifest_npcs_rejects_negative_count() -> None:
    reg = NPCRegistry()
    reg.register("test.rec", lambda config: _RecordingNPC())
    with pytest.raises(NPCError, match="non-negative"):
        resolve_manifest_npcs(
            ({"type": "test.rec", "count": -1},),
            registry=reg,
        )


def test_resolve_manifest_npcs_rejects_non_mapping_config() -> None:
    reg = NPCRegistry()
    reg.register("test.rec", lambda config: _RecordingNPC())
    with pytest.raises(NPCError, match="must be a mapping"):
        resolve_manifest_npcs(
            ({"type": "test.rec", "config": "not-a-map"},),
            registry=reg,
        )


# ---------------------------------------------------------------------------
# ABC defaults
# ---------------------------------------------------------------------------


def test_default_lifecycle_hooks_are_noops() -> None:
    """Subclasses that override only ``step`` can rely on default start/stop."""

    class MinimalNPC(NPC):
        def __init__(self) -> None:
            self.steps = 0

        def step(self, interface: Mapping[str, Any]) -> None:
            self.steps += 1

    npc = MinimalNPC()
    npc.start({"episode_id": "x"})  # default: no-op
    npc.step({})
    npc.stop()  # default: no-op
    assert npc.steps == 1


def test_npc_requires_llm_defaults_false() -> None:
    """Plain NPC subclasses do not opt into LLM injection."""

    class Plain(NPC):
        def step(self, interface: Mapping[str, Any]) -> None: ...

    assert Plain.requires_llm is False
    assert Plain().requires_llm is False


# ---------------------------------------------------------------------------
# AgentNPC
# ---------------------------------------------------------------------------


class _RecordingAgent:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(self, prompt: str) -> object:
        self.calls.append(prompt)
        return {"message": "ok"}


class _RecordingBackend:
    def __init__(self, *, label: str = "recording", reject_tools: bool = False) -> None:
        self.label = label
        self.reject_tools = reject_tools
        self.preflight_calls = 0
        self.builds: list[tuple[str, tuple[Callable[..., Any], ...]]] = []
        self.agent = _RecordingAgent()

    def preflight(self) -> None:
        self.preflight_calls += 1

    def build_agent(
        self,
        *,
        system_prompt: str,
        tools: Sequence[Callable[..., Any]] = (),
    ) -> Any:
        if self.reject_tools and tools:
            raise AgentBackendError("this backend rejects tools")
        self.builds.append((system_prompt, tuple(tools)))
        return self.agent


class _StubAgentNPC(AgentNPC):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.tool_calls: list[Mapping[str, Any]] = []

    def _build_tools(
        self,
        interface: Mapping[str, Any],
    ) -> Sequence[Callable[..., Any]]:
        self.tool_calls.append(dict(interface))

        def visit(path: str) -> str:
            return path

        return [visit]


def test_agent_npc_requires_llm_is_true_by_default() -> None:
    assert AgentNPC.requires_llm is True


def test_agent_npc_rejects_invalid_construction() -> None:
    with pytest.raises(ValueError, match="system_prompt"):
        _StubAgentNPC(system_prompt="", agent_backend=_RecordingBackend())
    with pytest.raises(ValueError, match="cadence_ticks"):
        _StubAgentNPC(
            system_prompt="x",
            cadence_ticks=0,
            agent_backend=_RecordingBackend(),
        )


def test_agent_npc_preflights_constructor_backend() -> None:
    backend = _RecordingBackend()
    _StubAgentNPC(system_prompt="x", agent_backend=backend)
    assert backend.preflight_calls == 1


def test_agent_npc_marks_broken_when_constructor_preflight_fails() -> None:
    class _BadBackend(_RecordingBackend):
        def preflight(self) -> None:
            raise AgentBackendError("boom")

    backend = _BadBackend()
    npc = _StubAgentNPC(system_prompt="x", agent_backend=backend)
    assert npc.broken_reason is not None
    assert "preflight failed" in npc.broken_reason


def test_agent_npc_acts_on_first_step_then_obeys_cadence() -> None:
    backend = _RecordingBackend()
    npc = _StubAgentNPC(
        system_prompt="be curious",
        cadence_ticks=3,
        agent_backend=backend,
    )
    iface: dict[str, Any] = {"base_url": "http://x"}

    npc.step(iface)
    assert backend.agent.calls == [npc._user_prompt(iface)]
    assert len(backend.builds) == 1

    npc.step(iface)
    npc.step(iface)
    assert len(backend.agent.calls) == 1  # cooldown

    npc.step(iface)
    assert len(backend.agent.calls) == 2
    assert len(backend.builds) == 1  # agent built once, reused


def test_agent_npc_captures_runtime_backend_from_context() -> None:
    runtime_backend = _RecordingBackend(label="runtime")
    npc = _StubAgentNPC(system_prompt="x", cadence_ticks=1)
    npc.start({"episode_id": "e", "agent_backend": runtime_backend})
    assert npc._runtime_backend is runtime_backend
    assert runtime_backend.preflight_calls == 1
    npc.step({})
    assert len(runtime_backend.builds) == 1


def test_agent_npc_constructor_backend_overrides_runtime() -> None:
    explicit = _RecordingBackend(label="explicit")
    runtime_backend = _RecordingBackend(label="runtime")
    npc = _StubAgentNPC(
        system_prompt="x",
        cadence_ticks=1,
        agent_backend=explicit,
    )
    npc.start({"episode_id": "e", "agent_backend": runtime_backend})
    npc.step({})
    # Constructor backend used, runtime backend ignored.
    assert len(explicit.builds) == 1
    assert len(runtime_backend.builds) == 0


def test_agent_npc_marks_broken_when_no_backend_anywhere() -> None:
    """No constructor backend + no runtime backend → broken at start()."""
    npc = _StubAgentNPC(system_prompt="x", cadence_ticks=1)
    npc.start({"episode_id": "e", "agent_backend": None})
    assert npc.broken_reason is not None
    assert "no AgentBackend configured" in npc.broken_reason
    npc.step({})  # short-circuited by broken flag
    npc.step({})


def test_agent_npc_marks_broken_when_runtime_preflight_fails() -> None:
    class _BadRuntimeBackend(_RecordingBackend):
        def preflight(self) -> None:
            raise AgentBackendError("runtime boom")

    npc = _StubAgentNPC(system_prompt="x", cadence_ticks=1)
    npc.start({"episode_id": "e", "agent_backend": _BadRuntimeBackend()})
    assert npc.broken_reason is not None
    assert "runtime backend preflight failed" in npc.broken_reason


def test_agent_npc_marks_broken_on_build_failure() -> None:
    """If backend.build_agent raises, NPC stays silent."""

    class _BuildFailBackend(_RecordingBackend):
        def build_agent(self, **_kwargs: Any) -> Any:
            raise AgentBackendError("build failed")

    backend = _BuildFailBackend()
    npc = _StubAgentNPC(system_prompt="x", cadence_ticks=1, agent_backend=backend)
    npc.step({})
    assert npc._broken is True
    assert npc.broken_reason is not None
    assert "failed to construct agent" in npc.broken_reason
    # Subsequent steps short-circuit; no further builds attempted.
    npc.step({})
    npc.step({})


def test_agent_npc_swallows_invocation_errors() -> None:
    """A throwing agent does not sink the episode."""

    class _ThrowingAgent(_RecordingAgent):
        def __call__(self, prompt: str) -> object:
            raise RuntimeError("model went poof")

    class _ThrowingBackend(_RecordingBackend):
        def build_agent(self, **kwargs: Any) -> Any:
            return _ThrowingAgent()

    npc = _StubAgentNPC(
        system_prompt="x",
        cadence_ticks=1,
        agent_backend=_ThrowingBackend(),
    )
    # Must not raise.
    npc.step({})
    npc.step({})


def test_agent_npc_stop_clears_agent_reference() -> None:
    npc = _StubAgentNPC(
        system_prompt="x",
        cadence_ticks=1,
        agent_backend=_RecordingBackend(),
    )
    npc.step({})
    assert npc._agent is not None
    npc.stop()
    assert npc._agent is None


def test_agent_npc_passes_system_prompt_and_tools_to_backend() -> None:
    backend = _RecordingBackend()
    npc = _StubAgentNPC(
        system_prompt="be curious",
        cadence_ticks=1,
        agent_backend=backend,
    )
    npc.step({"base_url": "http://x"})
    assert len(backend.builds) == 1
    system_prompt, tools = backend.builds[0]
    assert system_prompt == "be curious"
    assert len(tools) == 1  # the trivial visit() tool from _StubAgentNPC
