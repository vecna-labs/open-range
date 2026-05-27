"""Branch-coverage tests for openrange_pack_sdk.

Concrete subclasses + concrete recording implementations of the actual
Protocols and ABCs. No mocks, no patches, no test doubles.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import pytest
from graphschema import Ontology, WorldGraph
from openrange_pack_sdk import (
    NPC,
    AgentBackend,
    AgentBackendError,
    AgentNPC,
    AgentSession,
    Backing,
    Builder,
    BuildResult,
    EpisodeResult,
    FeasibilityVerdict,
    LLMRequest,
    LLMRequestError,
    Manifest,
    Pack,
    PackPrior,
    RuntimeHandle,
    TaskFamily,
    TaskSpec,
)


def _empty_ontology() -> Ontology:
    return Ontology(id="test@0")


def _empty_graph() -> WorldGraph:
    return WorldGraph(ontology=_empty_ontology().id)


class _NoopFamily(TaskFamily):
    id = "test.noop"
    pack_id = "test"

    def generate(
        self,
        graph: WorldGraph,
        manifest: Manifest,
        prior: PackPrior | None,
    ) -> list[TaskSpec]:
        del graph, manifest, prior
        return []

    def check_feasibility(
        self,
        graph: WorldGraph,
        task: TaskSpec,
    ) -> FeasibilityVerdict:
        del graph, task
        return FeasibilityVerdict(True)

    def check_success(
        self,
        graph: WorldGraph,
        task: TaskSpec,
        final_state: Mapping[str, Any],
    ) -> EpisodeResult:
        del graph, task, final_state
        return EpisodeResult(True)


class _NoopBuilder(Builder):
    def build(self, manifest: Manifest) -> BuildResult:
        del manifest
        return BuildResult(_empty_graph(), [])


class _NoopHandle:
    def reset(self) -> None: ...
    def surface(self) -> Mapping[str, Any]:
        return {}

    def poll_events(self) -> tuple[Mapping[str, Any], ...]:
        return ()

    def terminal(self) -> tuple[bool, str | None]:
        return False, None

    def checkpoint(self) -> Any:
        return None

    def restore(self, state: Any) -> None:
        del state

    def collect(self) -> Mapping[str, Any]:
        return {}

    def stop(self) -> None: ...


class _OneFamilyPack(Pack):
    id = "test"
    version = "v1"

    def ontology(self) -> Ontology:
        return _empty_ontology()

    def make_builder(self, prior: PackPrior | None) -> Builder:
        del prior
        return _NoopBuilder()

    def realize(self, graph: WorldGraph, backing: Backing) -> RuntimeHandle:
        del graph, backing
        return _NoopHandle()

    def task_families(self) -> list[TaskFamily]:
        return [_NoopFamily()]


class _NoFamiliesPack(_OneFamilyPack):
    def task_families(self) -> list[TaskFamily]:
        return []


class _RecordingLLMBackend:
    # No preflight() method — proves the LLMBackend Protocol stays
    # satisfied without it.

    def __init__(self) -> None:
        self.calls: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> Any:
        self.calls.append(request)
        from openrange_pack_sdk import LLMResult

        return LLMResult(text="ok")


class _PermissiveAgentBackend:
    def __init__(self, *, fail_preflight: bool = False) -> None:
        self.preflighted = 0
        self.built: list[tuple[str, Sequence[Callable[..., Any]]]] = []
        self.invoked: list[str] = []
        self._fail_preflight = fail_preflight

    def preflight(self) -> None:
        self.preflighted += 1
        if self._fail_preflight:
            raise AgentBackendError("intentional preflight failure")

    def build_agent(
        self,
        *,
        system_prompt: str,
        tools: Sequence[Callable[..., Any]] = (),
    ) -> AgentSession:
        self.built.append((system_prompt, list(tools)))
        recorded = self.invoked

        def session(prompt: str) -> Any:
            recorded.append(prompt)
            return "ok"

        return session


class _MinimalAgentNPC(AgentNPC):
    def _build_tools(
        self,
        interface: Mapping[str, Any],
    ) -> Sequence[Callable[..., Any]]:
        del interface
        return ()


class TestTaskFamilyDefaults:
    def test_available_mutations_default_returns_empty(self) -> None:
        family = _NoopFamily()
        graph = _empty_graph()
        snap_like = type(
            "FakeSnap",
            (),
            {"graph": graph, "tasks": (), "lineage": {}, "history": ()},
        )()
        assert family.available_mutations(snap_like, ()) == ()


class TestPackTaskFamilyLookup:
    def test_returns_family_when_match(self) -> None:
        pack = _OneFamilyPack()
        family = pack.task_family("test.noop")
        assert isinstance(family, _NoopFamily)

    def test_returns_none_when_no_match_with_families_present(self) -> None:
        pack = _OneFamilyPack()
        assert pack.task_family("unknown") is None

    def test_returns_none_when_no_families_at_all(self) -> None:
        pack = _NoFamiliesPack()
        assert pack.task_family("anything") is None

    def test_default_invariants_empty(self) -> None:
        pack = _OneFamilyPack()
        assert pack.invariants() == []

    def test_default_task_families_is_empty_list(self) -> None:
        class _MinimalPack(Pack):
            id = "minimal"
            version = "v0"

            def ontology(self) -> Ontology:
                return _empty_ontology()

            def make_builder(self, prior: PackPrior | None) -> Builder:
                del prior
                return _NoopBuilder()

            def realize(
                self,
                graph: WorldGraph,
                backing: Backing,
            ) -> RuntimeHandle:
                del graph, backing
                return _NoopHandle()

        assert _MinimalPack().task_families() == []


class TestBuilderDefaults:
    def test_repair_raises_by_default(self) -> None:
        builder = _NoopBuilder()
        with pytest.raises(NotImplementedError, match="repair"):
            builder.repair(BuildResult(_empty_graph(), []), errors=[], infeasible=[])

    def test_evolve_passes_patch_through(self) -> None:
        from graphschema import GraphPatch
        from openrange_pack_sdk import Mutation

        builder = _NoopBuilder()
        patch = GraphPatch()
        snap_like = type(
            "FakeSnap",
            (),
            {"graph": _empty_graph(), "tasks": (), "lineage": {}, "history": ()},
        )()
        mut = Mutation(patch=patch, direction="harden", relevance=1.0, family="test")
        out = builder.evolve(snap_like, mut)
        assert out is patch


class TestNPCActorId:
    def test_actor_id_falls_back_to_class_and_hash_when_unset(self) -> None:
        class _AnonymousNPC(NPC):
            def step(self, interface: Mapping[str, Any]) -> None:
                del interface

        npc = _AnonymousNPC()
        actor_id = npc.actor_id
        assert actor_id.startswith("_AnonymousNPC-")
        assert len(actor_id.split("-")[-1]) == 4

    def test_actor_id_uses_explicit_when_set(self) -> None:
        class _NamedNPC(NPC):
            def __init__(self) -> None:
                self._actor_id = "Alice"

            def step(self, interface: Mapping[str, Any]) -> None:
                del interface

        assert _NamedNPC().actor_id == "Alice"

    def test_npc_start_default_is_noop(self) -> None:
        class _Plain(NPC):
            def step(self, interface: Mapping[str, Any]) -> None:
                del interface

        npc = _Plain()
        npc.start({})
        npc.stop()


class TestAgentNPCLifecycle:
    def test_init_rejects_empty_system_prompt(self) -> None:
        with pytest.raises(ValueError, match="system_prompt"):
            _MinimalAgentNPC(system_prompt="", cadence_ticks=1)

    def test_init_rejects_zero_cadence(self) -> None:
        with pytest.raises(ValueError, match="cadence_ticks"):
            _MinimalAgentNPC(system_prompt="hi", cadence_ticks=0)

    def test_constructor_preflight_marks_broken_on_failure(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        backend = _PermissiveAgentBackend(fail_preflight=True)
        with caplog.at_level(logging.WARNING):
            npc = _MinimalAgentNPC(
                system_prompt="hi", cadence_ticks=1, agent_backend=backend
            )
        assert npc._broken
        assert npc.broken_reason is not None
        assert "preflight" in npc.broken_reason

    def test_mark_broken_is_reentrant(self, caplog: pytest.LogCaptureFixture) -> None:
        backend = _PermissiveAgentBackend(fail_preflight=True)
        with caplog.at_level(logging.WARNING):
            npc = _MinimalAgentNPC(
                system_prompt="hi", cadence_ticks=1, agent_backend=backend
            )
            npc._mark_broken("second reason")
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert "second reason" not in (npc.broken_reason or "")

    def test_start_short_circuits_when_already_broken(self) -> None:
        broken_backend = _PermissiveAgentBackend(fail_preflight=True)
        npc = _MinimalAgentNPC(
            system_prompt="hi", cadence_ticks=1, agent_backend=broken_backend
        )
        assert npc._broken
        runtime_backend = _PermissiveAgentBackend()
        npc.start({"agent_backend": runtime_backend})
        assert runtime_backend.preflighted == 0

    def test_start_marks_broken_when_no_backend_anywhere(self) -> None:
        npc = _MinimalAgentNPC(system_prompt="hi", cadence_ticks=1)
        npc.start({})
        assert npc._broken
        assert "no AgentBackend" in (npc.broken_reason or "")

    def test_start_marks_broken_when_runtime_preflight_fails(self) -> None:
        npc = _MinimalAgentNPC(system_prompt="hi", cadence_ticks=1)
        runtime_backend = _PermissiveAgentBackend(fail_preflight=True)
        npc.start({"agent_backend": runtime_backend})
        assert npc._broken
        assert "preflight failed" in (npc.broken_reason or "")

    def test_step_short_circuits_when_broken(self) -> None:
        broken_backend = _PermissiveAgentBackend(fail_preflight=True)
        npc = _MinimalAgentNPC(
            system_prompt="hi", cadence_ticks=1, agent_backend=broken_backend
        )
        npc.step({})

    def test_step_cooldown_path(self) -> None:
        backend = _PermissiveAgentBackend()
        npc = _MinimalAgentNPC(
            system_prompt="hi", cadence_ticks=3, agent_backend=backend
        )
        npc.start({"agent_backend": backend})
        npc.step({})  # builds + invokes once
        npc.step({})  # cooldown
        npc.step({})  # cooldown
        npc.step({})  # invokes again
        assert len(backend.invoked) == 2

    def test_step_swallows_transient_invoke_errors(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        class _FlakyBackend(_PermissiveAgentBackend):
            def build_agent(
                self,
                *,
                system_prompt: str,
                tools: Sequence[Callable[..., Any]] = (),
            ) -> AgentSession:
                del system_prompt, tools

                def session(prompt: str) -> Any:
                    del prompt
                    raise RuntimeError("transient")

                return session

        backend = _FlakyBackend()
        npc = _MinimalAgentNPC(
            system_prompt="hi", cadence_ticks=1, agent_backend=backend
        )
        npc.start({"agent_backend": backend})
        with caplog.at_level(logging.DEBUG):
            npc.step({})
        records = caplog.records
        assert any(
            "transient" in r.message or "tick failed" in r.message for r in records
        )
        assert not npc._broken

    def test_step_marks_broken_when_tool_construction_raises(self) -> None:
        class _ExplodingNPC(AgentNPC):
            def _build_tools(
                self,
                interface: Mapping[str, Any],
            ) -> Sequence[Callable[..., Any]]:
                del interface
                raise RuntimeError("kaboom")

        backend = _PermissiveAgentBackend()
        npc = _ExplodingNPC(system_prompt="hi", cadence_ticks=1, agent_backend=backend)
        npc.start({"agent_backend": backend})
        npc.step({})
        assert npc._broken
        assert "kaboom" in (npc.broken_reason or "")

    def test_build_agent_raises_when_backend_disappeared(self) -> None:
        npc = _MinimalAgentNPC(system_prompt="hi", cadence_ticks=1)
        with pytest.raises(AgentBackendError, match="start"):
            npc._build_agent(())

    def test_stop_clears_agent(self) -> None:
        backend = _PermissiveAgentBackend()
        npc = _MinimalAgentNPC(
            system_prompt="hi", cadence_ticks=1, agent_backend=backend
        )
        npc.start({"agent_backend": backend})
        npc.step({})
        assert npc._agent is not None
        npc.stop()
        assert npc._agent is None

    def test_default_user_prompt_is_non_empty(self) -> None:
        npc = _MinimalAgentNPC(system_prompt="hi", cadence_ticks=1)
        assert npc._user_prompt({})


class TestLLMRequestValidation:
    def test_accepts_serializable_schema(self) -> None:
        req = LLMRequest(prompt="x", json_schema={"type": "object"})
        assert req.json_schema == {"type": "object"}

    def test_rejects_unserializable_schema(self) -> None:
        with pytest.raises(LLMRequestError, match="JSON serializable"):
            LLMRequest(prompt="x", json_schema={"obj": object()})

    def test_as_prompt_without_system(self) -> None:
        assert LLMRequest(prompt="hi").as_prompt() == "hi"

    def test_as_prompt_with_system(self) -> None:
        out = LLMRequest(prompt="hi", system="sys").as_prompt()
        assert "sys" in out
        assert "hi" in out


class TestBuildEvent:
    def test_to_dict_without_refs(self) -> None:
        from openrange_pack_sdk import BuildEvent

        d = BuildEvent(0, "build", "x").to_dict()
        assert d == {"seq": 0, "phase": "build", "detail": "x"}

    def test_to_dict_with_refs(self) -> None:
        from openrange_pack_sdk import BuildEvent

        d = BuildEvent(1, "validate", "y", refs=("a", "b")).to_dict()
        assert d == {"seq": 1, "phase": "validate", "detail": "y", "refs": ["a", "b"]}


class TestErrorHierarchy:
    def test_all_descend_from_openrange_error(self) -> None:
        from openrange_pack_sdk import (
            AgentBackendError,
            LLMBackendError,
            LLMError,
            LLMRequestError,
            ManifestError,
            NPCError,
            OpenRangeError,
            PackError,
        )

        for exc in (
            ManifestError,
            PackError,
            LLMError,
            LLMRequestError,
            LLMBackendError,
            NPCError,
            AgentBackendError,
        ):
            assert issubclass(exc, OpenRangeError)

    def test_llm_backend_error_carries_returncode(self) -> None:
        from openrange_pack_sdk import LLMBackendError

        exc = LLMBackendError("boom", returncode=42)
        assert exc.returncode == 42
        exc2 = LLMBackendError("boom")
        assert exc2.returncode is None


class TestRuntimeCheckableProtocols:
    def test_runtime_handle_runtime_check(self) -> None:
        assert isinstance(_NoopHandle(), RuntimeHandle)

    def test_agent_backend_runtime_check(self) -> None:
        assert isinstance(_PermissiveAgentBackend(), AgentBackend)

    def test_llm_backend_runtime_check_without_preflight(self) -> None:
        from openrange_pack_sdk import LLMBackend

        assert isinstance(_RecordingLLMBackend(), LLMBackend)
