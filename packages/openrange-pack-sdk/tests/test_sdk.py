"""Branch-coverage tests for openrange_pack_sdk.

Concrete subclasses + concrete recording implementations of the actual
Protocols and ABCs. No mocks, no patches, no test doubles.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
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
    Snapshot,
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
        snap = Snapshot(
            snapshot_id="",
            ontology_id=graph.ontology,
            graph=graph,
            tasks=(),
            lineage={},
        )
        assert family.available_mutations(snap, ()) == ()


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
        graph = _empty_graph()
        snap = Snapshot(
            snapshot_id="",
            ontology_id=graph.ontology,
            graph=graph,
            tasks=(),
            lineage={},
        )
        mut = Mutation(patch=patch, direction="harden", relevance=1.0, family="test")
        out = builder.evolve(snap, mut)
        assert out is patch


class TestProceduralBuilder:
    def _make_builder(self, *, prior: PackPrior | None = None) -> tuple[Any, list[int]]:
        import random

        from openrange_pack_sdk import ProceduralBuilder

        seeds_seen: list[int] = []

        class _RecordingBuilder(ProceduralBuilder):
            def sample(self, rng: random.Random, manifest: Manifest) -> BuildResult:
                del manifest
                seeds_seen.append(rng.randint(0, 10_000_000))
                return BuildResult(_empty_graph(), [])

        return _RecordingBuilder(prior), seeds_seen

    def test_init_defaults(self) -> None:
        builder, _ = self._make_builder()
        assert builder.prior is None
        assert builder.current_seed == 0

    def test_init_stores_prior(self) -> None:
        prior = PackPrior(
            source="test",
            ontology=_empty_ontology(),
            topology={"n_services": 1},
        )
        builder, _ = self._make_builder(prior=prior)
        assert builder.prior is prior

    def test_build_uses_manifest_seed(self) -> None:
        builder, seeds_seen = self._make_builder()
        result = builder.build({"seed": 42})
        assert isinstance(result, BuildResult)
        assert builder.current_seed == 42
        # Same seed → deterministic sample output.
        builder2, seeds2 = self._make_builder()
        builder2.build({"seed": 42})
        assert seeds_seen == seeds2

    def test_build_defaults_missing_seed_to_zero(self) -> None:
        builder, _ = self._make_builder()
        builder.build({})
        assert builder.current_seed == 0

    def test_repair_increments_attempt_and_re_samples(self) -> None:
        builder, seeds_seen = self._make_builder()
        prev = builder.build({"seed": 100})
        # Attempt 0 → seed 100.
        assert builder.current_seed == 100
        builder.repair(prev, errors=[], infeasible=[])
        # Attempt 1 → seed 101 (different sample output).
        assert builder.current_seed == 101
        assert seeds_seen[0] != seeds_seen[1]

    def test_repair_chains_across_multiple_attempts(self) -> None:
        builder, _ = self._make_builder()
        prev = builder.build({"seed": 5})
        for _ in range(3):
            prev = builder.repair(prev, errors=[], infeasible=[])
        assert builder.current_seed == 5 + 3

    def test_custom_seed_key(self) -> None:
        import random

        from openrange_pack_sdk import ProceduralBuilder

        class _CustomSeed(ProceduralBuilder):
            def sample(self, rng: random.Random, manifest: Manifest) -> BuildResult:
                del rng, manifest
                return BuildResult(_empty_graph(), [])

        builder = _CustomSeed(seed_key="my_seed")
        builder.build({"my_seed": 7})
        assert builder.current_seed == 7


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


class TestMakeTaskHelper:
    def test_derives_ids_from_family(self) -> None:
        family = _NoopFamily()
        task = family.make_task(
            instruction="do X",
            entrypoints="ep1",
            goal_nodes="goal1",
        )
        assert task.id == "test.noop.0"
        assert task.feasibility_check == "test.noop"
        assert task.success_check == "test.noop"
        assert task.meta["family"] == "test.noop"
        assert task.meta["difficulty"] == 0.5

    def test_single_string_entrypoint_becomes_tuple(self) -> None:
        task = _NoopFamily().make_task(
            instruction="x", entrypoints="ep1", goal_nodes="g1"
        )
        assert task.entrypoints == ("ep1",)
        assert task.goal_nodes == ("g1",)

    def test_tuple_entrypoints_pass_through(self) -> None:
        task = _NoopFamily().make_task(
            instruction="x",
            entrypoints=("a", "b"),
            goal_nodes=("c", "d"),
        )
        assert task.entrypoints == ("a", "b")
        assert task.goal_nodes == ("c", "d")

    def test_index_changes_id(self) -> None:
        a = _NoopFamily().make_task(instruction="x", entrypoints="a", index=0)
        b = _NoopFamily().make_task(instruction="x", entrypoints="a", index=2)
        assert a.id == "test.noop.0"
        assert b.id == "test.noop.2"

    def test_explicit_meta_merges_with_derived_fields(self) -> None:
        task = _NoopFamily().make_task(
            instruction="x",
            entrypoints="a",
            difficulty=0.9,
            meta={"flag_secret": "s1", "kind": "api"},
        )
        assert task.meta == {
            "family": "test.noop",
            "difficulty": 0.9,
            "flag_secret": "s1",
            "kind": "api",
        }

    def test_string_index_supported(self) -> None:
        task = _NoopFamily().make_task(instruction="x", entrypoints="a", index="alice")
        assert task.id == "test.noop.alice"

    def test_no_meta_works(self) -> None:
        task = _NoopFamily().make_task(instruction="x", entrypoints="a")
        assert task.meta == {"family": "test.noop", "difficulty": 0.5}

    def test_default_goal_nodes_empty_tuple(self) -> None:
        task = _NoopFamily().make_task(instruction="x", entrypoints="a")
        assert task.goal_nodes == ()


class TestGraphHelpers:
    def test_edge_id_format(self) -> None:
        from openrange_pack_sdk import edge_id

        assert edge_id("a", "kind", "b") == "a__kind__b"

    def test_add_node_returns_node_and_inserts(self) -> None:
        from openrange_pack_sdk import add_node

        g = _empty_graph()
        node = add_node(g, kind="service", id="svc_a", attrs={"name": "alpha"})
        assert node.id == "svc_a"
        assert g.nodes["svc_a"] is node
        assert node.attrs == {"name": "alpha"}

    def test_add_node_defaults(self) -> None:
        from graphschema import Visibility
        from openrange_pack_sdk import add_node

        g = _empty_graph()
        node = add_node(g, kind="endpoint", id="ep_a")
        assert node.attrs == {}
        assert node.roles == set()
        assert node.visibility is Visibility.PUBLIC

    def test_add_node_with_roles_and_visibility(self) -> None:
        from graphschema import Role, Visibility
        from openrange_pack_sdk import add_node

        g = _empty_graph()
        node = add_node(
            g,
            kind="secret",
            id="s1",
            roles={Role.ACTOR},
            visibility=Visibility.HIDDEN,
        )
        assert node.roles == {Role.ACTOR}
        assert node.visibility is Visibility.HIDDEN

    def test_add_edge_returns_edge_and_inserts(self) -> None:
        from openrange_pack_sdk import add_edge, add_node

        g = _empty_graph()
        add_node(g, kind="service", id="svc")
        add_node(g, kind="host", id="host")
        edge = add_edge(g, kind="runs_on", src="svc", dst="host")
        assert edge.id == "svc__runs_on__host"
        assert g.edges["svc__runs_on__host"] is edge

    def test_add_edge_with_attrs(self) -> None:
        from openrange_pack_sdk import add_edge, add_node

        g = _empty_graph()
        add_node(g, kind="service", id="svc")
        add_node(g, kind="data_store", id="store")
        edge = add_edge(
            g, kind="backed_by", src="svc", dst="store", attrs={"mode": "rw"}
        )
        assert edge.attrs == {"mode": "rw"}


class TestWriteTree:
    def test_writes_files_under_root(self, tmp_path: Path) -> None:
        from openrange_pack_sdk import write_tree

        write_tree(tmp_path, {"a.txt": "alpha", "b.txt": "beta"})
        assert (tmp_path / "a.txt").read_text() == "alpha"
        assert (tmp_path / "b.txt").read_text() == "beta"

    def test_creates_intermediate_directories(self, tmp_path: Path) -> None:
        from openrange_pack_sdk import write_tree

        write_tree(tmp_path, {"deeply/nested/file.txt": "ok"})
        assert (tmp_path / "deeply" / "nested" / "file.txt").read_text() == "ok"

    def test_overwrites_existing_files(self, tmp_path: Path) -> None:
        from openrange_pack_sdk import write_tree

        (tmp_path / "existing.txt").write_text("old")
        write_tree(tmp_path, {"existing.txt": "new"})
        assert (tmp_path / "existing.txt").read_text() == "new"

    def test_empty_mapping_is_noop(self, tmp_path: Path) -> None:
        from openrange_pack_sdk import write_tree

        write_tree(tmp_path, {})
        assert list(tmp_path.iterdir()) == []


class TestManifestAccessors:
    def test_int_present(self) -> None:
        from openrange_pack_sdk import manifest_int

        assert manifest_int({"seed": 42}, "seed") == 42

    def test_int_missing_default(self) -> None:
        from openrange_pack_sdk import manifest_int

        assert manifest_int({}, "seed", default=7) == 7

    def test_int_rejects_bool(self) -> None:
        from openrange_pack_sdk import manifest_int

        assert manifest_int({"seed": True}, "seed", default=99) == 99

    def test_int_rejects_other_types(self) -> None:
        from openrange_pack_sdk import manifest_int

        assert manifest_int({"seed": "not int"}, "seed", default=3) == 3

    def test_str_present(self) -> None:
        from openrange_pack_sdk import manifest_str

        assert manifest_str({"goal": "x"}, "goal") == "x"

    def test_str_missing_default(self) -> None:
        from openrange_pack_sdk import manifest_str

        assert manifest_str({}, "goal", default="d") == "d"

    def test_str_rejects_other_types(self) -> None:
        from openrange_pack_sdk import manifest_str

        assert manifest_str({"goal": 42}, "goal", default="d") == "d"

    def test_bool_present(self) -> None:
        from openrange_pack_sdk import manifest_bool

        assert manifest_bool({"flag": True}, "flag") is True

    def test_bool_missing_default(self) -> None:
        from openrange_pack_sdk import manifest_bool

        assert manifest_bool({}, "flag", default=True) is True

    def test_bool_rejects_int(self) -> None:
        from openrange_pack_sdk import manifest_bool

        assert manifest_bool({"flag": 1}, "flag", default=False) is False

    def test_float_present(self) -> None:
        from openrange_pack_sdk import manifest_float

        assert manifest_float({"rate": 0.25}, "rate") == 0.25

    def test_float_promotes_int(self) -> None:
        from openrange_pack_sdk import manifest_float

        assert manifest_float({"n": 5}, "n") == 5.0

    def test_float_rejects_bool(self) -> None:
        from openrange_pack_sdk import manifest_float

        assert manifest_float({"x": True}, "x", default=1.5) == 1.5

    def test_float_missing_default(self) -> None:
        from openrange_pack_sdk import manifest_float

        assert manifest_float({}, "rate", default=0.1) == 0.1

    def test_float_rejects_other_types(self) -> None:
        from openrange_pack_sdk import manifest_float

        assert manifest_float({"x": "1.5"}, "x", default=2.0) == 2.0

    def test_list_present_copies(self) -> None:
        from openrange_pack_sdk import manifest_list

        original = [1, 2, 3]
        result = manifest_list({"xs": original}, "xs")
        assert result == [1, 2, 3]
        result.append(4)
        assert original == [1, 2, 3]

    def test_list_missing_returns_empty(self) -> None:
        from openrange_pack_sdk import manifest_list

        assert manifest_list({}, "xs") == []

    def test_list_missing_with_default(self) -> None:
        from openrange_pack_sdk import manifest_list

        default = ["a", "b"]
        result = manifest_list({}, "xs", default=default)
        assert result == ["a", "b"]
        result.append("c")
        assert default == ["a", "b"]

    def test_list_rejects_other_types(self) -> None:
        from openrange_pack_sdk import manifest_list

        assert manifest_list({"xs": "abc"}, "xs", default=[1]) == [1]


class TestMakeMutation:
    def test_make_mutation_tags_family(self) -> None:
        from graphschema import GraphPatch

        patch = GraphPatch()
        mut = _NoopFamily().make_mutation(
            direction="harden",
            relevance=0.7,
            patch=patch,
            note="add a vuln",
        )
        assert mut.family == "test.noop"
        assert mut.direction == "harden"
        assert mut.relevance == 0.7
        assert mut.patch is patch
        assert mut.note == "add a vuln"

    def test_make_mutation_default_note_empty(self) -> None:
        from graphschema import GraphPatch

        mut = _NoopFamily().make_mutation(
            direction="soften", relevance=0.1, patch=GraphPatch()
        )
        assert mut.note == ""


_SUBPROCESS_SCRIPT = (
    "import json, os, sys, time\n"
    "payload = {'ready': True, 'pid': os.getpid()}\n"
    "sys.stdout.write(json.dumps(payload) + '\\n')\n"
    "sys.stdout.flush()\n"
    "while True:\n"
    "    time.sleep(0.1)\n"
)


def _make_simple_subprocess_runtime() -> Any:
    from openrange_pack_sdk import SubprocessRuntime

    class _Simple(SubprocessRuntime):
        def prepare_env_files(self, graph: WorldGraph) -> Mapping[str, str]:
            del graph
            return {"ready.txt": "ok"}

        def subprocess_command(self, env_root: Any, agent_root: Any) -> Sequence[str]:
            import sys

            del env_root, agent_root
            return [sys.executable, "-c", _SUBPROCESS_SCRIPT]

        def parse_startup(self, stdout_line: str) -> Mapping[str, Any]:
            import json as _json

            return dict(_json.loads(stdout_line))

        def surface_extras(self) -> Mapping[str, Any]:
            return {"hello": "from pack"}

        def collect_extras(self) -> Mapping[str, Any]:
            return {"finalized_by": "test"}

    return _Simple(_empty_graph())


def _make_silent_subprocess_runtime() -> Any:
    """Subprocess that emits the contract-required newline then idles —
    proves the default parse_startup returns {} when there's no payload."""
    from openrange_pack_sdk import SubprocessRuntime

    class _Silent(SubprocessRuntime):
        def prepare_env_files(self, graph: WorldGraph) -> Mapping[str, str]:
            del graph
            return {}

        def subprocess_command(self, env_root: Any, agent_root: Any) -> Sequence[str]:
            import sys

            del env_root, agent_root
            return [
                sys.executable,
                "-c",
                "import sys, time\n"
                "sys.stdout.write('\\n')\n"
                "sys.stdout.flush()\n"
                "while True: time.sleep(0.1)\n",
            ]

    return _Silent(_empty_graph())


class TestSubprocessRuntime:
    def test_full_lifecycle(self) -> None:
        from pathlib import Path

        runtime = _make_simple_subprocess_runtime()
        try:
            runtime.reset()
            surface = runtime.surface()
            assert surface["ready"] is True
            assert "agent_root" in surface
            assert surface["hello"] == "from pack"
            assert Path(surface["agent_root"]).is_dir()
        finally:
            runtime.stop()

    def test_surface_raises_before_reset(self) -> None:
        runtime = _make_simple_subprocess_runtime()
        with pytest.raises(Exception, match="reset"):
            runtime.surface()

    def test_checkpoint_raises_before_reset(self) -> None:
        runtime = _make_simple_subprocess_runtime()
        with pytest.raises(Exception, match="reset"):
            runtime.checkpoint()

    def test_terminal_false_before_reset(self) -> None:
        runtime = _make_simple_subprocess_runtime()
        ok, reason = runtime.terminal()
        assert ok is False and reason is None

    def test_terminal_true_when_result_written(self) -> None:
        from pathlib import Path

        runtime = _make_simple_subprocess_runtime()
        try:
            runtime.reset()
            surface = runtime.surface()
            (Path(surface["agent_root"]) / "result.json").write_text("{}")
            ok, reason = runtime.terminal()
            assert ok is True
            assert reason == "agent wrote result"
        finally:
            runtime.stop()

    def test_collect_empty_before_reset(self) -> None:
        runtime = _make_simple_subprocess_runtime()
        assert runtime.collect() == {}

    def test_collect_returns_result_and_extras(self) -> None:
        from pathlib import Path

        runtime = _make_simple_subprocess_runtime()
        try:
            runtime.reset()
            agent_root = Path(runtime.surface()["agent_root"])
            (agent_root / "result.json").write_text('{"x": 1}')
            collected = runtime.collect()
            assert collected["result"] == {"x": 1}
            assert collected["finalized_by"] == "test"
            assert "agent_root" in collected
        finally:
            runtime.stop()

    def test_collect_silently_ignores_invalid_json_result(self) -> None:
        from pathlib import Path

        runtime = _make_simple_subprocess_runtime()
        try:
            runtime.reset()
            agent_root = Path(runtime.surface()["agent_root"])
            (agent_root / "result.json").write_text("{not valid")
            collected = runtime.collect()
            assert collected["result"] == {}
        finally:
            runtime.stop()

    def test_collect_treats_non_mapping_result_as_empty(self) -> None:
        from pathlib import Path

        runtime = _make_simple_subprocess_runtime()
        try:
            runtime.reset()
            agent_root = Path(runtime.surface()["agent_root"])
            (agent_root / "result.json").write_text("[1, 2]")
            collected = runtime.collect()
            assert collected["result"] == {}
        finally:
            runtime.stop()

    def test_checkpoint_then_restore_round_trips_agent_root(self) -> None:
        from pathlib import Path

        runtime = _make_simple_subprocess_runtime()
        try:
            runtime.reset()
            agent_root = Path(runtime.surface()["agent_root"])
            (agent_root / "scratch.txt").write_text("hello")
            ckpt = runtime.checkpoint()
            (agent_root / "scratch.txt").write_text("modified")
            runtime.restore(ckpt)
            assert (agent_root / "scratch.txt").read_text() == "hello"
        finally:
            runtime.stop()

    def test_restore_rejects_non_mapping(self) -> None:
        runtime = _make_simple_subprocess_runtime()
        try:
            runtime.reset()
            with pytest.raises(Exception, match="mapping"):
                runtime.restore("not a mapping")
        finally:
            runtime.stop()

    def test_restore_rejects_missing_snapshot_key(self) -> None:
        runtime = _make_simple_subprocess_runtime()
        try:
            runtime.reset()
            with pytest.raises(Exception, match="agent_root_snapshot"):
                runtime.restore({"wrong_key": "x"})
        finally:
            runtime.stop()

    def test_restore_rejects_missing_snapshot_on_disk(self) -> None:
        runtime = _make_simple_subprocess_runtime()
        try:
            runtime.reset()
            with pytest.raises(Exception, match="snapshot missing"):
                runtime.restore({"agent_root_snapshot": "/nonexistent/path"})
        finally:
            runtime.stop()

    def test_reset_twice_restarts_subprocess(self) -> None:
        runtime = _make_simple_subprocess_runtime()
        try:
            runtime.reset()
            first_agent_root = runtime.surface()["agent_root"]
            runtime.reset()
            second_agent_root = runtime.surface()["agent_root"]
            assert first_agent_root != second_agent_root
        finally:
            runtime.stop()

    def test_stop_is_idempotent(self) -> None:
        runtime = _make_simple_subprocess_runtime()
        runtime.reset()
        runtime.stop()
        runtime.stop()
        assert runtime.terminal() == (False, None)

    def test_silent_subprocess_has_empty_startup_info(self) -> None:
        runtime = _make_silent_subprocess_runtime()
        try:
            runtime.reset()
            surface = runtime.surface()
            assert "agent_root" in surface
            assert "ready" not in surface
        finally:
            runtime.stop()

    def test_poll_events_default_empty(self) -> None:
        runtime = _make_simple_subprocess_runtime()
        try:
            runtime.reset()
            assert runtime.poll_events() == ()
        finally:
            runtime.stop()

    def test_subprocess_that_exits_immediately_has_empty_startup(self) -> None:
        from openrange_pack_sdk import SubprocessRuntime

        class _Exits(SubprocessRuntime):
            def prepare_env_files(self, graph: WorldGraph) -> Mapping[str, str]:
                del graph
                return {}

            def subprocess_command(
                self, env_root: Any, agent_root: Any
            ) -> Sequence[str]:
                import sys

                del env_root, agent_root
                # Closes stdout without writing → reset's readline returns "".
                return [sys.executable, "-c", "import sys; sys.stdout.close()"]

        runtime = _Exits(_empty_graph())
        try:
            runtime.reset()
            surface = runtime.surface()
            assert "agent_root" in surface
        finally:
            runtime.stop()

    def test_restore_replaces_directory_under_agent_root(self) -> None:
        from pathlib import Path

        runtime = _make_simple_subprocess_runtime()
        try:
            runtime.reset()
            agent_root = Path(runtime.surface()["agent_root"])
            (agent_root / "sub").mkdir()
            (agent_root / "sub" / "f.txt").write_text("snapshot")
            ckpt = runtime.checkpoint()
            (agent_root / "sub" / "f.txt").write_text("modified")
            runtime.restore(ckpt)
            assert (agent_root / "sub" / "f.txt").read_text() == "snapshot"
        finally:
            runtime.stop()

    def test_restore_raises_when_called_before_reset(self) -> None:
        import tempfile
        from pathlib import Path

        runtime = _make_simple_subprocess_runtime()
        # Build a valid-looking snapshot on disk without ever calling reset.
        snap = Path(tempfile.mkdtemp(prefix="ckpt-fake-"))
        (snap / "agent").mkdir()
        try:
            with pytest.raises(Exception, match="before reset"):
                runtime.restore({"agent_root_snapshot": str(snap)})
        finally:
            import shutil

            shutil.rmtree(snap, ignore_errors=True)

    def test_collect_extras_default_is_empty(self) -> None:
        from openrange_pack_sdk import SubprocessRuntime

        class _Defaults(SubprocessRuntime):
            def prepare_env_files(self, graph: WorldGraph) -> Mapping[str, str]:
                del graph
                return {}

            def subprocess_command(
                self, env_root: Any, agent_root: Any
            ) -> Sequence[str]:
                import sys

                del env_root, agent_root
                return [
                    sys.executable,
                    "-c",
                    "import sys, time\nsys.stdout.write('\\n')\n"
                    "sys.stdout.flush()\nwhile True: time.sleep(0.1)\n",
                ]

        runtime = _Defaults(_empty_graph())
        try:
            runtime.reset()
            collected = runtime.collect()
            assert "agent_root" in collected
            assert collected["result"] == {}
            # collect_extras default → no extra keys beyond agent_root + result.
            assert set(collected.keys()) == {"agent_root", "result"}
        finally:
            runtime.stop()

    def test_subprocess_env_override(self) -> None:
        from openrange_pack_sdk import SubprocessRuntime

        captured: dict[str, str] = {}

        class _EnvRuntime(SubprocessRuntime):
            def prepare_env_files(self, graph: WorldGraph) -> Mapping[str, str]:
                del graph
                return {}

            def subprocess_command(
                self, env_root: Any, agent_root: Any
            ) -> Sequence[str]:
                import sys

                del env_root, agent_root
                return [
                    sys.executable,
                    "-c",
                    "import json, os, sys, time\n"
                    "payload = {'env': os.environ.get('PACK_X', '')}\n"
                    "sys.stdout.write(json.dumps(payload) + '\\n')\n"
                    "sys.stdout.flush()\n"
                    "while True: time.sleep(0.1)\n",
                ]

            def subprocess_env(self) -> Mapping[str, str]:
                return {"PACK_X": "from_pack"}

            def parse_startup(self, stdout_line: str) -> Mapping[str, Any]:
                import json as _json

                parsed: Mapping[str, Any] = dict(_json.loads(stdout_line))
                captured.update(parsed)
                return parsed

        runtime = _EnvRuntime(_empty_graph())
        try:
            runtime.reset()
            assert captured["env"] == "from_pack"
        finally:
            runtime.stop()

    def test_terminal_false_after_reset_when_no_result_written(self) -> None:
        runtime = _make_simple_subprocess_runtime()
        try:
            runtime.reset()
            ok, reason = runtime.terminal()
            assert ok is False
            assert reason is None
        finally:
            runtime.stop()

    def test_reset_preserves_existing_checkpoints(self) -> None:
        from pathlib import Path

        runtime = _make_simple_subprocess_runtime()
        try:
            runtime.reset()
            agent_root = Path(runtime.surface()["agent_root"])
            (agent_root / "scratch.txt").write_text("before-restart")
            ckpt = runtime.checkpoint()
            snap_path = Path(ckpt["agent_root_snapshot"])
            assert snap_path.exists()
            # reset() must NOT wipe the snapshot dir — restore() flows
            # like the webapp pack's call reset() before super().restore().
            runtime.reset()
            assert snap_path.exists()
            runtime.restore(ckpt)
            new_agent_root = Path(runtime.surface()["agent_root"])
            assert (new_agent_root / "scratch.txt").read_text() == "before-restart"
        finally:
            runtime.stop()

    def test_stop_clears_checkpoint_dirs(self) -> None:
        from pathlib import Path

        runtime = _make_simple_subprocess_runtime()
        runtime.reset()
        runtime.checkpoint()
        runtime.checkpoint()
        snap_paths = [
            Path(c)
            for c in [
                # _checkpoint_dirs holds Paths internally.
                str(p)
                for p in runtime._checkpoint_dirs
            ]
        ]
        assert all(p.exists() for p in snap_paths)
        runtime.stop()
        assert not any(p.exists() for p in snap_paths)

    def test_public_properties_are_none_before_reset(self) -> None:
        runtime = _make_simple_subprocess_runtime()
        assert runtime.env_root is None
        assert runtime.agent_root is None
        assert runtime.pack_root is None
        assert runtime.process is None

    def test_public_properties_after_reset(self) -> None:
        from pathlib import Path

        runtime = _make_simple_subprocess_runtime()
        try:
            runtime.reset()
            assert isinstance(runtime.env_root, Path)
            assert isinstance(runtime.agent_root, Path)
            assert isinstance(runtime.pack_root, Path)
            assert runtime.process is not None
            assert runtime.process.pid > 0
        finally:
            runtime.stop()

    def test_startup_timeout_raises_when_subprocess_silent(self) -> None:
        from openrange_pack_sdk import OpenRangeError, SubprocessRuntime

        class _NeverPrints(SubprocessRuntime):
            STARTUP_TIMEOUT_SECONDS = 0.2

            def prepare_env_files(self, graph: WorldGraph) -> Mapping[str, str]:
                del graph
                return {}

            def subprocess_command(
                self, env_root: Any, agent_root: Any
            ) -> Sequence[str]:
                import sys

                del env_root, agent_root
                # Sleeps without writing → readline must time out.
                return [sys.executable, "-c", "import time; time.sleep(5)"]

        runtime = _NeverPrints(_empty_graph())
        try:
            with pytest.raises(OpenRangeError, match="startup line"):
                runtime.reset()
        finally:
            runtime.stop()

    def test_stop_escalates_to_sigkill_when_sigterm_ignored(self) -> None:
        from openrange_pack_sdk import SubprocessRuntime

        class _IgnoresSigterm(SubprocessRuntime):
            GRACE_SECONDS = 0.2

            def prepare_env_files(self, graph: WorldGraph) -> Mapping[str, str]:
                del graph
                return {}

            def subprocess_command(
                self, env_root: Any, agent_root: Any
            ) -> Sequence[str]:
                import sys

                del env_root, agent_root
                return [
                    sys.executable,
                    "-c",
                    "import signal, sys, time\n"
                    "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
                    "sys.stdout.write('\\n')\n"
                    "sys.stdout.flush()\n"
                    "while True: time.sleep(0.1)\n",
                ]

        runtime = _IgnoresSigterm(_empty_graph())
        runtime.reset()
        process = runtime._process
        assert process is not None
        runtime.stop()
        # SIGKILL eventually reaped it — Popen.poll returns the exit code.
        assert process.poll() is not None

    def test_subprocess_popen_kwargs_adds_stdin_pipe(self) -> None:
        """Trading-style pack pattern: open stdin for two-way comms."""
        import json as _json
        import subprocess as _subprocess

        from openrange_pack_sdk import SubprocessRuntime

        class _StdinEcho(SubprocessRuntime):
            def prepare_env_files(self, graph: WorldGraph) -> Mapping[str, str]:
                del graph
                return {}

            def subprocess_command(
                self, env_root: Any, agent_root: Any
            ) -> Sequence[str]:
                import sys

                del env_root, agent_root
                # Reads one JSON line from stdin, echoes it on stdout
                # with an "echoed" marker; prints startup line first.
                return [
                    sys.executable,
                    "-c",
                    "import json, sys\n"
                    "sys.stdout.write('{}\\n')\n"
                    "sys.stdout.flush()\n"
                    "line = sys.stdin.readline()\n"
                    "payload = json.loads(line)\n"
                    "payload['echoed'] = True\n"
                    "sys.stdout.write(json.dumps(payload) + '\\n')\n"
                    "sys.stdout.flush()\n",
                ]

            def subprocess_popen_kwargs(self) -> Mapping[str, Any]:
                return {"stdin": _subprocess.PIPE}

        runtime = _StdinEcho(_empty_graph())
        try:
            runtime.reset()
            proc = runtime.process
            assert proc is not None
            assert proc.stdin is not None
            proc.stdin.write(_json.dumps({"hello": "world"}) + "\n")
            proc.stdin.flush()
            assert proc.stdout is not None
            response = _json.loads(proc.stdout.readline())
            assert response == {"hello": "world", "echoed": True}
        finally:
            runtime.stop()


def _make_simple_ondemand_runtime() -> Any:
    from openrange_pack_sdk import OnDemandRuntime

    class _SimpleOnDemand(OnDemandRuntime):
        def prepare_env_files(self, graph: WorldGraph) -> Mapping[str, str]:
            del graph
            return {"README.md": "# project", "src/main.py": "print('hi')\n"}

        def surface_extras(self) -> Mapping[str, Any]:
            return {"hello": "from on-demand pack"}

        def collect_extras(self) -> Mapping[str, Any]:
            return {"finalized_by": "ondemand"}

    return _SimpleOnDemand(_empty_graph())


class TestOnDemandRuntime:
    def test_full_lifecycle(self) -> None:
        runtime = _make_simple_ondemand_runtime()
        try:
            runtime.reset()
            surface = runtime.surface()
            assert "agent_root" in surface
            assert surface["hello"] == "from on-demand pack"
            agent_root = Path(surface["agent_root"])
            assert agent_root.is_dir()
            pack_root = runtime.pack_root
            assert pack_root is not None
            assert (pack_root / "README.md").read_text() == "# project"
            assert (pack_root / "src" / "main.py").read_text() == "print('hi')\n"
        finally:
            runtime.stop()

    def test_surface_raises_before_reset(self) -> None:
        runtime = _make_simple_ondemand_runtime()
        with pytest.raises(Exception, match="reset"):
            runtime.surface()

    def test_terminal_false_before_reset(self) -> None:
        runtime = _make_simple_ondemand_runtime()
        ok, reason = runtime.terminal()
        assert ok is False and reason is None

    def test_terminal_true_when_agent_writes_result(self) -> None:
        runtime = _make_simple_ondemand_runtime()
        try:
            runtime.reset()
            agent_root = Path(runtime.surface()["agent_root"])
            (agent_root / "result.json").write_text('{"status": "ok"}')
            ok, reason = runtime.terminal()
            assert ok is True
            assert reason == "agent wrote result"
        finally:
            runtime.stop()

    def test_collect_returns_result_and_extras(self) -> None:
        runtime = _make_simple_ondemand_runtime()
        try:
            runtime.reset()
            agent_root = Path(runtime.surface()["agent_root"])
            (agent_root / "result.json").write_text('{"passed": 3, "failed": 1}')
            collected = runtime.collect()
            assert collected["result"] == {"passed": 3, "failed": 1}
            assert collected["finalized_by"] == "ondemand"
        finally:
            runtime.stop()

    def test_checkpoint_restore_round_trips(self) -> None:
        runtime = _make_simple_ondemand_runtime()
        try:
            runtime.reset()
            agent_root = Path(runtime.surface()["agent_root"])
            (agent_root / "scratch.txt").write_text("v1")
            ckpt = runtime.checkpoint()
            (agent_root / "scratch.txt").write_text("v2")
            runtime.restore(ckpt)
            assert (agent_root / "scratch.txt").read_text() == "v1"
        finally:
            runtime.stop()

    def test_stop_drops_checkpoints(self) -> None:
        runtime = _make_simple_ondemand_runtime()
        runtime.reset()
        ckpt = runtime.checkpoint()
        snap_path = Path(ckpt["agent_root_snapshot"])
        assert snap_path.exists()
        runtime.stop()
        assert not snap_path.exists()

    def test_reset_preserves_checkpoints(self) -> None:
        runtime = _make_simple_ondemand_runtime()
        try:
            runtime.reset()
            ckpt = runtime.checkpoint()
            snap_path = Path(ckpt["agent_root_snapshot"])
            assert snap_path.exists()
            runtime.reset()
            assert snap_path.exists()
        finally:
            runtime.stop()

    def test_poll_events_default_empty(self) -> None:
        runtime = _make_simple_ondemand_runtime()
        try:
            runtime.reset()
            assert runtime.poll_events() == ()
        finally:
            runtime.stop()

    def test_swe_style_run_cmd_callable(self, tmp_path: Path) -> None:
        """End-to-end exercise of the SWE-pack pattern: harness exposes
        a run_cmd callable that shells out against the agent_root."""
        import subprocess as _subprocess

        from openrange_pack_sdk import OnDemandRuntime

        class _SWEPack(OnDemandRuntime):
            def prepare_env_files(self, graph: WorldGraph) -> Mapping[str, str]:
                del graph
                return {}

            def surface_extras(self) -> Mapping[str, Any]:
                agent_root = self.agent_root

                def run_cmd(argv: Sequence[str]) -> Mapping[str, Any]:
                    completed = _subprocess.run(
                        list(argv),
                        cwd=agent_root,
                        capture_output=True,
                        text=True,
                        timeout=5,
                    )
                    return {
                        "rc": completed.returncode,
                        "stdout": completed.stdout,
                        "stderr": completed.stderr,
                    }

                return {"run_cmd": run_cmd}

        runtime = _SWEPack(_empty_graph())
        try:
            runtime.reset()
            agent_root = Path(runtime.surface()["agent_root"])
            (agent_root / "hello.py").write_text("print('hello')")
            run_cmd = runtime.surface()["run_cmd"]
            result = run_cmd(["python3", "hello.py"])
            assert result["rc"] == 0
            assert "hello" in result["stdout"]
        finally:
            runtime.stop()
