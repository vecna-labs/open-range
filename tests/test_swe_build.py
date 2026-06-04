"""SWE *build* tasks (``swe.build``): the long-horizon sibling of ``swe.fix`` on
the same ``swe.repo@v1`` world. The held-out suite splits into unit tests that
**shape** (dense partial credit) and integration tests that **gate** success.

These tests pin the contract: the two families self-select on the suite shape so
exactly one claims a given world; admission's self-test proves the gold overlay
composes (greens every tier) while the bare skeleton fails the integration gate;
``check_success`` gates on integration but reports both tiers as subgoals; and
the training seam turns a half-built (units-only) episode into partial credit
*without* success — the dense signal that makes a long episode learnable.

Hermetic: the fixture ships a self-contained micro-project and grading shells out
to this interpreter's pytest — no network, no clone.
"""

from __future__ import annotations

from dataclasses import replace

from graphschema import WorldGraph, validate
from openrange_pack_sdk import TaskSpec
from swe import SweBuild, SweFix, SwePack, repo_ontology
from swe.builder import SweBuilder
from swe.instances import SweInstance, load_instance, to_graph

from openrange.core.episode import EpisodeReport
from openrange.training import episode_reward

_BUILD = "notes_app"
_FIX = "calc_sum"
_UNITS = (
    "tests/test_unit.py::test_store_put_get",
    "tests/test_unit.py::test_store_overwrite",
    "tests/test_unit.py::test_service_render",
)
_INTEG = (
    "tests/test_integration.py::test_add_then_get_roundtrip",
    "tests/test_integration.py::test_persists_via_store",
)

# A half-built service: each piece passes its unit test, but the pieces don't
# compose — add() returns the rendered note instead of an id and never persists,
# get() echoes its argument. Every unit test passes; every integration fails.
_UNITS_ONLY_SERVICE = (
    "class NoteService:\n"
    "    def __init__(self, store):\n"
    "        self.store = store\n"
    "        self._next = 0\n\n"
    "    def render(self, title, body):\n"
    '        return f"{title}: {body}"\n\n'
    "    def add(self, title, body):\n"
    "        return self.render(title, body)\n\n"
    "    def get(self, note_id):\n"
    "        return note_id\n"
)


def _graph_and_task() -> tuple[WorldGraph, TaskSpec]:
    graph = to_graph(load_instance(_BUILD))
    task = SweBuild().generate(graph, {}, None)[0]
    return graph, task


def _gold_tree(instance: SweInstance) -> dict[str, str]:
    return {**instance.base_files, **instance.gold_files}


def _units_only_tree(instance: SweInstance) -> dict[str, str]:
    return {**_gold_tree(instance), "notes/service.py": _UNITS_ONLY_SERVICE}


def _success(graph: WorldGraph, task: TaskSpec, tree: dict[str, str]):  # type: ignore[no-untyped-def]
    return SweBuild().check_success(
        graph, task, {"workspace_files": tree, "result": {"done": True}}
    )


def _error_codes(graph: WorldGraph) -> set[str]:
    return {i.code for i in validate(graph, repo_ontology(), SwePack().invariants())}


class TestFamilySelection:
    """Each family claims its own world shape; the builder runs both, so exactly
    one task is emitted per instance."""

    def test_build_world_emits_only_build_task(self) -> None:
        graph = to_graph(load_instance(_BUILD))
        assert len(SweBuild().generate(graph, {}, None)) == 1
        assert SweFix().generate(graph, {}, None) == []

    def test_fix_world_emits_only_fix_task(self) -> None:
        graph = to_graph(load_instance(_FIX))
        assert len(SweFix().generate(graph, {}, None)) == 1
        assert SweBuild().generate(graph, {}, None) == []

    def test_builder_emits_single_build_task(self) -> None:
        build = SweBuilder().build({"instance": _BUILD})
        assert [t.id for t in build.tasks] == ["swe.build.notes"]

    def test_task_wired_to_repo_and_suite(self) -> None:
        graph, task = _graph_and_task()
        repo = graph.by_kind("repo")[0]
        suite = graph.by_kind("test_suite")[0]
        assert task.entrypoints == (repo.id,)
        assert task.goal_nodes == (suite.id,)


class TestAdmission:
    def test_build_world_admits_through_all_layers(self) -> None:
        from openrange.core.admit import AdmissionFailure, admit

        result = admit(SwePack(), manifest={"instance": _BUILD}, max_repairs=0)
        assert not isinstance(result, AdmissionFailure)
        assert result.ontology_id == "swe.repo@v1"
        assert [t.id for t in result.tasks] == ["swe.build.notes"]


class TestInvariants:
    def test_build_only_world_passes_all(self) -> None:
        # A suite with no fail_to_pass but a non-empty integration gate is
        # well-formed — it grades the build.
        graph = to_graph(load_instance(_BUILD))
        errors = [
            i
            for i in validate(graph, repo_ontology(), SwePack().invariants())
            if i.severity == "error"
        ]
        assert errors == []

    def test_suite_grading_nothing_is_flagged(self) -> None:
        # Strip the integration gate from a build-only suite (its f2p is already
        # empty): now it grades nothing.
        instance = replace(load_instance(_BUILD), integration_tests=())
        assert "suite_grades_nothing" in _error_codes(to_graph(instance))

    def test_dangling_unit_test_id_is_flagged(self) -> None:
        graph = to_graph(load_instance(_BUILD))
        suite = graph.by_kind("test_suite")[0]
        suite.attrs["unit_tests"] = ["nope/test_missing.py::test_x"]
        assert "suite_test_id_dangling" in _error_codes(graph)

    def test_unit_integration_overlap_is_flagged(self) -> None:
        # A test id can shape (unit) or gate (integration), never both — the
        # tier overlap mirrors the existing F2P/P2P disjointness check.
        graph = to_graph(load_instance(_BUILD))
        suite = graph.by_kind("test_suite")[0]
        suite.attrs["integration_tests"] = [_UNITS[0], *_INTEG]
        assert "suite_tier_overlap" in _error_codes(graph)


class TestFeasibility:
    def test_build_world_is_feasible(self) -> None:
        graph, task = _graph_and_task()
        verdict = SweBuild().check_feasibility(graph, task)
        assert verdict.feasible, verdict.reason

    def test_precomposed_skeleton_is_infeasible(self) -> None:
        # If the skeleton already composes (base == gold), the integration gate
        # isn't real — the base-must-fail-integration half rejects the world.
        instance = load_instance(_BUILD)
        composed = replace(instance, base_files=_gold_tree(instance))
        graph = to_graph(composed)
        task = SweBuild().generate(graph, {}, None)[0]
        verdict = SweBuild().check_feasibility(graph, task)
        assert not verdict.feasible
        assert "skeleton already passes an integration test" in verdict.reason

    def test_non_composing_gold_is_infeasible(self) -> None:
        # A gold overlay that passes units but fails integration doesn't prove
        # the build is solvable.
        instance = load_instance(_BUILD)
        bad_gold = replace(
            instance,
            gold_files={**instance.gold_files, "notes/service.py": _UNITS_ONLY_SERVICE},
        )
        graph = to_graph(bad_gold)
        task = SweBuild().generate(graph, {}, None)[0]
        verdict = SweBuild().check_feasibility(graph, task)
        assert not verdict.feasible
        assert "gold overlay does not green" in verdict.reason


class TestGrading:
    def test_gold_tree_resolves_every_tier(self) -> None:
        instance = load_instance(_BUILD)
        graph, task = _graph_and_task()
        result = _success(graph, task, _gold_tree(instance))
        assert result.success
        assert all(result.subgoals.values())
        assert set(result.subgoals) == {*_UNITS, *_INTEG}

    def test_units_only_scores_but_does_not_resolve(self) -> None:
        instance = load_instance(_BUILD)
        graph, task = _graph_and_task()
        result = _success(graph, task, _units_only_tree(instance))
        # Integration gates: every unit passes, every integration fails, so the
        # episode does NOT resolve despite high partial credit.
        assert not result.success
        assert all(result.subgoals[t] is True for t in _UNITS)
        assert all(result.subgoals[t] is False for t in _INTEG)

    def test_skeleton_resolves_nothing(self) -> None:
        instance = load_instance(_BUILD)
        graph, task = _graph_and_task()
        result = _success(graph, task, dict(instance.base_files))
        assert not result.success
        assert not any(result.subgoals.values())

    def test_empty_workspace_fails_cleanly(self) -> None:
        graph, task = _graph_and_task()
        result = SweBuild().check_success(graph, task, {"result": {"done": True}})
        assert not result.success
        assert "no workspace" in result.reason


class TestRewardSeam:
    """The training seam turns the build's subgoal vector into a dense reward:
    units shape the partial credit, integration gates the 1.0."""

    def _reward(self, tree: dict[str, str]):  # type: ignore[no-untyped-def]
        graph, task = _graph_and_task()
        result = _success(graph, task, tree)
        report = EpisodeReport(
            snapshot_id="sha256:test", task_id=task.id, episode_result=result
        )
        return episode_reward(report), result

    def test_gold_earns_full_reward(self) -> None:
        reward, result = self._reward(_gold_tree(load_instance(_BUILD)))
        assert result.success
        assert reward.scalar == 1.0

    def test_units_only_earns_unit_fraction(self) -> None:
        reward, result = self._reward(_units_only_tree(load_instance(_BUILD)))
        assert not result.success
        # 3 unit + 2 integration subgoals; 3 pass -> 3/5 partial credit.
        assert reward.scalar == 0.6
        assert len(reward.components) == 5
        assert sum(1 for v in reward.components.values() if v >= 1.0) == 3
        assert sum(1 for v in reward.components.values() if v == 0.0) == 2

    def test_skeleton_earns_zero(self) -> None:
        reward, result = self._reward(dict(load_instance(_BUILD).base_files))
        assert not result.success
        assert reward.scalar == 0.0


class TestInstanceLoader:
    def test_load_build_instance_shape(self) -> None:
        instance = load_instance(_BUILD)
        assert isinstance(instance, SweInstance)
        assert instance.fail_to_pass == ()
        assert instance.unit_tests == _UNITS
        assert instance.integration_tests == _INTEG

    def test_to_graph_carries_tiers(self) -> None:
        suite = to_graph(load_instance(_BUILD)).by_kind("test_suite")[0]
        assert suite.attrs["unit_tests"] == list(_UNITS)
        assert suite.attrs["integration_tests"] == list(_INTEG)
