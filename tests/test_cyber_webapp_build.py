"""Integration tests for the ``webapp.build`` TaskFamily.

Real subprocess sandbox, real admitted snapshots, real graphs. No mocks.
"""

from __future__ import annotations

from typing import Any

import pytest
from cyber_webapp import WebappPack
from cyber_webapp.families.build import (
    _KIND_GENERATORS,
    KindGenerators,
    KindSpec,
    WebappBuild,
)
from cyber_webapp.families.build.contracts import (
    ContractCase,
    _check_items_list,
    _content_type_is_json,
    _parse_json_body,
    api_list_contract,
)
from cyber_webapp.families.build.grading import (
    CaseResult,
    ContractReport,
    grade_source,
)
from cyber_webapp.families.build.mutations import api_wrong_field_name
from cyber_webapp.families.build.reference import api_list_reference
from graphschema import Edge, Node, Visibility, WorldGraph
from openrange_pack_sdk import Snapshot, TaskSpec

from openrange.core.admit import admit


@pytest.fixture(scope="module")
def webapp_snapshot() -> Snapshot:
    pack = WebappPack()
    result = admit(
        pack,
        manifest={
            "world": {"goal": "stage1 build test"},
            "pack": {"id": "webapp"},
            "runtime": {"tick": {"mode": "off"}},
            "npc": [],
        },
    )
    assert isinstance(result, Snapshot), result
    return result


@pytest.fixture(scope="module")
def webapp_build_task(webapp_snapshot: Snapshot) -> TaskSpec:
    tasks = [t for t in webapp_snapshot.tasks if t.meta.get("family") == "webapp.build"]
    assert len(tasks) == 1, tasks
    return tasks[0]


@pytest.fixture
def family() -> WebappBuild:
    return WebappBuild()


class TestReferenceContractCoherence:
    @pytest.mark.parametrize("level", [1, 2, 3])
    def test_reference_passes_contract_at_level(self, level: int) -> None:
        report = grade_source(api_list_reference(level), api_list_contract(level))
        failed = [(c.description, c.reason) for c in report.cases if not c.passed]
        assert report.all_passed, failed

    @pytest.mark.parametrize("level", [1, 2, 3])
    def test_mutation_breaks_contract_at_level(self, level: int) -> None:
        mutated = api_wrong_field_name(api_list_reference(level))
        report = grade_source(mutated, api_list_contract(level))
        assert report.passed == 0
        assert all(not c.passed for c in report.cases)

    def test_lower_level_reference_fails_higher_contract(self) -> None:
        # L1 omits "count" → fails L2; L2 omits sorting → fails L3. Proves
        # each level is a real difficulty step, not cosmetic.
        assert not grade_source(api_list_reference(1), api_list_contract(2)).all_passed
        assert not grade_source(api_list_reference(2), api_list_contract(3)).all_passed

    def test_mutation_actually_changes_source(self) -> None:
        ref = api_list_reference(1)
        mut = api_wrong_field_name(ref)
        assert ref != mut
        assert '"items"' not in mut or '"results"' in mut


class TestContractPredicate:
    def test_passes_on_correct_response(self) -> None:
        ok, why = _check_items_list(
            200,
            {"Content-Type": "application/json"},
            b'{"items": []}',
            frozenset(),
        )
        assert ok, why

    def test_fails_on_wrong_status(self) -> None:
        ok, why = _check_items_list(
            500,
            {"Content-Type": "application/json"},
            b'{"items": []}',
            frozenset(),
        )
        assert not ok
        assert "status 500" in why

    def test_fails_on_non_json_content_type(self) -> None:
        ok, why = _check_items_list(
            200,
            {"Content-Type": "text/html"},
            b'{"items": []}',
            frozenset(),
        )
        assert not ok
        assert "Content-Type" in why

    def test_fails_when_body_not_json(self) -> None:
        ok, why = _check_items_list(
            200,
            {"Content-Type": "application/json"},
            b"not json at all",
            frozenset(),
        )
        assert not ok
        assert "items" in why or "json" in why.lower()

    def test_fails_when_body_json_not_dict(self) -> None:
        ok, why = _check_items_list(
            200,
            {"Content-Type": "application/json"},
            b"[1, 2, 3]",
            frozenset(),
        )
        assert not ok

    def test_fails_when_items_field_missing(self) -> None:
        ok, why = _check_items_list(
            200,
            {"Content-Type": "application/json"},
            b'{"results": []}',
            frozenset(),
        )
        assert not ok
        assert "items" in why

    def test_fails_when_items_not_list(self) -> None:
        ok, why = _check_items_list(
            200,
            {"Content-Type": "application/json"},
            b'{"items": "nope"}',
            frozenset(),
        )
        assert not ok
        assert "list" in why

    def test_fails_when_item_not_mapping(self) -> None:
        ok, why = _check_items_list(
            200,
            {"Content-Type": "application/json"},
            b'{"items": [1, 2]}',
            frozenset({"a"}),
        )
        assert not ok
        assert "mapping" in why or "int" in why

    def test_fails_when_item_missing_string_id(self) -> None:
        ok, why = _check_items_list(
            200,
            {"Content-Type": "application/json"},
            b'{"items": [{"name": "x"}]}',
            frozenset({"a"}),
        )
        assert not ok
        assert "id" in why

    def test_fails_when_item_ids_mismatch(self) -> None:
        ok, why = _check_items_list(
            200,
            {"Content-Type": "application/json"},
            b'{"items": [{"id": "x"}]}',
            frozenset({"a", "b"}),
        )
        assert not ok
        assert "ids" in why

    def test_content_type_case_insensitive(self) -> None:
        assert _content_type_is_json(
            {"content-type": "Application/JSON; charset=utf-8"}
        )
        assert not _content_type_is_json({"content-type": "text/plain"})
        assert not _content_type_is_json({})

    def test_parse_json_body_handles_bad_utf8(self) -> None:
        assert _parse_json_body(b"\xff\xfe garbage") is None

    def test_parse_json_body_handles_bad_json(self) -> None:
        assert _parse_json_body(b'{"not closed') is None

    def test_parse_json_body_returns_parsed(self) -> None:
        assert _parse_json_body(b'{"a": 1}') == {"a": 1}


class TestGrading:
    def _trivial_case(self) -> ContractCase:
        return ContractCase(
            description="trivial",
            query={},
            state={"records": {}},
            predicate=lambda s, h, b: (s == 200, f"status {s}"),
        )

    def test_correct_handler_passes(self) -> None:
        src = (
            "def handle(query, state):\n"
            "    return 200, {'Content-Type': 'application/json'}, b'{}'\n"
        )
        report = grade_source(src, (self._trivial_case(),))
        assert report.all_passed
        assert report.cases[0].status == 200
        assert report.cases[0].body_preview != ""

    def test_handler_returning_str_body_is_encoded(self) -> None:
        src = (
            "def handle(query, state):\n"
            "    return 200, {'Content-Type': 'application/json'}, '{}' \n"
        )
        report = grade_source(src, (self._trivial_case(),))
        assert report.all_passed

    def test_binary_body_survives_roundtrip(self) -> None:
        binary_case = ContractCase(
            description="binary fidelity",
            query={},
            state={},
            predicate=lambda s, h, b: (b == bytes(range(256)), f"body={b!r}"),
        )
        src = (
            "def handle(query, state):\n"
            "    return 200, {'Content-Type': 'application/octet-stream'}, "
            "bytes(range(256))\n"
        )
        report = grade_source(src, (binary_case,))
        assert report.all_passed, report.cases[0].reason

    def test_syntax_error_in_source(self) -> None:
        report = grade_source("def handle(q, s): return ???", (self._trivial_case(),))
        assert report.passed == 0
        assert "source did not load" in report.cases[0].reason

    def test_missing_handle_definition(self) -> None:
        report = grade_source("x = 1\n", (self._trivial_case(),))
        assert report.passed == 0
        assert "no callable 'handle'" in report.cases[0].reason

    def test_handle_not_callable(self) -> None:
        report = grade_source("handle = 42\n", (self._trivial_case(),))
        assert report.passed == 0
        assert "no callable 'handle'" in report.cases[0].reason

    def test_handler_raises(self) -> None:
        src = "def handle(query, state):\n    raise RuntimeError('boom')\n"
        report = grade_source(src, (self._trivial_case(),))
        assert report.passed == 0
        assert "handler raised" in report.cases[0].reason
        assert "RuntimeError" in report.cases[0].reason

    def test_handler_wrong_return_shape(self) -> None:
        src = "def handle(query, state):\n    return 200\n"
        report = grade_source(src, (self._trivial_case(),))
        assert report.passed == 0
        assert "handler raised" in report.cases[0].reason

    def test_handler_returns_none_body(self) -> None:
        src = (
            "def handle(query, state):\n"
            "    return 200, {'Content-Type': 'application/json'}, None\n"
        )
        report = grade_source(src, (self._trivial_case(),))
        assert report.passed == 0

    def test_infinite_loop_times_out(self) -> None:
        src = "def handle(query, state):\n    while True:\n        pass\n"
        report = grade_source(src, (self._trivial_case(),), timeout=0.5)
        assert report.passed == 0
        assert "timed out" in report.cases[0].reason

    def test_predicate_failure_records_reason(self) -> None:
        src = (
            "def handle(query, state):\n"
            "    return 500, {'Content-Type': 'application/json'}, b'{}'\n"
        )
        report = grade_source(src, (self._trivial_case(),))
        assert report.passed == 0
        assert "status 500" in report.cases[0].reason

    def test_subprocess_crash_with_no_output(self) -> None:
        src = "import os\nos._exit(7)\ndef handle(q, s): pass\n"
        report = grade_source(src, (self._trivial_case(),))
        assert report.passed == 0
        assert "subprocess exited 7" in report.cases[0].reason

    def test_subprocess_writes_non_json_then_exits_zero(self) -> None:
        src = (
            "import os, sys\n"
            "sys.__stdout__.write('not-json-output')\n"
            "sys.__stdout__.flush()\n"
            "os._exit(0)\n"
            "def handle(q, s): pass\n"
        )
        report = grade_source(src, (self._trivial_case(),))
        assert report.passed == 0
        assert "non-JSON" in report.cases[0].reason


class TestContractReport:
    def test_all_passed_false_when_total_zero(self) -> None:
        report = ContractReport(passed=0, total=0, cases=())
        assert not report.all_passed

    def test_all_passed_true_when_full(self) -> None:
        case = CaseResult("x", True, "", 200, "ok")
        report = ContractReport(passed=1, total=1, cases=(case,))
        assert report.all_passed

    def test_all_passed_false_when_partial(self) -> None:
        case = CaseResult("x", True, "", 200, "ok")
        report = ContractReport(passed=1, total=2, cases=(case, case))
        assert not report.all_passed


class TestFamilyAgainstRealWorld:
    def test_generate_emits_one_api_build_task(
        self,
        webapp_snapshot: Snapshot,
        webapp_build_task: TaskSpec,
    ) -> None:
        assert webapp_build_task.id == "webapp.build.0"
        assert webapp_build_task.feasibility_check == "webapp.build"
        assert webapp_build_task.success_check == "webapp.build"
        assert webapp_build_task.meta["family"] == "webapp.build"
        assert webapp_build_task.meta["kind"] == "api"
        assert "items" in webapp_build_task.instruction
        endpoint = webapp_snapshot.graph.nodes[webapp_build_task.goal_nodes[0]]
        service = webapp_snapshot.graph.nodes[webapp_build_task.entrypoints[0]]
        assert endpoint.kind == "endpoint"
        assert service.kind == "service"
        assert service.attrs.get("kind") == "api"

    def test_check_feasibility_passes(
        self,
        family: WebappBuild,
        webapp_snapshot: Snapshot,
        webapp_build_task: TaskSpec,
    ) -> None:
        verdict = family.check_feasibility(webapp_snapshot.graph, webapp_build_task)
        assert verdict.feasible, verdict.reason

    def test_check_success_on_reference_impl(
        self,
        family: WebappBuild,
        webapp_snapshot: Snapshot,
        webapp_build_task: TaskSpec,
    ) -> None:
        final_state = {"result": {"endpoint_impl": api_list_reference(1)}}
        result = family.check_success(
            webapp_snapshot.graph, webapp_build_task, final_state
        )
        assert result.success
        assert all(result.subgoals.values())

    def test_check_success_on_mutated_impl(
        self,
        family: WebappBuild,
        webapp_snapshot: Snapshot,
        webapp_build_task: TaskSpec,
    ) -> None:
        mutated = api_wrong_field_name(api_list_reference(1))
        final_state = {"result": {"endpoint_impl": mutated}}
        result = family.check_success(
            webapp_snapshot.graph, webapp_build_task, final_state
        )
        assert not result.success
        assert not any(result.subgoals.values())

    def test_check_success_no_result(
        self,
        family: WebappBuild,
        webapp_snapshot: Snapshot,
        webapp_build_task: TaskSpec,
    ) -> None:
        result = family.check_success(webapp_snapshot.graph, webapp_build_task, {})
        assert not result.success
        assert "result.json" in result.reason

    def test_check_success_result_not_mapping(
        self,
        family: WebappBuild,
        webapp_snapshot: Snapshot,
        webapp_build_task: TaskSpec,
    ) -> None:
        result = family.check_success(
            webapp_snapshot.graph, webapp_build_task, {"result": "not a dict"}
        )
        assert not result.success
        assert "result.json" in result.reason

    def test_check_success_missing_endpoint_impl(
        self,
        family: WebappBuild,
        webapp_snapshot: Snapshot,
        webapp_build_task: TaskSpec,
    ) -> None:
        result = family.check_success(
            webapp_snapshot.graph, webapp_build_task, {"result": {}}
        )
        assert not result.success
        assert "endpoint_impl" in result.reason

    def test_check_success_empty_endpoint_impl(
        self,
        family: WebappBuild,
        webapp_snapshot: Snapshot,
        webapp_build_task: TaskSpec,
    ) -> None:
        result = family.check_success(
            webapp_snapshot.graph,
            webapp_build_task,
            {"result": {"endpoint_impl": "   "}},
        )
        assert not result.success
        assert "endpoint_impl" in result.reason

    def test_check_success_non_string_endpoint_impl(
        self,
        family: WebappBuild,
        webapp_snapshot: Snapshot,
        webapp_build_task: TaskSpec,
    ) -> None:
        result = family.check_success(
            webapp_snapshot.graph,
            webapp_build_task,
            {"result": {"endpoint_impl": 42}},
        )
        assert not result.success
        assert "endpoint_impl" in result.reason

    def test_available_mutations_offers_harden_at_base_level(
        self,
        family: WebappBuild,
        webapp_snapshot: Snapshot,
    ) -> None:
        # The admitted world's endpoint defaults to level 1, so the only
        # move is harden (raise to level 2); soften has no room.
        muts = family.available_mutations(webapp_snapshot, ())
        directions = {m.direction for m in muts}
        assert directions == {"harden"}
        assert all(m.family == "webapp.build" for m in muts)


def _empty_graph() -> WorldGraph:
    from cyber_webapp.ontology import ONTOLOGY_ID

    return WorldGraph(ontology=ONTOLOGY_ID)


def _node(node_id: str, node_kind: str, **attrs: Any) -> Node:
    return Node(
        id=node_id,
        kind=node_kind,
        attrs=attrs,
        roles=set(),
        visibility=Visibility.PUBLIC,
        runtime={},
        meta={},
    )


def _edge(edge_id: str, kind: str, src: str, dst: str) -> Edge:
    return Edge(id=edge_id, kind=kind, src=src, dst=dst, attrs={})


def _api_world() -> WorldGraph:
    graph = _empty_graph()
    graph.nodes["svc_api"] = _node("svc_api", "service", kind="api", name="api")
    graph.nodes["ep_a"] = _node("ep_a", "endpoint", path="/x", method="GET")
    graph.edges["e1"] = _edge("e1", "exposes", "svc_api", "ep_a")
    return graph


def _api_task() -> TaskSpec:
    return TaskSpec(
        id="t",
        instruction="x",
        entrypoints=("svc_api",),
        goal_nodes=("ep_a",),
        feasibility_check="webapp.build",
        success_check="webapp.build",
    )


def _api_world_at_level(level: int) -> WorldGraph:
    graph = _empty_graph()
    graph.nodes["svc_api"] = _node("svc_api", "service", kind="api", name="api")
    graph.nodes["ep_a"] = _node(
        "ep_a", "endpoint", path="/x", method="GET", build_level=level
    )
    graph.edges["e1"] = _edge("e1", "exposes", "svc_api", "ep_a")
    return graph


def _snapshot(graph: WorldGraph) -> Snapshot:
    return Snapshot(
        snapshot_id="",
        ontology_id=graph.ontology,
        graph=graph,
        tasks=(),
        lineage={},
    )


class TestBuildCurriculum:
    def test_available_mutations_ignores_llm(self, family: WebappBuild) -> None:
        # Build curriculum is procedural; the offense-flavored LLM the pentest
        # family enriches with has no build signal and must not gate these.
        snap = _snapshot(_api_world_at_level(1))
        without = family.available_mutations(snap, ())
        with_llm = family.available_mutations(snap, (), llm=object())
        assert [m.direction for m in without] == [m.direction for m in with_llm]
        assert without and all(m.relevance > 0 for m in without)

    def test_level_ladder_directions(self, family: WebappBuild) -> None:
        def dirs(level: int) -> set[str]:
            snap = _snapshot(_api_world_at_level(level))
            return {m.direction for m in family.available_mutations(snap, ())}

        assert dirs(1) == {"harden"}  # floor: only up
        assert dirs(2) == {"harden", "soften"}
        assert dirs(3) == {"soften"}  # cap: only down

    def test_harden_patch_raises_build_level(self, family: WebappBuild) -> None:
        from graphschema import apply_patch

        snap = _snapshot(_api_world_at_level(1))
        harden = next(
            m for m in family.available_mutations(snap, ()) if m.direction == "harden"
        )
        graph = _api_world_at_level(1)
        apply_patch(graph, harden.patch)
        assert graph.nodes["ep_a"].attrs["build_level"] == 2

    def test_available_mutations_empty_when_no_target(
        self, family: WebappBuild
    ) -> None:
        assert family.available_mutations(_snapshot(_empty_graph()), ()) == ()

    def test_generate_instruction_reflects_level(self, family: WebappBuild) -> None:
        l2 = family.generate(_api_world_at_level(2), manifest={}, prior=None)[0]
        assert l2.meta["build_level"] == 2
        assert '"count"' in l2.instruction
        assert "ascending" not in l2.instruction

        l3 = family.generate(_api_world_at_level(3), manifest={}, prior=None)[0]
        assert l3.meta["build_level"] == 3
        assert "ascending" in l3.instruction


class TestPickTarget:
    def test_no_services_returns_none(self, family: WebappBuild) -> None:
        assert family._pick_target(_empty_graph()) is None

    def test_only_unsupported_kind_returns_none(self, family: WebappBuild) -> None:
        graph = _empty_graph()
        graph.nodes["svc_db"] = _node("svc_db", "service", kind="db", name="db")
        assert family._pick_target(graph) is None

    def test_supported_service_without_endpoints_returns_none(
        self,
        family: WebappBuild,
    ) -> None:
        graph = _empty_graph()
        graph.nodes["svc_api"] = _node("svc_api", "service", kind="api", name="api")
        assert family._pick_target(graph) is None

    def test_supported_service_with_endpoint_returns_target(
        self,
        family: WebappBuild,
    ) -> None:
        target = family._pick_target(_api_world())
        assert target is not None
        assert target.endpoint.id == "ep_a"
        assert target.service.id == "svc_api"
        assert target.kind == "api"
        assert target.level == 1

    def test_exposes_edge_to_missing_or_non_endpoint_is_skipped(
        self,
        family: WebappBuild,
    ) -> None:
        graph = _empty_graph()
        graph.nodes["svc_api"] = _node("svc_api", "service", kind="api", name="api")
        graph.nodes["host_x"] = _node("host_x", "host")
        graph.edges["e1"] = _edge("e1", "exposes", "svc_api", "host_x")
        graph.edges["e2"] = _edge("e2", "exposes", "svc_api", "ghost")
        assert family._pick_target(graph) is None


class TestResolveTargetBranches:
    def test_missing_entrypoint(self, family: WebappBuild) -> None:
        task = TaskSpec(
            id="t",
            instruction="x",
            entrypoints=(),
            goal_nodes=("ep_a",),
            feasibility_check="webapp.build",
            success_check="webapp.build",
        )
        verdict = family.check_feasibility(_api_world(), task)
        assert not verdict.feasible
        assert "entrypoint" in verdict.reason or "goal" in verdict.reason

    def test_missing_goal(self, family: WebappBuild) -> None:
        task = TaskSpec(
            id="t",
            instruction="x",
            entrypoints=("svc_api",),
            goal_nodes=(),
            feasibility_check="webapp.build",
            success_check="webapp.build",
        )
        verdict = family.check_feasibility(_api_world(), task)
        assert not verdict.feasible

    def test_entrypoint_node_missing(self, family: WebappBuild) -> None:
        task = TaskSpec(
            id="t",
            instruction="x",
            entrypoints=("ghost",),
            goal_nodes=("ep_a",),
            feasibility_check="webapp.build",
            success_check="webapp.build",
        )
        verdict = family.check_feasibility(_api_world(), task)
        assert not verdict.feasible
        assert "service" in verdict.reason

    def test_entrypoint_not_service_kind(self, family: WebappBuild) -> None:
        graph = _api_world()
        graph.nodes["host_x"] = _node("host_x", "host")
        task = TaskSpec(
            id="t",
            instruction="x",
            entrypoints=("host_x",),
            goal_nodes=("ep_a",),
            feasibility_check="webapp.build",
            success_check="webapp.build",
        )
        verdict = family.check_feasibility(graph, task)
        assert not verdict.feasible

    def test_goal_not_endpoint(self, family: WebappBuild) -> None:
        task = TaskSpec(
            id="t",
            instruction="x",
            entrypoints=("svc_api",),
            goal_nodes=("svc_api",),
            feasibility_check="webapp.build",
            success_check="webapp.build",
        )
        verdict = family.check_feasibility(_api_world(), task)
        assert not verdict.feasible
        assert "endpoint" in verdict.reason

    def test_service_does_not_expose_endpoint(self, family: WebappBuild) -> None:
        graph = _empty_graph()
        graph.nodes["svc_api"] = _node("svc_api", "service", kind="api", name="api")
        graph.nodes["ep_a"] = _node("ep_a", "endpoint", path="/x", method="GET")
        verdict = family.check_feasibility(graph, _api_task())
        assert not verdict.feasible
        assert "expose" in verdict.reason

    def test_unknown_kind(self, family: WebappBuild) -> None:
        graph = _empty_graph()
        graph.nodes["svc_x"] = _node("svc_x", "service", kind="auth", name="auth")
        graph.nodes["ep_a"] = _node("ep_a", "endpoint", path="/x", method="GET")
        graph.edges["e1"] = _edge("e1", "exposes", "svc_x", "ep_a")
        task = TaskSpec(
            id="t",
            instruction="x",
            entrypoints=("svc_x",),
            goal_nodes=("ep_a",),
            feasibility_check="webapp.build",
            success_check="webapp.build",
        )
        verdict = family.check_feasibility(graph, task)
        assert not verdict.feasible
        assert "auth" in verdict.reason or "build contract" in verdict.reason

    def test_check_success_unresolvable_target(self, family: WebappBuild) -> None:
        result = family.check_success(
            _empty_graph(), _api_task(), {"result": {"endpoint_impl": "x"}}
        )
        assert not result.success
        assert "unresolvable" in result.reason


class TestAdmissionValidityViaInjection:
    """The validity-check branches are exercised by constructing
    ``WebappBuild(generators=...)`` with deliberately ill-posed generators —
    no module-level monkey patching needed."""

    def _api_reference(self, level: int) -> str:
        return api_list_reference(level)

    def _api_contract(self, level: int) -> tuple[ContractCase, ...]:
        return api_list_contract(level)

    def test_feasibility_rejects_when_reference_fails_contract(self) -> None:
        def broken_reference(level: int) -> str:
            del level
            return "def handle(q, s): return 500, {}, b''"

        generators: KindGenerators = {
            "api": KindSpec(
                broken_reference, self._api_contract, (api_wrong_field_name,), 1
            ),
        }
        family = WebappBuild(generators=generators)
        verdict = family.check_feasibility(_api_world(), _api_task())
        assert not verdict.feasible
        assert "reference impl" in verdict.reason

    def test_feasibility_rejects_when_no_mutations_registered(self) -> None:
        generators: KindGenerators = {
            "api": KindSpec(self._api_reference, self._api_contract, (), 1),
        }
        family = WebappBuild(generators=generators)
        verdict = family.check_feasibility(_api_world(), _api_task())
        assert not verdict.feasible
        assert "no admission mutations" in verdict.reason

    def test_feasibility_rejects_when_a_mutation_does_not_break(self) -> None:
        def identity_mutation(source: str) -> str:
            return source

        generators: KindGenerators = {
            "api": KindSpec(
                self._api_reference,
                self._api_contract,
                (api_wrong_field_name, identity_mutation),
                1,
            ),
        }
        family = WebappBuild(generators=generators)
        verdict = family.check_feasibility(_api_world(), _api_task())
        assert not verdict.feasible
        assert "did not break" in verdict.reason
        assert "mutation 1" in verdict.reason

    def test_feasibility_passes_when_all_mutations_break(self) -> None:
        family = WebappBuild(generators=_KIND_GENERATORS)
        verdict = family.check_feasibility(_api_world(), _api_task())
        assert verdict.feasible


class TestGenerateNoTarget:
    def test_generate_returns_empty_when_no_supported_service(
        self,
        family: WebappBuild,
    ) -> None:
        graph = _empty_graph()
        graph.nodes["svc_db"] = _node("svc_db", "service", kind="db", name="db")
        out = family.generate(graph, manifest={}, prior=None)
        assert out == []
