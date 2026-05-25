"""Tests for the layered admission loop.

Covers:
- structural / conformance / invariant failures rejected with the right Issue.codes
- task-binding failures (dangling entrypoint, hidden entrypoint, dangling goal)
- task feasibility called per task via the TaskFamily handle
- repair budget: a builder that fails twice then succeeds is admitted
- Snapshot is content-addressed (snapshot_id == graph.content_hash())
- build history records every phase per attempt
- the same world admits two TaskFamilies (build + pentest with different entrypoints)
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from openrange.core.admit import (
    AdmissionFailure,
    Snapshot,
    admit,
    snapshot_to_dict,
    validate_task_bindings,
)
from openrange.core.pack import (
    Backing,
    Builder,
    BuildResult,
    EpisodeResult,
    FeasibilityVerdict,
    Manifest,
    Pack,
    PackPrior,
    RuntimeHandle,
    TaskFamily,
    TaskSpec,
)
from openrange.world_ir import (
    AttrSpec,
    AttrType,
    Edge,
    EdgeKind,
    Issue,
    Node,
    NodeKind,
    Ontology,
    Visibility,
    WorldGraph,
)

# ---------------------------------------------------------------------------
# A tiny test-only ontology / pack — domain-free, exercises all the seams.
# ---------------------------------------------------------------------------


_TEST_ONTOLOGY = Ontology(
    id="test@1",
    node_kinds={
        "repo": NodeKind(
            "repo",
            attrs={
                "name": AttrSpec(AttrType.STRING, required=True),
            },
        ),
        "endpoint": NodeKind(
            "endpoint",
            attrs={
                "path": AttrSpec(AttrType.STRING, required=True),
            },
        ),
        "secret": NodeKind(
            "secret",
            attrs={
                "kind": AttrSpec(AttrType.STRING, required=True),
            },
        ),
    },
    edge_kinds={
        "exposes": EdgeKind("exposes", endpoints=[("repo", "endpoint")]),
        "holds": EdgeKind("holds", endpoints=[("endpoint", "secret")]),
    },
)


def _build_test_graph() -> WorldGraph:
    """A small world: repo -> endpoint -> secret. The secret is HIDDEN."""
    g = WorldGraph(ontology="test@1")
    g.add_node(Node("repo.a", "repo", attrs={"name": "alpha"}))
    g.add_node(Node("ep.login", "endpoint", attrs={"path": "/login"}))
    g.add_node(
        Node("sec.flag", "secret", attrs={"kind": "flag"}, visibility=Visibility.HIDDEN)
    )
    g.add_edge(Edge("e1", "exposes", "repo.a", "ep.login"))
    g.add_edge(Edge("e2", "holds", "ep.login", "sec.flag"))
    return g


class _NoopHandle:
    """A RuntimeHandle stub — admit() never realizes, so methods aren't called."""

    def reset(self) -> None: ...
    def surface(self) -> Mapping[str, Any]:
        return {}

    def collect(self) -> Mapping[str, Any]:
        return {}

    def stop(self) -> None: ...


class _BuildFamily(TaskFamily):
    id = "test.build"
    pack_id = "test"

    def generate(
        self, graph: WorldGraph, manifest: Manifest, prior: PackPrior | None
    ) -> list[TaskSpec]:
        repo = next(iter(graph.by_kind("repo")), None)
        endpoint = next(iter(graph.by_kind("endpoint")), None)
        if repo is None or endpoint is None:
            return []
        return [
            TaskSpec(
                id="test.build.0",
                instruction="Add an endpoint and confirm it serves 200.",
                entrypoints=(repo.id,),
                goal_nodes=(endpoint.id,),
                feasibility_check="test.build",
                success_check="test.build",
                meta={"family": "test.build", "difficulty": 0.3},
            )
        ]

    def check_feasibility(
        self, graph: WorldGraph, task: TaskSpec
    ) -> FeasibilityVerdict:
        # entrypoint exists and is a repo
        if not task.entrypoints:
            return FeasibilityVerdict(False, "no entrypoint")
        ep = graph.nodes.get(task.entrypoints[0])
        if ep is None or ep.kind != "repo":
            return FeasibilityVerdict(False, "entrypoint is not a repo")
        return FeasibilityVerdict(True)

    def check_success(
        self, graph: WorldGraph, task: TaskSpec, final_state: Mapping[str, Any]
    ) -> EpisodeResult:
        return EpisodeResult(success=bool(final_state.get("smoke_ok")))


class _PentestFamily(TaskFamily):
    id = "test.pentest"
    pack_id = "test"

    def generate(
        self, graph: WorldGraph, manifest: Manifest, prior: PackPrior | None
    ) -> list[TaskSpec]:
        endpoint = next(iter(graph.by_kind("endpoint")), None)
        secret = next(iter(graph.by_kind("secret")), None)
        if endpoint is None or secret is None:
            return []
        return [
            TaskSpec(
                id="test.pentest.0",
                instruction="Recover the hidden flag.",
                entrypoints=(endpoint.id,),
                goal_nodes=(secret.id,),
                feasibility_check="test.pentest",
                success_check="test.pentest",
                meta={"family": "test.pentest", "difficulty": 0.7},
            )
        ]

    def check_feasibility(
        self, graph: WorldGraph, task: TaskSpec
    ) -> FeasibilityVerdict:
        # there must be a `holds` edge from the entrypoint to the goal
        for e in graph.out_edges(task.entrypoints[0], "holds"):
            if e.dst in task.goal_nodes:
                return FeasibilityVerdict(True)
        return FeasibilityVerdict(False, "no holds chain to goal")

    def check_success(
        self, graph: WorldGraph, task: TaskSpec, final_state: Mapping[str, Any]
    ) -> EpisodeResult:
        return EpisodeResult(success=bool(final_state.get("flag_found")))


class _StaticBuilder(Builder):
    """A builder that returns a fixed BuildResult — no repair."""

    def __init__(self, result: BuildResult) -> None:
        self._result = result

    def build(self, manifest: Manifest) -> BuildResult:
        return self._result


class _CountingBuilder(Builder):
    """A builder that fails N times before succeeding. Exercises repair."""

    def __init__(
        self, failures_before_success: int, good: BuildResult, bad: BuildResult
    ) -> None:
        self._left = failures_before_success
        self._good = good
        self._bad = bad
        self.repair_calls = 0

    def build(self, manifest: Manifest) -> BuildResult:
        if self._left <= 0:
            return self._good
        self._left -= 1
        return self._bad

    def repair(
        self, prev: BuildResult, errors: list[Issue], infeasible: list[str]
    ) -> BuildResult:
        self.repair_calls += 1
        if self._left <= 0:
            return self._good
        self._left -= 1
        return self._bad


class _TestPack(Pack):
    id = "test"
    version = "0.1.0"

    def __init__(
        self,
        builder: Builder,
        families: list[TaskFamily] | None = None,
        invariants_: list[Any] | None = None,
    ) -> None:
        self._builder = builder
        self._families = (
            families
            if families is not None
            else [
                _BuildFamily(),
                _PentestFamily(),
            ]
        )
        self._invariants = invariants_ or []

    def ontology(self) -> Ontology:
        return _TEST_ONTOLOGY

    def invariants(self) -> list[Any]:
        return list(self._invariants)

    def make_builder(self, prior: PackPrior | None) -> Builder:
        return self._builder

    def realize(self, graph: WorldGraph, backing: Backing) -> RuntimeHandle:
        return _NoopHandle()

    def task_families(self) -> list[TaskFamily]:
        return list(self._families)


# ---------------------------------------------------------------------------
# Happy path: admit one task
# ---------------------------------------------------------------------------


def _good_build_result(family_id: str) -> BuildResult:
    g = _build_test_graph()
    if family_id == "test.build":
        tasks = _BuildFamily().generate(g, {}, None)
    else:
        tasks = _PentestFamily().generate(g, {}, None)
    return BuildResult(graph=g, tasks=tasks, admission_meta={"seed": 42})


def test_admit_returns_snapshot_for_admittable_world() -> None:
    pack = _TestPack(
        _StaticBuilder(_good_build_result("test.pentest")), families=[_PentestFamily()]
    )
    snap = admit(pack, manifest={"goal": "test"})
    assert isinstance(snap, Snapshot)
    assert snap.tasks[0].id == "test.pentest.0"


def test_snapshot_id_equals_graph_content_hash() -> None:
    pack = _TestPack(
        _StaticBuilder(_good_build_result("test.pentest")), families=[_PentestFamily()]
    )
    snap = admit(pack, manifest={})
    assert isinstance(snap, Snapshot)
    assert snap.snapshot_id == snap.graph.content_hash()


def test_admit_records_history_per_phase() -> None:
    pack = _TestPack(
        _StaticBuilder(_good_build_result("test.pentest")), families=[_PentestFamily()]
    )
    snap = admit(pack, manifest={})
    assert isinstance(snap, Snapshot)
    phases = [e.phase for e in snap.history]
    assert phases == ["build", "validate", "feasibility", "freeze"]


def test_lineage_carries_manifest_and_pack_info() -> None:
    pack = _TestPack(
        _StaticBuilder(_good_build_result("test.pentest")), families=[_PentestFamily()]
    )
    snap = admit(pack, manifest={"goal": "demo", "seed": 7})
    assert isinstance(snap, Snapshot)
    assert snap.lineage["pack"] == "test"
    assert snap.lineage["pack_version"] == "0.1.0"
    assert snap.lineage["attempts"] == 1
    assert snap.lineage["manifest"] == {"goal": "demo", "seed": 7}
    assert snap.lineage["seed"] == 42  # admission_meta merged in


# ---------------------------------------------------------------------------
# Cross-domain: same world, two TaskFamilies, different entrypoints
# ---------------------------------------------------------------------------


def test_two_families_admit_on_one_world() -> None:
    """The single best test of the new shape: one world, two families, two
    tasks with different entrypoints. Mirrors crossdomain.py act 3."""
    g = _build_test_graph()
    tasks = []
    tasks.extend(_BuildFamily().generate(g, {}, None))
    tasks.extend(_PentestFamily().generate(g, {}, None))
    result = BuildResult(graph=g, tasks=tasks, admission_meta={"seed": 0})
    pack = _TestPack(_StaticBuilder(result))
    snap = admit(pack, manifest={})
    assert isinstance(snap, Snapshot), snap
    assert {t.feasibility_check for t in snap.tasks} == {
        "test.build",
        "test.pentest",
    }
    # The two tasks entrypoint DIFFERENT node-kinds in the SAME world.
    entrypoint_kinds = {snap.graph.nodes[t.entrypoints[0]].kind for t in snap.tasks}
    assert entrypoint_kinds == {"repo", "endpoint"}


# ---------------------------------------------------------------------------
# Failure paths
# ---------------------------------------------------------------------------


def test_admit_fails_on_dangling_entrypoint() -> None:
    g = _build_test_graph()
    bad_task = TaskSpec(
        id="x.0",
        instruction="bad",
        entrypoints=("nonexistent",),
        goal_nodes=("sec.flag",),
        feasibility_check="test.pentest",
        success_check="test.pentest",
    )
    result = BuildResult(graph=g, tasks=[bad_task])
    pack = _TestPack(_StaticBuilder(result), families=[_PentestFamily()])
    out = admit(pack, manifest={}, max_repairs=0)
    assert isinstance(out, AdmissionFailure)
    assert any(i.code == "task_dangling_entrypoint" for i in out.issues)


def test_admit_fails_on_hidden_entrypoint() -> None:
    g = _build_test_graph()
    # entrypoint a HIDDEN secret node — disallowed
    bad_task = TaskSpec(
        id="x.0",
        instruction="bad",
        entrypoints=("sec.flag",),  # secret is HIDDEN
        goal_nodes=("ep.login",),
        feasibility_check="test.pentest",
        success_check="test.pentest",
    )
    result = BuildResult(graph=g, tasks=[bad_task])
    pack = _TestPack(_StaticBuilder(result), families=[_PentestFamily()])
    out = admit(pack, manifest={}, max_repairs=0)
    assert isinstance(out, AdmissionFailure)
    assert any(i.code == "task_hidden_entrypoint" for i in out.issues)


def test_admit_allows_hidden_goal() -> None:
    """The flag is HIDDEN; discovering it is the point. Should admit."""
    pack = _TestPack(
        _StaticBuilder(_good_build_result("test.pentest")), families=[_PentestFamily()]
    )
    snap = admit(pack, manifest={})
    assert isinstance(snap, Snapshot)
    # confirm the goal is in fact hidden
    goal = snap.graph.nodes[snap.tasks[0].goal_nodes[0]]
    assert goal.visibility is Visibility.HIDDEN


def test_admit_fails_on_infeasible_task() -> None:
    """Build a graph that lacks the `holds` edge the pentest family needs."""
    g = WorldGraph(ontology="test@1")
    g.add_node(Node("ep.lonely", "endpoint", attrs={"path": "/"}))
    g.add_node(
        Node(
            "sec.unreachable",
            "secret",
            attrs={"kind": "flag"},
            visibility=Visibility.HIDDEN,
        )
    )
    # no `holds` edge — pentest's check_feasibility returns False
    task = TaskSpec(
        id="t.0",
        instruction="reach the flag",
        entrypoints=("ep.lonely",),
        goal_nodes=("sec.unreachable",),
        feasibility_check="test.pentest",
        success_check="test.pentest",
    )
    result = BuildResult(graph=g, tasks=[task])
    pack = _TestPack(_StaticBuilder(result), families=[_PentestFamily()])
    out = admit(pack, manifest={}, max_repairs=0)
    assert isinstance(out, AdmissionFailure)
    assert "t.0" in out.infeasible_tasks


def test_admit_fails_when_pack_does_not_offer_named_family() -> None:
    """A task names `feasibility_check="something.unknown"` — the pack does
    not declare that family, so the task is infeasible."""
    g = _build_test_graph()
    task = TaskSpec(
        id="t.0",
        instruction="impossible",
        entrypoints=("repo.a",),
        goal_nodes=("ep.login",),
        feasibility_check="ghost.family",
        success_check="ghost.family",
    )
    result = BuildResult(graph=g, tasks=[task])
    pack = _TestPack(_StaticBuilder(result), families=[_BuildFamily()])
    out = admit(pack, manifest={}, max_repairs=0)
    assert isinstance(out, AdmissionFailure)
    assert "t.0" in out.infeasible_tasks


def test_admit_runs_pack_invariants() -> None:
    """A pack invariant that fails surfaces in AdmissionFailure.issues."""

    def needs_two_endpoints(g: WorldGraph) -> list[Issue]:
        if len(g.by_kind("endpoint")) < 2:
            return [
                Issue(
                    "error",
                    "needs_two_endpoints",
                    "world must have at least 2 endpoints",
                    "graph",
                )
            ]
        return []

    pack = _TestPack(
        _StaticBuilder(_good_build_result("test.pentest")),
        families=[_PentestFamily()],
        invariants_=[needs_two_endpoints],
    )
    out = admit(pack, manifest={}, max_repairs=0)
    assert isinstance(out, AdmissionFailure)
    assert any(i.code == "needs_two_endpoints" for i in out.issues)


# ---------------------------------------------------------------------------
# Repair budget
# ---------------------------------------------------------------------------


def _bad_build_result() -> BuildResult:
    """Bad: entrypoint dangling."""
    g = _build_test_graph()
    task = TaskSpec(
        id="bad.0",
        instruction="bad",
        entrypoints=("does.not.exist",),
        goal_nodes=("sec.flag",),
        feasibility_check="test.pentest",
        success_check="test.pentest",
    )
    return BuildResult(graph=g, tasks=[task])


def test_admit_repair_succeeds_within_budget() -> None:
    """Builder fails once, then succeeds. With max_repairs=2 it should admit."""
    builder = _CountingBuilder(
        failures_before_success=1,
        good=_good_build_result("test.pentest"),
        bad=_bad_build_result(),
    )
    pack = _TestPack(builder, families=[_PentestFamily()])
    snap = admit(pack, manifest={}, max_repairs=2)
    assert isinstance(snap, Snapshot), snap
    assert builder.repair_calls == 1
    # history: build, validate, feas, repair, build... actually:
    # build, validate, feas, repair, validate, feas, freeze
    phases = [e.phase for e in snap.history]
    assert phases == [
        "build",
        "validate",
        "feasibility",
        "repair",
        "validate",
        "feasibility",
        "freeze",
    ]


def test_admit_repair_exhausts_budget() -> None:
    builder = _CountingBuilder(
        failures_before_success=5,
        good=_good_build_result("test.pentest"),
        bad=_bad_build_result(),
    )
    pack = _TestPack(builder, families=[_PentestFamily()])
    out = admit(pack, manifest={}, max_repairs=2)
    assert isinstance(out, AdmissionFailure)
    assert out.attempts == 3
    assert builder.repair_calls == 2


def test_admit_default_repair_raises_when_called() -> None:
    """A builder that doesn't override repair() should NotImplementedError on
    the first repair attempt — admission cannot retry."""

    class _Bare(Builder):
        def build(self, manifest: Manifest) -> BuildResult:
            return _bad_build_result()

    pack = _TestPack(_Bare(), families=[_PentestFamily()])
    with pytest.raises(NotImplementedError):
        admit(pack, manifest={}, max_repairs=1)


# ---------------------------------------------------------------------------
# validate_task_bindings standalone
# ---------------------------------------------------------------------------


def test_validate_task_bindings_returns_empty_for_clean_tasks() -> None:
    g = _build_test_graph()
    tasks = _PentestFamily().generate(g, {}, None)
    issues = validate_task_bindings(g, tasks)
    assert issues == []


def test_validate_task_bindings_flags_dangling_goal() -> None:
    g = _build_test_graph()
    task = TaskSpec(
        id="t.0",
        instruction="x",
        entrypoints=("ep.login",),
        goal_nodes=("ghost",),
        feasibility_check="test.pentest",
        success_check="test.pentest",
    )
    issues = validate_task_bindings(g, [task])
    assert any(i.code == "task_dangling_goal" for i in issues)


# ---------------------------------------------------------------------------
# snapshot_to_dict wire shape
# ---------------------------------------------------------------------------


def test_snapshot_to_dict_round_trip_shape() -> None:
    pack = _TestPack(
        _StaticBuilder(_good_build_result("test.pentest")), families=[_PentestFamily()]
    )
    snap = admit(pack, manifest={"goal": "demo"})
    assert isinstance(snap, Snapshot)
    d = snapshot_to_dict(snap)
    assert d["snapshot_id"] == snap.snapshot_id
    assert d["ontology_id"] == "test@1"
    # nodes are sorted by id
    nodes = d["graph"]["nodes"]
    assert [n["id"] for n in nodes] == sorted(n["id"] for n in nodes)
    # hidden nodes carry visibility on the wire
    hidden = [n for n in nodes if n["id"] == "sec.flag"][0]
    assert hidden.get("visibility") == "hidden"
    # public nodes omit visibility
    public = [n for n in nodes if n["id"] == "ep.login"][0]
    assert "visibility" not in public
    # history phases preserved
    assert [h["phase"] for h in d["history"]] == [
        "build",
        "validate",
        "feasibility",
        "freeze",
    ]
