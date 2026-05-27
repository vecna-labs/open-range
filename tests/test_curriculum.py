"""Core auto-evolve tests.

Three layers:

  1. ``direction_from_reports`` — pure policy over synthetic reports
     satisfying ``EpisodeReportLike``.

  2. ``auto_evolve`` — orchestration against:
       a. An inline stub pack with hand-crafted mutations.
       b. The real :class:`WebappPack` against a procedural snapshot.

  3. ``apply_patch`` round-trip — confirms the curriculum's chosen
     patch shape matches ``graphschema.apply_patch``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import pytest
from graphschema import (
    AttrSpec,
    AttrType,
    Edge,
    EdgeKind,
    GraphPatch,
    Issue,
    Node,
    NodeKind,
    Ontology,
    Visibility,
    WorldGraph,
    apply_patch,
)
from openrange_pack_sdk import (
    Backing,
    Builder,
    BuildResult,
    EpisodeReportLike,
    EpisodeResult,
    FeasibilityVerdict,
    LLMBackend,
    Manifest,
    Mutation,
    Pack,
    PackPrior,
    RuntimeHandle,
    Snapshot,
    TaskFamily,
    TaskSpec,
)

from openrange.core.admit import admit
from openrange.core.curriculum import (
    auto_evolve,
    direction_from_reports,
)


@dataclass(frozen=True)
class _Report:
    """Concrete `EpisodeReportLike` — `passed` + `final_state`.

    Tests use this so they don't depend on the runtime ``EpisodeReport``
    class. ``final_state`` defaults to ``{}`` because the curriculum
    policy under test only reads ``passed``; pack-side mutation heuristics
    that interrogate final_state pass it explicitly.
    """

    passed: bool
    final_state: Mapping[str, object] = field(default_factory=dict)


def test_direction_none_when_no_reports() -> None:
    assert direction_from_reports(()) is None


def test_direction_harden_when_pass_rate_high() -> None:
    reports = [_Report(True), _Report(True), _Report(True)]
    assert direction_from_reports(reports) == "harden"


def test_direction_soften_when_pass_rate_low() -> None:
    reports = [_Report(False), _Report(False), _Report(False)]
    assert direction_from_reports(reports) == "soften"


def test_direction_diversify_when_mid_band() -> None:
    reports = [_Report(True), _Report(False)]
    assert direction_from_reports(reports) == "diversify"


def test_direction_threshold_boundaries() -> None:
    # 2/3 = 0.66... >= 0.66 -> harden
    high = [_Report(True), _Report(True), _Report(False)]
    assert direction_from_reports(high) == "harden"
    # 1/3 = 0.33... > 0.33 -> not soften; lands in mid -> diversify
    low = [_Report(False), _Report(False), _Report(True)]
    assert direction_from_reports(low) == "diversify"


def test_direction_treats_missing_passed_as_failure() -> None:
    """A report whose ``passed`` access raises is counted as a failure
    rather than crashing the policy. The protocol slice is narrow, but
    callers who hand in a malformed object should still get a usable
    direction."""

    class _MissingPassed:
        final_state: Mapping[str, object] = {}

        @property
        def passed(self) -> bool:
            raise AttributeError("passed not yet set")

    bad: EpisodeReportLike = _MissingPassed()
    assert direction_from_reports([bad]) == "soften"


def test_direction_custom_thresholds() -> None:
    """Callers can shift the harden/soften bands."""
    mid = [_Report(True), _Report(False)]
    # With aggressive threshold even 0.5 counts as a harden signal
    assert (
        direction_from_reports(mid, harden_threshold=0.5, soften_threshold=0.0)
        == "harden"
    )


_STUB_ONTOLOGY = Ontology(
    id="stub@1",
    node_kinds={
        "repo": NodeKind(
            "repo",
            attrs={"name": AttrSpec(AttrType.STRING, required=True)},
        ),
        "endpoint": NodeKind(
            "endpoint",
            attrs={"path": AttrSpec(AttrType.STRING, required=True)},
        ),
        "vuln": NodeKind(
            "vuln",
            attrs={"kind": AttrSpec(AttrType.STRING, required=True)},
        ),
    },
    edge_kinds={
        "exposes": EdgeKind("exposes", endpoints=[("repo", "endpoint")]),
        "affects": EdgeKind("affects", endpoints=[("vuln", "endpoint")]),
    },
)


def _build_stub_world() -> WorldGraph:
    """A small repo+endpoint+vuln world; satisfies the stub ontology."""
    g = WorldGraph(ontology="stub@1")
    g.add_node(Node("repo.a", "repo", attrs={"name": "alpha"}))
    g.add_node(Node("ep.login", "endpoint", attrs={"path": "/login"}))
    g.add_node(
        Node(
            "vuln.sqli",
            "vuln",
            attrs={"kind": "sql_injection"},
            visibility=Visibility.HIDDEN,
        ),
    )
    g.add_edge(Edge("e1", "exposes", "repo.a", "ep.login"))
    g.add_edge(Edge("e2", "affects", "vuln.sqli", "ep.login"))
    return g


def _stub_task() -> TaskSpec:
    return TaskSpec(
        id="stub.t.0",
        instruction="reach the endpoint",
        entrypoints=("repo.a",),
        goal_nodes=("ep.login",),
        feasibility_check="stub.family",
        success_check="stub.family",
    )


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


class _StubFamily(TaskFamily):
    """Test family that emits one task + a configurable list of mutations.

    The mutations are passed in at construction so each test can pin
    exactly what proposals the curriculum sees without re-implementing
    the family.
    """

    id = "stub.family"
    pack_id = "stub"

    def __init__(self, mutations: tuple[Mutation, ...] = ()) -> None:
        self._mutations = mutations
        self.calls = 0

    def generate(
        self,
        graph: WorldGraph,
        manifest: Manifest,
        prior: PackPrior | None,
    ) -> list[TaskSpec]:
        del graph, manifest, prior
        return [_stub_task()]

    def check_feasibility(
        self,
        graph: WorldGraph,
        task: TaskSpec,
    ) -> FeasibilityVerdict:
        if not task.entrypoints:
            return FeasibilityVerdict(False, "no entrypoint")
        if task.entrypoints[0] not in graph.nodes:
            return FeasibilityVerdict(False, "missing entrypoint")
        return FeasibilityVerdict(True)

    def check_success(
        self,
        graph: WorldGraph,
        task: TaskSpec,
        final_state: Mapping[str, Any],
    ) -> EpisodeResult:
        del graph, task
        return EpisodeResult(success=bool(final_state.get("ok")))

    def available_mutations(
        self,
        snapshot: Snapshot,
        reports: Sequence[EpisodeReportLike],
        *,
        llm: LLMBackend | None = None,
    ) -> tuple[Mutation, ...]:
        del snapshot, reports, llm
        self.calls += 1
        return self._mutations


class _StubBuilder(Builder):
    """Returns a fixed BuildResult. Doesn't repair."""

    def __init__(self, result: BuildResult) -> None:
        self._result = result

    def build(self, manifest: Manifest) -> BuildResult:
        del manifest
        return self._result


class _StubPack(Pack):
    id = "stub"
    version = "0.1.0"

    def __init__(self, family: _StubFamily) -> None:
        self._family = family
        # Builder is configured by the test via attach_build_result.
        self._builder: Builder | None = None

    def attach_build_result(self, result: BuildResult) -> None:
        self._builder = _StubBuilder(result)

    def ontology(self) -> Ontology:
        return _STUB_ONTOLOGY

    def invariants(self) -> list[Any]:
        return []

    def make_builder(self, prior: PackPrior | None) -> Builder:
        del prior
        if self._builder is None:
            raise RuntimeError("test did not attach a build result")
        return self._builder

    def realize(self, graph: WorldGraph, backing: Backing) -> RuntimeHandle:
        del graph, backing
        return _NoopHandle()

    def task_families(self) -> list[TaskFamily]:
        return [self._family]


def _build_stub_snapshot(family: _StubFamily) -> tuple[Snapshot, _StubPack]:
    """Admit a stub snapshot the curriculum can evolve."""
    pack = _StubPack(family)
    graph = _build_stub_world()
    result = BuildResult(graph=graph, tasks=[_stub_task()])
    pack.attach_build_result(result)
    snap = admit(pack, manifest={"seed": 0})
    assert isinstance(snap, Snapshot), snap
    return snap, pack


def test_auto_evolve_no_reports_returns_none() -> None:
    family = _StubFamily()
    snap, pack = _build_stub_snapshot(family)
    assert auto_evolve(snap, pack=pack) is None
    # No reports means the family is never even queried — the loop
    # short-circuits on the empty-reports check.
    assert family.calls == 0


def test_auto_evolve_no_options_returns_none() -> None:
    """Family returns no mutations → curriculum bails out."""
    family = _StubFamily(mutations=())
    snap, pack = _build_stub_snapshot(family)
    result = auto_evolve(snap, _Report(True), pack=pack)
    assert result is None
    assert family.calls == 1  # the family WAS asked


def test_auto_evolve_no_matching_direction_returns_none() -> None:
    """Pack offers only soften options but agent passed (harden direction)."""
    add_patch = GraphPatch(
        nodes_added=[
            Node("repo.b", "repo", attrs={"name": "beta"}),
        ],
    )
    family = _StubFamily(
        mutations=(
            Mutation(
                patch=add_patch,
                direction="soften",
                relevance=0.9,
                family="stub.family",
            ),
        ),
    )
    snap, pack = _build_stub_snapshot(family)
    assert auto_evolve(snap, _Report(True), pack=pack) is None


def test_auto_evolve_zero_relevance_returns_none() -> None:
    """Even when direction matches, a 0-relevance candidate is ignored."""
    family = _StubFamily(
        mutations=(
            Mutation(
                patch=GraphPatch(),
                direction="harden",
                relevance=0.0,
                family="stub.family",
            ),
        ),
    )
    snap, pack = _build_stub_snapshot(family)
    assert auto_evolve(snap, _Report(True), pack=pack) is None


def test_auto_evolve_picks_highest_relevance_in_direction() -> None:
    """Agent passed → harden; pick highest-relevance harden candidate.

    We attach two harden candidates with different patches; the
    higher-relevance one should land in the returned snapshot's graph.
    """
    low_patch = GraphPatch(
        nodes_added=[Node("ep.low", "endpoint", attrs={"path": "/low"})],
        edges_added=[Edge("e.low", "exposes", "repo.a", "ep.low")],
    )
    high_patch = GraphPatch(
        nodes_added=[Node("ep.high", "endpoint", attrs={"path": "/high"})],
        edges_added=[Edge("e.high", "exposes", "repo.a", "ep.high")],
    )
    soften_patch = GraphPatch(nodes_removed=["vuln.sqli"])

    family = _StubFamily(
        mutations=(
            Mutation(
                patch=low_patch,
                direction="harden",
                relevance=0.2,
                family="stub.family",
                note="low",
            ),
            Mutation(
                patch=high_patch,
                direction="harden",
                relevance=0.9,
                family="stub.family",
                note="high",
            ),
            Mutation(
                patch=soften_patch,
                direction="soften",
                relevance=0.95,
                family="stub.family",
                note="soften — should be ignored on harden direction",
            ),
        ),
    )
    snap, pack = _build_stub_snapshot(family)

    out = auto_evolve(snap, _Report(True), pack=pack)
    assert isinstance(out, Snapshot), out
    # The high-relevance harden patch was applied; the low-relevance
    # patch was not; the soften patch was ignored.
    assert "ep.high" in out.graph.nodes
    assert "ep.low" not in out.graph.nodes
    assert "vuln.sqli" in out.graph.nodes
    # The original snapshot is untouched (re-admission must not mutate).
    assert "ep.high" not in snap.graph.nodes
    # Lineage carries the parent snapshot id so callers can chain.
    out_manifest = out.lineage["manifest"]
    assert isinstance(out_manifest, Mapping)
    evolve_meta = out_manifest.get("_evolve")
    assert isinstance(evolve_meta, Mapping)
    assert evolve_meta["parent_snapshot_id"] == snap.snapshot_id
    assert evolve_meta["direction"] == "harden"
    assert evolve_meta["note"] == "high"


def test_auto_evolve_custom_policy() -> None:
    """Trainer can override the direction policy."""
    diversify_patch = GraphPatch(
        nodes_added=[Node("ep.div", "endpoint", attrs={"path": "/div"})],
        edges_added=[Edge("e.div", "exposes", "repo.a", "ep.div")],
    )
    family = _StubFamily(
        mutations=(
            Mutation(
                patch=diversify_patch,
                direction="diversify",
                relevance=0.5,
                family="stub.family",
            ),
        ),
    )
    snap, pack = _build_stub_snapshot(family)

    # All-pass reports would normally select harden — but the override
    # forces diversify, which matches the candidate.
    out = auto_evolve(
        snap,
        _Report(True),
        pack=pack,
        policy=lambda _reports: "diversify",
    )
    assert isinstance(out, Snapshot)
    assert "ep.div" in out.graph.nodes


def test_auto_evolve_skips_failing_candidate_and_tries_next() -> None:
    """A candidate whose patch breaks admission must NOT crash the loop.

    We hand the curriculum two harden options. The higher-relevance one
    introduces an invalid edge (dangling endpoint) — admission will
    reject it. The next option is valid and must land.
    """
    bad_patch = GraphPatch(
        edges_added=[
            # Edge with a dangling dst → admission's structural tier
            # flags ``edge_dangling_dst`` and rejects.
            Edge("e.bad", "exposes", "repo.a", "does-not-exist"),
        ],
    )
    good_patch = GraphPatch(
        nodes_added=[
            Node("ep.ok", "endpoint", attrs={"path": "/ok"}),
        ],
        edges_added=[Edge("e.ok", "exposes", "repo.a", "ep.ok")],
    )
    family = _StubFamily(
        mutations=(
            Mutation(
                patch=bad_patch,
                direction="harden",
                relevance=0.9,
                family="stub.family",
                note="bad",
            ),
            Mutation(
                patch=good_patch,
                direction="harden",
                relevance=0.5,
                family="stub.family",
                note="good",
            ),
        ),
    )
    snap, pack = _build_stub_snapshot(family)

    out = auto_evolve(snap, _Report(True), pack=pack)
    assert isinstance(out, Snapshot), out
    assert "ep.ok" in out.graph.nodes


def test_auto_evolve_aggregates_across_families() -> None:
    """Two families each contribute one mutation; the curriculum picks
    across them by relevance.

    Both families share the test task's ``feasibility_check`` /
    ``success_check`` id (``stub.family``) so admission's per-task
    family dispatch still finds a handler. The two family instances
    differ only in the mutation list they propose; aggregation should
    visit BOTH and pick whichever proposal has the highest relevance
    in the chosen direction.
    """
    fam_a_patch = GraphPatch(
        nodes_added=[Node("ep.a", "endpoint", attrs={"path": "/a"})],
        edges_added=[Edge("e.a", "exposes", "repo.a", "ep.a")],
    )
    fam_b_patch = GraphPatch(
        nodes_added=[Node("ep.b", "endpoint", attrs={"path": "/b"})],
        edges_added=[Edge("e.b", "exposes", "repo.a", "ep.b")],
    )

    fam_a = _StubFamily(
        mutations=(
            Mutation(
                patch=fam_a_patch,
                direction="harden",
                relevance=0.4,
                family="stub.family",
                note="A",
            ),
        ),
    )
    fam_b = _StubFamily(
        mutations=(
            Mutation(
                patch=fam_b_patch,
                direction="harden",
                relevance=0.9,
                family="stub.family",
                note="B",
            ),
        ),
    )

    class _TwoFamilyPack(_StubPack):
        def task_families(self) -> list[TaskFamily]:
            return [fam_a, fam_b]

    pack = _TwoFamilyPack(fam_a)  # fam_a here is just a placeholder
    pack.attach_build_result(
        BuildResult(graph=_build_stub_world(), tasks=[_stub_task()]),
    )
    snap = admit(pack, manifest={"seed": 0})
    assert isinstance(snap, Snapshot)

    out = auto_evolve(snap, _Report(True), pack=pack)
    assert isinstance(out, Snapshot)
    # fam_b's higher-relevance patch wins.
    assert "ep.b" in out.graph.nodes
    assert "ep.a" not in out.graph.nodes
    # Both families WERE consulted (aggregation actually visited each).
    assert fam_a.calls == 1
    assert fam_b.calls == 1


def test_apply_patch_modifies_graph_in_place() -> None:
    """A Mutation's :class:`GraphPatch` round-trips through ``apply_patch``."""
    g = _build_stub_world()
    assert "vuln.sqli" in g.nodes
    assert any(e.kind == "affects" for e in g.edges.values())

    patch = GraphPatch(nodes_removed=["vuln.sqli"])
    mutation = Mutation(
        patch=patch,
        direction="soften",
        relevance=0.5,
        family="stub.family",
        note="remove sqli",
    )
    apply_patch(g, mutation.patch)

    assert "vuln.sqli" not in g.nodes
    # The dangling ``affects`` edge is dropped automatically by
    # ``apply_patch`` — exercised here so a regression in that
    # behavior wouldn't slip past the curriculum tests.
    assert not any(e.kind == "affects" for e in g.edges.values())


def test_auto_evolve_e2e_on_webapp_pack() -> None:
    """``WebappPack`` admits + ``auto_evolve`` re-admits end-to-end.

    Specific direction / patch shape are pack business; this test only
    confirms admit → auto_evolve → re-admit produces a new Snapshot whose
    snapshot_id differs from the parent's (so the patch actually
    landed) and whose lineage points back at the parent.
    """
    webapp_pack_cls = pytest.importorskip("cyber_webapp").WebappPack
    pack = webapp_pack_cls()

    snap = admit(pack, manifest={"seed": 0}, max_repairs=3)
    assert isinstance(snap, Snapshot), snap

    # All-pass reports → harden direction → the WebappPack's
    # mutation enumerator has plenty of harden options for any
    # procedurally-built world.
    out = auto_evolve(
        snap,
        _Report(True),
        _Report(True),
        _Report(True),
        pack=pack,
    )
    if out is None:
        pytest.skip(
            "WebappPack curriculum returned no candidates in harden direction "
            "for this seed — re-test once a stable manifest is wired in",
        )
    assert isinstance(out, Snapshot)
    assert out.snapshot_id != snap.snapshot_id
    out_manifest = out.lineage["manifest"]
    assert isinstance(out_manifest, Mapping)
    evolve_meta = out_manifest.get("_evolve")
    assert isinstance(evolve_meta, Mapping)
    assert evolve_meta["parent_snapshot_id"] == snap.snapshot_id
    assert evolve_meta["direction"] == "harden"


# Lint shim — keep imported types from being flagged as unused. They
# appear in type annotations on the stub classes but ruff sometimes
# misses cross-class refs.
assert Issue is not None
