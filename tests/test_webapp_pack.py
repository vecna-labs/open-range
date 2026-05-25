"""Cross-domain integration test for the webapp pack.

The single best test of the new OpenRange shape: one webapp world graph
admits BOTH `webapp.build` and `webapp.pentest` task families, each
entrypointing a different node-kind in the same world.

This mirrors `.claude/bbg-openrange/crossdomain.py` (the design's
canonical demo) against the real `WebappPack` class. If this passes,
the load-bearing design property holds: domain lives in the
TaskFamily, not in the Pack.
"""

from __future__ import annotations

from webapp import WebappBuild, WebappBuilder, WebappPack, WebappPentest

from openrange import (
    AdmissionFailure,
    Snapshot,
    Visibility,
    admit,
)
from openrange.ontologies.bbg import BBG_ONTOLOGY_ID  # for distill-flywheel test

# ---------------------------------------------------------------------------
# Sanity: pack identity and surface
# ---------------------------------------------------------------------------


def test_pack_identity() -> None:
    p = WebappPack()
    assert p.id == "webapp"
    assert p.version == "0.1.0"
    assert p.ontology().id == "webapp@0.1.0"
    family_ids = {f.id for f in p.task_families()}
    assert family_ids == {"webapp.build", "webapp.pentest"}


def test_pack_resolves_family_by_id() -> None:
    p = WebappPack()
    assert isinstance(p.task_family("webapp.build"), WebappBuild)
    assert isinstance(p.task_family("webapp.pentest"), WebappPentest)
    assert p.task_family("webapp.ghost") is None


def test_invariants_are_registered() -> None:
    p = WebappPack()
    names = {fn.__name__ for fn in p.invariants()}
    assert "no_orphan_nodes" in names
    assert "secret_must_be_held" in names
    assert "service_must_own_repo_and_expose_endpoint" in names


# ---------------------------------------------------------------------------
# Admission: the full happy path with BOTH families on ONE world
# ---------------------------------------------------------------------------


def test_default_world_admits_both_families() -> None:
    """One WebappPack, default manifest -> snapshot with two tasks
    (webapp.build and webapp.pentest) on the same world graph."""
    pack = WebappPack()
    snap = admit(pack, manifest={"seed": 0})
    assert isinstance(snap, Snapshot), snap

    families = {t.feasibility_check for t in snap.tasks}
    assert families == {"webapp.build", "webapp.pentest"}, families

    # The two tasks entrypoint DIFFERENT node-kinds in the SAME world.
    entrypoint_kinds = {snap.graph.nodes[t.entrypoints[0]].kind for t in snap.tasks}
    assert entrypoint_kinds == {"repo", "endpoint"}, entrypoint_kinds


def test_snapshot_is_content_addressed() -> None:
    """Two builds with the same seed share a snapshot_id."""
    pack = WebappPack()
    snap_a = admit(pack, manifest={"seed": 7})
    snap_b = admit(pack, manifest={"seed": 7})
    assert isinstance(snap_a, Snapshot)
    assert isinstance(snap_b, Snapshot)
    assert snap_a.snapshot_id == snap_b.snapshot_id


def test_different_seeds_give_different_snapshot_ids() -> None:
    """v1 builder varies only the flag value with the seed; different seed
    => different flag value => different content hash."""
    pack = WebappPack()
    snap_a = admit(pack, manifest={"seed": 0})
    snap_b = admit(pack, manifest={"seed": 42})
    assert isinstance(snap_a, Snapshot)
    assert isinstance(snap_b, Snapshot)
    assert snap_a.snapshot_id != snap_b.snapshot_id


def test_snapshot_history_records_all_phases() -> None:
    pack = WebappPack()
    snap = admit(pack, manifest={"seed": 0})
    assert isinstance(snap, Snapshot)
    phases = [e.phase for e in snap.history]
    assert phases == ["build", "validate", "feasibility", "freeze"]


def test_lineage_carries_builder_provenance() -> None:
    pack = WebappPack()
    snap = admit(pack, manifest={"seed": 0})
    assert isinstance(snap, Snapshot)
    assert snap.lineage["pack"] == "webapp"
    assert snap.lineage["pack_version"] == "0.1.0"
    assert snap.lineage["attempts"] == 1
    assert snap.lineage["builder"] == "webapp.v1"


# ---------------------------------------------------------------------------
# The task graph: bindings match the design
# ---------------------------------------------------------------------------


def test_build_task_entrypoints_repo_goal_is_endpoint() -> None:
    snap = admit(WebappPack(), manifest={"seed": 0})
    assert isinstance(snap, Snapshot)
    build_task = next(t for t in snap.tasks if t.feasibility_check == "webapp.build")
    assert snap.graph.nodes[build_task.entrypoints[0]].kind == "repo"
    assert snap.graph.nodes[build_task.goal_nodes[0]].kind == "endpoint"


def test_pentest_task_entrypoints_endpoint_goal_is_secret() -> None:
    snap = admit(WebappPack(), manifest={"seed": 0})
    assert isinstance(snap, Snapshot)
    pentest = next(t for t in snap.tasks if t.feasibility_check == "webapp.pentest")
    ep = snap.graph.nodes[pentest.entrypoints[0]]
    secret = snap.graph.nodes[pentest.goal_nodes[0]]
    assert ep.kind == "endpoint"
    assert ep.visibility is Visibility.PUBLIC  # entrypoint must be visible
    assert secret.kind == "secret"
    assert secret.visibility is Visibility.HIDDEN  # discovery is the point


# ---------------------------------------------------------------------------
# Feasibility checks called via the family handles
# ---------------------------------------------------------------------------


def test_build_feasibility_rejects_dangling_repo() -> None:
    from openrange import TaskSpec

    snap = admit(WebappPack(), manifest={"seed": 0})
    assert isinstance(snap, Snapshot)
    bogus = TaskSpec(
        id="build.bogus",
        instruction="x",
        entrypoints=("repo.ghost",),
        goal_nodes=(snap.tasks[0].goal_nodes[0],),
        feasibility_check="webapp.build",
        success_check="webapp.build",
    )
    verdict = WebappBuild().check_feasibility(snap.graph, bogus)
    assert not verdict.feasible


def test_pentest_feasibility_rejects_world_without_weakness() -> None:
    from openrange import GraphPatch, apply_patch

    snap = admit(WebappPack(), manifest={"seed": 0})
    assert isinstance(snap, Snapshot)
    # remove the weakness — the chain to the flag should fail
    g = snap.graph
    weak_ids = [n.id for n in g.by_kind("weakness")]
    apply_patch(g, GraphPatch(nodes_removed=weak_ids))
    pentest = next(t for t in snap.tasks if t.feasibility_check == "webapp.pentest")
    verdict = WebappPentest().check_feasibility(g, pentest)
    assert not verdict.feasible
    assert "weakness" in verdict.reason


# ---------------------------------------------------------------------------
# Success checks (without a real realizer)
# ---------------------------------------------------------------------------


def test_pentest_success_when_flag_matches() -> None:
    snap = admit(WebappPack(), manifest={"seed": 0})
    assert isinstance(snap, Snapshot)
    pentest = next(t for t in snap.tasks if t.success_check == "webapp.pentest")
    flag = snap.graph.nodes[pentest.goal_nodes[0]]
    expected = flag.attrs["value_ref"]
    result = WebappPentest().check_success(
        snap.graph,
        pentest,
        final_state={
            "flag_from_response": expected,
            "requests_made": ["POST /login"],
        },
    )
    assert result.success
    assert result.subgoals["matched_flag"] is True


def test_pentest_failure_when_flag_missing() -> None:
    snap = admit(WebappPack(), manifest={"seed": 0})
    assert isinstance(snap, Snapshot)
    pentest = next(t for t in snap.tasks if t.success_check == "webapp.pentest")
    result = WebappPentest().check_success(
        snap.graph,
        pentest,
        final_state={},
    )
    assert not result.success


def test_build_success_when_smoke_passes() -> None:
    snap = admit(WebappPack(), manifest={"seed": 0})
    assert isinstance(snap, Snapshot)
    build_task = next(t for t in snap.tasks if t.success_check == "webapp.build")
    ok = WebappBuild().check_success(
        snap.graph,
        build_task,
        final_state={"endpoint_serves_200": True},
    )
    assert ok.success
    bad = WebappBuild().check_success(
        snap.graph,
        build_task,
        final_state={"endpoint_serves_200": False},
    )
    assert not bad.success


# ---------------------------------------------------------------------------
# Invariants: rejected admissions
# ---------------------------------------------------------------------------


def test_admission_rejects_world_missing_owned_by() -> None:
    """Hand-craft a broken world (service with no owned_by) by stripping
    the edge after build. Admission should reject via the pack invariant.
    """
    from openrange import BuildResult, GraphPatch, apply_patch

    WebappPack()
    builder = WebappBuilder(prior=None)
    base = builder.build({"seed": 0})
    # remove the owned_by edge
    apply_patch(base.graph, GraphPatch(edges_removed=["e.svc-repo"]))

    class _Fixed:
        def __init__(self, br: BuildResult) -> None:
            self._br = br

        def build(self, manifest: object) -> BuildResult:
            return self._br

        def repair(self, *a: object, **k: object) -> BuildResult:
            raise NotImplementedError

    # bolt the fixed builder onto a fresh Pack instance
    class _Patched(WebappPack):
        def make_builder(self, prior: object) -> object:  # type: ignore[override]
            return _Fixed(base)

    out = admit(_Patched(), manifest={"seed": 0}, max_repairs=0)
    assert isinstance(out, AdmissionFailure)
    assert any(i.code == "service_no_repo" for i in out.issues)


# ---------------------------------------------------------------------------
# The flywheel seam: distill -> PackPrior -> Pack.make_builder
# ---------------------------------------------------------------------------


def test_make_builder_accepts_none_prior() -> None:
    """The boot path — Pack.make_builder(prior=None) returns a Builder
    that admits without ever needing a BBG."""
    pack = WebappPack()
    builder = pack.make_builder(None)
    result = builder.build({"seed": 0})
    assert result.graph.nodes
    assert result.tasks


def test_make_builder_accepts_distilled_prior() -> None:
    """The flywheel path — a hand-built BBG-shaped graph distills into a
    PackPrior that the builder accepts without crashing.

    v1 builder ignores the prior content; v2 will read topology /
    task_seeds. This test just proves the seam is wired."""
    from openrange import Edge, Node, WorldGraph, distill

    bbg = WorldGraph(ontology=BBG_ONTOLOGY_ID)
    bbg.add_node(
        Node(
            "thing.x",
            "thing",
            attrs={
                "label": "x",
                "provenance": "trajectory",
                "status": "salient",
                "kind_hint": "endpoint",
                "category": "place",
                "first_seen": 0,
                "visits": 1,
                "explored": True,
            },
        )
    )
    bbg.add_node(
        Node(
            "thing.y",
            "thing",
            attrs={
                "label": "y",
                "provenance": "trajectory",
                "status": "salient",
                "kind_hint": "secret",
                "category": "object",
                "first_seen": 1,
                "visits": 1,
                "explored": True,
            },
        )
    )
    bbg.add_edge(
        Edge(
            "e.xy",
            "traversed",
            "thing.x",
            "thing.y",
            attrs={"outcome": "productive", "count": 1},
        )
    )

    prior = distill(bbg)
    assert "endpoint" in prior.topology["node_kind_freq"]

    pack = WebappPack()
    builder = pack.make_builder(prior)
    result = builder.build({"seed": 0})
    assert result.admission_meta["prior_source"].startswith("bbg@0.1.0 :: ")
