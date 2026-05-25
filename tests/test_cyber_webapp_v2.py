"""Integration test for the new-shape cyber webapp pack.

Two flavors of tests:

  1. Hand-built world fixtures + a stub `_StubWebappPack` exercise the
     ontology, invariants, families, and admission against a minimal
     graph that touches every shape we care about. Useful for testing
     invariants in isolation and for hand-controlled coverage.

  2. End-to-end tests against the REAL `WebappPack` class run the full
     procedural builder + sampling + admission pipeline. These are the
     load-bearing demonstrations that the new shape works at every
     layer of the pack.

The single load-bearing assertion across both: one cyber webapp world
admits BOTH `webapp.build` and `webapp.pentest` task families with
different entrypoint kinds. That's the cross-domain story from
`.claude/bbg-openrange/crossdomain.py` act 3 applied to the cyber pack.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from cyber_webapp.families.build import WebappBuild
from cyber_webapp.families.pentest import WebappPentest
from cyber_webapp.invariants import (
    no_orphan_nodes,
    oracle_path_exists,
    secret_must_be_held,
)
from cyber_webapp.ontology_v2 import ONTOLOGY_ID, webapp_ontology

from openrange.core.admit import Snapshot, admit
from openrange.core.pack import (
    Backing,
    Builder,
    BuildResult,
    Manifest,
    Pack,
    PackPrior,
    RuntimeHandle,
    TaskFamily,
    TaskSpec,
)
from openrange.world_ir import (
    Edge,
    Issue,
    Node,
    Ontology,
    Role,
    Visibility,
    WorldGraph,
)

# ---------------------------------------------------------------------------
# A minimal hand-built world graph honoring the cyber ontology.
# ---------------------------------------------------------------------------


def _build_cyber_world() -> WorldGraph:
    """One repo / host / public auth service / login endpoint / store /
    record / hidden flag / hidden SQLi vuln. The minimal shape that
    satisfies every pack invariant AND admits both task families.
    """
    g = WorldGraph(ontology=ONTOLOGY_ID)

    g.add_node(
        Node(
            "host.web",
            "host",
            attrs={"hostname": "web-01", "os": "linux", "zone": "dmz"},
        )
    )
    g.add_node(
        Node(
            "svc.auth",
            "service",
            attrs={
                "name": "auth-service",
                "kind": "auth",
                "language": "python",
                "exposure": "public",
            },
            roles={Role.ACTOR},
        )
    )
    g.add_node(
        Node(
            "ep.login",
            "endpoint",
            attrs={
                "path": "/login",
                "method": "POST",
                "auth_required": True,
                "behavior_ref": "auth/login",
            },
        )
    )
    g.add_node(
        Node(
            "store.users",
            "data_store",
            attrs={"name": "users", "kind": "sql", "engine": "sqlite"},
        )
    )
    g.add_node(
        Node(
            "rec.admin",
            "record",
            attrs={"key": "users/admin", "fields": {"name": "admin"}},
        )
    )
    g.add_node(
        Node(
            "sec.flag",
            "secret",
            attrs={
                "kind": "flag",
                "value_ref": "FLAG{cyber-webapp-v2-demo}",
                "description": "admin flag",
            },
            visibility=Visibility.HIDDEN,
        )
    )
    g.add_node(
        Node(
            "wk.sqli",
            "vulnerability",
            attrs={
                "kind": "sql_injection",
                "family": "code_web",
                "params": {"site": "POST /login"},
            },
            visibility=Visibility.HIDDEN,
        )
    )

    g.add_edge(Edge("e.svc-host", "runs_on", "svc.auth", "host.web"))
    g.add_edge(Edge("e.svc-ep", "exposes", "svc.auth", "ep.login"))
    g.add_edge(
        Edge(
            "e.svc-store",
            "backed_by",
            "svc.auth",
            "store.users",
            attrs={"mode": "readwrite"},
        )
    )
    g.add_edge(Edge("e.store-rec", "contains", "store.users", "rec.admin"))
    g.add_edge(
        Edge(
            "e.rec-sec",
            "holds",
            "rec.admin",
            "sec.flag",
            attrs={"field": "secret_token"},
        )
    )
    g.add_edge(
        Edge(
            "e.wk-ep",
            "affects",
            "wk.sqli",
            "ep.login",
            attrs={"injection_site": "username"},
        )
    )
    return g


# ---------------------------------------------------------------------------
# A stub WebappPack — the real one lands in Phase 2e (__init__.py rewrite).
# This one wires together the new ontology + invariants + families against
# a hand-built world so we can exercise admit() through the cyber shape.
# ---------------------------------------------------------------------------


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


class _StubBuilder(Builder):
    """v2 builder stub — emits the hand-built world + both families'
    tasks. The real procedural builder lands in Phase 2c."""

    def __init__(self, prior: PackPrior | None) -> None:
        self._prior = prior

    def build(self, manifest: Manifest) -> BuildResult:
        del manifest
        g = _build_cyber_world()
        tasks: list[TaskSpec] = []
        tasks.extend(WebappBuild().generate(g, {}, self._prior))
        tasks.extend(WebappPentest().generate(g, {}, self._prior))
        return BuildResult(
            graph=g,
            tasks=tasks,
            admission_meta={"builder": "cyber.webapp.v2.stub"},
        )


class _StubWebappPack(Pack):
    id = "webapp"
    version = "v2-stub"

    def ontology(self) -> Ontology:
        return webapp_ontology()

    def invariants(self):  # type: ignore[no-untyped-def]
        return [
            no_orphan_nodes,
            secret_must_be_held,
            oracle_path_exists,
        ]

    def make_builder(self, prior: PackPrior | None) -> Builder:
        return _StubBuilder(prior)

    def realize(self, graph: WorldGraph, backing: Backing) -> RuntimeHandle:
        del graph, backing
        return _NoopHandle()

    def task_families(self) -> list[TaskFamily]:
        return [WebappBuild(), WebappPentest()]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_cyber_ontology_v2_is_valid() -> None:
    """The new cyber ontology declares 10 node kinds + 11 edge kinds
    (the two old `affects` rows collapse into one EdgeKind with 2
    endpoint pairs)."""
    o = webapp_ontology()
    assert o.id == "cyber.webapp@v1"
    assert set(o.node_kinds) == {
        "host",
        "service",
        "endpoint",
        "account",
        "credential",
        "secret",
        "vulnerability",
        "network",
        "data_store",
        "record",
    }
    assert "affects" in o.edge_kinds
    # affects allows both endpoint -- target -- and service-- target shapes
    assert ("vulnerability", "endpoint") in o.edge_kinds["affects"].endpoints
    assert ("vulnerability", "service") in o.edge_kinds["affects"].endpoints


def test_cyber_world_admits_both_families_through_new_admit() -> None:
    """The single load-bearing test: one hand-built cyber world admits
    BOTH webapp.build and webapp.pentest with different entrypoints."""
    pack = _StubWebappPack()
    snap = admit(pack, manifest={"goal": "demo"})
    assert isinstance(snap, Snapshot), snap

    families = {t.feasibility_check for t in snap.tasks}
    assert families == {"webapp.build", "webapp.pentest"}, families

    # The two tasks entrypoint DIFFERENT node-kinds in the SAME world.
    entrypoint_kinds = {snap.graph.nodes[t.entrypoints[0]].kind for t in snap.tasks}
    assert entrypoint_kinds == {"service", "endpoint"}, entrypoint_kinds


def test_pentest_feasibility_walks_the_chain() -> None:
    g = _build_cyber_world()
    tasks = WebappPentest().generate(g, {}, None)
    assert len(tasks) == 1
    verdict = WebappPentest().check_feasibility(g, tasks[0])
    assert verdict.feasible, verdict.reason


def test_pentest_success_matches_value_ref() -> None:
    g = _build_cyber_world()
    pentest_task = WebappPentest().generate(g, {}, None)[0]
    flag_node = g.nodes[pentest_task.goal_nodes[0]]
    expected_flag = flag_node.attrs["value_ref"]
    result = WebappPentest().check_success(
        g,
        pentest_task,
        final_state={
            "flag_from_response": expected_flag,
            "requests_made": ["POST /login"],
        },
    )
    assert result.success
    assert result.subgoals["matched_flag"] is True


def test_pentest_failure_when_flag_missing() -> None:
    g = _build_cyber_world()
    pentest_task = WebappPentest().generate(g, {}, None)[0]
    result = WebappPentest().check_success(g, pentest_task, final_state={})
    assert not result.success


def test_build_feasibility_requires_exposed_endpoint() -> None:
    g = _build_cyber_world()
    tasks = WebappBuild().generate(g, {}, None)
    assert len(tasks) == 1
    verdict = WebappBuild().check_feasibility(g, tasks[0])
    assert verdict.feasible


def test_build_success_when_smoke_passes() -> None:
    g = _build_cyber_world()
    build_task = WebappBuild().generate(g, {}, None)[0]
    ok = WebappBuild().check_success(
        g,
        build_task,
        final_state={"endpoint_serves_200": True},
    )
    assert ok.success
    bad = WebappBuild().check_success(
        g,
        build_task,
        final_state={"endpoint_serves_200": False},
    )
    assert not bad.success


# ---------------------------------------------------------------------------
# Invariants
# ---------------------------------------------------------------------------


def test_no_orphan_nodes_passes_for_clean_world() -> None:
    issues: list[Issue] = no_orphan_nodes(_build_cyber_world())
    assert issues == []


def test_no_orphan_nodes_flags_disconnected_account() -> None:
    g = _build_cyber_world()
    # Drop an account with no edges: orphan.
    g.add_node(
        Node(
            "acct.lonely",
            "account",
            attrs={"username": "lonely", "role": "user"},
        )
    )
    issues = no_orphan_nodes(g)
    assert any(i.code == "orphan_node" for i in issues)


def test_secret_must_be_held_passes_for_clean_world() -> None:
    issues = secret_must_be_held(_build_cyber_world())
    assert issues == []


def test_oracle_path_exists_passes_for_clean_world() -> None:
    issues = oracle_path_exists(_build_cyber_world())
    assert issues == []


def test_oracle_path_exists_fails_when_vuln_removed() -> None:
    g = _build_cyber_world()
    g.nodes.pop("wk.sqli", None)
    g.edges.pop("e.wk-ep", None)
    issues = oracle_path_exists(g)
    assert any(i.code == "no_oracle_chain" for i in issues)


# ---------------------------------------------------------------------------
# End-to-end tests against the REAL WebappPack — exercises the full
# procedural sampler + builder + families pipeline.
# ---------------------------------------------------------------------------


def test_real_webapp_pack_identity() -> None:
    """The pack registers under id `webapp`, ships two families."""
    from cyber_webapp import WebappPack

    pack = WebappPack()
    assert pack.id == "webapp"
    assert pack.version == "v2"
    assert pack.ontology().id == "cyber.webapp@v1"
    assert {f.id for f in pack.task_families()} == {
        "webapp.build",
        "webapp.pentest",
    }


def test_real_webapp_pack_admits_with_procedural_sampler() -> None:
    """The procedural sampler + builder + families pipeline produces
    a snapshot with both families' tasks against a non-trivial graph."""
    from cyber_webapp import WebappPack

    pack = WebappPack()
    snap = admit(pack, manifest={"seed": 0}, max_repairs=3)
    assert isinstance(snap, Snapshot), snap
    # The procedural sampler produces a real-shaped world.
    assert len(snap.graph.nodes) >= 10
    assert len(snap.graph.edges) >= 8
    # Two tasks from two families.
    families = {t.feasibility_check for t in snap.tasks}
    assert families == {"webapp.build", "webapp.pentest"}
    # Different entrypoint kinds.
    entrypoint_kinds = {snap.graph.nodes[t.entrypoints[0]].kind for t in snap.tasks}
    assert entrypoint_kinds == {"service", "endpoint"}


def test_real_webapp_pack_seed_is_deterministic() -> None:
    """Same seed -> same snapshot id (content-addressed)."""
    from cyber_webapp import WebappPack

    snap_a = admit(WebappPack(), manifest={"seed": 7})
    snap_b = admit(WebappPack(), manifest={"seed": 7})
    assert isinstance(snap_a, Snapshot)
    assert isinstance(snap_b, Snapshot)
    assert snap_a.snapshot_id == snap_b.snapshot_id


def test_real_webapp_pack_seed_yields_distinct_worlds() -> None:
    """Different seeds -> different snapshot ids."""
    from cyber_webapp import WebappPack

    snap_a = admit(WebappPack(), manifest={"seed": 0})
    snap_b = admit(WebappPack(), manifest={"seed": 42})
    assert isinstance(snap_a, Snapshot)
    assert isinstance(snap_b, Snapshot)
    assert snap_a.snapshot_id != snap_b.snapshot_id


def test_real_webapp_pack_lineage_carries_pack_provenance() -> None:
    """The Snapshot's lineage captures pack id, version, attempt count,
    and the builder's admission_meta (seed, prior source, etc.)."""
    from cyber_webapp import WebappPack

    snap = admit(WebappPack(), manifest={"seed": 0})
    assert isinstance(snap, Snapshot)
    assert snap.lineage["pack"] == "webapp"
    assert snap.lineage["pack_version"] == "v2"
    assert snap.lineage["builder"] == "cyber.webapp.v2"
    assert snap.lineage["seed"] == 0
    assert "prior_source" in snap.lineage


def test_real_webapp_pack_history_records_all_phases() -> None:
    """admit() records build / validate / feasibility / freeze phases."""
    from cyber_webapp import WebappPack

    snap = admit(WebappPack(), manifest={"seed": 0})
    assert isinstance(snap, Snapshot)
    phases = [e.phase for e in snap.history]
    # First successful pass: 4 phases. With repairs: more.
    assert phases[0] == "build"
    assert "validate" in phases
    assert "feasibility" in phases
    assert phases[-1] == "freeze"
