"""Integration tests for the cyber webapp pack.

Two flavors of tests:

  1. Hand-built world fixtures + a stub :class:`_StubWebappPack`
     exercise the ontology, invariants, families, and admission
     against a minimal graph.

  2. End-to-end tests against :class:`WebappPack` run the full
     procedural builder + sampling + admission pipeline.

The load-bearing assertion: one cyber webapp world admits BOTH
``webapp.build`` and ``webapp.pentest`` task families with different
entrypoint kinds — "one pack, many TaskFamilies" applied to this pack.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from cyber_webapp.families.build import WebappBuild
from cyber_webapp.families.pentest import WebappPentest
from cyber_webapp.invariants import (
    credential_reuse_binding,
    no_orphan_nodes,
    oracle_path_exists,
    secret_must_be_held,
    sqli_targets_db_backed_service,
)
from cyber_webapp.ontology import ONTOLOGY_ID, webapp_ontology
from graphschema import (
    Edge,
    Issue,
    Node,
    Ontology,
    Role,
    Visibility,
    WorldGraph,
)
from openrange_pack_sdk import (
    Backing,
    Builder,
    BuildResult,
    Manifest,
    Pack,
    PackPrior,
    RuntimeHandle,
    Snapshot,
    TaskFamily,
    TaskSpec,
)

from openrange.core.admit import admit


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
                "public_url": "/svc/auth/login",
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

    g.add_node(
        Node(
            "svc.api",
            "service",
            attrs={
                "name": "api-service",
                "kind": "api",
                "language": "python",
                "exposure": "internal",
            },
            roles={Role.ACTOR},
        )
    )
    g.add_node(
        Node(
            "ep.api.items",
            "endpoint",
            attrs={
                "path": "/api/items",
                "public_url": "/svc/api/api/items",
                "method": "GET",
                "auth_required": False,
                "behavior_ref": "api/list",
            },
        )
    )

    g.add_edge(Edge("e.svc-host", "runs_on", "svc.auth", "host.web"))
    g.add_edge(Edge("e.api-host", "runs_on", "svc.api", "host.web"))
    g.add_edge(Edge("e.svc-ep", "exposes", "svc.auth", "ep.login"))
    g.add_edge(Edge("e.api-ep", "exposes", "svc.api", "ep.api.items"))
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
    """Stub builder — emits the hand-built world + both families' tasks."""

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


def test_cyber_ontology_is_valid() -> None:
    """The cyber ontology declares the expected node and edge kinds."""
    o = webapp_ontology()
    assert o.id == "cyber.webapp@v2"
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


def _binding_chain(*, producer: str | None = "v_leak") -> WorldGraph:
    # leak -enables-> gate; the gate's endpoint requires a credential. `producer`
    # is the vuln that `produces` that credential (None = nobody produces it).
    g = WorldGraph(ontology=ONTOLOGY_ID)
    g.add_node(
        Node(
            "v_leak",
            "vulnerability",
            attrs={"kind": "credential_leak"},
            visibility=Visibility.HIDDEN,
        )
    )
    g.add_node(
        Node(
            "v_gate",
            "vulnerability",
            attrs={"kind": "credential_gated_flag"},
            visibility=Visibility.HIDDEN,
        )
    )
    g.add_node(Node("ep_gate", "endpoint", attrs={"path": "/internal/vault"}))
    g.add_node(Node("cred", "credential", attrs={"kind": "token", "value_ref": "tok0"}))
    g.add_edge(Edge("e_aff", "affects", "v_gate", "ep_gate"))
    g.add_edge(Edge("e_en", "enables", "v_leak", "v_gate"))
    g.add_edge(Edge("e_req", "requires_credential", "ep_gate", "cred"))
    if producer is not None:
        g.add_edge(Edge("e_prod", "produces", producer, "cred"))
    return g


def test_credential_binding_accepts_a_valid_chain() -> None:
    assert credential_reuse_binding(_binding_chain()) == []


def test_credential_binding_rejects_an_unproduced_credential() -> None:
    issues = credential_reuse_binding(_binding_chain(producer=None))
    assert any(i.code == "credential_binding" for i in issues)


def test_credential_binding_rejects_a_producer_not_strictly_earlier() -> None:
    # The consuming gate produces its own required credential — not obtainable
    # before the gate, so the binding is unsatisfiable.
    issues = credential_reuse_binding(_binding_chain(producer="v_gate"))
    assert any(i.code == "credential_binding" for i in issues)


def test_credential_binding_rejects_a_swapped_chain_kind() -> None:
    # A diversify swap that rewrites a chain hop's kind in place (keeping its
    # produces edge) must be caught, or an unsolvable world admits clean.
    g = _binding_chain()
    g.nodes["v_leak"].attrs["kind"] = "idor"
    assert any(i.code == "credential_binding" for i in credential_reuse_binding(g))


def test_credential_binding_rejects_two_producers() -> None:
    g = _binding_chain()
    g.add_node(
        Node(
            "v_leak2",
            "vulnerability",
            attrs={"kind": "credential_leak"},
            visibility=Visibility.HIDDEN,
        )
    )
    g.add_edge(Edge("e_prod2", "produces", "v_leak2", "cred"))
    assert any(i.code == "credential_binding" for i in credential_reuse_binding(g))


def test_credential_binding_rejects_when_no_gate_vuln() -> None:
    # The required endpoint's only vuln is no longer a gate kind (a swap rewrote
    # it), so nothing actually consumes the credential it gates on.
    g = _binding_chain()
    g.nodes["v_gate"].attrs["kind"] = "idor"
    assert any(i.code == "credential_binding" for i in credential_reuse_binding(g))


def test_credential_binding_rejects_when_producer_cannot_reach_gate() -> None:
    # Producer is a valid hop but no enables path reaches the gate (an enables
    # cycle that never arrives), so the gate's credential is unobtainable.
    g = WorldGraph(ontology=ONTOLOGY_ID)
    for vid, kind in (
        ("v_leak", "credential_leak"),
        ("v_relay", "credential_gated_relay"),
        ("v_gate", "credential_gated_flag"),
    ):
        g.add_node(
            Node(
                vid,
                "vulnerability",
                attrs={"kind": kind},
                visibility=Visibility.HIDDEN,
            )
        )
    g.add_node(Node("ep_gate", "endpoint", attrs={"path": "/internal/vault"}))
    g.add_node(Node("cred", "credential", attrs={"kind": "token", "value_ref": "t"}))
    g.add_edge(Edge("e_prod", "produces", "v_leak", "cred"))
    g.add_edge(Edge("e_aff", "affects", "v_gate", "ep_gate"))
    g.add_edge(Edge("e_req", "requires_credential", "ep_gate", "cred"))
    g.add_edge(Edge("e_en1", "enables", "v_leak", "v_relay"))
    g.add_edge(Edge("e_en2", "enables", "v_relay", "v_leak"))
    assert any(i.code == "credential_binding" for i in credential_reuse_binding(g))


def test_real_webapp_pack_identity() -> None:
    """The pack registers under id `webapp`, ships two families."""
    from cyber_webapp import WebappPack

    pack = WebappPack()
    assert pack.id == "webapp"
    assert pack.version == "v2"
    assert pack.ontology().id == "cyber.webapp@v2"
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
    and the builder's admission_meta (seed, prior source, world difficulty)."""
    from cyber_webapp import WebappPack
    from cyber_webapp.difficulty import world_difficulty

    snap = admit(WebappPack(), manifest={"seed": 0})
    assert isinstance(snap, Snapshot)
    assert snap.lineage["pack"] == "webapp"
    assert snap.lineage["pack_version"] == "v2"
    assert snap.lineage["builder"] == "cyber.webapp.v2"
    assert snap.lineage["seed"] == 0
    assert "prior_source" in snap.lineage
    # the #322 solve-path-cost metric is persisted, queryable without recompute
    assert snap.lineage["world_difficulty"] == float(world_difficulty(snap.graph))


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


def test_sampler_fills_public_url_on_every_endpoint() -> None:
    from cyber_webapp import WebappPack

    snap = admit(WebappPack(), manifest={"seed": 42})
    assert isinstance(snap, Snapshot)
    endpoints = [n for n in snap.graph.nodes.values() if n.kind == "endpoint"]
    assert endpoints
    for ep in endpoints:
        assert isinstance(ep.attrs.get("public_url"), str)
        assert ep.attrs["public_url"].startswith("/")


def test_public_url_matches_owning_service_exposure() -> None:
    """Public-exposure services serve at the root path; internal services
    are mounted under `/svc/<name>`. The convention lives in the graph,
    set at sampling time."""
    from cyber_webapp import WebappPack

    snap = admit(WebappPack(), manifest={"seed": 42})
    assert isinstance(snap, Snapshot)
    service_of_endpoint = {
        e.dst: snap.graph.nodes[e.src]
        for e in snap.graph.edges.values()
        if e.kind == "exposes"
    }
    for ep in snap.graph.by_kind("endpoint"):
        service = service_of_endpoint[ep.id]
        public_url = ep.attrs["public_url"]
        path = ep.attrs["path"]
        if service.attrs["exposure"] == "public":
            assert public_url == path
        else:
            assert public_url == f"/svc/{service.attrs['name']}{path}"


def test_pentest_instruction_uses_public_url_not_internal_path() -> None:
    """The task instruction names the URL the agent will actually hit."""
    from cyber_webapp import WebappPack

    snap = admit(WebappPack(), manifest={"seed": 42})
    assert isinstance(snap, Snapshot)
    pentest = next(t for t in snap.tasks if t.feasibility_check == "webapp.pentest")
    ep = snap.graph.nodes[pentest.entrypoints[0]]
    assert ep.attrs["public_url"] in pentest.instruction


def test_realizer_serves_endpoints_at_their_public_url() -> None:
    """End-to-end: the live HTTP server mounts each endpoint at exactly the
    `public_url` the graph declares. No hidden URL convention."""
    import time
    import urllib.error
    import urllib.request

    from cyber_webapp import WebappPack

    snap = admit(WebappPack(), manifest={"seed": 42})
    assert isinstance(snap, Snapshot)
    pack = WebappPack()
    handle = pack.realize(snap.graph, Backing.PROCESS)
    handle.reset()
    surface = handle.surface()
    base_url = str(surface["base_url"])
    try:
        time.sleep(0.3)
        for ep in snap.graph.by_kind("endpoint"):
            url = base_url + str(ep.attrs["public_url"])
            try:
                with urllib.request.urlopen(url, timeout=5) as r:
                    status = r.status
            except urllib.error.HTTPError as e:
                status = e.code
            assert status != 404, (
                f"endpoint {ep.id} declared public_url={ep.attrs['public_url']!r} "
                f"but the realizer returned 404"
            )
    finally:
        handle.stop()


def test_invariant_rejects_sqli_on_service_without_data_store() -> None:
    """A SQLi vuln on an endpoint whose service has no `backed_by` data
    store is structurally infeasible — the generated handler would
    query a non-existent table."""
    g = WorldGraph(ontology=ONTOLOGY_ID)
    g.add_node(
        Node(
            "host.web",
            "host",
            attrs={"hostname": "h", "os": "linux", "zone": "dmz"},
        )
    )
    g.add_node(
        Node(
            "svc.web",
            "service",
            attrs={
                "name": "web",
                "kind": "web",
                "language": "python",
                "exposure": "public",
            },
        )
    )
    g.add_node(
        Node(
            "ep.search",
            "endpoint",
            attrs={
                "path": "/search",
                "public_url": "/search",
                "method": "GET",
                "auth_required": False,
                "behavior_ref": "web.default",
            },
        )
    )
    g.add_edge(Edge("e0", "exposes", "svc.web", "ep.search"))
    g.add_edge(Edge("e1", "runs_on", "svc.web", "host.web"))
    g.add_node(
        Node(
            "vuln.sqli",
            "vulnerability",
            attrs={"kind": "sql_injection", "family": "code_web", "params": {}},
            visibility=Visibility.HIDDEN,
        )
    )
    g.add_edge(Edge("e2", "affects", "vuln.sqli", "ep.search"))

    issues = sqli_targets_db_backed_service(g)
    codes = {i.code for i in issues}
    assert "sqli_without_db_backing" in codes


def test_invariant_allows_sqli_on_db_backed_service() -> None:
    """The same vuln on a service WITH backed_by data_store is fine."""
    g = WorldGraph(ontology=ONTOLOGY_ID)
    g.add_node(
        Node(
            "host.api",
            "host",
            attrs={"hostname": "h", "os": "linux", "zone": "dmz"},
        )
    )
    g.add_node(
        Node(
            "svc.api",
            "service",
            attrs={
                "name": "api",
                "kind": "api",
                "language": "python",
                "exposure": "internal",
            },
        )
    )
    g.add_node(
        Node(
            "ep.search",
            "endpoint",
            attrs={
                "path": "/search",
                "public_url": "/svc/api/search",
                "method": "GET",
                "auth_required": False,
                "behavior_ref": "api.default",
            },
        )
    )
    g.add_node(
        Node(
            "ds.users",
            "data_store",
            attrs={"name": "users", "kind": "sql", "engine": "sqlite"},
        )
    )
    g.add_edge(Edge("e0", "exposes", "svc.api", "ep.search"))
    g.add_edge(Edge("e1", "runs_on", "svc.api", "host.api"))
    g.add_edge(Edge("e2", "backed_by", "svc.api", "ds.users"))
    g.add_node(
        Node(
            "vuln.sqli",
            "vulnerability",
            attrs={"kind": "sql_injection", "family": "code_web", "params": {}},
            visibility=Visibility.HIDDEN,
        )
    )
    g.add_edge(Edge("e3", "affects", "vuln.sqli", "ep.search"))

    assert sqli_targets_db_backed_service(g) == []


def test_sampler_never_places_sqli_on_non_db_backed_service() -> None:
    """Across multiple seeds, every sampled SQLi vuln targets an endpoint
    on a DB-backed service. The sampler honors the same constraint the
    invariant enforces."""
    from cyber_webapp import WebappPack

    pack = WebappPack()
    for seed in range(20):
        snap = admit(pack, manifest={"seed": seed})
        if not isinstance(snap, Snapshot):
            continue
        graph = snap.graph
        db_backed = {e.src for e in graph.edges.values() if e.kind == "backed_by"}
        service_of = {e.dst: e.src for e in graph.edges.values() if e.kind == "exposes"}
        for vuln in graph.by_kind("vulnerability"):
            if vuln.attrs.get("kind") != "sql_injection":
                continue
            for affects in graph.out_edges(vuln.id, "affects"):
                target = graph.nodes[affects.dst]
                if target.kind == "service":
                    svc_id = target.id
                else:
                    svc_id = service_of[target.id]
                assert svc_id in db_backed, (
                    f"seed={seed}: SQLi vuln {vuln.id} targets {target.id} "
                    f"on service {svc_id} which has no backed_by data_store"
                )
