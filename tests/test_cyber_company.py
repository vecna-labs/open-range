"""Company worlds (DESIGN.md §11): a believable multi-service estate the agent recons
and pivots through. Generation + a PROCESS solve here; the docker-gated test proves the
same recon→pivot recovers the flag across real containers."""

from __future__ import annotations

import dataclasses
import json
import re
import shutil
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pytest
from cyber_webapp import (
    NetworkedContainerWebappRuntime,
    WebappPack,
    _is_networked,
    monotone_chain_gate,
)
from cyber_webapp.codegen.seeding import project_seed
from cyber_webapp.difficulty import world_difficulty
from cyber_webapp.invariants import unique_vuln_per_endpoint
from cyber_webapp.mutation import available_mutations
from cyber_webapp.reference_solver import solve_chain
from graphschema import Edge, Node, Visibility, WorldGraph
from openrange_pack_sdk import Backing, PoolableRuntime, Snapshot
from openrange_trl import EpisodeEnv

from examples.tools import WEB_TOOLS
from examples.verify import consequence_gate, verdict
from openrange.core.admit import admit
from openrange.core.curriculum import _clone_graph, auto_evolve
from openrange.core.episode import EpisodeReport, EpisodeService
from openrange.pool import (
    _MAX_PRIORITY,
    EvalPool,
    RoundMetrics,
    RunRound,
    WorldPool,
    run_pool_curriculum,
)

_COMPANY_MANIFEST = {
    "pack": {"id": "webapp"},
    "runtime": {"tick": {"mode": "off"}},
    "npc": [],
    "seed": 3,
    "company": True,
}
_DEFAULT_MANIFEST = {
    "pack": {"id": "webapp"},
    "runtime": {"tick": {"mode": "off"}},
    "npc": [],
    "seed": 3,
}
_LATERAL_MANIFEST = {**_COMPANY_MANIFEST, "lateral_movement": True}


def _admit(manifest: dict[str, object]) -> Snapshot:
    snap = admit(WebappPack(), manifest=manifest, max_repairs=3)
    assert isinstance(snap, Snapshot), snap
    return snap


def _public_service(graph: WorldGraph) -> Node:
    return next(
        n for n in graph.by_kind("service") if n.attrs.get("exposure") == "public"
    )


def _get(base_url: str, path: str) -> tuple[int, str]:
    try:
        with urllib.request.urlopen(f"{base_url}{path}", timeout=15) as resp:
            return resp.status, resp.read().decode()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode()


def _flag_from(body: str) -> str:
    try:
        obj = json.loads(body)
    except json.JSONDecodeError:
        obj = None
    if isinstance(obj, dict):
        for key in ("credential", "data", "flag", "secret"):
            value = obj.get(key)
            if isinstance(value, str):
                return value
    found = re.search(r"(?:ghp_|sk_live_|AKIA)[A-Za-z0-9_]+|[0-9a-f-]{32,36}", body)
    return found.group(0) if found else ""


def _pentest_only(_evolved: Snapshot, mutation: object) -> bool:
    return getattr(mutation, "family", None) == "webapp.pentest"


def _breach_report(pack: WebappPack, work_dir: Path, snap: Snapshot) -> EpisodeReport:
    task = next(t for t in snap.tasks if t.meta.get("family") == "webapp.pentest")
    svc = EpisodeService(pack, work_dir)
    env = EpisodeEnv(service=svc, snapshots={snap.snapshot_id: snap}, tools=WEB_TOOLS)
    try:
        entry = str(snap.graph.nodes[task.entrypoints[0]].attrs["public_url"])
        env.reset(snapshot_id=snap.snapshot_id, task_id=task.id)
        env.http_get(entry)
        trace = solve_chain(snap.graph, lambda p: env.http_get(p).split("\n", 1)[-1])
        env.submit(json.dumps({"flag": _flag_from(trace.terminal)}))
        env._finalize()
        assert env.report is not None
        return env.report
    finally:
        svc.close()


def _reach_only_report(
    pack: WebappPack, work_dir: Path, snap: Snapshot
) -> EpisodeReport:
    task = next(t for t in snap.tasks if t.meta.get("family") == "webapp.pentest")
    svc = EpisodeService(pack, work_dir)
    env = EpisodeEnv(service=svc, snapshots={snap.snapshot_id: snap}, tools=WEB_TOOLS)
    try:
        entry = str(snap.graph.nodes[task.entrypoints[0]].attrs["public_url"])
        env.reset(snapshot_id=snap.snapshot_id, task_id=task.id)
        env.http_get(entry)
        env._finalize()
        assert env.report is not None
        return env.report
    finally:
        svc.close()


def _solve_round(
    pack: WebappPack,
    work_dir: Path,
    rows: list[dict[str, object]],
    snapshots: list[Snapshot],
) -> dict[tuple[str, str], list[EpisodeReport]]:
    by_id = {s.snapshot_id: s for s in snapshots}
    svc = EpisodeService(pack, work_dir, warm_capacity=len(by_id))
    env = EpisodeEnv(service=svc, snapshots=by_id, tools=WEB_TOOLS)
    out: dict[tuple[str, str], list[EpisodeReport]] = {}
    try:
        for row in rows:
            sid, tid = str(row["snapshot_id"]), str(row["task_id"])
            snap = by_id[sid]
            task = next(t for t in snap.tasks if t.id == tid)
            entry = str(snap.graph.nodes[task.entrypoints[0]].attrs["public_url"])
            env.reset(snapshot_id=sid, task_id=tid)
            env.http_get(entry)
            trace = solve_chain(
                snap.graph, lambda p: env.http_get(p).split("\n", 1)[-1]
            )
            env.submit(json.dumps({"flag": _flag_from(trace.terminal)}))
            env._finalize()
            assert env.report is not None
            out.setdefault((sid, tid), []).append(env.report)
        return out
    finally:
        svc.close()


def test_company_world_is_multi_service_and_segmented() -> None:
    graph = _admit(_COMPANY_MANIFEST).graph
    services = list(graph.by_kind("service"))
    assert len(services) >= 6  # a believable estate, not the minimal pair

    networks = {n.attrs.get("name"): n for n in graph.by_kind("network")}
    assert set(networks) == {"dmz", "internal"}  # segmented, not one flat segment
    public = _public_service(graph)

    def nets_of(svc: Node) -> set[str]:
        return {e.dst for e in graph.out_edges(svc.id, "connected_to")}

    assert nets_of(public) == {"net_dmz"}
    for svc in services:
        if svc.attrs.get("exposure") != "public":
            assert nets_of(svc) == {"net_internal"}

    assert _is_networked(graph)  # routes to the per-service networked runtime


def test_company_plants_recon_that_names_internal_hosts() -> None:
    graph = _admit(_COMPANY_MANIFEST).graph
    recon = next(
        n
        for n in graph.by_kind("vulnerability")
        if n.attrs.get("kind") == "config_disclosure"
    )
    # The recon sits on the public service (the agent's only entry).
    recon_eps = {e.dst for e in graph.out_edges(recon.id, "affects")}
    public_eps = {e.dst for e in graph.out_edges(_public_service(graph).id, "exposes")}
    assert recon_eps <= public_eps

    internal_names = {
        str(n.attrs.get("name"))
        for n in graph.by_kind("service")
        if n.attrs.get("exposure") != "public"
    }
    disclosed = set(recon.attrs["params"]["internal_services"])
    assert disclosed == internal_names  # names every internal host, incl. the flag's

    ssrf = next(
        n for n in graph.by_kind("vulnerability") if n.attrs.get("kind") == "ssrf"
    )
    assert str(ssrf.attrs["params"]["internal_host"]) in disclosed


def test_company_solves_on_process(tmp_path: Path) -> None:
    snap = _admit(_COMPANY_MANIFEST)
    graph = snap.graph
    flag = str(graph.nodes["secret_flag"].attrs["value_ref"])
    ssrf = next(
        n for n in graph.by_kind("vulnerability") if n.attrs.get("kind") == "ssrf"
    )
    flag_host = str(ssrf.attrs["params"]["internal_host"])
    recon = next(n for n in graph.by_kind("endpoint") if n.id.endswith("_recon"))
    recon_url = str(recon.attrs["public_url"])

    pentest = next(t for t in snap.tasks if t.meta.get("family") == "webapp.pentest")
    svc = EpisodeService(WebappPack(), tmp_path)
    try:
        handle = svc.start_episode(snap, pentest.id)
        base = str(svc.surface(handle)["base_url"])

        # Recon discloses the internal estate (the flag host among them) — but not the
        # flag itself; that still takes the pivot.
        status, recon_body = _get(base, recon_url)
        assert status == 200
        assert flag_host in recon_body
        assert flag not in recon_body

        trace = solve_chain(graph, lambda p: _get(base, p)[1])
        assert flag in trace.terminal
        assert all(flag not in probe for probe in trace.probes)
    finally:
        svc.close()


def test_company_reward_surface_grades_the_breach(tmp_path: Path) -> None:
    # The notebook's reward surface (examples/trl_grpo_cyber.ipynb §5), pinned on the
    # company world: reaching the storefront earns 1/3, a wrong flag 2/3, and the full
    # recon→pivot→exfiltrate breach 1.0 (passed). That spread is the GRPO gradient.
    snap = _admit(_COMPANY_MANIFEST)
    graph = snap.graph
    flag = str(graph.nodes["secret_flag"].attrs["value_ref"])
    pentest = next(t for t in snap.tasks if t.meta.get("family") == "webapp.pentest")
    entry = str(graph.nodes[pentest.entrypoints[0]].attrs["public_url"])

    services: list[EpisodeService] = []

    def _env() -> EpisodeEnv:
        svc = EpisodeService(WebappPack(), tmp_path / f"env{len(services)}")
        services.append(svc)
        env = EpisodeEnv(
            service=svc, snapshots={snap.snapshot_id: snap}, tools=WEB_TOOLS
        )
        env.reset(snapshot_id=snap.snapshot_id, task_id=pentest.id)
        return env

    def _reward(env: EpisodeEnv) -> float:
        env._finalize()
        return env.reward

    try:
        reach = _env()
        reach.http_get(entry)
        assert _reward(reach) == pytest.approx(1 / 3)  # reached_endpoint only

        wrong = _env()
        wrong.http_get(entry)
        wrong.submit(json.dumps({"flag": "not-the-real-flag"}))
        assert _reward(wrong) == pytest.approx(2 / 3)  # + extracted_anything, but wrong

        breach = _env()
        breach.http_get(entry)
        trace = solve_chain(graph, lambda p: breach.http_get(p).split("\n", 1)[-1])
        assert flag in trace.terminal  # genuinely exfiltrated over the wire
        breach.submit(json.dumps({"flag": flag}))
        assert _reward(breach) == 1.0
        assert breach.report is not None and breach.report.passed
    finally:
        for svc in services:
            svc.close()


def test_warm_pool_reuses_a_poolable_world(tmp_path: Path) -> None:
    # A poolable world (the company: no write/exec vuln) is booted once and reused
    # across episodes on one EpisodeService — as a TRL env is reused across training
    # steps. stop_episode keeps it warm; the next start_episode hands back the SAME
    # runtime after a cheap reset_episode, and each episode still grades the breach.
    snap = _admit(_COMPANY_MANIFEST)
    pentest = next(t for t in snap.tasks if t.meta.get("family") == "webapp.pentest")
    flag = str(snap.graph.nodes["secret_flag"].attrs["value_ref"])
    entry = str(snap.graph.nodes[pentest.entrypoints[0]].attrs["public_url"])

    svc = EpisodeService(WebappPack(), tmp_path)
    env = EpisodeEnv(service=svc, snapshots={snap.snapshot_id: snap}, tools=WEB_TOOLS)
    warm: list[object] = []
    try:
        for _ in range(2):
            env.reset(snapshot_id=snap.snapshot_id, task_id=pentest.id)
            env.http_get(entry)
            trace = solve_chain(
                snap.graph, lambda p: env.http_get(p).split("\n", 1)[-1]
            )
            assert flag in trace.terminal  # exfiltrated over the wire, each episode
            env.submit(json.dumps({"flag": flag}))
            env._finalize()
            assert env.report is not None and env.report.passed
            warm.append(svc._warm[snap.snapshot_id])  # kept warm, not torn down
        assert warm[0] is warm[1]  # the SAME world was reused, not rebooted
    finally:
        svc.close()
    assert not svc._warm  # close() evicts the warm world


def test_write_exec_world_is_not_poolable() -> None:
    # A command_injection world runs an agent-driven shell on a writable container,
    # so its state can cross episodes — it must never be kept warm. A response-leak
    # world (sql_injection) only reads immutable state, so it is poolable.
    cmdi = WebappPack().realize(
        _admit(
            {
                **_DEFAULT_MANIFEST,
                "loot_shapes": {"file": 1, "db": 0},
                "vuln_kinds": {"command_injection": 1},
            }
        ).graph,
        Backing.PROCESS,
    )
    assert isinstance(cmdi, PoolableRuntime) and not cmdi.poolable()
    sqli = WebappPack().realize(
        _admit(
            {
                **_DEFAULT_MANIFEST,
                "loot_shapes": {"db": 1, "file": 0},
                "vuln_kinds": {"sql_injection": 1},
            }
        ).graph,
        Backing.PROCESS,
    )
    assert isinstance(sqli, PoolableRuntime) and sqli.poolable()


def test_world_difficulty_rises_with_chain_depth() -> None:
    flat = world_difficulty(_admit(_DEFAULT_MANIFEST).graph)
    company = world_difficulty(_admit(_COMPANY_MANIFEST).graph)
    lateral = world_difficulty(
        _admit({**_COMPANY_MANIFEST, "lateral_movement": True}).graph
    )
    assert flat < company < lateral


def test_warm_cache_is_a_bounded_lru(tmp_path: Path) -> None:
    snaps = [_admit({**_COMPANY_MANIFEST, "seed": s}) for s in (1, 2, 3)]
    assert len({s.snapshot_id for s in snaps}) == 3
    svc = EpisodeService(WebappPack(), tmp_path, warm_capacity=2)
    try:
        for snap in snaps:
            pentest = next(
                t for t in snap.tasks if t.meta.get("family") == "webapp.pentest"
            )
            svc.stop_episode(svc.start_episode(snap, pentest.id))
        warm_ids = list(svc._warm)
        assert len(warm_ids) == 2
        assert snaps[0].snapshot_id not in warm_ids
        assert {snaps[1].snapshot_id, snaps[2].snapshot_id} == set(warm_ids)
    finally:
        svc.close()


def test_pool_round_keeps_the_easy_tail() -> None:
    # The mix floor is enforced at round composition, so the easiest world stays in
    # even when its priority is zeroed.
    pack = WebappPack()
    pool = WorldPool.seed(
        pack,
        [
            _DEFAULT_MANIFEST,
            _COMPANY_MANIFEST,
            {**_COMPANY_MANIFEST, "lateral_movement": True},
        ],
        difficulty_fn=lambda s: float(world_difficulty(s.graph)),
        family="webapp.pentest",
        max_size=8,
        mix_floor=0.5,
    )
    easiest = min(pool._members.values(), key=lambda m: m.difficulty)
    for member in pool._members.values():
        member.priority = 0.0 if member is easiest else 1.0
    rows = pool.round_rows(groups=2, num_generations=1)
    chosen = {row["snapshot_id"] for row in rows}
    assert easiest.snapshot.snapshot_id in chosen


def test_pool_curriculum_grows_bounds_and_keeps_a_mix(tmp_path: Path) -> None:
    pack = WebappPack()
    seeds = [{**_COMPANY_MANIFEST, "seed": s} for s in range(4)]
    round_no = [0]

    def run_round(
        rows: list[dict[str, object]], snapshots: list[Snapshot]
    ) -> dict[tuple[str, str], list[EpisodeReport]]:
        round_no[0] += 1
        return _solve_round(pack, tmp_path / f"r{round_no[0]}", rows, snapshots)

    def build_and_run() -> tuple[WorldPool, float]:
        pool = WorldPool.seed(
            pack,
            seeds,
            difficulty_fn=lambda s: float(world_difficulty(s.graph)),
            family="webapp.pentest",
            max_size=5,
        )
        assert len(pool) == 4
        seed_min = min(m.difficulty for m in pool._members.values())
        run_pool_curriculum(
            pool,
            run_round,
            rounds=2,
            pack=pack,
            groups=3,
            num_generations=2,
            gate=_pentest_only,
        )
        return pool, seed_min

    pool, seed_min = build_and_run()
    diffs = [m.difficulty for m in pool._members.values()]
    assert 4 < len(pool) <= 5
    assert min(diffs) == seed_min
    assert max(diffs) > seed_min
    assert pool.keys() == build_and_run()[0].keys()


def test_grown_child_survives_a_full_pool(tmp_path: Path) -> None:
    # A child must not be evicted the round it is born: older members are forced
    # above it on staleness, yet eviction falls on one of them.
    pack = WebappPack()
    pool = WorldPool.seed(
        pack,
        [
            {**_COMPANY_MANIFEST, "seed": 0},
            {**_COMPANY_MANIFEST, "lateral_movement": True, "seed": 1},
            {**_COMPANY_MANIFEST, "lateral_movement": True, "seed": 2},
        ],
        difficulty_fn=lambda s: float(world_difficulty(s.graph)),
        family="webapp.pentest",
        max_size=3,
    )
    assert len(pool) == 3
    original = pool.keys()
    easiest = min(pool._members.values(), key=lambda m: m.difficulty)
    for member in pool._members.values():
        if member is not easiest:
            member.priority = 1.5
    report = _breach_report(pack, tmp_path, easiest.snapshot)
    pool.update({easiest.key: [report]}, pack=pack, gate=_pentest_only)
    assert len(pool) == 3
    assert pool.keys() - original


def test_staleness_priority_is_capped() -> None:
    pack = WebappPack()
    pool = WorldPool.seed(
        pack,
        [_DEFAULT_MANIFEST, _COMPANY_MANIFEST],
        difficulty_fn=lambda s: float(world_difficulty(s.graph)),
        family="webapp.pentest",
        max_size=8,
    )
    for _ in range(40):
        pool.update({}, pack=pack, gate=_pentest_only)
    priorities = [m.priority for m in pool._members.values()]
    assert max(priorities) == _MAX_PRIORITY
    assert all(p <= _MAX_PRIORITY for p in priorities)


def test_round_rows_never_exceeds_groups() -> None:
    # mix_floor is a fraction: a value above 1 must not inflate a round past its
    # group budget.
    pack = WebappPack()
    pool = WorldPool.seed(
        pack,
        [{**_COMPANY_MANIFEST, "seed": s} for s in range(6)],
        difficulty_fn=lambda s: float(world_difficulty(s.graph)),
        family="webapp.pentest",
        max_size=8,
        mix_floor=2.0,
    )
    assert len(pool) == 6
    rows = pool.round_rows(groups=1, num_generations=2)
    groups = {(row["snapshot_id"], row["task_id"]) for row in rows}
    assert len(groups) == 1
    assert pool.round_rows(groups=0, num_generations=2) == []


def test_pool_holds_when_no_harder_world_admits(tmp_path: Path) -> None:
    pack = WebappPack()
    pool = WorldPool.seed(
        pack,
        [_COMPANY_MANIFEST],
        difficulty_fn=lambda s: float(world_difficulty(s.graph)),
        family="webapp.pentest",
        max_size=5,
    )
    member = next(iter(pool._members.values()))
    before = pool.keys()
    report = _breach_report(pack, tmp_path, member.snapshot)
    capped = pool.update(
        {member.key: [report]}, pack=pack, gate=lambda _evolved, _mutation: False
    )
    assert pool.keys() == before
    assert capped is True


def test_regrowing_the_same_parent_does_not_duplicate(tmp_path: Path) -> None:
    # Evolution is deterministic: the same parent yields the same child key, so the
    # second growth is a no-op rather than a duplicate.
    pack = WebappPack()
    pool = WorldPool.seed(
        pack,
        [_COMPANY_MANIFEST],
        difficulty_fn=lambda s: float(world_difficulty(s.graph)),
        family="webapp.pentest",
        max_size=5,
    )
    parent = next(iter(pool._members.values()))
    pool.update(
        {parent.key: [_breach_report(pack, tmp_path / "a", parent.snapshot)]},
        pack=pack,
        gate=_pentest_only,
    )
    grew_to = pool.keys()
    assert len(grew_to) == 2
    pool.update(
        {parent.key: [_breach_report(pack, tmp_path / "b", parent.snapshot)]},
        pack=pack,
        gate=_pentest_only,
    )
    assert pool.keys() == grew_to


def test_evolution_hardens_a_world_the_agent_masters(tmp_path: Path) -> None:
    pack = WebappPack()
    pool = WorldPool.seed(
        pack,
        [_LATERAL_MANIFEST],
        difficulty_fn=lambda s: float(world_difficulty(s.graph)),
        family="webapp.pentest",
        max_size=8,
    )
    member = next(iter(pool._members.values()))
    before = pool.keys()
    report = _breach_report(pack, tmp_path, member.snapshot)
    assert report.passed
    pool.update({member.key: [report]}, pack=pack, gate=_pentest_only)
    new = pool.keys() - before
    assert len(new) == 1
    child = pool._members[next(iter(new))]
    evolve = child.snapshot.lineage["_evolve"]
    assert evolve["direction"] == "harden"
    assert child.difficulty > member.difficulty


def test_evolution_softens_the_world_the_agent_is_stuck_on(tmp_path: Path) -> None:
    pack = WebappPack()
    pool = WorldPool.seed(
        pack,
        [_COMPANY_MANIFEST, _LATERAL_MANIFEST],
        difficulty_fn=lambda s: float(world_difficulty(s.graph)),
        family="webapp.pentest",
        max_size=8,
    )
    by_diff = sorted(pool._members.values(), key=lambda m: m.difficulty)
    easy, hard = by_diff[0], by_diff[-1]
    before = pool.keys()
    reports = {
        easy.key: [_breach_report(pack, tmp_path / "easy", easy.snapshot)],
        hard.key: [_reach_only_report(pack, tmp_path / "hard", hard.snapshot)],
    }
    assert reports[easy.key][0].passed and not reports[hard.key][0].passed
    pool.update(reports, pack=pack, gate=_pentest_only)
    new = pool.keys() - before
    assert len(new) == 1
    child = pool._members[next(iter(new))]
    evolve = child.snapshot.lineage["_evolve"]
    assert evolve["parent_snapshot_id"] == hard.snapshot.snapshot_id
    assert evolve["direction"] == "soften"
    assert child.difficulty < hard.difficulty


def test_evolution_selects_whichever_world_the_agent_struggles_with(
    tmp_path: Path,
) -> None:
    def evolve_with_stuck(stuck_is_hard: bool) -> tuple[str, str]:
        pack = WebappPack()
        pool = WorldPool.seed(
            pack,
            [_COMPANY_MANIFEST, _LATERAL_MANIFEST],
            difficulty_fn=lambda s: float(world_difficulty(s.graph)),
            family="webapp.pentest",
            max_size=8,
        )
        by_diff = sorted(pool._members.values(), key=lambda m: m.difficulty)
        easy, hard = by_diff[0], by_diff[-1]
        stuck, solved = (hard, easy) if stuck_is_hard else (easy, hard)
        before = pool.keys()
        tag = "hard" if stuck_is_hard else "easy"
        reports = {
            solved.key: [
                _breach_report(pack, tmp_path / f"{tag}-solved", solved.snapshot)
            ],
            stuck.key: [
                _reach_only_report(pack, tmp_path / f"{tag}-stuck", stuck.snapshot)
            ],
        }
        pool.update(reports, pack=pack, gate=_pentest_only)
        child = pool._members[next(iter(pool.keys() - before))]
        parent = str(child.snapshot.lineage["_evolve"]["parent_snapshot_id"])
        return parent, stuck.snapshot.snapshot_id

    evolved_parent, stuck_id = evolve_with_stuck(stuck_is_hard=True)
    assert evolved_parent == stuck_id
    evolved_parent, stuck_id = evolve_with_stuck(stuck_is_hard=False)
    assert evolved_parent == stuck_id


def _drop_credential_leak(snap: Snapshot) -> Snapshot:
    graph = _clone_graph(snap.graph)
    dead = {
        n.id
        for n in graph.by_kind("vulnerability")
        if n.attrs.get("kind") == "credential_leak"
    }
    for nid in dead:
        del graph.nodes[nid]
    stale = [e.id for e in list(graph.edges.values()) if e.src in dead or e.dst in dead]
    for eid in stale:
        del graph.edges[eid]
    return dataclasses.replace(snap, graph=graph)


def test_consequence_gate_admits_a_solvable_evolution(tmp_path: Path) -> None:
    pack = WebappPack()
    parent = _admit(_LATERAL_MANIFEST)
    report = _breach_report(pack, tmp_path / "parent", parent)
    gate = consequence_gate(pack, tmp_path / "gate")
    child = auto_evolve(parent, report, pack=pack, gate=gate, max_repairs=3)
    assert child is not None
    assert _breach_report(pack, tmp_path / "child", child).passed


def test_consequence_gate_rejects_an_unsolvable_world(tmp_path: Path) -> None:
    pack = WebappPack()
    broken = _drop_credential_leak(_admit(_LATERAL_MANIFEST))
    task = next(t for t in broken.tasks if t.meta.get("family") == "webapp.pentest")
    entry = str(broken.graph.nodes[task.entrypoints[0]].attrs["public_url"])
    svc = EpisodeService(pack, tmp_path)
    try:
        handle = svc.start_episode(broken, task.id)
        base = str(svc.surface(handle)["base_url"])
        assert not verdict(broken.graph, base, entry).accepted
    finally:
        svc.close()


def test_unique_vuln_invariant_flags_a_duplicate() -> None:
    graph = _admit(_LATERAL_MANIFEST).graph
    assert unique_vuln_per_endpoint(graph) == []
    vuln = next(iter(graph.by_kind("vulnerability")))
    target = next(e.dst for e in graph.out_edges(vuln.id, "affects"))
    graph.add_node(
        Node(
            id="vuln_dup",
            kind="vulnerability",
            attrs={"kind": vuln.attrs.get("kind"), "family": "code_web", "params": {}},
            visibility=Visibility.HIDDEN,
        )
    )
    graph.edges["e_dup"] = Edge(id="e_dup", kind="affects", src="vuln_dup", dst=target)
    codes = [i.code for i in unique_vuln_per_endpoint(graph)]
    assert codes == ["duplicate_vuln_on_endpoint"]


def test_no_duplicate_same_kind_vuln_on_one_endpoint() -> None:
    for seed in range(30):
        for extra in ({}, {"company": True}):
            graph = _admit({**_DEFAULT_MANIFEST, "seed": seed, **extra}).graph
            seen: set[tuple[str, str]] = set()
            for v in graph.by_kind("vulnerability"):
                target = next((e.dst for e in graph.out_edges(v.id, "affects")), "")
                key = (str(v.attrs.get("kind")), str(target))
                assert key not in seen, f"seed={seed} {extra}: duplicate {key}"
                seen.add(key)


def test_evolution_never_removes_or_swaps_the_recon() -> None:
    graph = _admit(_LATERAL_MANIFEST).graph
    recon_ids = {
        n.id
        for n in graph.by_kind("vulnerability")
        if n.attrs.get("kind") == "config_disclosure"
    }
    assert recon_ids
    threatening = [
        m
        for m in available_mutations(graph, "webapp.pentest", [])
        if m.direction in ("soften", "diversify")
    ]
    touched = {
        nid
        for m in threatening
        for nid in (*m.patch.nodes_removed, *(n.id for n in m.patch.nodes_updated))
    }
    assert recon_ids.isdisjoint(touched)
    assert touched


def test_append_a_hop_deepens_the_chain_and_stays_solvable(tmp_path: Path) -> None:
    pack = WebappPack()
    parent = _admit(_LATERAL_MANIFEST)
    parent_diff = world_difficulty(parent.graph)
    report = _breach_report(pack, tmp_path / "parent", parent)
    assert report.passed
    child = auto_evolve(
        parent, report, pack=pack, gate=monotone_chain_gate(parent), max_repairs=3
    )
    assert child is not None
    # +10 is the chain-depth weight: a hop was appended, not an off-path decoy (+1).
    assert world_difficulty(child.graph) - parent_diff >= 10
    assert _breach_report(pack, tmp_path / "child", child).passed


def test_append_a_hop_keeps_the_flag_owned_under_scoped_seeding(tmp_path: Path) -> None:
    # Cross-backing parity: under the per-service scoped seed CONTAINER uses (not the
    # shared PROCESS seed), the new flag-gate host must OWN the flag store, not just
    # serve it.
    pack = WebappPack()
    parent = _admit(_LATERAL_MANIFEST)
    report = _breach_report(pack, tmp_path, parent)
    child = auto_evolve(
        parent, report, pack=pack, gate=monotone_chain_gate(parent), max_repairs=3
    )
    assert child is not None
    g = child.graph
    term = next(
        n
        for n in g.by_kind("vulnerability")
        if n.attrs.get("kind") == "credential_gated_flag"
    )
    term_ep = next(
        e.dst for e in g.edges.values() if e.kind == "affects" and e.src == term.id
    )
    gate_host = next(
        e.src for e in g.edges.values() if e.kind == "exposes" and e.dst == term_ep
    )
    assert project_seed(g, only_services=frozenset({gate_host})).get("flag")


def test_monotone_chain_gate_requires_one_more_hop(tmp_path: Path) -> None:
    pack = WebappPack()
    parent = _admit(_LATERAL_MANIFEST)
    any_mutation = available_mutations(parent.graph, "webapp.pentest", [])[0]
    gate = monotone_chain_gate(parent)
    assert not gate(parent, any_mutation)
    assert not monotone_chain_gate(_admit(_DEFAULT_MANIFEST))(parent, any_mutation)
    report = _breach_report(pack, tmp_path, parent)
    deeper = auto_evolve(parent, report, pack=pack, gate=gate, max_repairs=3)
    assert deeper is not None
    assert gate(deeper, any_mutation)


def test_pool_grows_a_deeper_chain_under_the_monotone_gate(tmp_path: Path) -> None:
    pack = WebappPack()
    pool = WorldPool.seed(
        pack,
        [_LATERAL_MANIFEST],
        difficulty_fn=lambda s: float(world_difficulty(s.graph)),
        family="webapp.pentest",
        max_size=4,
    )
    assert len(pool) == 1
    seed_diff = next(iter(pool._members.values())).difficulty
    round_no = [0]

    def run_round(
        rows: list[dict[str, object]], snapshots: list[Snapshot]
    ) -> dict[tuple[str, str], list[EpisodeReport]]:
        round_no[0] += 1
        return _solve_round(pack, tmp_path / f"r{round_no[0]}", rows, snapshots)

    run_pool_curriculum(
        pool,
        run_round,
        rounds=1,
        pack=pack,
        groups=1,
        num_generations=2,
        gate_factory=monotone_chain_gate,
    )
    diffs = [m.difficulty for m in pool._members.values()]
    assert len(pool) == 2
    assert max(diffs) - seed_diff >= 10


def test_pool_chain_deepens_until_internal_hosts_run_out(tmp_path: Path) -> None:
    pack = WebappPack()
    pool = WorldPool.seed(
        pack,
        [_LATERAL_MANIFEST],
        difficulty_fn=lambda s: float(world_difficulty(s.graph)),
        family="webapp.pentest",
        max_size=12,
    )
    start = max(m.difficulty for m in pool._members.values())
    round_no = [0]

    def run_round(
        rows: list[dict[str, object]], snapshots: list[Snapshot]
    ) -> dict[tuple[str, str], list[EpisodeReport]]:
        round_no[0] += 1
        return _solve_round(pack, tmp_path / f"r{round_no[0]}", rows, snapshots)

    run_pool_curriculum(
        pool, run_round, rounds=5, pack=pack, groups=1, num_generations=2
    )
    assert max(m.difficulty for m in pool._members.values()) >= start + 10


def test_held_out_eval_pool_is_fenced_and_measured(tmp_path: Path) -> None:
    pack = WebappPack()
    train = WorldPool.seed(
        pack,
        [{**_COMPANY_MANIFEST, "seed": s} for s in (0, 1)],
        difficulty_fn=lambda s: float(world_difficulty(s.graph)),
        family="webapp.pentest",
        max_size=5,
    )
    held_out = EvalPool.seed(
        pack,
        [{**_COMPANY_MANIFEST, "seed": 2}, _LATERAL_MANIFEST],
        difficulty_fn=lambda s: float(world_difficulty(s.graph)),
        family="webapp.pentest",
    )
    assert len(held_out) == 2
    eval_keys = held_out.keys()
    assert not (train.keys() & eval_keys)
    round_no = [0]

    def run_round(
        rows: list[dict[str, object]], snapshots: list[Snapshot]
    ) -> dict[tuple[str, str], list[EpisodeReport]]:
        round_no[0] += 1
        return _solve_round(pack, tmp_path / f"r{round_no[0]}", rows, snapshots)

    metrics = run_pool_curriculum(
        train,
        run_round,
        rounds=2,
        pack=pack,
        groups=2,
        num_generations=2,
        gate=_pentest_only,
        eval_pool=held_out,
    )
    assert len(metrics) == 2
    # The scripted solver breaches every world, so both rates are 1.0 and the gap is
    # 0; the wiring (both measured, eval set fenced) is what is under test.
    assert all(m.train_solve_rate == 1.0 for m in metrics)
    assert all(m.held_out_solve_rate == 1.0 for m in metrics)
    assert all(m.generalization_gap == 0.0 for m in metrics)
    assert not any(m.frontier_capped for m in metrics)
    assert held_out.keys() == eval_keys
    assert not (train.keys() & eval_keys)


def test_generalization_gap_is_train_minus_held_out() -> None:
    assert RoundMetrics(0.8, 0.5).generalization_gap == pytest.approx(0.3)
    assert RoundMetrics(0.8).generalization_gap is None


def test_eval_round_measures_the_held_out_pool(tmp_path: Path) -> None:
    # The held-out pool is measured through eval_round, never the training
    # run_round — so a real trainer can't accidentally learn on it.
    pack = WebappPack()
    train = WorldPool.seed(
        pack,
        [{**_COMPANY_MANIFEST, "seed": s} for s in (0, 1)],
        difficulty_fn=lambda s: float(world_difficulty(s.graph)),
        family="webapp.pentest",
        max_size=5,
    )
    held_out = EvalPool.seed(
        pack,
        [{**_COMPANY_MANIFEST, "seed": 2}],
        difficulty_fn=lambda s: float(world_difficulty(s.graph)),
        family="webapp.pentest",
    )
    trained: set[tuple[str, str]] = set()
    evaluated: set[tuple[str, str]] = set()
    round_no = [0]

    def recording(into: set[tuple[str, str]], label: str) -> RunRound:
        def run(
            rows: list[dict[str, object]], snapshots: list[Snapshot]
        ) -> dict[tuple[str, str], list[EpisodeReport]]:
            round_no[0] += 1
            for row in rows:
                into.add((str(row["snapshot_id"]), str(row["task_id"])))
            work = tmp_path / f"{label}{round_no[0]}"
            return _solve_round(pack, work, rows, snapshots)

        return run

    metrics = run_pool_curriculum(
        train,
        recording(trained, "t"),
        rounds=1,
        pack=pack,
        groups=2,
        num_generations=2,
        gate=_pentest_only,
        eval_pool=held_out,
        eval_round=recording(evaluated, "e"),
    )
    assert metrics[0].held_out_solve_rate is not None
    assert held_out.keys() <= evaluated
    assert held_out.keys().isdisjoint(trained)


def test_services_are_realistically_named() -> None:
    # Coherence (DESIGN.md §2: realism is procedural-first, from curated pools): names
    # read like a real company estate, not the mechanical api1/db2 shape.
    from cyber_webapp.sampling import _SERVICE_NAMES_BY_KIND

    graph = _admit(_COMPANY_MANIFEST).graph
    for svc in graph.by_kind("service"):
        name, kind = str(svc.attrs["name"]), str(svc.attrs["kind"])
        pool = _SERVICE_NAMES_BY_KIND[kind]
        assert name in pool or name.startswith(pool[0] + "-")  # pool name or -indexed


def test_accounts_are_real_people() -> None:
    # Coherence (DESIGN.md §2: alice@corp.example): background accounts are real people
    # at the company domain, not admin / user1.
    graph = _admit(_COMPANY_MANIFEST).graph
    accounts = list(graph.by_kind("account"))
    assert accounts
    for acct in accounts:
        username = str(acct.attrs["username"])
        assert "@" in username and "." in username.split("@")[0]
        assert not username.startswith("user")


def test_default_world_stays_one_flat_segment() -> None:
    # The company preset is opt-in: a default world is unchanged — one network, no
    # recon disclosure.
    graph = _admit(_DEFAULT_MANIFEST).graph
    networks = {n.attrs.get("name") for n in graph.by_kind("network")}
    assert networks == {"main"}
    kinds = {n.attrs.get("kind") for n in graph.by_kind("vulnerability")}
    assert "config_disclosure" not in kinds


def test_company_world_is_deterministic() -> None:
    # Same builder + manifest + seed -> the same world, byte for byte (the recon path
    # is sampled, so this guards it stays content-addressed).
    a, b = _admit(_COMPANY_MANIFEST), _admit(_COMPANY_MANIFEST)
    assert a.snapshot_id == b.snapshot_id


def test_company_world_admits_across_seeds() -> None:
    # The preset is robust across the seed space, not just the pinned seed: every seed
    # yields a solvable networked company world with the recon disclosure wired. A
    # stray vuln_kinds override cannot strip the SSRF either.
    for seed in range(12):
        snap = _admit({**_COMPANY_MANIFEST, "seed": seed, "vuln_kinds": {"idor": 9}})
        kinds = {n.attrs.get("kind") for n in snap.graph.by_kind("vulnerability")}
        assert len(list(snap.graph.by_kind("service"))) >= 6
        assert _is_networked(snap.graph)
        assert {"ssrf", "metadata_credential_leak", "config_disclosure"} <= kinds


def _docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        probe = subprocess.run(
            ["docker", "info"], capture_output=True, timeout=10, check=False
        )
    except Exception:  # noqa: BLE001 - any failure means "no"
        return False
    return probe.returncode == 0


@pytest.mark.skipif(not _docker_available(), reason="docker engine not reachable")
def test_company_solves_across_real_containers() -> None:
    # The same recon→pivot recovers the flag across real per-service containers: the
    # flag lives in an internal container the host can't address; only the SSRF pivot
    # over the docker network reaches it.
    snap = _admit(_COMPANY_MANIFEST)
    graph = snap.graph
    flag = str(graph.nodes["secret_flag"].attrs["value_ref"])

    recon = next(n for n in graph.by_kind("endpoint") if n.id.endswith("_recon"))
    internal_names = {
        str(n.attrs.get("name"))
        for n in graph.by_kind("service")
        if n.attrs.get("exposure") != "public"
    }

    runtime = WebappPack().realize(graph, Backing.CONTAINER)
    assert isinstance(runtime, NetworkedContainerWebappRuntime)
    try:
        runtime.reset()
        base = str(runtime.surface()["base_url"])

        # Recon works on real containers too (cross-backing parity): it discloses the
        # internal estate but never the flag.
        status, recon_body = _get(base, str(recon.attrs["public_url"]))
        assert status == 200, recon_body
        assert set(json.loads(recon_body)["upstreams"]) == internal_names
        assert flag not in recon_body

        trace = solve_chain(graph, lambda p: _get(base, p)[1])
        assert flag in trace.terminal
        assert all(flag not in probe for probe in trace.probes)
        final = runtime.collect()
        assert "secret_flag" in final["leaked_secret_ids"]
    finally:
        runtime.stop()


@pytest.mark.skipif(not _docker_available(), reason="docker engine not reachable")
def test_warm_pool_reuses_real_containers(tmp_path: Path) -> None:
    # On the CONTAINER backing, a poolable company world is booted once and reused:
    # reset_episode truncates the in-container request logs over docker exec, so a
    # second episode on the SAME containers re-exfiltrates cleanly — no full reboot.
    snap = _admit(_COMPANY_MANIFEST)
    pentest = next(t for t in snap.tasks if t.meta.get("family") == "webapp.pentest")
    flag = str(snap.graph.nodes["secret_flag"].attrs["value_ref"])
    entry = str(snap.graph.nodes[pentest.entrypoints[0]].attrs["public_url"])

    svc = EpisodeService(WebappPack(), tmp_path, backing=Backing.CONTAINER)
    env = EpisodeEnv(service=svc, snapshots={snap.snapshot_id: snap}, tools=WEB_TOOLS)
    warm: list[object] = []
    try:
        for _ in range(2):
            env.reset(snapshot_id=snap.snapshot_id, task_id=pentest.id)
            env.http_get(entry)
            trace = solve_chain(
                snap.graph, lambda p: env.http_get(p).split("\n", 1)[-1]
            )
            assert flag in trace.terminal
            env.submit(json.dumps({"flag": flag}))
            env._finalize()
            assert env.report is not None and env.report.passed
            warm.append(svc._warm[snap.snapshot_id])
        assert (
            warm[0] is warm[1]
        )  # the same per-service containers reused, not rebooted
    finally:
        svc.close()
    assert not svc._warm
