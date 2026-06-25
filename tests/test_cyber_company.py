"""Company worlds (DESIGN.md §11): a believable multi-service estate the agent recons
and pivots through. Generation + a PROCESS solve here; the docker-gated test proves the
same recon→pivot recovers the flag across real containers."""

from __future__ import annotations

import dataclasses
import functools
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
from cyber_webapp.difficulty import (
    _DECOY_CAP,
    _FANOUT_CAP,
    _W_HOP,
    _entry_ssrf,
    world_difficulty,
)
from cyber_webapp.invariants import unique_vuln_per_endpoint
from cyber_webapp.mutation import _oracle_path_targets, available_mutations
from cyber_webapp.realize_admit import classify_service_admission
from cyber_webapp.reference_solver import solve_chain
from cyber_webapp.verify import accepts, verdict
from graphschema import Edge, Node, Visibility, WorldGraph
from openrange_pack_sdk import Backing, PoolableRuntime, Snapshot

from openrange.core.admit import admit
from openrange.core.curriculum import (
    _clone_graph,
    auto_evolve,
    consequence_gate,
    consequence_seed_gate,
)
from openrange.core.episode import EpisodeHandle, EpisodeReport, EpisodeService
from openrange.pool import (
    _MAX_PRIORITY,
    EvalPool,
    RoundMetrics,
    RunRound,
    WorldPool,
    run_pool_curriculum,
)
from openrange.training import episode_reward

_COMPANY_MANIFEST = {
    "pack": {"id": "webapp"},
    "runtime": {"tick": {"mode": "off"}},
    "npc": [],
    "seed": 3,
    "topology": "company",
}
_DEFAULT_MANIFEST = {
    "pack": {"id": "webapp"},
    "runtime": {"tick": {"mode": "off"}},
    "npc": [],
    "seed": 3,
}
_LATERAL_MANIFEST = {**_COMPANY_MANIFEST, "topology": "chain"}


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


def _http(base: str, path: str) -> str:
    try:
        with urllib.request.urlopen(base + path, timeout=15) as resp:
            raw: bytes = resp.read()
    except urllib.error.HTTPError as exc:
        raw = exc.read()
    return raw.decode("utf-8", "replace")


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
    found = re.search(
        r"sk_live_[A-Za-z0-9]+|ghp_[A-Za-z0-9]+|AKIA[A-Z0-9]+"
        r"|xoxb-[0-9]+-[0-9]+-[A-Za-z0-9]+"
        r"|[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
        r"|[0-9a-f]{40}",
        body,
    )
    return found.group(0) if found else ""


def _pentest_only(_evolved: Snapshot, mutation: object) -> bool:
    return getattr(mutation, "family", None) == "webapp.pentest"


def _breach_report(pack: WebappPack, work_dir: Path, snap: Snapshot) -> EpisodeReport:
    task = next(t for t in snap.tasks if t.meta.get("family") == "webapp.pentest")
    svc = EpisodeService(pack, work_dir)
    try:
        handle = svc.start_episode(snap, task.id)
        base = svc.base_url(handle)
        entry = str(snap.graph.nodes[task.entrypoints[0]].attrs["public_url"])
        _http(base, entry)
        trace = solve_chain(snap.graph, lambda p: _http(base, p))
        (svc.solver_root(handle) / "result.json").write_text(
            json.dumps({"flag": _flag_from(trace.terminal)}), encoding="utf-8"
        )
        return svc.stop_episode(handle)
    finally:
        svc.close()


def _reach_only_report(
    pack: WebappPack, work_dir: Path, snap: Snapshot
) -> EpisodeReport:
    task = next(t for t in snap.tasks if t.meta.get("family") == "webapp.pentest")
    svc = EpisodeService(pack, work_dir)
    try:
        handle = svc.start_episode(snap, task.id)
        base = svc.base_url(handle)
        entry = str(snap.graph.nodes[task.entrypoints[0]].attrs["public_url"])
        _http(base, entry)
        return svc.stop_episode(handle)
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
    out: dict[tuple[str, str], list[EpisodeReport]] = {}
    try:
        for row in rows:
            sid, tid = str(row["snapshot_id"]), str(row["task_id"])
            snap = by_id[sid]
            task = next(t for t in snap.tasks if t.id == tid)
            handle = svc.start_episode(snap, tid)
            base = svc.base_url(handle)
            entry = str(snap.graph.nodes[task.entrypoints[0]].attrs["public_url"])
            _http(base, entry)
            trace = solve_chain(snap.graph, functools.partial(_http, base))
            (svc.solver_root(handle) / "result.json").write_text(
                json.dumps({"flag": _flag_from(trace.terminal)}), encoding="utf-8"
            )
            out.setdefault((sid, tid), []).append(svc.stop_episode(handle))
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
    assert (
        internal_names <= disclosed
    )  # names every real internal host, incl. the flag's
    chaff = disclosed - internal_names
    assert chaff  # ...padded with decoy hosts, so the page isn't a perfect oracle
    service_names = {str(n.attrs.get("name")) for n in graph.by_kind("service")}
    assert not (chaff & service_names)  # the decoys are not real services

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

    def _start() -> tuple[EpisodeService, EpisodeHandle]:
        svc = EpisodeService(WebappPack(), tmp_path / f"env{len(services)}")
        services.append(svc)
        return svc, svc.start_episode(snap, pentest.id)

    try:
        svc, reach = _start()
        _http(svc.base_url(reach), entry)
        report = svc.stop_episode(reach)
        assert episode_reward(report).scalar == pytest.approx(1 / 3)  # reached only

        svc, wrong = _start()
        _http(svc.base_url(wrong), entry)
        (svc.solver_root(wrong) / "result.json").write_text(
            json.dumps({"flag": "not-the-real-flag"}), encoding="utf-8"
        )
        report = svc.stop_episode(wrong)
        # + extracted_anything, but wrong
        assert episode_reward(report).scalar == pytest.approx(2 / 3)

        svc, breach = _start()
        base = svc.base_url(breach)
        _http(base, entry)
        trace = solve_chain(graph, lambda p: _http(base, p))
        assert flag in trace.terminal  # genuinely exfiltrated over the wire
        (svc.solver_root(breach) / "result.json").write_text(
            json.dumps({"flag": flag}), encoding="utf-8"
        )
        report = svc.stop_episode(breach)
        assert episode_reward(report).scalar == 1.0
        assert report.passed
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
    warm: list[object] = []
    try:
        for _ in range(2):
            handle = svc.start_episode(snap, pentest.id)
            base = svc.base_url(handle)
            _http(base, entry)
            trace = solve_chain(snap.graph, functools.partial(_http, base))
            assert flag in trace.terminal  # exfiltrated over the wire, each episode
            (svc.solver_root(handle) / "result.json").write_text(
                json.dumps({"flag": flag}), encoding="utf-8"
            )
            assert svc.stop_episode(handle).passed
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
                "loot": {"file": 1, "db": 0},
                "vuln": {"pin": [{"kind": "command_injection"}]},
            }
        ).graph,
        Backing.PROCESS,
    )
    assert isinstance(cmdi, PoolableRuntime) and not cmdi.poolable()
    sqli = WebappPack().realize(
        _admit(
            {
                **_DEFAULT_MANIFEST,
                "loot": {"db": 1, "file": 0},
                "vuln": {"pin": [{"kind": "sql_injection"}]},
            }
        ).graph,
        Backing.PROCESS,
    )
    assert isinstance(sqli, PoolableRuntime) and sqli.poolable()


def test_consecutive_seeds_never_collide() -> None:
    # Distinct seeds must produce distinct worlds AND distinct flags — no two seeds
    # (least of all the consecutive N, N+1 pairs an earlier audit flagged) may collapse
    # to a byte-identical world or reuse the same secret. The high-entropy loot path and
    # flag templates make this hold; this pins it across every topology.
    for manifest in (_DEFAULT_MANIFEST, _COMPANY_MANIFEST, _LATERAL_MANIFEST):
        hashes: dict[str, int] = {}
        flags: dict[str, int] = {}
        for seed in range(20):
            graph = _admit({**manifest, "seed": seed}).graph
            h = graph.content_hash()
            f = str(graph.nodes["secret_flag"].attrs["value_ref"])
            assert h not in hashes, (
                f"{manifest.get('topology', 'flat')}: seeds "
                f"{hashes.get(h)} and {seed} are byte-identical"
            )
            assert f not in flags, (
                f"{manifest.get('topology', 'flat')}: seeds "
                f"{flags.get(f)} and {seed} reuse the same flag"
            )
            hashes[h] = seed
            flags[f] = seed


def test_diversify_keeps_the_oracle_loot_compatible() -> None:
    # Diversify rotates the on-path technique, not just off-path decoys — but a swap of
    # the flag-READING oracle must stay within the loot's exploit shapes (file-loot ->
    # file_read/code_exec; db-loot -> response_leak) or it strands the flag. Off-path
    # decoys stay free to become any class.
    from cyber_webapp.mutation import _affects_target_id, _oracle_allowed_shapes
    from cyber_webapp.vulnerabilities import CATALOG
    from graphschema import apply_patch

    for loot, shapes in (
        ("file", {"file_read", "code_exec"}),
        ("db", {"response_leak"}),
    ):
        other = "db" if loot == "file" else "file"
        oracle_swaps = 0
        for seed in range(10):
            graph = _admit(
                {**_DEFAULT_MANIFEST, "seed": seed, "loot": {loot: 1, other: 0}}
            ).graph
            assert _oracle_allowed_shapes(graph) == frozenset(shapes)
            oracle_endpoints, _ = _oracle_path_targets(graph)
            for mut in available_mutations(graph, "webapp.pentest", []):
                if mut.direction != "diversify":
                    continue
                vulns = [
                    n for n in mut.patch.nodes_updated if n.kind == "vulnerability"
                ]
                if not vulns:
                    continue
                evolved = _clone_graph(graph)
                apply_patch(evolved, mut.patch)
                if _affects_target_id(evolved, vulns[0].id) in oracle_endpoints:
                    oracle_swaps += 1
                    swapped_shape = CATALOG[vulns[0].attrs["kind"]].shape
                    assert swapped_shape in shapes, (
                        f"{loot} seed {seed}: oracle swap to {vulns[0].attrs['kind']} "
                        f"({swapped_shape}) breaks loot compatibility"
                    )
        assert oracle_swaps, f"{loot}: diversify never rotated the on-path oracle"


def test_diversify_oracle_swap_keeps_the_world_solvable(tmp_path: Path) -> None:
    # The behavioral guarantee behind the structural loot-shape constraint: realizing a
    # world after an on-path diversify swap still leaks the flag — including the rare
    # endpoint that hosts a co-located sibling the body-shaped method-flip touches (db
    # seed 11). The world stays solvable after rotating the technique the agent learns
    from cyber_webapp.mutation import _affects_target_id
    from graphschema import apply_patch

    pack = WebappPack()

    def world_accepts(snap: Snapshot, sub: str) -> bool:
        svc = EpisodeService(pack, tmp_path / sub)
        try:
            task = next(
                t for t in snap.tasks if t.meta.get("family") == "webapp.pentest"
            )
            return accepts(snap, svc.base_url(svc.start_episode(snap, task.id)))
        finally:
            svc.close()

    checked = 0
    for loot, lo in (("db", 12), ("file", 6)):
        other = "db" if loot == "file" else "file"
        for seed in range(lo):
            base = _admit(
                {**_DEFAULT_MANIFEST, "seed": seed, "loot": {loot: 1, other: 0}}
            )
            if not world_accepts(base, f"b{loot}{seed}"):
                continue  # only worlds whose base is already solvable
            oracle_endpoints, _ = _oracle_path_targets(base.graph)
            for mut in available_mutations(base.graph, "webapp.pentest", []):
                if mut.direction != "diversify":
                    continue
                vulns = [
                    n for n in mut.patch.nodes_updated if n.kind == "vulnerability"
                ]
                if not vulns:
                    continue
                evolved = _clone_graph(base.graph)
                apply_patch(evolved, mut.patch)
                if _affects_target_id(evolved, vulns[0].id) not in oracle_endpoints:
                    continue
                nsnap = dataclasses.replace(base, graph=evolved)
                assert world_accepts(nsnap, f"s{loot}{seed}{checked}"), mut.note
                checked += 1
    assert checked, "no on-path diversify swap was exercised"


def test_eval_pool_seed_gate_admits_solvable_drops_unsolvable(tmp_path: Path) -> None:
    # The held-out EvalPool gets the same seed-gate guarantee as the train pool: a
    # solvable world is kept, a rejected one dropped — so an unsolvable world can never
    # silently corrupt the generalization measurement.
    pack = WebappPack()
    seeds = [_COMPANY_MANIFEST, {**_COMPANY_MANIFEST, "topology": "chain"}]

    kept = EvalPool.seed(
        pack,
        seeds,
        difficulty_fn=lambda s: float(world_difficulty(s.graph)),
        family="webapp.pentest",
        seed_gate=consequence_seed_gate(pack, tmp_path / "keep", accepts),
    )
    assert len(kept) == len(seeds)

    dropped = EvalPool.seed(
        pack,
        seeds,
        difficulty_fn=lambda s: float(world_difficulty(s.graph)),
        family="webapp.pentest",
        seed_gate=lambda _snapshot: False,
    )
    assert len(dropped) == 0


def test_diversify_swap_params_are_deterministic() -> None:
    # available_mutations is pure: enumerating diversify swaps on the same graph twice
    # yields byte-identical patch params (the swap seeds its rng from a hash of
    # kind+target, not a live rng), so a curriculum's evolved lineage is reproducible.
    graph = _admit(_LATERAL_MANIFEST).graph

    def diversify_params() -> dict[str, object]:
        out: dict[str, object] = {}
        for mut in available_mutations(graph, "webapp.pentest", []):
            if mut.direction != "diversify":
                continue
            for node in mut.patch.nodes_updated:
                if node.kind == "vulnerability":
                    out[node.id] = (node.attrs["kind"], node.attrs["params"])
        return out

    assert diversify_params() == diversify_params()
    assert diversify_params()  # a chain world does offer diversify swaps


def test_world_difficulty_rises_with_chain_depth() -> None:
    flat = world_difficulty(_admit(_DEFAULT_MANIFEST).graph)
    company = world_difficulty(_admit(_COMPANY_MANIFEST).graph)
    lateral = world_difficulty(_admit(_LATERAL_MANIFEST).graph)
    assert flat < company < lateral


def _add_off_path_vuln(graph: WorldGraph, i: int) -> None:
    graph.add_node(
        Node(
            f"svc_decoy{i}",
            "service",
            attrs={"name": f"decoy{i}", "exposure": "internal"},
        )
    )
    graph.add_node(
        Node(f"ep_decoy{i}", "endpoint", attrs={"path": f"/d{i}", "method": "GET"})
    )
    graph.add_node(
        Node(
            f"v_decoy{i}",
            "vulnerability",
            attrs={"kind": "sql_injection", "params": {}},
            visibility=Visibility.HIDDEN,
        )
    )
    graph.add_edge(Edge(f"ed{i}exp", "exposes", f"svc_decoy{i}", f"ep_decoy{i}"))
    graph.add_edge(Edge(f"ed{i}aff", "affects", f"v_decoy{i}", f"ep_decoy{i}"))


def _add_oracle_sibling_decoy(graph: WorldGraph) -> None:
    service = next(iter(_oracle_path_targets(graph)[1]))
    graph.add_node(
        Node(
            "ep_oracle_sibling", "endpoint", attrs={"path": "/admin", "method": "POST"}
        )
    )
    graph.add_edge(Edge("e_oracle_sib_exp", "exposes", service, "ep_oracle_sibling"))
    graph.add_node(
        Node(
            "vuln_command_injection_9",
            "vulnerability",
            attrs={"kind": "command_injection", "params": {}},
            visibility=Visibility.HIDDEN,
        )
    )
    graph.add_edge(
        Edge(
            "e_oracle_sib_aff",
            "affects",
            "vuln_command_injection_9",
            "ep_oracle_sibling",
        )
    )


def test_off_path_decoys_cannot_outrank_a_real_hop() -> None:
    graph = _admit(_LATERAL_MANIFEST).graph
    base = world_difficulty(graph)
    for i in range(20):
        _add_off_path_vuln(graph, i)
    delta = world_difficulty(graph) - base
    # Each decoy is a vuln on its own internal host. Both responses saturate — the vuln
    # term at _DECOY_CAP, the host-triage term at _FANOUT_CAP — so 20 decoys add no more
    # than a believable estate's worth and can never out-rank a real credential hop.
    assert delta <= _DECOY_CAP + _FANOUT_CAP
    assert delta < _W_HOP


def test_an_oracle_sibling_decoy_is_never_scored_on_path() -> None:
    for manifest in (_DEFAULT_MANIFEST, _LATERAL_MANIFEST):
        graph = _admit(manifest).graph
        base = world_difficulty(graph)
        _add_oracle_sibling_decoy(graph)
        assert world_difficulty(graph) - base < 1


def test_a_chain_with_no_way_in_scores_like_a_flat_world() -> None:
    graph = _admit(_LATERAL_MANIFEST).graph
    flat = world_difficulty(_admit(_DEFAULT_MANIFEST).graph)
    ssrf = _entry_ssrf(graph)
    assert ssrf is not None
    internal_ep = next(
        edge.dst
        for svc in graph.by_kind("service")
        if svc.attrs.get("exposure") != "public"
        for edge in graph.out_edges(svc.id, "exposes")
    )
    for affects in graph.out_edges(ssrf.id, "affects"):
        affects.dst = internal_ep
    assert _entry_ssrf(graph) is None
    assert world_difficulty(graph) <= flat


def test_blind_recon_is_harder_than_recon_given() -> None:
    given = world_difficulty(_admit(_COMPANY_MANIFEST).graph)
    blind = world_difficulty(_admit({**_COMPANY_MANIFEST, "recon": "none"}).graph)
    assert blind > given


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
            {**_COMPANY_MANIFEST, "topology": "chain"},
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


def test_seed_gate_admits_solvable_drops_unsolvable(tmp_path: Path) -> None:
    # A structurally-admitted world enters the pool only when its reference breach
    # actually leaks: the real consequence verdict keeps every solvable seed, while a
    # gate that rejects everything drops them all — so a live-unsolvable world (the gap
    # structural admission alone left open) can never seed the curriculum.
    pack = WebappPack()
    seeds = [_COMPANY_MANIFEST, {**_COMPANY_MANIFEST, "topology": "chain"}]

    kept = WorldPool.seed(
        pack,
        seeds,
        difficulty_fn=lambda s: float(world_difficulty(s.graph)),
        family="webapp.pentest",
        max_size=8,
        seed_gate=consequence_seed_gate(pack, tmp_path / "keep", accepts),
    )
    assert len(kept) == len(seeds)  # the real verdict admits every solvable world

    dropped = WorldPool.seed(
        pack,
        seeds,
        difficulty_fn=lambda s: float(world_difficulty(s.graph)),
        family="webapp.pentest",
        max_size=8,
        seed_gate=lambda _snapshot: False,
    )
    assert len(dropped) == 0  # a rejected world never enters the pool


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
            {**_COMPANY_MANIFEST, "topology": "chain", "seed": 1},
            {**_COMPANY_MANIFEST, "topology": "chain", "seed": 2},
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
    gate = consequence_gate(pack, tmp_path / "gate", accepts)
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


def test_pool_curriculum_only_evolves_breach_verified_worlds(tmp_path: Path) -> None:
    # Brick 1 -- self-verifying by default: with the consequence gate wired into the
    # curriculum, every world the pool evolves into must still leak via the reference
    # breach, so training never inherits a world that hardened its way to unsolvable.
    pack = WebappPack()
    seeds = [{**_COMPANY_MANIFEST, "seed": s} for s in range(3)]
    pool = WorldPool.seed(
        pack,
        seeds,
        difficulty_fn=lambda s: float(world_difficulty(s.graph)),
        family="webapp.pentest",
        max_size=8,
    )
    seed_ids = {m.snapshot.snapshot_id for m in pool._members.values()}
    round_no = [0]

    def run_round(
        rows: list[dict[str, object]], snapshots: list[Snapshot]
    ) -> dict[tuple[str, str], list[EpisodeReport]]:
        round_no[0] += 1
        return _solve_round(pack, tmp_path / f"r{round_no[0]}", rows, snapshots)

    verify = consequence_gate(pack, tmp_path / "gate", accepts)
    run_pool_curriculum(
        pool,
        run_round,
        rounds=2,
        pack=pack,
        groups=3,
        num_generations=2,
        gate=lambda snap, mut: _pentest_only(snap, mut) and verify(snap, mut),
    )

    final_ids = {m.snapshot.snapshot_id for m in pool._members.values()}
    assert final_ids - seed_ids, "the gate rejected every evolution; nothing to verify"

    svc = EpisodeService(pack, tmp_path / "check")
    try:
        for member in pool._members.values():
            task = next(
                t
                for t in member.snapshot.tasks
                if t.meta.get("family") == "webapp.pentest"
            )
            handle = svc.start_episode(member.snapshot, task.id)
            verified = accepts(member.snapshot, svc.base_url(handle))
            svc.stop_episode(handle)
            assert verified, f"unverified world in pool: {member.snapshot.snapshot_id}"
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
        for extra in ({}, {"topology": "company"}):
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
    # A real appended hop adds the full chain-hop weight, far above any off-path decoy.
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
    # difficulty_gain is the honest "is the frontier still moving" read, distinct from
    # frontier_capped: every round admits a harder child (capped False), but company
    # worlds have no credential walk, so the gain is the small cosmetic decoy creep —
    # well under a real credential hop — exposing the plateau the boolean can't.
    gains = [m.difficulty_gain for m in metrics]
    assert all(g is not None and 0 <= g < _W_HOP for g in gains)
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
    # yields a solvable networked company world with the recon disclosure wired. The
    # company topology forces its own SSRF vuln shape, so a vuln override can't even be
    # passed (it is rejected before it could strip the SSRF).
    for seed in range(12):
        snap = _admit({**_COMPANY_MANIFEST, "seed": seed})
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
        # Every real internal host is disclosed, padded with decoy hosts (the page is a
        # candidate set, not a perfect oracle) — and it never leaks the flag itself.
        assert internal_names <= set(json.loads(recon_body)["upstreams"])
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
    warm: list[object] = []
    try:
        for _ in range(2):
            handle = svc.start_episode(snap, pentest.id)
            base = svc.base_url(handle)
            _http(base, entry)
            trace = solve_chain(snap.graph, functools.partial(_http, base))
            assert flag in trace.terminal
            (svc.solver_root(handle) / "result.json").write_text(
                json.dumps({"flag": flag}), encoding="utf-8"
            )
            assert svc.stop_episode(handle).passed
            warm.append(svc._warm[snap.snapshot_id])
        assert (
            warm[0] is warm[1]
        )  # the same per-service containers reused, not rebooted
    finally:
        svc.close()
    assert not svc._warm


_FLAT_SQLI = {
    "pack": {"id": "webapp"},
    "runtime": {"tick": {"mode": "off"}},
    "npc": [],
    "vuln": {"pin": [{"kind": "sql_injection"}]},
    "loot": {"db": 1, "file": 0},
}


def _flat_sqli_with_sibling_db() -> Snapshot:
    for seed in range(16):
        snap = _admit({**_FLAT_SQLI, "seed": seed})
        svc_of_ep = {
            e.dst: e.src for e in snap.graph.edges.values() if e.kind == "exposes"
        }
        db_eps = sum(
            1
            for ep in snap.graph.by_kind("endpoint")
            if snap.graph.nodes.get(svc_of_ep.get(ep.id, "")) is not None
            and snap.graph.nodes[svc_of_ep[ep.id]].attrs.get("kind") == "db"
        )
        if db_eps >= 2:
            return snap
    pytest.skip("no flat sqli world with a sibling db endpoint in range(16)")


def test_benign_db_endpoint_never_serves_the_flag(tmp_path: Path) -> None:
    # A sibling default-db endpoint once served the flag for ?key=<flagkey> with
    # no exploit — hence the probe with the flag's own key.
    snap = _flat_sqli_with_sibling_db()
    graph = snap.graph
    holds = {e.src: e.dst for e in graph.edges.values() if e.kind == "holds"}
    flag_record = next(rid for rid, sid in holds.items() if sid == "secret_flag")
    flag_key = str(graph.nodes[flag_record].attrs["key"])
    flag = str(graph.nodes["secret_flag"].attrs["value_ref"])

    task = next(t for t in snap.tasks if t.meta.get("family") == "webapp.pentest")
    svc = EpisodeService(WebappPack(), tmp_path)
    try:
        base = svc.base_url(svc.start_episode(snap, task.id))
        for ep in graph.by_kind("endpoint"):
            url = str(ep.attrs["public_url"])
            assert flag not in _http(base, f"{url}?key={flag_key}")
            assert flag not in _http(base, url)
        entry = str(graph.nodes[task.entrypoints[0]].attrs["public_url"])
        outcome = verdict(graph, base, entry)
        assert outcome.accepted
        assert "no benign endpoint leaks" in outcome.reason
    finally:
        svc.close()


def test_whole_world_verdict_rejects_a_sibling_leak() -> None:
    graph = _admit({**_FLAT_SQLI, "seed": 0}).graph
    flag = str(graph.nodes["secret_flag"].attrs["value_ref"])
    outcome = classify_service_admission(
        graph,
        oracle_exploit_body=flag,
        oracle_benign_body="{}",
        benign_endpoint_bodies={"/orders": "{}", "/leak": flag},
        root_ok=True,
    )
    assert not outcome.accepted
    assert "benign endpoint" in outcome.reason


def test_evolved_snapshot_persists_world_difficulty(tmp_path: Path) -> None:
    pack = WebappPack()
    seeds = [{**_COMPANY_MANIFEST, "seed": s} for s in range(4)]
    round_no = [0]

    def run_round(
        rows: list[dict[str, object]], snapshots: list[Snapshot]
    ) -> dict[tuple[str, str], list[EpisodeReport]]:
        round_no[0] += 1
        return _solve_round(pack, tmp_path / f"r{round_no[0]}", rows, snapshots)

    pool = WorldPool.seed(
        pack,
        seeds,
        difficulty_fn=lambda s: float(world_difficulty(s.graph)),
        family="webapp.pentest",
        max_size=5,
    )
    run_pool_curriculum(
        pool,
        run_round,
        rounds=2,
        pack=pack,
        groups=3,
        num_generations=2,
        gate=_pentest_only,
    )
    evolved = [
        s
        for s in pool.snapshots()
        if (s.lineage.get("_evolve") or {}).get("kind") == "patch"
    ]
    assert evolved
    for snap in evolved:
        stamped = snap.lineage.get("world_difficulty")
        assert stamped == pytest.approx(float(world_difficulty(snap.graph)))
