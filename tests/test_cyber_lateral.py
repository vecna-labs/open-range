"""Credential-reuse lateral movement (DESIGN.md §11): the SSRF becomes an agent-driven
internal proxy, and the chain is SYNTHESIZED at a sampled depth from one composable
primitive — an entry host leaks a credential, each gated host relays the next, the last
serves the flag. One preset synthesizes 1-, 2-, 3-hop chains. The flag is reachable ONLY
through the final gate. PROCESS solves here; the docker-gated test proves it on real
containers."""

from __future__ import annotations

import shutil
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pytest
from cyber_webapp import NetworkedContainerWebappRuntime, WebappPack, _is_networked
from cyber_webapp.consequence import guarded_values
from cyber_webapp.reference_solver import solve_chain
from graphschema import Visibility, WorldGraph
from openrange_pack_sdk import Backing, Snapshot

from openrange.core.admit import admit
from openrange.core.episode import EpisodeService


def _manifest(seed: int = 3) -> dict[str, object]:
    return {
        "pack": {"id": "webapp"},
        "runtime": {"tick": {"mode": "off"}},
        "npc": [],
        "seed": seed,
        "topology": "chain",
    }


def _admit(seed: int = 3) -> Snapshot:
    snap = admit(WebappPack(), manifest=_manifest(seed), max_repairs=3)
    assert isinstance(snap, Snapshot), snap
    return snap


def _chain_depth(graph: WorldGraph) -> int:
    return sum(
        1
        for n in graph.by_kind("vulnerability")
        if n.attrs.get("kind") in ("credential_gated_relay", "credential_gated_flag")
    )


def test_soften_collapses_one_credential_hop(tmp_path: Path) -> None:
    # Soften can rescue a chain-stuck agent by collapsing the last hop: the chain gets
    # exactly one shorter, stays solvable (the promoted relay now serves the flag) on
    # both backings, and is strictly easier — validated through the real re-admit path.
    from cyber_webapp.difficulty import world_difficulty
    from cyber_webapp.mutation import _soften_remove_hop_mutation
    from cyber_webapp.verify import accepts

    from openrange.core.curriculum import _evolve_snapshot

    pack = WebappPack()
    snap = admit(
        pack,
        manifest={**_manifest(1), "chain": {"depth": {"min": 2, "max": 2}}},
        max_repairs=3,
    )
    assert isinstance(snap, Snapshot)
    parent_depth = _chain_depth(snap.graph)
    assert parent_depth >= 2

    mut = _soften_remove_hop_mutation(snap.graph, "webapp.pentest", 0.95)
    assert mut is not None and mut.direction == "soften"
    evolved = _evolve_snapshot(snap, pack, mut, max_repairs=3)
    assert evolved is not None and evolved.snapshot_id != snap.snapshot_id
    eg = evolved.graph

    assert _chain_depth(eg) == parent_depth - 1  # exactly one hop shorter
    assert world_difficulty(eg) < world_difficulty(snap.graph)  # strictly easier
    orphans = [
        n
        for n in eg.nodes
        if not any(e.src == n or e.dst == n for e in eg.edges.values())
    ]
    assert not orphans

    svc = EpisodeService(pack, tmp_path)
    try:
        task = next(
            t for t in evolved.tasks if t.meta.get("family") == "webapp.pentest"
        )
        handle = svc.start_episode(evolved, task.id)
        assert accepts(evolved, svc.base_url(handle))
    finally:
        svc.close()

    again_mut = _soften_remove_hop_mutation(snap.graph, "webapp.pentest", 0.95)
    assert again_mut is not None
    again = _evolve_snapshot(snap, pack, again_mut, max_repairs=3)
    assert (
        again is not None and again.snapshot_id == evolved.snapshot_id
    )  # deterministic


def test_soften_hop_collapse_needs_a_relay() -> None:
    # A depth-1 chain (leak -> flag, no relay) has no hop to collapse, so the move is
    # offered and decoy-removal stays the only soften.
    from cyber_webapp.mutation import _soften_remove_hop_mutation

    snap = admit(
        WebappPack(),
        manifest={**_manifest(1), "chain": {"depth": {"min": 1, "max": 1}}},
        max_repairs=3,
    )
    assert isinstance(snap, Snapshot)
    assert _chain_depth(snap.graph) == 1
    assert _soften_remove_hop_mutation(snap.graph, "webapp.pentest", 0.95) is None


def test_chain_credentials_are_public_wired_graph_nodes() -> None:
    graph = _admit().graph
    # Scope to the chain's tokens — the NPC `password` credentials are a
    # separate credential system that this binding does not own.
    creds = [n for n in graph.by_kind("credential") if n.attrs.get("kind") == "token"]
    assert creds, "a lateral world mints a credential node per hop"
    referenced = {e.src for e in graph.edges.values()} | {
        e.dst for e in graph.edges.values()
    }
    for cred in creds:
        assert cred.visibility == Visibility.PUBLIC
        assert cred.attrs.get("kind") == "token"
        produced = [
            e for e in graph.edges.values() if e.kind == "produces" and e.dst == cred.id
        ]
        required = [
            e
            for e in graph.edges.values()
            if e.kind == "requires_credential" and e.dst == cred.id
        ]
        assert len(produced) == 1, cred.id
        assert len(required) == 1, cred.id
        assert cred.id in referenced


def test_admission_rejects_a_broken_credential_binding() -> None:
    # Runs the full validate() pipeline (ontology + invariants), not the
    # invariant alone — proves the gate fires at admission.
    from graphschema import validate

    pack = WebappPack()
    graph = _admit().graph
    produces = [e for e in graph.edges.values() if e.kind == "produces"]
    assert produces
    del graph.edges[produces[0].id]
    issues = validate(graph, pack.ontology(), pack.invariants())
    assert any(i.code == "credential_binding" and i.severity == "error" for i in issues)


def test_admission_rejects_a_drifted_credential_value() -> None:
    # The credential node is the single source of truth: if a gate's param-string
    # token drifts from the node's value_ref, admission rejects it even though the
    # structural binding is still intact.
    import dataclasses

    from cyber_webapp.invariants import credential_value_binding
    from graphschema import validate

    pack = WebappPack()
    graph = _admit().graph
    assert not credential_value_binding(graph)  # a fresh world is value-consistent

    gate = next(
        n
        for n in graph.by_kind("vulnerability")
        if n.attrs.get("kind") in ("credential_gated_relay", "credential_gated_flag")
    )
    drifted = {**gate.attrs, "params": {**gate.attrs["params"], "credential": "WRONG"}}
    graph.nodes[gate.id] = dataclasses.replace(gate, attrs=drifted)

    assert any(i.code == "credential_value" for i in credential_value_binding(graph))
    issues = validate(graph, pack.ontology(), pack.invariants())
    assert any(i.code == "credential_value" and i.severity == "error" for i in issues)


def test_admission_rejects_a_short_circuitable_chain() -> None:
    # The flag must live only in the gated HIDDEN secret; planting the real value in a
    # reachable node (here the loot record) would let one response-leak skip the chain.
    import dataclasses

    from cyber_webapp.invariants import flag_confined_to_gate
    from graphschema import validate

    pack = WebappPack()
    graph = _admit().graph
    assert not flag_confined_to_gate(graph)  # a fresh chain confines the flag

    flag = str(graph.nodes["secret_flag"].attrs["value_ref"])
    record = next(
        graph.nodes[e.src]
        for e in graph.edges.values()
        if e.kind == "holds" and e.dst == "secret_flag"
    )
    leaky = {**record.attrs, "fields": {**record.attrs["fields"], "value": flag}}
    graph.nodes[record.id] = dataclasses.replace(record, attrs=leaky)

    assert any(i.code == "flag_short_circuit" for i in flag_confined_to_gate(graph))
    issues = validate(graph, pack.ontology(), pack.invariants())
    assert any(i.code == "flag_short_circuit" and i.severity == "error" for i in issues)


def test_chain_credentials_are_not_guarded() -> None:
    # PUBLIC credential nodes must NOT join the guarded set — a HIDDEN token would
    # be swept in, and the leak handler serving it would trip the verifier on a
    # benign probe (every lateral world would then be rejected as trivial).
    graph = _admit().graph
    guarded = set(guarded_values(graph))
    assert guarded, "the flag must be guarded"
    assert not (guarded & {n.id for n in graph.by_kind("credential")})


def _get(base: str, path: str) -> str:
    try:
        with urllib.request.urlopen(f"{base}{path}", timeout=15) as resp:
            return str(resp.read().decode())
    except urllib.error.HTTPError as exc:
        return exc.read().decode()


def _enables_chain_kinds(graph: WorldGraph) -> list[str]:
    # Walk the single enables path from the ssrf and return the kinds in order.
    by_id = {n.id: n for n in graph.by_kind("vulnerability")}
    out = {e.src: e.dst for e in graph.edges.values() if e.kind == "enables"}
    node: str | None = next(
        v.id for v in by_id.values() if v.attrs.get("kind") == "ssrf"
    )
    kinds: list[str] = []
    seen: set[str] = set()
    while node is not None and node not in seen:
        seen.add(node)
        kinds.append(str(by_id[node].attrs.get("kind")))
        node = out.get(node)
    return kinds


def test_lateral_chain_is_synthesized_and_wired() -> None:
    graph = _admit().graph
    assert _is_networked(graph)
    ssrf = next(
        n for n in graph.by_kind("vulnerability") if n.attrs.get("kind") == "ssrf"
    )
    assert "internal_hosts" in ssrf.attrs["params"]  # proxy mode, agent-driven

    # The enables path is ssrf -> credential_leak -> (relay ->)* -> gated_flag at the
    # sampled depth — exactly one leak entry and one terminal flag gate.
    kinds = _enables_chain_kinds(graph)
    assert kinds[0] == "ssrf"
    assert kinds[1] == "credential_leak"
    assert kinds[-1] == "credential_gated_flag"
    assert all(k == "credential_gated_relay" for k in kinds[2:-1])
    assert _chain_depth(graph) == len(kinds) - 2  # relays + the terminal gate

    # The flag record's value is a decoy — the real flag only lives in the gated secret.
    flag = str(graph.nodes["secret_flag"].attrs["value_ref"])
    record = next(
        graph.nodes[e.src]
        for e in graph.edges.values()
        if e.kind == "holds" and e.dst == "secret_flag"
    )
    assert record.attrs["fields"]["value"] != flag


def test_lateral_chain_pivots_inward_by_tier() -> None:
    # Structure coherence: the chain pivots INWARD — its hosts' tiers never decrease
    # (web < api < auth < db) — so lateral movement reads architecturally.
    graph = _admit().graph
    tier = {"web": 1, "api": 2, "auth": 3, "db": 4}
    out = {e.src: e.dst for e in graph.edges.values() if e.kind == "enables"}
    ep_of_vuln = {e.src: e.dst for e in graph.edges.values() if e.kind == "affects"}
    svc_of_ep = {e.dst: e.src for e in graph.edges.values() if e.kind == "exposes"}
    by_kind = {n.attrs.get("kind"): n.id for n in graph.by_kind("vulnerability")}
    node: str | None = by_kind["credential_leak"]
    tiers: list[int] = []
    while node is not None and graph.nodes.get(node) is not None:
        ep = ep_of_vuln.get(node)
        svc = svc_of_ep.get(ep) if ep is not None else None
        if svc is not None:
            tiers.append(tier.get(str(graph.nodes[svc].attrs.get("kind")), 0))
        node = out.get(node)
    assert tiers == sorted(tiers)  # non-decreasing — the pivot moves toward the data


def test_lateral_synthesizes_varied_depth() -> None:
    # One preset, many depths: the engine synthesizes a distribution, not a fixed shape.
    depths = {_chain_depth(_admit(seed).graph) for seed in range(12)}
    assert len(depths) >= 2
    assert depths <= {1, 2, 3}


def test_lateral_solves_on_process(tmp_path: Path) -> None:
    snap = _admit()
    graph = snap.graph
    flag = str(graph.nodes["secret_flag"].attrs["value_ref"])

    pentest = next(t for t in snap.tasks if t.meta.get("family") == "webapp.pentest")
    svc = EpisodeService(WebappPack(), tmp_path)
    try:
        handle = svc.start_episode(snap, pentest.id)
        base = str(svc.surface(handle)["base_url"])
        trace = solve_chain(graph, lambda p: _get(base, p))
        assert flag in trace.terminal
        assert all(flag not in probe for probe in trace.probes)
    finally:
        svc.close()


def test_lateral_world_is_deterministic() -> None:
    assert _admit().snapshot_id == _admit().snapshot_id


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
def test_lateral_solves_across_real_containers() -> None:
    # The real sim-to-real target: the credential chain reused across real per-service
    # containers, each hop a real fetch over the docker network.
    snap = _admit()
    graph = snap.graph
    flag = str(graph.nodes["secret_flag"].attrs["value_ref"])

    runtime = WebappPack().realize(graph, Backing.CONTAINER)
    assert isinstance(runtime, NetworkedContainerWebappRuntime)
    try:
        runtime.reset()
        base = str(runtime.surface()["base_url"])
        trace = solve_chain(graph, lambda p: _get(base, p))
        assert flag in trace.terminal
        assert all(flag not in probe for probe in trace.probes)
        assert "secret_flag" in runtime.collect()["leaked_secret_ids"]
    finally:
        runtime.stop()


def test_chain_soften_is_reachable_from_real_reports(tmp_path: Path) -> None:
    # The soften must be OFFERED from a real foothold-engagement report, not merely
    # work in isolation -- otherwise it's dead in the loop.
    from cyber_webapp.mutation import _REMOVE_RELEVANCE_FLOOR, available_mutations

    pack = WebappPack()
    snap = admit(
        pack,
        manifest={**_manifest(1), "chain": {"depth": {"min": 2, "max": 2}}},
        max_repairs=3,
    )
    assert isinstance(snap, Snapshot)
    graph = snap.graph
    task = next(t for t in snap.tasks if t.meta.get("family") == "webapp.pentest")

    svc = EpisodeService(pack, tmp_path)
    try:
        handle = svc.start_episode(snap, task.id)
        entry = str(graph.nodes[task.entrypoints[0]].attrs["public_url"])
        _get(svc.base_url(handle), entry)  # reaches the foothold without finishing
        report = svc.stop_episode(handle)
    finally:
        svc.close()

    options = available_mutations(graph, "webapp.pentest", [report])
    hop = next(
        (
            m
            for m in options
            if m.direction == "soften" and m.note.startswith("collapse")
        ),
        None,
    )
    assert hop is not None
    assert hop.relevance > _REMOVE_RELEVANCE_FLOOR
