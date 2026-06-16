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
        "lateral_movement": True,
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
