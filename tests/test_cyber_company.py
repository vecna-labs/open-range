"""Company worlds (DESIGN.md §11): a believable multi-service estate the agent recons
and pivots through. Generation + a PROCESS solve here; the docker-gated test proves the
same recon→pivot recovers the flag across real containers."""

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
from cyber_webapp.reference_solver import solve_chain
from graphschema import Node, WorldGraph
from openrange_pack_sdk import Backing, Snapshot
from openrange_trl import EpisodeEnv

from examples.tools import WEB_TOOLS
from openrange.core.admit import admit
from openrange.core.episode import EpisodeService

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
    # The notebook's reward surface (examples/trl_grpo_cyber.ipynb §4), pinned on the
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
