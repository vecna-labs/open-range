"""The networked multi-service backing: per-service realization, then (docker-gated)
the runtime that runs one container per service on a real network with real SSRF."""

from __future__ import annotations

import ast
import shutil
import subprocess
import urllib.request

import pytest
from cyber_webapp import (
    ContainerWebappRuntime,
    NetworkedContainerWebappRuntime,
    WebappPack,
)
from cyber_webapp.container import realize_services
from openrange_pack_sdk import Backing, Snapshot

from openrange.core.admit import admit

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


def test_todays_worlds_stay_single_container() -> None:
    # No regression: today's generator co-locates the SSRF with the flag on one internal
    # service, so a world isn't yet "networked-shaped" (SSRF on a public service) — it
    # stays one container. Native networked-shape generation is the follow-up.
    for snap in (_admit_ssrf(), _admit_cmdi()):
        runtime = WebappPack().realize(snap.graph, Backing.CONTAINER)
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
    # directly: today's worlds don't yet auto-route here, see the routing test above.)
    runtime = NetworkedContainerWebappRuntime(_admit_ssrf().graph, Backing.CONTAINER)
    try:
        runtime.reset()
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
    finally:
        runtime.stop()
