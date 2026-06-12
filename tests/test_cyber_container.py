"""M1 — the container backing, first brick (DESIGN.md §9, #252).

`image_files` packages a world's rendered app into a container build context. The
docker-gated test then proves the real thing: build the image, run the container, and
recover the flag by exploiting the world over HTTP. (This brick containerizes the
existing in-memory app; making the exploits hit the container's *real* fs/shell is the
next M1 increment.)
"""

from __future__ import annotations

import shutil
import subprocess
import time
import urllib.request
from pathlib import Path

import pytest
from cyber_webapp import WebappPack
from cyber_webapp.container import BASE_IMAGE, image_files
from cyber_webapp.realize_admit import cmdi_exploit_and_benign
from openrange_pack_sdk import Snapshot

from openrange.core.admit import admit


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


def test_image_files_packages_the_world() -> None:
    files = image_files(_admit_cmdi().graph)
    assert set(files) == {"Dockerfile", "app.py", "seed.json"}
    assert BASE_IMAGE in files["Dockerfile"]
    assert "def handle" in files["app.py"]
    assert '"--port", "8000"' in files["Dockerfile"]


def _docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        probe = subprocess.run(
            ["docker", "info"], capture_output=True, timeout=10, check=False
        )
    except Exception:  # noqa: BLE001 - a best-effort probe; any failure means "no"
        return False
    return probe.returncode == 0


def _wait_ready(base: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            urllib.request.urlopen(base + "/", timeout=2)
            return
        except OSError:  # URLError is an OSError subclass
            time.sleep(0.3)
    raise AssertionError(f"container did not become ready at {base}")


@pytest.mark.skipif(not _docker_available(), reason="docker engine not reachable")
def test_world_runs_in_a_container_and_is_exploited(tmp_path: Path) -> None:
    snap = _admit_cmdi()
    graph = snap.graph
    context = tmp_path / "ctx"
    context.mkdir()
    for name, content in image_files(graph).items():
        (context / name).write_text(content, encoding="utf-8")

    tag = f"openrange-m1-{snap.snapshot_id[:12]}"
    container_id = ""
    try:
        subprocess.run(
            ["docker", "build", "-q", "-t", tag, str(context)],
            check=True,
            capture_output=True,
            timeout=600,
        )
        started = subprocess.run(
            ["docker", "run", "-d", "-p", "0:8000", tag],
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        container_id = started.stdout.strip()
        mapping = subprocess.run(
            ["docker", "port", container_id, "8000"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        base = f"http://127.0.0.1:{mapping.rsplit(':', 1)[-1]}"
        _wait_ready(base, timeout=30)

        exploit_path, _benign = cmdi_exploit_and_benign(graph)
        expected = str(graph.nodes["secret_flag"].attrs["value_ref"])
        body = urllib.request.urlopen(base + exploit_path, timeout=10).read().decode()
        assert expected in body, body[:200]
    finally:
        if container_id:
            subprocess.run(["docker", "rm", "-f", container_id], capture_output=True)
        subprocess.run(["docker", "rmi", "-f", tag], capture_output=True)
