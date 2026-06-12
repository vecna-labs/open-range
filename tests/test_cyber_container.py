"""M1 — the container backing (DESIGN.md §9, #252).

`image_files` packages a world's rendered app into a container build context. The
docker-gated tests then prove the real thing: build the image, run the container, and
recover the flag by exploiting the world over HTTP. The container sets OPENRANGE_REALFS,
so the file-read shape (path_traversal, xxe) and the cmdi readers hit a REAL filesystem:
a real `open()` and real OS path resolution, not the in-memory dict. The stdlib
`image_files_realfs` variant additionally proves a REAL `sh -c` for command_injection.
"""

from __future__ import annotations

import contextlib
import posixpath
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Iterator, Sequence
from pathlib import Path
from urllib.parse import quote

import pytest
from cyber_webapp import WebappPack
from cyber_webapp.container import (
    BASE_IMAGE,
    image_files,
    image_files_realfs,
    realfs_cmdi_app,
)
from cyber_webapp.realize_admit import cmdi_exploit_and_benign
from graphschema import Node, WorldGraph
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


def _http_get(url: str) -> str:
    # The response body regardless of status — a neutralized traversal answers 403/404,
    # which urlopen raises on; we still want to assert the flag is NOT in that body.
    try:
        return urllib.request.urlopen(url, timeout=10).read().decode()
    except urllib.error.HTTPError as exc:
        return exc.read().decode()


def _wait_ready(base: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            urllib.request.urlopen(base + "/", timeout=2)
            return
        except OSError:  # URLError is an OSError subclass
            time.sleep(0.3)
    raise AssertionError(f"container did not become ready at {base}")


@contextlib.contextmanager
def _container(
    build_files: dict[str, str],
    tmp_path: Path,
    tag: str,
    *,
    env: Sequence[tuple[str, str]] = (),
) -> Iterator[str]:
    # Build the given image build-context, run it (with any -e env), and yield the base
    # URL once it answers. Cleans up image + container regardless of outcome.
    context = tmp_path / "ctx"
    context.mkdir()
    for name, content in build_files.items():
        (context / name).write_text(content, encoding="utf-8")
    run_cmd = ["docker", "run", "-d", "-p", "0:8000"]
    for key, value in env:
        run_cmd += ["-e", f"{key}={value}"]
    run_cmd.append(tag)
    container_id = ""
    try:
        subprocess.run(
            ["docker", "build", "-q", "-t", tag, str(context)],
            check=True,
            capture_output=True,
            timeout=600,
        )
        started = subprocess.run(
            run_cmd, check=True, capture_output=True, text=True, timeout=60
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
        yield base
    finally:
        if container_id:
            subprocess.run(["docker", "rm", "-f", container_id], capture_output=True)
        subprocess.run(["docker", "rmi", "-f", tag], capture_output=True)


def _cmdi_params(graph: WorldGraph) -> dict[str, object]:
    vuln = next(
        n
        for n in graph.by_kind("vulnerability")
        if n.attrs.get("kind") == "command_injection"
    )
    params = vuln.attrs["params"]
    assert isinstance(params, dict)
    return params


def _pin_context(graph: WorldGraph, context: str) -> None:
    params = _cmdi_params(graph)
    params["inj_context"] = context
    params["quote"] = "'"


def _exploit_for(graph: WorldGraph, context: str) -> str:
    # The context-matching exploit path; mutates params transiently to shape its payload
    _pin_context(graph, context)
    exploit_path, _benign = cmdi_exploit_and_benign(graph)
    return exploit_path


@pytest.mark.skipif(not _docker_available(), reason="docker engine not reachable")
def test_world_runs_in_a_container_and_is_exploited(tmp_path: Path) -> None:
    snap = _admit_cmdi()
    graph = snap.graph
    tag = f"openrange-m1-{snap.snapshot_id[:12]}"
    with _container(image_files(graph), tmp_path, tag) as base:
        exploit_path, _benign = cmdi_exploit_and_benign(graph)
        expected = str(graph.nodes["secret_flag"].attrs["value_ref"])
        body = urllib.request.urlopen(base + exploit_path, timeout=10).read().decode()
    assert expected in body, body[:200]


def test_realfs_cmdi_app_is_a_valid_real_shell_app() -> None:
    import ast

    source = realfs_cmdi_app(_admit_cmdi().graph)
    ast.parse(source)  # the generated app is valid Python
    assert "subprocess.run" in source  # a real shell, not the in-memory emulation
    assert "OPENRANGE_FLAG" in source  # the flag arrives at run time, not in the image


@pytest.mark.skipif(not _docker_available(), reason="docker engine not reachable")
def test_real_shell_container_recovers_a_real_file_flag(tmp_path: Path) -> None:
    snap = _admit_cmdi()
    graph = snap.graph
    _pin_context(graph, "separator")  # a clean `; cat <path>` exploit
    flag = str(graph.nodes["secret_flag"].attrs["value_ref"])
    exploit_path, benign_path = cmdi_exploit_and_benign(graph)

    tag = f"openrange-m1-realfs-{snap.snapshot_id[:12]}"
    env = [("OPENRANGE_FLAG", flag)]
    with _container(image_files_realfs(graph), tmp_path, tag, env=env) as base:
        # A real `cat` against the real filesystem recovers the real file's flag.
        exploit_body = (
            urllib.request.urlopen(base + exploit_path, timeout=10).read().decode()
        )
        benign_body = (
            urllib.request.urlopen(base + benign_path, timeout=10).read().decode()
        )
    assert flag in exploit_body, exploit_body[:200]
    assert flag not in benign_body


@pytest.mark.skipif(not _docker_available(), reason="docker engine not reachable")
@pytest.mark.parametrize(
    ("live", "wrong"),
    [
        ("separator", "substitution"),
        ("substitution", "separator"),
        ("quoted", "separator"),
    ],
)
def test_real_shell_contexts_are_mutually_exclusive(
    live: str, wrong: str, tmp_path: Path
) -> None:
    # The §6 contexts hold over a REAL shell, not just the in-memory emulation: a world
    # built for one injection context is exploited by THAT context's payload and is NOT
    # exploited by another context's payload (the wrong vectors are filtered before sh).
    snap = _admit_cmdi()
    graph = snap.graph
    flag = str(graph.nodes["secret_flag"].attrs["value_ref"])

    matching = _exploit_for(graph, live)
    mismatched = _exploit_for(graph, wrong)
    _pin_context(graph, live)  # the image must be built from the live context

    tag = f"openrange-m1-ctx-{live}-{snap.snapshot_id[:8]}"
    env = [("OPENRANGE_FLAG", flag)]
    with _container(image_files_realfs(graph), tmp_path, tag, env=env) as base:
        hit = urllib.request.urlopen(base + matching, timeout=10).read().decode()
        miss = urllib.request.urlopen(base + mismatched, timeout=10).read().decode()
    assert flag in hit, hit[:200]  # the matching context's exploit lands
    assert flag not in miss  # a wrong-context exploit is filtered out


# --- file_read shape over a real filesystem (generalize past command_injection) ------


def _admit_path_traversal() -> Snapshot:
    snap = admit(
        WebappPack(),
        manifest={
            "pack": {"id": "webapp"},
            "runtime": {"tick": {"mode": "off"}},
            "npc": [],
            "seed": 7,
            "loot_shapes": {"file": 1, "db": 0},
            "vuln_kinds": {"path_traversal": 1},
        },
        max_repairs=3,
    )
    assert isinstance(snap, Snapshot), snap
    return snap


def _pt_vuln(graph: WorldGraph) -> Node:
    return next(
        n
        for n in graph.by_kind("vulnerability")
        if n.attrs.get("kind") == "path_traversal"
    )


def _flag_file_path(graph: WorldGraph) -> str:
    # The file whose content is the flag, in the projected seed's file map.
    from cyber_webapp.codegen.seeding import project_seed

    flag = str(graph.nodes["secret_flag"].attrs["value_ref"])
    files = project_seed(graph)["files"]
    assert isinstance(files, dict)
    for path, content in files.items():
        if content == flag:
            return str(path)
    raise AssertionError("no seed file holds the flag")


def _pt_url(graph: WorldGraph, payload: str) -> str:
    vuln = _pt_vuln(graph)
    params = vuln.attrs["params"]
    assert isinstance(params, dict)
    endpoint_id = next(e.dst for e in graph.out_edges(vuln.id, "affects"))
    public_url = str(graph.nodes[endpoint_id].attrs["public_url"])
    param = str(params["target_param"])
    return f"{public_url}?{param}={quote(payload, safe='')}"


@pytest.mark.skipif(not _docker_available(), reason="docker engine not reachable")
@pytest.mark.parametrize("confinement", ["absolute_only", "relative", "dotdot_filter"])
def test_path_traversal_reads_a_real_file_in_a_container(
    confinement: str, tmp_path: Path
) -> None:
    # The file_read shape is REAL on the generated app in a container: a path-traversal
    # escape is a real open() against the real container fs, and the three confinement
    # contexts stay mutually exclusive over it — each accepts ONE technique and
    # neutralizes the others, so a wrong-technique payload recovers nothing.
    snap = _admit_path_traversal()
    graph = snap.graph
    flag = str(graph.nodes["secret_flag"].attrs["value_ref"])
    vuln = _pt_vuln(graph)
    params = vuln.attrs["params"]
    assert isinstance(params, dict)
    params["confinement"] = confinement

    base_dir = str(params["base_dir"])
    flag_path = _flag_file_path(graph)
    relchain = posixpath.relpath(flag_path, base_dir)
    assert ".." in relchain  # the flag is reachable only by escaping base_dir
    payloads = {
        # confinement: (the technique that escapes, a technique it neutralizes)
        "absolute_only": (flag_path, relchain),
        "relative": (relchain, flag_path),
        "dotdot_filter": (relchain.replace("../", "....//"), relchain),
    }
    matching, wrong = payloads[confinement]

    tag = f"openrange-m1-pt-{confinement}-{snap.snapshot_id[:8]}"
    with _container(image_files(graph), tmp_path, tag) as base:
        hit = _http_get(base + _pt_url(graph, matching))
        miss = _http_get(base + _pt_url(graph, wrong))
    assert flag in hit, hit[:200]  # real open() recovers the real file via this escape
    assert flag not in miss  # a wrong-technique payload this confinement neutralizes
