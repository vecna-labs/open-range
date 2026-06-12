"""The container backing for a webapp world.

`image_files` packages a world's rendered app into a container build context. The
docker-gated tests then prove the real thing: build the image, run the container, and
recover the flag by exploiting the world over HTTP. The container sets OPENRANGE_REALFS,
so the app's surfaces go real on the one generated app: the file-read shape
(path_traversal, xxe) does a real `open()` with real OS path resolution, and
command_injection runs a real `sh -c` — both with their mutually-exclusive
injection / confinement contexts intact.
"""

from __future__ import annotations

import contextlib
import json
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
from cyber_webapp import ContainerWebappRuntime, WebappPack
from cyber_webapp.codegen import _realize_graph
from cyber_webapp.container import (
    BASE_IMAGE,
    hardening_run_args,
    image_files,
    required_apt_packages,
)
from cyber_webapp.realize_admit import cmdi_exploit_and_benign
from graphschema import Node, WorldGraph
from openrange_pack_sdk import Backing, EpisodeResult, Snapshot

from openrange.core.admit import admit
from openrange.core.episode import EpisodeService


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


_BASE_COMMAND_PACKAGE = {
    "ping": "iputils-ping",
    "nslookup": "dnsutils",
    "dig": "dnsutils",
    "host": "dnsutils",
    "traceroute": "traceroute",
}


def test_required_apt_packages_scopes_to_the_worlds_cmdi_tool() -> None:
    cmdi = _admit_cmdi().graph
    vuln = next(
        n
        for n in cmdi.by_kind("vulnerability")
        if n.attrs.get("kind") == "command_injection"
    )
    params = vuln.attrs["params"]
    assert isinstance(params, dict)
    expected = _BASE_COMMAND_PACKAGE[str(params["base_command"])]
    assert required_apt_packages(cmdi) == {expected}
    # a file-read world runs no server-side OS command → its image installs none
    assert required_apt_packages(_admit_path_traversal().graph) == set()


def test_hardening_run_args_drops_privileges_and_caps_resources() -> None:
    args = hardening_run_args()
    assert args[args.index("--cap-drop") + 1] == "ALL"
    assert "no-new-privileges" in args
    assert "--memory" in args and "--cpus" in args and "--pids-limit" in args


def test_required_apt_packages_skips_malformed_and_unmapped() -> None:
    # A cmdi vuln whose params aren't a mapping, or whose base_command isn't a known
    # diagnostic tool, contributes nothing — no crash, no bogus package.
    graph = _admit_cmdi().graph
    vuln = next(
        n
        for n in graph.by_kind("vulnerability")
        if n.attrs.get("kind") == "command_injection"
    )
    vuln.attrs["params"] = "not-a-mapping"
    assert required_apt_packages(graph) == set()
    vuln.attrs["params"] = {"base_command": "whoami", "target_param": "q"}
    assert required_apt_packages(graph) == set()


def test_dockerfile_installs_os_tools_only_when_a_vuln_needs_them() -> None:
    cmdi_df = image_files(_admit_cmdi().graph)["Dockerfile"]
    pt_df = image_files(_admit_path_traversal().graph)["Dockerfile"]
    assert "apt-get install" in cmdi_df  # cmdi world needs its diagnostic tool
    assert "apt-get" not in pt_df  # a file-read world stays lean — no OS tools
    assert "pip install --no-cache-dir jinja2" in pt_df  # the app's one structural dep


def test_every_sampled_base_command_has_an_apt_package() -> None:
    # Lockstep guard: every base_command the sampler picks must map to a package, or a
    # cmdi world ships without the tool its real `sh -c` needs.
    from cyber_webapp.container import _CMDI_APT_PACKAGES
    from cyber_webapp.sampling import _COMMAND_INJECTION_BASE

    assert set(_COMMAND_INJECTION_BASE) <= set(_CMDI_APT_PACKAGES)


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
        body = urllib.request.urlopen(url, timeout=10).read()
    except urllib.error.HTTPError as exc:
        body = exc.read()
    return bytes(body).decode()


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
    run_cmd = ["docker", "run", "-d", "-p", "0:8000", *hardening_run_args()]
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


@pytest.mark.skipif(not _docker_available(), reason="docker engine not reachable")
def test_world_container_is_hardened(tmp_path: Path) -> None:
    # The world runs attacker-controlled code, so it is contained: all capabilities
    # dropped, no privilege escalation, and memory / pid caps set — verified both on the
    # run config and behaviourally inside the container. It stays exploitable over HTTP
    # under these flags (every other docker test here runs with the same _container).
    snap = _admit_cmdi()
    graph = snap.graph
    tag = f"openrange-m1-harden-{snap.snapshot_id[:8]}"
    with _container(image_files(graph), tmp_path, tag) as base:
        cid = subprocess.run(
            ["docker", "ps", "-q", "--filter", f"ancestor={tag}"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.split()[0]
        host = json.loads(
            subprocess.run(
                ["docker", "inspect", cid],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            ).stdout
        )[0]["HostConfig"]
        assert host["CapDrop"] == ["ALL"], host["CapDrop"]
        assert any("no-new-privileges" in opt for opt in host.get("SecurityOpt") or [])
        assert host["Memory"] > 0 and host["PidsLimit"] and host["PidsLimit"] > 0

        # Behavioural: effective capabilities are actually all-zero in the container.
        status = subprocess.run(
            ["docker", "exec", cid, "cat", "/proc/self/status"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout
        cap_eff = next(ln for ln in status.splitlines() if ln.startswith("CapEff:"))
        assert cap_eff.split()[1].strip("0") == "", cap_eff

        # Still exploitable under the hardening — containment doesn't break the vuln.
        exploit_path, _benign = cmdi_exploit_and_benign(graph)
        body = _http_get(base + exploit_path)
    assert str(graph.nodes["secret_flag"].attrs["value_ref"]) in body, body[:200]


def test_generated_app_has_a_real_shell_cmdi_branch() -> None:
    import ast

    source = _realize_graph(_admit_cmdi().graph)["app.py"]
    ast.parse(source)  # the generated app is valid Python
    assert "subprocess.run" in source  # real shell, gated by OPENRANGE_REALFS
    assert 'os.environ.get("OPENRANGE_REALFS")' in source  # the CONTAINER toggle


@pytest.mark.skipif(not _docker_available(), reason="docker engine not reachable")
def test_real_shell_container_recovers_a_real_file_flag(tmp_path: Path) -> None:
    snap = _admit_cmdi()
    graph = snap.graph
    _pin_context(graph, "separator")  # a clean `; cat <path>` exploit
    flag = str(graph.nodes["secret_flag"].attrs["value_ref"])
    exploit_path, benign_path = cmdi_exploit_and_benign(graph)

    tag = f"openrange-m1-realfs-{snap.snapshot_id[:12]}"
    with _container(image_files(graph), tmp_path, tag) as base:
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
    # The injection contexts hold over a REAL shell, not just the in-memory emulation: a
    # world built for one context is exploited by THAT context's payload and NOT by
    # another's (the wrong vectors are filtered before sh).
    snap = _admit_cmdi()
    graph = snap.graph
    flag = str(graph.nodes["secret_flag"].attrs["value_ref"])

    matching = _exploit_for(graph, live)
    mismatched = _exploit_for(graph, wrong)
    _pin_context(graph, live)  # the image must be built from the live context

    tag = f"openrange-m1-ctx-{live}-{snap.snapshot_id[:8]}"
    with _container(image_files(graph), tmp_path, tag) as base:
        hit = urllib.request.urlopen(base + matching, timeout=10).read().decode()
        miss = urllib.request.urlopen(base + mismatched, timeout=10).read().decode()
    assert flag in hit, hit[:200]  # the matching context's exploit lands
    assert flag not in miss  # a wrong-context exploit is filtered out


# --- CONTAINER backing wired as a runtime: it grades identically to PROCESS -----------


def _run_pentest_episode(
    snapshot: Snapshot,
    task_id: str,
    backing: Backing,
    root: Path,
    exploit_path: str,
    flag: str,
) -> EpisodeResult:
    # Drive one pentest episode end to end on the given backing: start it, run the
    # exploit over its live HTTP surface, submit the recovered flag, return the result.
    service = EpisodeService(WebappPack(), root, backing=backing)
    try:
        handle = service.start_episode(snapshot, task_id)
        surface = service.surface(handle)
        base_url = str(surface["base_url"])
        solver_root = Path(str(surface["solver_root"]))
        body = (
            urllib.request.urlopen(base_url + exploit_path, timeout=20).read().decode()
        )
        assert flag in body, f"{backing}: {body[:200]}"
        (solver_root / "result.json").write_text(
            json.dumps({"flag": flag}), encoding="utf-8"
        )
        report = service.stop_episode(handle)
    finally:
        service.close()
    return report.episode_result


def test_container_runtime_rejects_non_container_backing() -> None:
    with pytest.raises(NotImplementedError):
        ContainerWebappRuntime(_admit_cmdi().graph, Backing.PROCESS)


def test_container_runtime_is_inert_before_reset() -> None:
    # No container yet (no docker touched): the log read is None and stop() is a clean
    # no-op — nothing built or running to tear down.
    runtime = ContainerWebappRuntime(_admit_cmdi().graph, Backing.CONTAINER)
    assert runtime._read_log_bytes() is None
    runtime.stop()  # must not raise with nothing built/running


@pytest.mark.skipif(not _docker_available(), reason="docker engine not reachable")
def test_container_runtime_reuses_the_image_across_resets() -> None:
    # The image builds once and is reused on later resets; each reset brings up a fresh
    # container on a fresh published port.
    runtime = ContainerWebappRuntime(_admit_cmdi().graph, Backing.CONTAINER)
    try:
        runtime.reset()
        first = str(runtime.surface()["base_url"])
        runtime.reset()  # image already built → rebuild is skipped
        second = str(runtime.surface()["base_url"])
        assert first.startswith("http://127.0.0.1:")
        assert second.startswith("http://127.0.0.1:")
    finally:
        runtime.stop()


@pytest.mark.skipif(not _docker_available(), reason="docker engine not reachable")
def test_container_and_process_backings_grade_identically(tmp_path: Path) -> None:
    # The load-bearing parity check: the SAME snapshot + SAME exploit grades identically
    # on PROCESS (in-memory emulation) and CONTAINER (a real shell in a container).
    # Only fidelity changes between the backings, not the task surface.
    snap = _admit_cmdi()
    graph = snap.graph
    _pin_context(graph, "separator")
    flag = str(graph.nodes["secret_flag"].attrs["value_ref"])
    task = next(t for t in snap.tasks if t.meta.get("family") == "webapp.pentest")
    exploit_path, _benign = cmdi_exploit_and_benign(graph)

    process = _run_pentest_episode(
        snap, task.id, Backing.PROCESS, tmp_path / "proc", exploit_path, flag
    )
    container = _run_pentest_episode(
        snap, task.id, Backing.CONTAINER, tmp_path / "cont", exploit_path, flag
    )

    assert process.success is True  # the exploit really solves the world
    assert container.success == process.success
    assert container.subgoals == process.subgoals  # identical grade across backings


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
