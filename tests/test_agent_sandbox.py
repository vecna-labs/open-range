"""The agent runs its own commands in its sandbox; OpenRange ships no tools.

The pure tests pin the binding contract (an HTTP world joins the network by alias; a
code world is bind-mounted; an opaque world binds nothing) and the lifecycle guards,
with no docker. The gated tests prove the model end to end on a real engine: an agent
exploits a containerized cyber world from its sandbox using its **own** ``curl`` — no
shipped tool — and edits a workspace as its own filesystem through the bind mount.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import subprocess
import tempfile
import time
import urllib.request
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from cyber_webapp import WebappPack
from cyber_webapp.container import hardening_run_args, image_files
from cyber_webapp.realize_admit import cmdi_exploit_and_benign
from openrange_pack_sdk import Snapshot
from openrange_trl import AgentSandbox, SandboxError
from openrange_trl import sandbox as sandbox_mod

from openrange.core.admit import admit

# --- pure: the binding contract and lifecycle guards (no docker) ----------------------


class TestSurfaceBinding:
    def test_http_world_joins_the_network_by_alias(self) -> None:
        sandbox = AgentSandbox({"base_url": "http://target:8000"}, network="net-x")
        assert sandbox._surface_bindings("agent-1") == [
            "--network",
            "net-x",
            "--network-alias",
            "agent-1",
        ]

    def test_http_world_without_a_network_is_refused(self) -> None:
        # An HTTP world is reached only over the wire; with no network to join there is
        # no target to reach, so this fails fast rather than starting a blind sandbox.
        with pytest.raises(SandboxError, match="network"):
            AgentSandbox({"base_url": "http://target:8000"})._surface_bindings("a")

    def test_code_world_bind_mounts_the_workspace(self, tmp_path: Path) -> None:
        sandbox = AgentSandbox({"solver_root": str(tmp_path)})
        assert sandbox._surface_bindings("agent-1") == [
            "-v",
            f"{tmp_path.resolve()}:/workspace",
            "-w",
            "/workspace",
        ]

    def test_opaque_world_binds_nothing(self) -> None:
        assert AgentSandbox({})._surface_bindings("agent-1") == []


class TestLifecycleGuards:
    def test_run_before_start_is_refused(self) -> None:
        with pytest.raises(SandboxError, match="not started"):
            AgentSandbox({}).run("echo hi")

    def test_close_before_start_is_a_noop(self) -> None:
        AgentSandbox({}).close()  # nothing started — must not raise


class TestLeakSafety:
    def test_track_and_untrack_manage_the_sweep_set(self) -> None:
        # track_resource records a (kind, name) for the atexit sweep; untrack_resource
        # drops it once teardown removed the resource. Pure set logic — no docker.
        sandbox_mod._TRACKED.clear()
        try:
            sandbox_mod.track_resource("container", "c-x")
            sandbox_mod.track_resource("network", "n-x")
            assert ("container", "c-x") in sandbox_mod._TRACKED
            assert ("network", "n-x") in sandbox_mod._TRACKED
            sandbox_mod.untrack_resource("container", "c-x")
            assert ("container", "c-x") not in sandbox_mod._TRACKED
        finally:
            sandbox_mod._TRACKED.clear()


# --- gated: the real engine (build + run real containers) -----------------------------


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


gated = pytest.mark.skipif(
    not _docker_available(), reason="docker engine not reachable"
)


def _bind_mount_writeback_works() -> bool:
    # Whether a container's writes to a host bind mount sync back. True on native Linux
    # (CI); false where the OS temp dir isn't file-shared with the docker VM (e.g. the
    # macOS /var/folders TMPDIR under Docker Desktop) — there reads leak through but
    # writes don't, so a mount-writeback assertion would be a false negative.
    probe = Path(tempfile.mkdtemp())
    try:
        (probe / "p").write_text("0", encoding="utf-8")
        subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "--user",
                f"{os.getuid()}:{os.getgid()}",  # mirror the sandbox's non-root user
                "-v",
                f"{probe}:/w",
                "-w",
                "/w",
                "python:3.13-slim",
                "bash",
                "-lc",
                "echo 1 > p",
            ],
            check=False,  # a failed write (no perms / no file-share) just means False
            capture_output=True,
            timeout=60,
        )
        return (probe / "p").read_text(encoding="utf-8").strip() == "1"
    finally:
        shutil.rmtree(probe, ignore_errors=True)


def _admit_cmdi() -> Snapshot:
    snap = admit(
        WebappPack(),
        manifest={
            "pack": {"id": "webapp"},
            "runtime": {"tick": {"mode": "off"}},
            "npc": [],
            "seed": 7,
            "loot": {"file": 1, "db": 0},
            "vuln": {"pin": [{"kind": "command_injection"}]},
        },
        max_repairs=3,
    )
    assert isinstance(snap, Snapshot), snap
    return snap


def _wait_ready(base: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            urllib.request.urlopen(base + "/", timeout=2)
            return
        except OSError:  # URLError is an OSError subclass
            time.sleep(0.3)
    raise AssertionError(f"world did not become ready at {base}")


@contextlib.contextmanager
def _world_on_network(
    build_files: dict[str, str], tmp_path: Path, tag: str, network: str, alias: str
) -> Iterator[None]:
    # Build the world image and run it on the given network under `alias` (so a sandbox
    # on the same network reaches it by name), also host-published so we can wait ready.
    context = tmp_path / "world-ctx"
    context.mkdir()
    for name, content in build_files.items():
        (context / name).write_text(content, encoding="utf-8")
    subprocess.run(
        ["docker", "build", "-q", "-t", tag, str(context)],
        check=True,
        capture_output=True,
        timeout=600,
    )
    cid = subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "-p",
            "0:8000",
            "--network",
            network,
            "--network-alias",
            alias,
            *hardening_run_args(),
            tag,
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    ).stdout.strip()
    try:
        mapping = subprocess.run(
            ["docker", "port", cid, "8000"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        _wait_ready(f"http://127.0.0.1:{mapping.rsplit(':', 1)[-1]}", timeout=30)
        yield
    finally:
        subprocess.run(["docker", "rm", "-f", cid], capture_output=True)
        subprocess.run(["docker", "rmi", "-f", tag], capture_output=True)


@gated
def test_agent_exploits_an_http_world_from_its_sandbox(tmp_path: Path) -> None:
    # The model end to end: the world is a network target; the agent, in its own
    # sandbox, exploits it with its OWN curl and recovers the flag. No shipped tool.
    snap = _admit_cmdi()
    graph = snap.graph
    exploit_req, _benign = cmdi_exploit_and_benign(graph)
    flag = str(graph.nodes["secret_flag"].attrs["value_ref"])

    target = f"http://target:8000{exploit_req.path}"
    if exploit_req.method == "POST":
        curl = (
            f"curl -s -X POST -H 'Content-Type: {exploit_req.content_type}' "
            f"--data '{exploit_req.body}' '{target}'"
        )
    else:
        curl = f"curl -s '{target}'"

    network = f"openrange-agent-net-{snap.snapshot_id[:12]}"
    tag = f"openrange-agent-world-{snap.snapshot_id[:12]}"
    subprocess.run(
        ["docker", "network", "create", network],
        check=True,
        capture_output=True,
        timeout=30,
    )
    try:
        with (
            _world_on_network(image_files(graph), tmp_path, tag, network, "target"),
            AgentSandbox({"base_url": "http://target:8000"}, network=network) as sb,
        ):
            result = sb.run(curl)
        assert result.exit_code == 0, result.output
        assert flag in result.output, result.output[:300]
    finally:
        subprocess.run(["docker", "network", "rm", network], capture_output=True)


@gated
def test_agent_edits_a_code_world_as_its_filesystem(tmp_path: Path) -> None:
    # A code world is the agent's filesystem: it edits the bind-mounted workspace, and
    # the change is visible back on the host (the foundation for edit-and-grade).
    if not _bind_mount_writeback_works():
        pytest.skip("docker bind-mount writeback unavailable (e.g. macOS TMPDIR)")
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "a.py").write_text("x = 1\n", encoding="utf-8")
    with AgentSandbox({"solver_root": str(workspace)}) as sandbox:
        edit = sandbox.run("echo 'y = 2' >> a.py")
        assert edit.exit_code == 0, edit.output
        shown = sandbox.run("cat a.py")
        assert "y = 2" in shown.output
    assert "y = 2" in (workspace / "a.py").read_text(encoding="utf-8")


@gated
def test_sandbox_is_hardened_reuses_the_image_and_cleans_up(tmp_path: Path) -> None:
    surface = {"solver_root": str(tmp_path)}
    first = AgentSandbox(surface)
    first.start()
    cname = first._cname
    assert cname is not None
    try:
        record = json.loads(
            subprocess.run(
                ["docker", "inspect", cname],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            ).stdout
        )[0]
        host = record["HostConfig"]
        assert host["CapDrop"] == ["ALL"], host["CapDrop"]
        assert any("no-new-privileges" in opt for opt in host.get("SecurityOpt") or [])
        assert host["Memory"] > 0 and host["PidsLimit"] and host["PidsLimit"] > 0
        assert record["Config"]["User"] == f"{os.getuid()}:{os.getgid()}"  # never root

        with pytest.raises(SandboxError, match="already started"):
            first.start()  # the guard holds — no second container for one sandbox

        # The image is content-tagged, so a second sandbox reuses it (no rebuild).
        second = AgentSandbox(surface)
        second.start()
        second.close()
    finally:
        first.close()

    gone = subprocess.run(
        ["docker", "inspect", cname], capture_output=True, text=True, timeout=10
    )
    assert gone.returncode != 0  # close() removed the container


@gated
def test_the_atexit_sweep_removes_a_tracked_container_and_network(
    tmp_path: Path,
) -> None:
    # The shutdown safety net: the sweep force-removes every resource this process
    # tracked but didn't tear down. Track a real container (via start) and a real
    # network, sweep, and assert both are gone and the set is cleared — no test doubles.
    sandbox_mod._TRACKED.clear()
    network = f"openrange-agent-net-sweep-{uuid.uuid4().hex[:8]}"
    subprocess.run(
        ["docker", "network", "create", network],
        check=True,
        capture_output=True,
        timeout=30,
    )
    sandbox = AgentSandbox({"solver_root": str(tmp_path)})
    sandbox.start()
    cname = sandbox._cname
    assert cname is not None
    sandbox_mod.track_resource("network", network)
    try:
        sandbox_mod._sweep_tracked()
        assert not sandbox_mod._TRACKED  # the sweep clears what it handled
        assert (
            subprocess.run(
                ["docker", "inspect", cname], capture_output=True, timeout=10
            ).returncode
            != 0
        )
        assert (
            subprocess.run(
                ["docker", "network", "inspect", network],
                capture_output=True,
                timeout=10,
            ).returncode
            != 0
        )
    finally:
        sandbox_mod._TRACKED.clear()
        subprocess.run(["docker", "rm", "-f", cname], capture_output=True)
        subprocess.run(["docker", "network", "rm", network], capture_output=True)


@gated
def test_a_leaked_sandbox_is_discoverable_by_its_label(tmp_path: Path) -> None:
    # The interruption backstop on the real engine: even without a clean teardown the
    # container is found by the documented label filter, so an operator (or the one-line
    # prune) can reclaim it. We start without closing, then assert the filter finds it.
    sandbox = AgentSandbox({"solver_root": str(tmp_path)})
    sandbox.start()
    cname = sandbox._cname
    assert cname is not None
    try:
        listed = subprocess.run(
            [
                "docker",
                "ps",
                "-q",
                "--filter",
                "label=openrange.sandbox=1",
                "--filter",
                f"name={cname}",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout
        assert listed.strip(), "labelled sandbox not found by the documented filter"
    finally:
        sandbox.close()
