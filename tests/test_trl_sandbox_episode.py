"""The sandbox, wired into the bring-your-own-tools seam (#288) through ``EpisodeEnv``.

A *brought* tool runs in the episode's own throwaway sandbox, the trainer runs no agent
command, and it plugs into the same ``tools=`` seam (#288) with the trainer unchanged.
The shell tool below is the user's — the package ships none; it just reads the ``run``
capability the env injects into the live surface. The gated test boots a REAL cyber
episode on the CONTAINER backing and an agent recovers the flag with its OWN curl, over
the network, from its sandbox — then the real grader returns 1.0.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import pytest
from cyber_webapp import NetworkedContainerWebappRuntime, WebappPack
from cyber_webapp.realize_admit import cmdi_exploit_and_benign
from graphschema import WorldGraph
from openrange_pack_sdk import Backing, Snapshot
from openrange_trl import EpisodeEnv, SandboxError

from openrange.core.admit import admit
from openrange.core.episode import EpisodeService

_CMDI_MANIFEST = {
    "pack": {"id": "webapp"},
    "runtime": {"tick": {"mode": "off"}},
    "npc": [],
    "seed": 7,
    "loot_shapes": {"file": 1, "db": 0},
    "vuln_kinds": {"command_injection": 1},
}

# An SSRF on a public endpoint pivoting to an internal service -> _is_networked is True,
# so this world realizes as the NetworkedContainerWebappRuntime (public + internal
# containers on one net) rather than a single container. seed 3 matches the proven
# networked tests (tests/test_cyber_networked.py).
_SSRF_MANIFEST = {
    "pack": {"id": "webapp"},
    "runtime": {"tick": {"mode": "off"}},
    "npc": [],
    "seed": 3,
    "vuln_kinds": {"ssrf": 1},
}


# -- brought tools: shell runs in the sandbox, submit records the answer for grading --


def shell(surface: Mapping[str, Any], command: str) -> str:
    """Run a shell command on the agent's own sandbox machine.

    Args:
        command: the shell command to run.
    """
    return str(surface["run"](command).output)


def submit(surface: Mapping[str, Any], flag: str) -> str:
    """Submit the recovered flag for grading.

    Args:
        flag: the flag value to submit.
    """
    path = Path(str(surface["solver_root"])) / "result.json"
    path.write_text(json.dumps({"flag": flag}), encoding="utf-8")
    return "submitted"


def _admit_cmdi() -> Snapshot:
    snap = admit(WebappPack(), manifest=_CMDI_MANIFEST, max_repairs=3)
    assert isinstance(snap, Snapshot), snap
    return snap


def _admit_ssrf() -> Snapshot:
    snap = admit(WebappPack(), manifest=_SSRF_MANIFEST, max_repairs=3)
    assert isinstance(snap, Snapshot), snap
    return snap


def _pentest_task_id(snapshot: Snapshot) -> str:
    task = next(t for t in snapshot.tasks if t.meta.get("family") == "webapp.pentest")
    return task.id


def _ssrf_exploit(graph: WorldGraph) -> tuple[str, str, str]:
    """The (public path, query param, payload URL) for the world's networked SSRF.

    Built from the sampled filter the same way the proven networked tests do, so the
    agent's own curl drives the cross-service pivot to the internal flag.
    """
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


def test_a_sandboxed_http_world_needs_a_container(tmp_path: Path) -> None:
    # No docker: a PROCESS cyber world has a base_url but no container to network the
    # sandbox onto, so enabling the sandbox fails fast with a clear message.
    snap = _admit_cmdi()
    service = EpisodeService(WebappPack(), tmp_path / "svc")  # PROCESS backing
    env = EpisodeEnv(
        service=service,
        snapshots={snap.snapshot_id: snap},
        tools=[shell, submit],
        sandbox=True,
    )
    try:
        with pytest.raises(SandboxError, match="CONTAINER"):
            env.reset(snapshot_id=snap.snapshot_id, task_id=_pentest_task_id(snap))
    finally:
        service.close()


# -- gated: the real engine ------------------------------------------------------------


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
    # Whether a container's writes to a host bind mount sync back (false under the macOS
    # /var/folders TMPDIR, where reads leak through but writes don't); see the same
    # probe in test_agent_sandbox.py.
    probe = Path(tempfile.mkdtemp())
    try:
        (probe / "p").write_text("0", encoding="utf-8")
        subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "--user",
                f"{os.getuid()}:{os.getgid()}",
                "-v",
                f"{probe}:/w",
                "-w",
                "/w",
                "python:3.13-slim",
                "bash",
                "-lc",
                "echo 1 > p",
            ],
            check=False,
            capture_output=True,
            timeout=60,
        )
        return (probe / "p").read_text(encoding="utf-8").strip() == "1"
    finally:
        shutil.rmtree(probe, ignore_errors=True)


def _openrange_resources(kind: str) -> set[str]:
    # Names of all openrange-* docker networks (or -a containers), for a leak check: a
    # clean teardown leaves none of an episode's own behind.
    args = (
        ["network", "ls", "--format", "{{.Name}}"]
        if kind == "network"
        else ["ps", "-a", "--format", "{{.Names}}"]
    )
    out = subprocess.run(["docker", *args], capture_output=True, text=True, timeout=10)
    return {n for n in out.stdout.split() if n.startswith("openrange-")}


def _network_is_internal(name: str) -> bool:
    out = subprocess.run(
        ["docker", "network", "inspect", "--format", "{{.Internal}}", name],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    return out.returncode == 0 and out.stdout.strip() == "true"


@gated
def test_byo_shell_tool_exploits_a_real_episode_in_its_sandbox(tmp_path: Path) -> None:
    # End to end through the real harness: a real CONTAINER cyber world. The agent, in
    # its OWN sandbox, exploits it with its OWN curl over the network; the trainer runs
    # no agent command, and the real grader returns full reward.
    snap = _admit_cmdi()
    graph = snap.graph
    exploit_path, _benign = cmdi_exploit_and_benign(graph)
    flag = str(graph.nodes["secret_flag"].attrs["value_ref"])

    service = EpisodeService(WebappPack(), tmp_path / "svc", backing=Backing.CONTAINER)
    env = EpisodeEnv(
        service=service,
        snapshots={snap.snapshot_id: snap},
        tools=[shell, submit],
        sandbox=True,
    )
    try:
        obs = env.reset(snapshot_id=snap.snapshot_id, task_id=_pentest_task_id(snap))
        assert "http://target:" in obs  # the brief points at the in-network alias

        out = env.shell(f"curl -s 'http://target:8000{exploit_path}'")
        assert flag in out  # recovered over the wire, from the sandbox, no shipped tool
        assert env.submit(flag) == "submitted"

        env._finalize()
        assert env.reward == 1.0
        assert env.report is not None and env.report.passed
    finally:
        service.close()


@gated
def test_the_sandbox_can_reach_the_target_but_not_the_host_or_internet(
    tmp_path: Path,
) -> None:
    # The per-episode net is --internal — no gateway — so the agent (untrusted code)
    # reaches the target by alias but has no route off the network: no host, no
    # internet, no other episode's published ports.
    snap = _admit_cmdi()
    service = EpisodeService(WebappPack(), tmp_path / "svc", backing=Backing.CONTAINER)
    env = EpisodeEnv(
        service=service,
        snapshots={snap.snapshot_id: snap},
        tools=[shell, submit],
        sandbox=True,
    )
    try:
        env.reset(snapshot_id=snap.snapshot_id, task_id=_pentest_task_id(snap))
        reachable = env.shell(
            "curl -s -o /dev/null -w '%{http_code}' http://target:8000/"
        )
        assert "200" in reachable, reachable  # the target is reachable on the net
        # No route off the internal network: 1.1.1.1 needs no DNS, --max-time bounds a
        # hang, and any non-zero exit means egress was refused. A bare (non-internal)
        # network would connect (EXIT=0) and fail this — that is the regression guard.
        egress = env.shell("curl --max-time 5 -s http://1.1.1.1; echo EXIT=$?")
        assert "EXIT=0" not in egress, egress
        env._finalize()
    finally:
        service.close()


@gated
def test_byo_shell_tool_pivots_an_ssrf_networked_world_in_its_sandbox(
    tmp_path: Path,
) -> None:
    # The multi-service analogue of the cmdi case: an SSRF world routes to the
    # NetworkedContainerWebappRuntime (a public + an internal container on the world's
    # own net). With sandbox=True the public container is DOUBLE-attached — its world
    # net (so its SSRF handler still pivots to the internal flag) AND a second,
    # per-episode agent net (so the sandbox reaches it by the `target` alias). The agent
    # drives the whole cross-service pivot from its sandbox with its own curl, and the
    # real grader returns full reward — the untested double-attach, proven end to end.
    snap = _admit_ssrf()
    graph = snap.graph
    assert isinstance(
        WebappPack().realize(graph, Backing.CONTAINER), NetworkedContainerWebappRuntime
    )  # ssrf on a public endpoint -> _is_networked -> the networked backing
    flag = str(graph.nodes["secret_flag"].attrs["value_ref"])
    path, param, payload = _ssrf_exploit(graph)

    before_nets = _openrange_resources("network")
    before_cons = _openrange_resources("container")

    service = EpisodeService(WebappPack(), tmp_path / "svc", backing=Backing.CONTAINER)
    env = EpisodeEnv(
        service=service,
        snapshots={snap.snapshot_id: snap},
        tools=[shell, submit],
        sandbox=True,
    )
    agent_net: str | None = None
    target: str | None = None
    try:
        obs = env.reset(snapshot_id=snap.snapshot_id, task_id=_pentest_task_id(snap))
        assert "http://target:" in obs  # the brief points at the in-network alias
        agent_net, target = env._network, env._target_container
        assert agent_net is not None and target is not None

        # The per-episode agent net is --internal: the agent reaches the target by alias
        # but has no route to the host or internet (the world-net pivot is unaffected).
        assert _network_is_internal(agent_net)
        reachable = env.shell(
            "curl -s -o /dev/null -w '%{http_code}' http://target:8000/"
        )
        assert "200" in reachable, reachable  # public service reachable by the alias
        egress = env.shell("curl --max-time 5 -s http://1.1.1.1; echo EXIT=$?")
        assert "EXIT=0" not in egress, egress

        # The SSRF pivot: the agent's curl hits the public service, which fetches the
        # flag from the internal service across the world net and echoes it back — over
        # the double-attached container, from the sandbox, no shipped tool.
        url = f"http://target:8000{path}?{urlencode({param: payload})}"
        out = env.shell(f"curl -s '{url}'")
        assert flag in out, out  # recovered across the container boundary
        assert env.submit(flag) == "submitted"

        env._finalize()
        assert env.reward == 1.0
        assert env.report is not None and env.report.passed
    finally:
        service.close()

    # Teardown leaks nothing: the agent net + the double-attached target are gone, and
    # no openrange-* network/container this episode created survives.
    assert agent_net not in _openrange_resources("network")
    assert target not in _openrange_resources("container")
    assert _openrange_resources("network") <= before_nets
    assert _openrange_resources("container") <= before_cons


@gated
def test_a_code_world_is_edited_through_the_sandbox(tmp_path: Path) -> None:
    # The same seam is domain-agnostic: a code world mounts into the sandbox, so a
    # brought shell tool edits the workspace and the change lands on the host tree the
    # grader reads. (Skips where the host temp dir isn't docker-file-shared.)
    if not _bind_mount_writeback_works():
        pytest.skip("docker bind-mount writeback unavailable (e.g. macOS TMPDIR)")
    from swe import SwePack

    snap = admit(SwePack(), manifest={"instance": "calc_sum"}, max_repairs=0)
    assert isinstance(snap, Snapshot), snap
    service = EpisodeService(SwePack(), tmp_path / "svc")  # PROCESS — a workspace world
    env = EpisodeEnv(
        service=service, snapshots={snap.snapshot_id: snap}, tools=[shell], sandbox=True
    )
    try:
        env.reset(snapshot_id=snap.snapshot_id, task_id=snap.tasks[0].id)
        # A clean append prints nothing; the proof is the change landing on the host
        # tree the grader reads, not the (empty) command output.
        env.shell("echo '# edited in the sandbox' >> calc/core.py")
        solver_root = Path(str(env._surface["solver_root"]))  # type: ignore[index]
        assert "edited in the sandbox" in (solver_root / "calc/core.py").read_text()
    finally:
        service.close()
