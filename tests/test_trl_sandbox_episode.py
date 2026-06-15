"""The sandbox, wired into the bring-your-own-tools seam (#288) through ``EpisodeEnv``.

This is the #290 close: a *brought* tool runs in the episode's own throwaway sandbox,
the trainer runs no agent command, and it plugs into the same ``tools=`` seam with the
trainer unchanged. The shell tool below is the user's — the package ships none; it just
reads the ``run`` capability the env injects into the live surface. The gated test boots
a REAL cyber episode on the CONTAINER backing and an agent recovers the flag with its
OWN curl, over the network, from its sandbox — then the real grader returns 1.0.
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

import pytest
from cyber_webapp import WebappPack
from cyber_webapp.realize_admit import cmdi_exploit_and_benign
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


def _pentest_task_id(snapshot: Snapshot) -> str:
    task = next(t for t in snapshot.tasks if t.meta.get("family") == "webapp.pentest")
    return task.id


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


@gated
def test_byo_shell_tool_exploits_a_real_episode_in_its_sandbox(tmp_path: Path) -> None:
    # The #290 close, end to end through the real harness: a real CONTAINER cyber world,
    # the agent in its OWN sandbox exploits it with its OWN curl over the network, the
    # trainer runs no agent command, and the real grader returns full reward.
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
    # #290 criterion 1 ENFORCED, not just claimed: the per-episode net is --internal, so
    # the agent (untrusted code) reaches the target by alias but has no route off the
    # network — no host, no internet, no other episode's published ports.
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
