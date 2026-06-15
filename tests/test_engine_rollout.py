"""The async engine, end to end on real episodes — concurrency, grading, isolation.

No model: a deterministic policy (a real :data:`Policy`-shaped callable, not a mock)
issues the known cmdi exploit + submit over real CONTAINER cyber worlds. Proves a
rollout grades to 1.0, that ``batch`` truly runs episodes concurrently (an
``asyncio.Barrier`` all N must reach at once — serialization would hang), and that a
no-op rollout scores 0.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
from cyber_webapp import WebappPack
from cyber_webapp.realize_admit import cmdi_exploit_and_benign
from openrange_pack_sdk import Backing, Snapshot
from openrange_trl import make_environment_factory
from openrange_trl.engine import Action, AsyncRollout, Finish, Policy, ToolCall, batch

from openrange.core.admit import admit
from openrange.training import Trajectory

_CMDI_MANIFEST = {
    "pack": {"id": "webapp"},
    "runtime": {"tick": {"mode": "off"}},
    "npc": [],
    "seed": 7,
    "loot_shapes": {"file": 1, "db": 0},
    "vuln_kinds": {"command_injection": 1},
}


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


def _solver(
    exploit_path: str, flag: str, *, gate: asyncio.Barrier | None = None
) -> Policy:
    # A fresh per-rollout policy: curl the exploit, submit the flag, finish. `gate`, if
    # given, blocks the first action until every rollout has reached it (overlap proof).
    state = {"step": 0}

    async def policy(
        messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> Action:
        step = state["step"]
        state["step"] += 1
        if step == 0:
            if gate is not None:
                await gate.wait()
            cmd = f"curl -s 'http://target:8000{exploit_path}'"
            return ToolCall(id="c1", name="shell", arguments={"command": cmd})
        if step == 1:
            return ToolCall(id="c2", name="submit", arguments={"flag": flag})
        return Finish(content="done")

    return policy


@gated
def test_a_rollout_solves_a_real_cyber_episode(tmp_path: Path) -> None:
    snap = _admit_cmdi()
    exploit_path, _ = cmdi_exploit_and_benign(snap.graph)
    flag = str(snap.graph.nodes["secret_flag"].attrs["value_ref"])
    factory = make_environment_factory(
        WebappPack(),
        [snap],
        tmp_path / "envs",
        tools=[shell, submit],
        backing=Backing.CONTAINER,
        sandbox=True,
    )
    rollout = AsyncRollout(
        factory,
        policy=_solver(exploit_path, flag),
        snapshot_id=snap.snapshot_id,
        task_id=_pentest_task_id(snap),
    )
    [trajectory] = asyncio.run(batch([rollout], concurrency=1))
    assert trajectory.reward.scalar == 1.0
    assert trajectory.success
    assert trajectory.snapshot_id == snap.snapshot_id


@gated
def test_batch_runs_episodes_concurrently(tmp_path: Path) -> None:
    n = 3
    snap = _admit_cmdi()
    exploit_path, _ = cmdi_exploit_and_benign(snap.graph)
    flag = str(snap.graph.nodes["secret_flag"].attrs["value_ref"])
    task_id = _pentest_task_id(snap)
    factory = make_environment_factory(
        WebappPack(),
        [snap],
        tmp_path / "envs",
        tools=[shell, submit],
        backing=Backing.CONTAINER,
        sandbox=True,
    )

    async def go() -> list[Trajectory]:
        gate = asyncio.Barrier(n)  # releases only when all n rollouts arrive at once
        rollouts = [
            AsyncRollout(
                factory,
                policy=_solver(exploit_path, flag, gate=gate),
                snapshot_id=snap.snapshot_id,
                task_id=task_id,
            )
            for _ in range(n)
        ]
        # If batch() serialized, the first rollout blocks at the barrier forever and
        # this times out; overlap is what lets all n arrive and release.
        return await asyncio.wait_for(batch(rollouts, concurrency=n), timeout=240)

    trajectories = asyncio.run(go())
    assert len(trajectories) == n
    assert all(t.reward.scalar == 1.0 for t in trajectories)


@gated
def test_a_rollout_that_does_nothing_scores_zero(tmp_path: Path) -> None:
    snap = _admit_cmdi()
    factory = make_environment_factory(
        WebappPack(),
        [snap],
        tmp_path / "envs",
        tools=[shell, submit],
        backing=Backing.CONTAINER,
        sandbox=True,
    )

    async def idle(
        messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> Finish:
        return Finish(content="")

    rollout = AsyncRollout(
        factory,
        policy=idle,
        snapshot_id=snap.snapshot_id,
        task_id=_pentest_task_id(snap),
    )
    [trajectory] = asyncio.run(batch([rollout], concurrency=1))
    assert trajectory.reward.scalar == 0.0
    assert not trajectory.success


def _agent_resource_count() -> tuple[int, int]:
    # The leak-prone per-episode resources: sandbox containers + their --internal nets.
    containers = subprocess.run(
        ["docker", "ps", "-aq", "--filter", "name=openrange-agent"],
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.split()
    networks = subprocess.run(
        ["docker", "network", "ls", "-q", "--filter", "name=openrange-agent-net"],
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.split()
    return len(containers), len(networks)


@gated
def test_batch_cleans_up_when_a_rollout_fails(tmp_path: Path) -> None:
    # A mid-batch failure must leak nothing: a sibling raises after both worlds and
    # sandboxes are up, yet every container and per-episode network is torn down (and
    # the error surfaces). Guards the failure path the happy-path tests can't.
    snap = _admit_cmdi()
    exploit_path, _ = cmdi_exploit_and_benign(snap.graph)
    flag = str(snap.graph.nodes["secret_flag"].attrs["value_ref"])
    task_id = _pentest_task_id(snap)
    factory = make_environment_factory(
        WebappPack(),
        [snap],
        tmp_path / "envs",
        tools=[shell, submit],
        backing=Backing.CONTAINER,
        sandbox=True,
    )

    async def boom(
        messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> Finish:
        raise RuntimeError("model server down")

    before = _agent_resource_count()
    rollouts = [
        AsyncRollout(
            factory,
            policy=_solver(exploit_path, flag),
            snapshot_id=snap.snapshot_id,
            task_id=task_id,
        ),
        AsyncRollout(
            factory, policy=boom, snapshot_id=snap.snapshot_id, task_id=task_id
        ),
    ]
    with pytest.raises(RuntimeError, match="model server down"):
        asyncio.run(batch(rollouts, concurrency=2))
    assert _agent_resource_count() == before  # no leaked container or network
