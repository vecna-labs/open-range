"""``run_react`` + ``AsyncRollout`` guards over a real PROCESS world (no docker).

A PROCESS cyber world boots an in-process HTTP server (no docker), so the loop and the
rollout guards are driven over a real ``EpisodeEnv`` without a sandbox or a model.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
from cyber_webapp import WebappPack
from openrange_pack_sdk import Snapshot
from openrange_trl import EpisodeEnv, make_environment_factory
from openrange_trl.engine import AsyncRollout, Finish, ToolCall
from openrange_trl.engine.async_utils import BlockingPool
from openrange_trl.engine.react import run_react

from openrange.core.admit import admit
from openrange.core.episode import EpisodeService

_MANIFEST = {
    "pack": {"id": "webapp"},
    "runtime": {"tick": {"mode": "off"}},
    "npc": [],
    "seed": 0,
    "loot_shapes": {"db": 1, "file": 0},
    "vuln_kinds": {"sql_injection": 1},
}


def note(surface: Mapping[str, Any], text: str = "ok") -> str:
    """Record a note.

    Args:
        text: the note text.
    """
    return f"noted: {text}"


def _snapshot() -> Snapshot:
    snap = admit(WebappPack(), manifest=_MANIFEST)
    assert isinstance(snap, Snapshot), snap
    return snap


def _pentest_task_id(snap: Snapshot) -> str:
    return next(t.id for t in snap.tasks if t.meta.get("family") == "webapp.pentest")


def test_run_react_runs_to_max_iters_and_reports_unknown_tools(tmp_path: Path) -> None:
    # A policy that keeps calling tools and never finishes runs the loop to max_iters;
    # an unknown tool name fails soft instead of crashing the rollout.
    snap = _snapshot()
    service = EpisodeService(WebappPack(), tmp_path / "svc")  # PROCESS backing
    env = EpisodeEnv(service=service, snapshots={snap.snapshot_id: snap}, tools=[note])
    pool = BlockingPool(1)
    calls = {"i": 0}

    async def policy(
        messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> ToolCall:
        calls["i"] += 1
        if calls["i"] == 1:
            return ToolCall(id="c1", name="note", arguments={"text": "hi"})
        return ToolCall(id="c2", name="ghost", arguments={})

    try:
        env.reset(snapshot_id=snap.snapshot_id, task_id=_pentest_task_id(snap))
        messages = asyncio.run(
            run_react(
                env,
                policy=policy,
                tool_schemas=[],
                first_prompt="go",
                max_iters=3,
                pool=pool,
            )
        )
    finally:
        pool.close()
        service.close()

    tool_outputs = [m["content"] for m in messages if m["role"] == "tool"]
    assert len(tool_outputs) == 3  # the loop ran the full max_iters (never finished)
    assert "noted: hi" in tool_outputs[0]
    assert "error: no tool named 'ghost'" in tool_outputs[1]


def test_instruction_falls_back_when_no_task_matches(tmp_path: Path) -> None:
    snap = _snapshot()
    service = EpisodeService(WebappPack(), tmp_path / "svc")
    env = EpisodeEnv(service=service, snapshots={snap.snapshot_id: snap}, tools=[note])

    async def policy(
        messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> Finish:
        return Finish(content="")

    rollout = AsyncRollout(
        lambda: env, policy=policy, snapshot_id=snap.snapshot_id, task_id="no-such-task"
    )
    try:
        assert rollout._instruction(env) == ""  # no task matches → empty instruction
    finally:
        service.close()


def test_run_or_eval_before_init_is_refused(tmp_path: Path) -> None:
    factory = make_environment_factory(
        WebappPack(), [_snapshot()], tmp_path / "envs", tools=[note]
    )

    async def policy(
        messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> Finish:
        return Finish(content="")

    rollout = AsyncRollout(factory, policy=policy)
    with pytest.raises(RuntimeError, match="not initialized"):
        rollout._require_env()


def test_rollout_aclose_is_a_safe_noop_before_init() -> None:
    async def policy(
        messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> Finish:
        return Finish(content="")

    rollout = AsyncRollout(lambda: pytest.fail("factory must not run"), policy=policy)
    pool = BlockingPool(1)
    try:
        asyncio.run(rollout.aclose(pool))  # env is None → tears down nothing, no raise
    finally:
        pool.close()
