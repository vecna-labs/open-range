"""The example harness briefing renders the live interface contract.

A harness must tell its agent which interface a world presents and where; the
shared `agent_briefing` helper turns the task + surface into that prompt, adapting
to whatever the world declares (HTTP target, file workspace, ...).
"""

from __future__ import annotations

from pathlib import Path
from urllib.request import urlopen

from openrange_pack_sdk import Snapshot, TaskSpec

from examples._briefing import agent_briefing
from openrange.core.episode import AgentTurn
from openrange.runtime import EpisodeContext, OpenRangeRun, RunConfig


def _task() -> TaskSpec:
    return TaskSpec(
        id="t0",
        instruction="Recover the hidden admin flag via GET /svc/orders-db/backup.",
        entrypoints=("ep.backup",),
        goal_nodes=("secret.flag",),
        feasibility_check="webapp.pentest",
        success_check="webapp.pentest",
    )


def test_briefing_gives_the_agent_the_http_target() -> None:
    ctx = EpisodeContext(
        task=_task(),
        surface={"base_url": "http://127.0.0.1:51991", "solver_root": "/tmp/ep0"},
    )
    briefing = agent_briefing(ctx)
    assert _task().instruction in briefing
    assert "http://127.0.0.1:51991" in briefing
    assert "over HTTP" in briefing
    # An HTTP world points the agent at the URL, not a directory to edit.
    assert "working in the directory" not in briefing


def test_briefing_gives_a_file_world_its_workspace() -> None:
    ctx = EpisodeContext(
        task=_task(),
        surface={"solver_root": "/tmp/ws"},
    )
    briefing = agent_briefing(ctx)
    assert _task().instruction in briefing
    assert "/tmp/ws" in briefing
    assert "base_url" not in briefing and "over HTTP" not in briefing


def test_briefing_is_just_the_instruction_when_surface_is_opaque() -> None:
    # A world that declares neither a URL nor a workspace (e.g. a pure MCP/sim
    # surface) still briefs the task; the harness binds the rest itself.
    ctx = EpisodeContext(task=_task(), surface={"mcp_endpoint": "stdio://x"})
    assert agent_briefing(ctx) == _task().instruction


def test_briefing_delivers_the_live_target_through_run_episode(tmp_path: Path) -> None:
    # End-to-end via the real harness seam the example evals use: run_episode boots
    # a cyber world, the briefing carries the LIVE base_url the static instruction
    # can't, and a solver given ONLY the briefing reaches the running server.
    run = OpenRangeRun(RunConfig(tmp_path / "run", dashboard=False))
    snapshot = run.build(
        {
            "pack": {"id": "webapp"},
            "runtime": {"tick": {"mode": "off"}},
            "npc": [],
            "seed": 0,
        }
    )
    assert isinstance(snapshot, Snapshot), snapshot
    task = next(t for t in snapshot.tasks if t.meta.get("family") == "webapp.pentest")
    seen: dict[str, bool] = {}

    def solve(ctx: EpisodeContext) -> AgentTurn:
        brief = agent_briefing(ctx)
        seen["url_in_brief"] = ctx.base_url in brief
        with urlopen(ctx.base_url + "/", timeout=10) as resp:  # reach via the brief
            seen["reached"] = resp.status == 200
        return AgentTurn(message="probed")

    run.run_episode(snapshot, solve, task_id=task.id)
    assert seen["url_in_brief"]  # the live, dynamic base_url made it into the brief
    assert seen["reached"]  # a solver reached the server with only the brief
