"""The example harness briefing renders the live interface contract.

A harness must tell its agent which interface a world presents and where; the
shared `agent_briefing` helper turns the task + surface into that prompt, adapting
to whatever the world declares (HTTP target, file workspace, ...).
"""

from __future__ import annotations

from openrange_pack_sdk import TaskSpec

from examples._briefing import agent_briefing
from openrange.runtime import EpisodeContext


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
        surface={
            "base_url": "http://127.0.0.1:51991",
            "solver_root": "/tmp/ep0",
            "http_get": lambda p: b"",
        },
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
        surface={"solver_root": "/tmp/ws", "run_tests": lambda t: {}},
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
