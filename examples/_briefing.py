"""Render the world's interface contract into an agent prompt.

A harness owns the agent loop, so it must tell its agent which interface the
world presents and where — an agent handed only the instruction (a path, not the
dynamic host:port) can't reach the target. ``agent_briefing`` adds that: the task
plus a one-line statement of the live ``surface`` (HTTP ``base_url`` / file root).
"""

from __future__ import annotations

from openrange.runtime import EpisodeContext


def agent_briefing(ctx: EpisodeContext) -> str:
    """The task plus the live interface contract, for any harness's agent."""
    parts = [ctx.task.instruction]
    base_url = ctx.surface.get("base_url")
    solver_root = ctx.surface.get("solver_root")
    if isinstance(base_url, str):
        parts.append(
            f"The target web service is running at {base_url} — "
            "interact with it over HTTP."
        )
    elif solver_root is not None:
        parts.append(f"You are working in the directory {solver_root}.")
    return "\n\n".join(parts)
