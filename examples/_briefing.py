"""Render the world's interface contract into an agent prompt.

OpenRange exposes a world through its pack-defined ``surface`` (an HTTP
``base_url``, a file ``solver_root``, an MCP endpoint, ...) plus the ``task``.
A harness owns the agent loop, so it must tell its agent *which* interface this
world presents and *where* — otherwise an agent handed only the instruction
(which names a path like ``GET /svc/orders-db/backup``, not the dynamic
host:port) can't reach the target.

``agent_briefing`` is that glue: the task instruction plus a one-line statement
of the live interface, adapted to whatever the world declares. It reads the
surface with ``.get`` (not the raising ``EpisodeContext`` accessors) so the same
helper works for an HTTP world, a file world, or both.
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
