"""The domain-agnostic agent loop: model → tool call → observation → … until done.

Tools are dispatched through ``getattr(env, name)`` — the exact brought-tool methods
``EpisodeEnv`` reflects to TRL — so the sandbox seam, fail-soft handling, and turn
recording all carry over. The tool call runs on the blocking pool (``docker exec``),
never the event loop.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any

from openrange_trl import EpisodeEnv
from openrange_trl.engine.async_utils import BlockingPool
from openrange_trl.engine.protocol import Action, Finish

Policy = Callable[[list[dict[str, Any]], list[dict[str, Any]]], Awaitable[Action]]


async def run_react(
    env: EpisodeEnv,
    *,
    policy: Policy,
    tool_schemas: list[dict[str, Any]],
    first_prompt: str,
    max_iters: int,
    pool: BlockingPool,
) -> list[dict[str, Any]]:
    """Drive ``env`` with ``policy`` until it finishes (or ``max_iters``); return the
    message log. Tool effects + turn recording happen on ``env`` itself."""
    messages: list[dict[str, Any]] = [{"role": "user", "content": first_prompt}]
    for _ in range(max_iters):
        action = await policy(messages, tool_schemas)
        if isinstance(action, Finish):
            messages.append({"role": "assistant", "content": action.content})
            break
        messages.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": action.id,
                        "type": "function",
                        "function": {
                            "name": action.name,
                            "arguments": json.dumps(action.arguments),
                        },
                    }
                ],
            }
        )
        output = await pool.run(_call_tool, env, action.name, action.arguments)
        messages.append({"role": "tool", "tool_call_id": action.id, "content": output})
    return messages


def _call_tool(env: EpisodeEnv, name: str, arguments: dict[str, Any]) -> str:
    method = getattr(env, name, None)
    if method is None:
        return f"error: no tool named {name!r}"
    return str(method(**arguments))
