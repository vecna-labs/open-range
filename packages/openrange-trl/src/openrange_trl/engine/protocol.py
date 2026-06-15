"""Pure shaping of OpenAI chat/completions requests and responses.

Kept free of I/O so every branch is unit-testable with plain dicts; ``OpenAIBackend``
is only the thin HTTP round-trip around these.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ToolCall:
    """The model asked to call a tool."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True, slots=True)
class Finish:
    """The model returned a final answer (no tool call) — the loop ends."""

    content: str


Action = ToolCall | Finish


def build_chat_request(
    model: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
) -> dict[str, Any]:
    """The /v1/chat/completions request body for one turn."""
    return {"model": model, "messages": messages, "tools": tools}


def parse_chat_response(payload: dict[str, Any]) -> Action:
    """The next :data:`Action` from a chat/completions response — a tool call if the
    model emitted one, else a final answer. Unparseable tool arguments degrade to an
    empty dict (the tool then fails soft) rather than crashing the rollout."""
    message = payload["choices"][0]["message"]
    tool_calls = message.get("tool_calls")
    if tool_calls:
        call = tool_calls[0]
        fn = call["function"]
        raw = fn.get("arguments") or ""
        try:
            arguments = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            arguments = {}
        return ToolCall(id=call.get("id", ""), name=fn["name"], arguments=arguments)
    return Finish(content=message.get("content") or "")
