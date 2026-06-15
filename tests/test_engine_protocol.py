"""Pure chat/completions request/response shaping — every branch, with real dicts."""

from __future__ import annotations

from typing import Any

from openrange_trl.engine import Finish, ToolCall
from openrange_trl.engine.protocol import build_chat_request, parse_chat_response


def test_build_chat_request_carries_model_messages_and_tools() -> None:
    msgs = [{"role": "user", "content": "go"}]
    tools = [{"type": "function", "function": {"name": "shell"}}]
    assert build_chat_request("m", msgs, tools) == {
        "model": "m",
        "messages": msgs,
        "tools": tools,
    }


def _tool_message(name: str, arguments: str | None) -> dict[str, Any]:
    fn: dict[str, Any] = {"name": name}
    if arguments is not None:
        fn["arguments"] = arguments
    return {"choices": [{"message": {"tool_calls": [{"id": "c1", "function": fn}]}}]}


def test_parse_returns_a_toolcall_with_parsed_arguments() -> None:
    action = parse_chat_response(_tool_message("shell", '{"command": "ls"}'))
    assert action == ToolCall(id="c1", name="shell", arguments={"command": "ls"})


def test_parse_toolcall_with_empty_arguments_is_an_empty_dict() -> None:
    action = parse_chat_response(_tool_message("submit", ""))
    assert isinstance(action, ToolCall)
    assert action.arguments == {}


def test_parse_toolcall_with_malformed_arguments_degrades_to_empty() -> None:
    # A model emitting non-JSON args shouldn't crash the rollout — the tool fails soft.
    action = parse_chat_response(_tool_message("submit", "{not json"))
    assert isinstance(action, ToolCall)
    assert action.arguments == {}


def test_parse_returns_finish_when_no_tool_call() -> None:
    payload = {"choices": [{"message": {"content": "the answer"}}]}
    assert parse_chat_response(payload) == Finish(content="the answer")


def test_parse_returns_empty_finish_when_content_is_null() -> None:
    payload = {"choices": [{"message": {"content": None}}]}
    assert parse_chat_response(payload) == Finish(content="")
