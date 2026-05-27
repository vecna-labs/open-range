"""Behavioral contracts for the build-family grader.

A contract is a tuple of ``TestCase``s. Each case pins down a ``(query, state)``
input and a predicate over the handler's ``(status, headers, body)`` return.

Contracts are keyed by service kind: an ``api`` endpoint has a different
behavioral spec than an ``auth`` or ``web`` endpoint. The family uses the
contract both at admission time (to validate the task is well-posed: clean
reference passes, mutation breaks) and at success time (to grade the agent's
submitted handler source).
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

Predicate = Callable[[int, Mapping[str, str], bytes], tuple[bool, str]]


@dataclass(frozen=True, slots=True)
class ContractCase:
    description: str
    query: Mapping[str, str]
    state: Mapping[str, Any]
    predicate: Predicate


def _content_type_is_json(headers: Mapping[str, str]) -> bool:
    for key, value in headers.items():
        if key.lower() == "content-type" and "application/json" in value.lower():
            return True
    return False


def _parse_json_body(body: bytes) -> object | None:
    try:
        parsed: object = json.loads(body.decode("utf-8"))
    except json.JSONDecodeError, UnicodeDecodeError:
        return None
    return parsed


def _check_items_list(
    status: int,
    headers: Mapping[str, str],
    body: bytes,
    expected_ids: frozenset[str],
) -> tuple[bool, str]:
    if status != 200:
        return False, f"status {status}, expected 200"
    if not _content_type_is_json(headers):
        return False, "Content-Type is not application/json"
    parsed = _parse_json_body(body)
    if not isinstance(parsed, dict) or "items" not in parsed:
        return False, f"body missing 'items' field: {body[:120]!r}"
    items = parsed["items"]
    if not isinstance(items, list):
        return False, f"'items' is {type(items).__name__}, expected list"
    got_ids: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            return False, f"item is {type(item).__name__}, expected mapping"
        item_id = item.get("id")
        if not isinstance(item_id, str):
            return False, f"item missing string 'id': {item!r}"
        got_ids.add(item_id)
    if got_ids != set(expected_ids):
        return (
            False,
            f"item ids {sorted(got_ids)}, expected {sorted(expected_ids)}",
        )
    return True, ""


def api_list_contract() -> tuple[ContractCase, ...]:
    """Spec for an api-kind list endpoint: GET → 200 JSON with every record
    surfaced under top-level field ``items``."""
    return (
        ContractCase(
            description="empty records returns empty items list",
            query={},
            state={"records": {}},
            predicate=lambda s, h, b: _check_items_list(s, h, b, frozenset()),
        ),
        ContractCase(
            description="single record appears under its id",
            query={},
            state={"records": {"alpha": {"name": "Alpha"}}},
            predicate=lambda s, h, b: _check_items_list(s, h, b, frozenset({"alpha"})),
        ),
        ContractCase(
            description="multiple records all appear",
            query={},
            state={
                "records": {
                    "a": {"v": "1"},
                    "b": {"v": "2"},
                    "c": {"v": "3"},
                },
            },
            predicate=lambda s, h, b: _check_items_list(
                s, h, b, frozenset({"a", "b", "c"})
            ),
        ),
    )
