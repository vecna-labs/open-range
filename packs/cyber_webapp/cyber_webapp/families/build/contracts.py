"""Behavioral contracts for the build-family grader.

A contract is a tuple of ``TestCase``s. Each case pins down a ``(query, state)``
input and a predicate over the handler's ``(status, headers, body)`` return.

Contracts are keyed by service kind AND difficulty level. The level is the
curriculum knob the build family hardens/softens: an ``api`` endpoint at
level 1 only has to list records; level 2 also requires a top-level
``count``; level 3 also requires the items sorted by id. Each level is a
strict superset of the one below, so the same reference handler that passes
level N passes every level below it. The family uses the contract both at
admission time (clean reference passes, bug-injecting mutation breaks) and
at success time (grade the agent's submitted handler).
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
    except (json.JSONDecodeError, UnicodeDecodeError):
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


API_MAX_LEVEL = 3


def _check_api(
    status: int,
    headers: Mapping[str, str],
    body: bytes,
    expected_ids: frozenset[str],
    level: int,
) -> tuple[bool, str]:
    ok, why = _check_items_list(status, headers, body, expected_ids)
    if not ok:
        return ok, why
    parsed = _parse_json_body(body)
    assert isinstance(parsed, dict)  # _check_items_list already proved this
    items = parsed["items"]
    if level >= 2:
        count = parsed.get("count")
        if count != len(items):
            return False, f"'count' is {count!r}, expected {len(items)}"
    if level >= 3:
        ids = [item["id"] for item in items]
        if ids != sorted(ids):
            return False, f"items not sorted by id ascending: {ids}"
    return True, ""


def api_list_contract(level: int) -> tuple[ContractCase, ...]:
    """Spec for an api-kind list endpoint at ``level``.

    L1: GET → 200 JSON, every record under top-level ``items``.
    L2: also a top-level ``count`` equal to the number of items.
    L3: also ``items`` sorted by ``id`` ascending. The multi-record case
    inserts ids out of order so a handler that skips sorting fails L3.
    """

    def case(description: str, records: dict[str, dict[str, str]]) -> ContractCase:
        expected = frozenset(records)
        return ContractCase(
            description=description,
            query={},
            state={"records": records},
            predicate=lambda s, h, b: _check_api(s, h, b, expected, level),
        )

    return (
        case("empty records returns empty items list", {}),
        case("single record appears under its id", {"alpha": {"name": "Alpha"}}),
        case(
            "multiple records all appear",
            {"c": {"v": "3"}, "a": {"v": "1"}, "b": {"v": "2"}},
        ),
    )
