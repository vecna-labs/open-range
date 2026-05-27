"""Canonical clean handler implementations per service kind.

Used at admission time to validate that a kind's contract is well-posed: the
reference must pass its own contract. Never shown to the agent.
"""

from __future__ import annotations

_API_LIST_REFERENCE = """def handle(query, state):
    import json
    del query
    records = state.get("records", {})
    if not isinstance(records, dict):
        records = {}
    items = [{"id": key, **value} for key, value in records.items()]
    body = json.dumps({"items": items}).encode("utf-8")
    return 200, {"Content-Type": "application/json"}, body
"""


def api_list_reference() -> str:
    return _API_LIST_REFERENCE
