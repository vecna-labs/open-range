"""Canonical clean handler implementations per service kind.

Used at admission time to validate that a kind's contract is well-posed: the
reference must pass its own contract. Never shown to the agent.
"""

from __future__ import annotations


def api_list_reference(level: int) -> str:
    """A handler that satisfies the api-list contract at ``level``.

    Built cumulatively so it passes every level at or below ``level``: L2
    adds the ``count`` field, L3 sorts the items by id.
    """
    # Field names are double-quoted to match the admission mutation
    # (api_wrong_field_name renames the literal "items").
    lines = [
        "def handle(query, state):",
        "    import json",
        "    del query",
        '    records = state.get("records", {})',
        "    if not isinstance(records, dict):",
        "        records = {}",
        '    items = [{"id": key, **value} for key, value in records.items()]',
    ]
    if level >= 3:
        lines.append('    items.sort(key=lambda item: item["id"])')
    lines.append('    payload = {"items": items}')
    if level >= 2:
        lines.append('    payload["count"] = len(items)')
    lines.append("    body = json.dumps(payload).encode('utf-8')")
    lines.append('    return 200, {"Content-Type": "application/json"}, body')
    return "\n".join(lines) + "\n"
