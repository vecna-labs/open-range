"""Derive a chat/completions tool schema for a brought tool — transformers-free.

The sync TRL path lets TRL reflect the tools via ``transformers.get_json_schema``;
the async engine must stay transformers-free, so it builds the same shape here from
the tool's signature + Google-style docstring. The first parameter (the live surface
``EpisodeEnv`` injects) is dropped by POSITION, matching ``_tool_method``'s ``[1:]`` —
the contract is positional, not "a param literally named ``surface``".
"""

from __future__ import annotations

import inspect
import re
from collections.abc import Callable
from typing import Any

# Annotations may be runtime types or strings (``from __future__ import annotations``),
# so map by the type's name rather than the type object.
_JSON_TYPES = {"str": "string", "int": "integer", "float": "number", "bool": "boolean"}


def tool_schema(fn: Callable[..., str]) -> dict[str, Any]:
    """The OpenAI tool dict for a brought tool ``fn(surface, **params)``."""
    params = list(inspect.signature(fn).parameters.values())[1:]
    descriptions = _arg_descriptions(fn.__doc__ or "")
    properties: dict[str, Any] = {}
    required: list[str] = []
    for p in params:
        prop: dict[str, Any] = {"type": _json_type(p.annotation)}
        if p.name in descriptions:
            prop["description"] = descriptions[p.name]
        properties[p.name] = prop
        if p.default is inspect.Parameter.empty:
            required.append(p.name)
    return {
        "type": "function",
        "function": {
            "name": fn.__name__,
            "description": _summary(fn.__doc__ or ""),
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


def _json_type(annotation: Any) -> str:
    if annotation is inspect.Parameter.empty:
        return "string"
    name = (
        annotation
        if isinstance(annotation, str)
        else getattr(annotation, "__name__", "str")
    )
    return _JSON_TYPES.get(name, "string")


def _summary(doc: str) -> str:
    # The text before the Google-style ``Args:`` block is the tool description.
    head = re.split(r"\n\s*Args:", doc, maxsplit=1)[0]
    return " ".join(head.split())


def _arg_descriptions(doc: str) -> dict[str, str]:
    out: dict[str, str] = {}
    in_args = False
    for line in doc.splitlines():
        if line.strip() == "Args:":
            in_args = True
            continue
        if in_args:
            match = re.match(r"\s+(\w+):\s*(.+)", line)
            if match:
                out[match.group(1)] = match.group(2).strip()
            elif line.strip():
                break
    return out
