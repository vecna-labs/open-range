"""The vendored tool-schema builder turns a brought tool into a chat/completions dict.

Pure — no transformers, no HTTP. Pins the position-drop of the injected surface arg,
per-parameter descriptions from the Google docstring, and the type mapping (including
string annotations under ``from __future__ import annotations``).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from openrange_trl.engine import tool_schema
from openrange_trl.engine.schema import _json_type


def recon(ctx: Mapping[str, Any], path: str, depth: int = 1) -> str:
    """Fetch a path on the target.

    Args:
        path: the request path.
        depth: how deep to crawl.
    """
    return ""


def test_tool_schema_drops_the_first_arg_by_position_and_reads_the_docstring() -> None:
    # The first param is named `ctx` (not `surface`) on purpose: the contract is
    # positional, so `ctx` must be dropped and never appear in the schema.
    assert tool_schema(recon) == {
        "type": "function",
        "function": {
            "name": "recon",
            "description": "Fetch a path on the target.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "the request path."},
                    "depth": {"type": "integer", "description": "how deep to crawl."},
                },
                "required": ["path"],  # `depth` has a default → optional
            },
        },
    }


def test_tool_schema_falls_back_with_no_docstring() -> None:
    def bare(surface: Any, value: Any) -> str:
        return ""

    schema = tool_schema(bare)["function"]
    assert schema["description"] == ""
    assert schema["parameters"]["properties"] == {"value": {"type": "string"}}
    assert schema["parameters"]["required"] == ["value"]


def test_json_type_defaults_to_string_for_missing_or_unknown() -> None:
    import inspect

    assert _json_type(inspect.Parameter.empty) == "string"  # no annotation
    assert _json_type(dict) == "string"  # unmapped type


def with_trailing_block(surface: Any, x: str) -> str:
    """Do a thing.

    Args:
        x: the x value.

    Returns:
        nothing useful.
    """
    return ""


def test_arg_parsing_stops_at_the_next_docstring_block() -> None:
    # A non-"name:" line after the Args block (here ``Returns:``) ends arg parsing.
    props = tool_schema(with_trailing_block)["function"]["parameters"]["properties"]
    assert props == {"x": {"type": "string", "description": "the x value."}}
