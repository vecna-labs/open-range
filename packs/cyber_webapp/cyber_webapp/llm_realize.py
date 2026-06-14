"""Ask an LLM to realize a vuln handler, per class (DESIGN.md §9, #260).

Procedural architects the world and places the flag; the LLM only fills in ONE node's
handler with a varied-but-valid implementation. The realization has to be vulnerable to
the SAME sampled injection context the reference solver will exploit, so each prompt
reads that context off the vuln's params. This module owns only the *request* — the host
runs the LLM and then admits the result dynamically (`realize_admit.classify_admission`
with `reference_solver.exploit_and_benign`), since running an episode is a host concern.
"""

from __future__ import annotations

from collections.abc import Mapping

from graphschema import WorldGraph
from openrange_pack_sdk import LLMRequest, PackError

from cyber_webapp.reference_solver import _flag_record_key, _vuln_of_kind

# The classes a prompt exists for. command_injection is the first realized class (#266);
# sql_injection and path_traversal extend it across the response-leak and file-read
# families. The rest follow the same shape.
REALIZABLE_KINDS = ("command_injection", "sql_injection", "path_traversal")

_HANDLER_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["handler"],
    "properties": {"handler": {"type": "string"}},
}

_SYSTEM = (
    "You author one level for an authorized, local security-training gym "
    "(standard CTF authoring / defensive security education)."
)

_RETURN = (
    "- Return (status:int, headers:dict, body:bytes). Standard library only.\n"
    'Return JSON: {"handler": "<the full def handle source>"}.'
)


def _cmdi_prompt(
    param: str, flag_path: str, ctx: str, params: Mapping[str, object]
) -> str:
    if ctx == "substitution":
        technique = (
            "The handler strips ';', '|' and '&' separators, so it must stay "
            f"exploitable via command substitution, e.g. '$(cat {flag_path})'."
        )
    elif ctx == "quoted":
        q = str(params.get("quote", "'"))
        technique = (
            f"The input lands inside a {q}-quoted shell argument; it must be "
            f"exploitable by breaking out: {q}; cat {flag_path}; echo {q}."
        )
    else:
        technique = (
            "It must be exploitable by chaining a command after a ';' "
            f"separator, e.g. '127.0.0.1; cat {flag_path}'."
        )
    return (
        "Write a Python `def handle(query, state):` for an OS command-injection "
        "level.\n"
        "- query is dict[str, list[str]] (a parsed query string).\n"
        "- state['files'] maps file paths to contents; the flag is the file at "
        f"{flag_path!r}.\n"
        f"- Read user input from query[{param!r}] and run it as a vulnerable "
        "shell call would.\n"
        f"- {technique}\n"
        "- A benign value (e.g. '127.0.0.1') must NOT return the flag.\n" + _RETURN
    )


def _sqli_prompt(param: str, table: str, leak_column: str, ctx: str) -> str:
    union = f"UNION SELECT key, {leak_column} FROM {table} -- "
    if ctx == "numeric":
        wrap, example = "an unquoted", f"0 {union}"
    elif ctx == "double":
        wrap, example = 'a double-quoted (")', f'" {union}'
    else:
        wrap, example = "a single-quoted (')", f"' {union}"
    return (
        "Write a Python `def handle(query, state):` for a SQL-injection lookup "
        "level.\n"
        "- query is dict[str, list[str]].\n"
        "- state['db'] is a sqlite3 connection whose rows read as dicts; "
        "state['schema']['key_column'] is the key column.\n"
        f"- The records table is {table!r} with the key column and a "
        f"{leak_column!r} column; the flag is the value in that column for one row.\n"
        f"- Concatenate query[{param!r}] into the WHERE clause with NO "
        f"parameterization, in {wrap} string context, so a UNION payload "
        f"surfaces every row, e.g. {example}\n"
        "- A benign id must return only its own (non-flag) row.\n" + _RETURN
    )


def _path_prompt(param: str, flag_path: str, base_dir: str, ctx: str) -> str:
    if ctx == "relative":
        technique = (
            "Re-anchor an absolute path under base_dir, but let a relative '../' "
            f"chain climb out, so enough '../' segments reach {flag_path!r}."
        )
    elif ctx == "dotdot_filter":
        technique = (
            "Strip '../' exactly once (so '....//' collapses back to '../' and "
            "survives) and re-anchor absolute paths; only the '....//' form escapes."
        )
    else:
        technique = (
            "Strip relative '../' chains but honor an absolute path "
            f"(posixpath.join drops the base), so an absolute {flag_path!r} leaks."
        )
    return (
        "Write a Python `def handle(query, state):` for a path-traversal "
        "file-serving level.\n"
        "- query is dict[str, list[str]].\n"
        "- state['files'] maps absolute file paths to contents; the flag is at "
        f"{flag_path!r}.\n"
        f"- Join query[{param!r}] onto base_dir {base_dir!r} without proper "
        "confinement.\n"
        f"- {technique}\n"
        "- A benign filename must NOT return the flag.\n" + _RETURN
    )


def realization_request(graph: WorldGraph, kind: str) -> LLMRequest:
    """The LLM request to realize `kind`'s handler, tailored to its sampled context.

    Raises if `kind` has no prompt yet (see `REALIZABLE_KINDS`). The host runs this
    against an `LLMBackend` and passes the returned handler through the admission gate.
    """
    vuln = _vuln_of_kind(graph, kind)
    params = vuln.attrs["params"]
    if not isinstance(params, Mapping):
        raise PackError(f"{kind} vuln has no params mapping")
    param = str(params["target_param"])
    if kind == "command_injection":
        ctx = str(params.get("inj_context", "separator"))
        prompt = _cmdi_prompt(param, _flag_record_key(graph), ctx, params)
    elif kind == "sql_injection":
        ctx = str(params.get("context", "single"))
        prompt = _sqli_prompt(
            param, str(params["table"]), str(params["leak_column"]), ctx
        )
    elif kind == "path_traversal":
        ctx = str(params.get("confinement", "absolute_only"))
        prompt = _path_prompt(
            param, _flag_record_key(graph), str(params["base_dir"]), ctx
        )
    else:
        raise PackError(f"no LLM realization prompt for kind {kind!r}")
    return LLMRequest(prompt=prompt, system=_SYSTEM, json_schema=_HANDLER_SCHEMA)


def handler_from_result(parsed_json: Mapping[str, object] | None) -> str:
    """The handler source out of an LLM result's parsed JSON, or '' if absent."""
    handler = (parsed_json or {}).get("handler")
    return handler if isinstance(handler, str) else ""
