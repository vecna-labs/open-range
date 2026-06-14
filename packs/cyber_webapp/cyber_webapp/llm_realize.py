"""Ask an LLM to realize a vuln handler, per class (DESIGN.md §9, #260).

Procedural architects the world and places the flag; the LLM only fills in ONE node's
handler with a varied-but-valid implementation. The realization has to be vulnerable to
the SAME sampled injection context the reference solver will exploit, so each prompt
reads that context off the vuln's params. This module owns only the *request* — the host
runs the LLM and then admits the result dynamically (`realize_admit.classify_admission`
with `reference_solver.exploit_and_benign`), since running an episode is a host concern.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

from graphschema import WorldGraph
from openrange_pack_sdk import LLMRequest, PackError, Snapshot

from cyber_webapp.realize_admit import classify_admission
from cyber_webapp.reference_solver import _flag_record_key, _vuln_of_kind

# The classes a prompt exists for. command_injection is the first realized class (#266);
# sql_injection and path_traversal extend it across the response-leak and file-read
# families. The rest follow the same shape.
REALIZABLE_KINDS = (
    "command_injection",
    "sql_injection",
    "path_traversal",
    "xxe",
    "ssti",
    "idor",
    "broken_authz",
    "weak_credentials",
    "ssrf",
)

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


def _xxe_prompt(
    param: str, flag_path: str, ctx: str, params: Mapping[str, object]
) -> str:
    if ctx == "wrapped_root":
        root = str(params["root_element"])
        technique = (
            "Only text inside the sampled child element is reflected, so the "
            f"entity must be nested as <{root}>&e;</{root}> under any root; "
            "text elsewhere is ignored."
        )
        payload = (
            f'<!DOCTYPE w [<!ENTITY e SYSTEM "file://{flag_path}">]>'
            f"<w><{root}>&e;</{root}></w>"
        )
    elif ctx == "scheme_prefix":
        scheme = str(params["uri_scheme"])
        technique = (
            f"The resolver honors only a SYSTEM id carrying the {scheme!r} "
            "scheme, so the SYSTEM literal must use that wrapper instead of "
            "'file://'; a 'file://' id resolves to empty."
        )
        payload = f'<!DOCTYPE r [<!ENTITY e SYSTEM "{scheme}{flag_path}">]><r>&e;</r>'
    else:
        technique = (
            "The entity is reflected from the root element's direct text, so a "
            "classic 'file://' SYSTEM entity referenced in the root leaks, e.g. "
            "<r>&e;</r>."
        )
        payload = f'<!DOCTYPE r [<!ENTITY e SYSTEM "file://{flag_path}">]><r>&e;</r>'
    return (
        "Write a Python `def handle(query, state):` for an XXE (XML external "
        "entity) level.\n"
        "- query is dict[str, list[str]] (a parsed query string).\n"
        "- state['files'] maps absolute file paths to contents; the flag is the "
        f"file at {flag_path!r}.\n"
        f"- Parse query[{param!r}] with a real SAX parser "
        "(xml.sax) with external general entities enabled "
        "(feature_external_ges), resolving a SYSTEM entity against "
        "state['files'] and reflecting parsed element text.\n"
        f"- {technique}\n"
        f"- It must be exploitable by exactly this document: {payload}\n"
        "- A benign document with no entity (e.g. <r>ping</r>) must NOT return "
        "the flag.\n" + _RETURN
    )


def _ssti_prompt(param: str, flag_path: str, ctx: str) -> str:
    access = "config[" + repr(flag_path) + "]"
    if ctx == "comment":
        technique = (
            "Wrap the input inside a Jinja comment, source = '{# ' + inp + "
            "' #}', which the engine discards; so the payload must CLOSE the "
            f"comment to reach print context, e.g. #}}{{{{ {access} }}}}{{#."
        )
    elif ctx == "expr":
        technique = (
            "Splice the input INTO an existing print expression around a string "
            "literal, source = \"{{ '[' ~ \" + inp + \" ~ ']' }}\"; it is "
            "already between {{ and }}, so only a bare expression joined with ~ "
            f"works, e.g. {access}."
        )
    else:
        technique = (
            "Splice the input as the value of a quoted data field, source = "
            "'{\"v\": \"' + inp + '\"}', then render the whole string; the "
            "field value is template DATA, so a print TAG fires in place, e.g. "
            f"{{{{ {access} }}}}."
        )
    return (
        "Write a Python `def handle(query, state):` for a server-side template "
        "injection level.\n"
        "- query is dict[str, list[str]].\n"
        "- state['files'] maps file paths to contents and is passed to the "
        f"template render as `config`; the flag is the file at {flag_path!r}.\n"
        f"- Read user input from query[{param!r}] and render it with a "
        "jinja2.sandbox.SandboxedEnvironment (import it INSIDE def handle).\n"
        f"- {technique}\n"
        "- A benign value (e.g. 'status') must NOT return the flag.\n" + _RETURN
    )


def _idor_prompt(param: str, flag_key: str, ctx: str, ref_prefix: str) -> str:
    if ctx == "base64":
        technique = (
            "The reference is base64-decoded into the db key, so the exploit "
            f"sends base64 of {flag_key!r} to reach the flag record."
        )
    elif ctx == "prefixed":
        technique = (
            f"The reference is a {ref_prefix!r}-prefixed key (the prefix is "
            f"stripped to get the db key), so {ref_prefix + flag_key!r} reaches "
            "the flag record."
        )
    else:
        technique = (
            "The reference is the raw db key, so sending the flag record's key "
            f"{flag_key!r} directly reaches it."
        )
    return (
        "Write a Python `def handle(query, state):` for an insecure direct "
        "object reference (IDOR) lookup level.\n"
        "- query is dict[str, list[str]].\n"
        "- state['db'] is a sqlite3 connection whose rows read as dicts; "
        "state['schema'] has 'table', 'key_column' and 'value_column'.\n"
        f"- Look up the record by the key column using query[{param!r}]; the "
        f"flag is the value column of the record whose key is {flag_key!r}.\n"
        "- Decode the client reference into the db key, then run a parametrized "
        "lookup with NO ownership/authorization check, so any referenced "
        "record (including the flag's) is returned.\n"
        f"- {technique}\n"
        "- A benign reference to a non-existent record must NOT return the "
        "flag.\n" + _RETURN
    )


def _broken_authz_prompt(
    header: str, expected: str, leak_field: str, ctx: str, params: Mapping[str, object]
) -> str:
    pool = list(params.get("confirm_pool", []))
    confirm = str(params.get("confirm_param", "X-Confirm"))
    confirm_value = str(params.get("confirm_value", "1"))
    if ctx == "encoded_token":
        technique = (
            f"Grant only when query[{header!r}] equals the HEX encoding of "
            f"{expected!r} (i.e. {expected!r}.encode().hex()) AND none of the "
            f"confirm params {pool!r} is present."
        )
    elif ctx == "dual_factor":
        technique = (
            f"Grant only when query[{header!r}] equals {expected!r} AND "
            f"query[{confirm!r}] equals {confirm_value!r}; both are trusted "
            "client-supplied values with no real provenance check."
        )
    else:  # single_token
        technique = (
            f"Grant only when query[{header!r}] equals {expected!r} AND none of "
            f"the confirm params {pool!r} is present."
        )
    return (
        "Write a Python `def handle(query, state):` for a broken-authorization "
        "level.\n"
        "- query is dict[str, list[str]] (a parsed query string); request headers "
        "are NOT available, so the trusted role is a query parameter.\n"
        "- state['secrets'] maps field names to values; the flag is the secret at "
        f"state['secrets'][{leak_field!r}].\n"
        f"- Trust the client-supplied role in query[{header!r}] with no real check. "
        f"{technique}\n"
        f"- On grant, return 200 with JSON {{{leak_field!r}: <that secret>}}; "
        "otherwise return 403 and do NOT include the secret.\n"
        "- A request with no role (a benign request) must return 403, not the "
        "flag.\n" + _RETURN
    )


def _weak_credentials_prompt(
    user_param: str, password_param: str, flag_key: str, ctx: str
) -> str:
    if ctx == "combined":
        carrier = (
            f'Read one param query[{password_param!r}] carrying "user:pass" '
            "colon-joined in clear; split on the first ':' into user and "
            "password."
        )
    elif ctx == "basic":
        carrier = (
            f"Read one param query[{password_param!r}] carrying "
            'base64("user:pass") (HTTP Basic style); base64-decode it, then '
            "split on the first ':' into user and password."
        )
    else:  # pair
        carrier = (
            f"Read the username from query[{user_param!r}] and the password "
            f"from query[{password_param!r}] as two separate params."
        )
    return (
        "Write a Python `def handle(query, state):` for a weak/default "
        "credentials login level.\n"
        "- query is dict[str, list[str]] (a parsed query string).\n"
        "- state['secrets'] is a dict; the flag is state['secrets']['flag'].\n"
        f"- {carrier}\n"
        "- A default credential pair is wrongly accepted: when the submitted "
        "user and password equal the known-weak pair, return the flag (e.g. as "
        'JSON {"token": state["secrets"]["flag"]}).\n'
        "- A request with no/blank or wrong credentials must NOT return the "
        "flag (return 401).\n" + _RETURN
    )


def _ssrf_prompt(param: str, ctx: str, params: Mapping[str, object]) -> str:
    host = str(params["internal_host"])
    if ctx == "scheme_block":
        technique = (
            "Refuse http/https outright and every scheme but gopher, so ONLY a "
            f"non-web URL reaches it, e.g. 'gopher://{host}/_admin'."
        )
    elif ctx == "host_allowlist":
        allowed = str(params["allowed_host"])
        technique = (
            "Require http(s) AND that the raw URL string contain the allowed "
            f"host {allowed!r}, but resolve to the URL's real hostname, so a "
            f"credential trick reaches it: 'http://{allowed}@{host}/'."
        )
    else:  # decimal_ip
        dec = str(params.get("internal_decimal", ""))
        technique = (
            "Require http(s) and match the destination host against the bare "
            f"decimal-integer form of the internal IPv4, so 'http://{dec}/' "
            "reaches it while the dotted-quad / hostname forms do not."
        )
    return (
        "Write a Python `def handle(query, state):` for a server-side request "
        "forgery (SSRF) level.\n"
        "- query is dict[str, list[str]].\n"
        f"- Treat query[{param!r}][0] as a URL fetched server-side; parse it "
        "with urllib.parse.urlparse (import INSIDE the function).\n"
        f"- The internal target host is {host!r}; only a URL that PASSES the "
        "filter AND resolves to that host reaches the secret. On reach, return "
        'state["secrets"]["flag"] as JSON {"data": <flag>} — that flag is '
        "the value this level hides.\n"
        f"- {technique}\n"
        "- A benign URL (e.g. 'http://example.com/') must NOT return the flag.\n"
        + _RETURN
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
    # Most classes inject through one query param; broken_authz/weak_credentials carry
    # their own param names instead, so this stays optional.
    param = str(params.get("target_param", ""))
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
    elif kind == "xxe":
        ctx = str(params.get("entity_context", "element_content"))
        prompt = _xxe_prompt(param, _flag_record_key(graph), ctx, params)
    elif kind == "ssti":
        ctx = str(params.get("render_sink", "attribute"))
        prompt = _ssti_prompt(param, _flag_record_key(graph), ctx)
    elif kind == "idor":
        ctx = str(params.get("ref_context", "direct"))
        prompt = _idor_prompt(
            param, _flag_record_key(graph), ctx, str(params.get("ref_prefix", ""))
        )
    elif kind == "broken_authz":
        ctx = str(params.get("trust_context", "single_token"))
        prompt = _broken_authz_prompt(
            str(params["trust_header"]),
            str(params["expected_value"]),
            str(params["leak_field"]),
            ctx,
            params,
        )
    elif kind == "weak_credentials":
        ctx = str(params.get("cred_format", "pair"))
        prompt = _weak_credentials_prompt(
            str(params["user_param"]),
            str(params["password_param"]),
            _flag_record_key(graph),
            ctx,
        )
    elif kind == "ssrf":
        ctx = str(params.get("ssrf_filter", "decimal_ip"))
        prompt = _ssrf_prompt(param, ctx, params)
    else:
        raise PackError(f"no LLM realization prompt for kind {kind!r}")
    return LLMRequest(prompt=prompt, system=_SYSTEM, json_schema=_HANDLER_SCHEMA)


def handler_from_result(parsed_json: Mapping[str, object] | None) -> str:
    """The handler source out of an LLM result's parsed JSON, or '' if absent."""
    handler = (parsed_json or {}).get("handler")
    return handler if isinstance(handler, str) else ""


def realize_world(
    snapshot: Snapshot,
    propose: Callable[[WorldGraph, str], str],
    run_exploit: Callable[[str], tuple[str, str]],
) -> Snapshot:
    """Generate-verify-freeze: turn a procedural snapshot into an LLM-realized one.

    For each realizable vuln: `propose` a handler, have the host `run_exploit` boot the
    world and return the intended-exploit and benign response bodies, and keep the
    handler only if the consequence verifier accepts it (leaks on the exploit, not on a
    benign request) — otherwise fall back to the procedural template. The result is
    re-frozen to a new content-addressed snapshot recording the realized kinds in
    lineage. The host injects `propose` (the LLM) and `run_exploit` (booting an episode
    is a host concern), so the pack stays transport-free. Mutates `snapshot.graph` — use
    the returned snapshot.
    """
    graph = snapshot.graph
    realized: list[str] = []
    for kind in REALIZABLE_KINDS:
        vuln = next(
            (n for n in graph.by_kind("vulnerability") if n.attrs.get("kind") == kind),
            None,
        )
        if vuln is None:
            continue
        handler = propose(graph, kind)
        if not handler.strip():
            continue
        vuln.attrs["realized_handler"] = handler
        exploit_body, benign_body = run_exploit(kind)
        if classify_admission(graph, exploit_body, benign_body).accepted:
            realized.append(kind)
        else:
            del vuln.attrs["realized_handler"]  # rejected — keep the template
    return Snapshot(
        snapshot_id=graph.content_hash(),
        ontology_id=snapshot.ontology_id,
        graph=graph,
        tasks=snapshot.tasks,
        lineage={**dict(snapshot.lineage), "realized_handlers": tuple(realized)},
        history=snapshot.history,
    )
