"""Ask an LLM to realize a vuln handler, per class (DESIGN.md §9, #260).

Procedural architects the world and places the flag; the LLM only fills in ONE node's
handler with a varied-but-valid implementation. The realization has to be vulnerable to
the SAME sampled injection context the reference solver will exploit, so each prompt
reads that context off the vuln's params. This module owns only the *request* — the host
runs the LLM and then admits the result dynamically (`realize_admit.classify_admission`
with `reference_solver.exploit_and_benign`), since running an episode is a host concern.
"""

from __future__ import annotations

import dataclasses
import random
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass

from graphschema import Node, WorldGraph
from openrange_pack_sdk import LLMBackend, LLMRequest, PackError, Snapshot

from cyber_webapp.codegen.handlers import _extract_handle_body
from cyber_webapp.realize_admit import (
    classify_admission_with_control,
    classify_service_admission,
)
from cyber_webapp.reference_solver import (
    _flag_record_key,
    _vuln_of_kind,
    control_request,
    exploit_and_benign,
    exploit_recipe,
    wrap_payload,
)
from cyber_webapp.verify import perform, verdict_authored

# Injected by the host: booting an episode needs the runtime, which the pack must not
# import, so it takes the boot as a callback yielding the world's base_url.
BootEpisode = Callable[[Snapshot, str], AbstractContextManager[str]]

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
    raw_pool = params.get("confirm_pool", [])
    pool = list(raw_pool) if isinstance(raw_pool, list | tuple) else []
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


def _is_valid_handler(src: str) -> bool:
    # A real LLM sometimes emits unparseable Python; codegen renders the handler by
    # AST-parsing it, so an invalid one crashes the episode boot. Reject it up front.
    try:
        _extract_handle_body(src)
    except PackError:
        return False
    return True


_EXPLOIT_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["exploit", "benign"],
    "properties": {
        "exploit": {"type": "string"},
        "benign": {"type": "string"},
    },
}

_EXPLOIT_RETURN = (
    'Return JSON: {"exploit": "<value to send>", "benign": "<non-leaking value>"}.'
)


def exploit_request(graph: WorldGraph, kind: str) -> LLMRequest:
    """The LLM request to author an (exploit, benign) pair for `kind`. The recipe
    -- the technique plus the flag's LOCATION, never its value -- is read off the vuln's
    meta if the world carries one (an LLM-built world supplies its own, #261), else
    derived; so one generic prompt covers every kind."""
    vuln = _vuln_of_kind(graph, kind)
    endpoint_id = next(e.dst for e in graph.out_edges(vuln.id, "affects"))
    endpoint = graph.nodes[endpoint_id]
    where = f"{endpoint.attrs.get('method', 'GET')} {endpoint.attrs['public_url']}"
    recipe = str(vuln.meta.get("exploit_recipe") or exploit_recipe(graph, kind))
    return LLMRequest(
        prompt=(
            f"Target endpoint: {where}.\n{recipe}\n"
            "Write the exploit value to send and a benign value that does not leak.\n"
            + _EXPLOIT_RETURN
        ),
        system=_SYSTEM,
        json_schema=_EXPLOIT_SCHEMA,
    )


def exploit_from_result(parsed_json: Mapping[str, object] | None) -> tuple[str, str]:
    """The (exploit, benign) payloads from an LLM result's parsed JSON, or ('', '')."""
    data = parsed_json or {}
    exploit, benign = data.get("exploit"), data.get("benign")
    return (
        exploit if isinstance(exploit, str) else "",
        benign if isinstance(benign, str) else "",
    )


def realize_world(
    snapshot: Snapshot,
    propose: Callable[[WorldGraph, str], str],
    run_probes: Callable[[str], tuple[str, str, str | None]],
) -> Snapshot:
    """Generate-verify-freeze: turn a procedural snapshot into an LLM-realized one.

    For each realizable vuln: `propose` a handler, have the host `run_probes` boot the
    world and return the (exploit, benign, control) response bodies, and keep the
    handler only if the gate accepts it — the exploit leaks the flag, a benign request
    does not, and the faithfulness control computes (so a faked/hard-coded handler is
    rejected) — otherwise fall back to the procedural template. The result is re-frozen
    to a new content-addressed snapshot recording the realized kinds in lineage. The
    host injects `propose` (the LLM) and `run_probes` (booting an episode is a host
    concern), so the pack stays transport-free. Mutates `snapshot.graph` — use the
    returned snapshot.
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
        if not handler.strip() or not _is_valid_handler(handler):
            continue
        vuln.attrs["realized_handler"] = handler
        exploit_body, benign_body, control_body = run_probes(kind)
        control = control_request(graph, kind)
        verdict = classify_admission_with_control(
            graph,
            exploit_body,
            benign_body,
            control_body,
            control.expected if control else None,
        )
        if verdict.accepted:
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


def realize_with_backend(
    snapshot: Snapshot,
    backend: LLMBackend,
    run_probes: Callable[[str], tuple[str, str, str | None]],
) -> Snapshot:
    """LLM-realize a snapshot's handlers via this pack's prompts, gated by the verifier.

    Proposes each realizable vuln's handler from ``backend`` (this pack's
    ``realization_request`` + ``handler_from_result``) and runs ``realize_world``'s
    generate-verify-freeze. The host injects ``run_probes`` — booting an episode to run
    the verify probes stays a host concern, so the pack stays transport-free.
    """

    def propose(graph: WorldGraph, kind: str) -> str:
        result = backend.complete(realization_request(graph, kind))
        return handler_from_result(result.parsed_json)

    return realize_world(snapshot, propose, run_probes)


def pentest_probes(
    snapshot: Snapshot,
    boot: BootEpisode,
) -> Callable[[str], tuple[str, str, str | None]]:
    """Build ``realize_world``'s ``run_probes`` for a pentest world: boot the (mutated)
    world via ``boot`` and return the reference exploit, benign and faithfulness-control
    response bodies for ``kind``. The graph is read live each call so an injected
    handler is exercised; the host supplies ``boot`` so the pack stays transport-free.
    """
    task = next(t for t in snapshot.tasks if t.meta.get("family") == "webapp.pentest")

    def run_probes(kind: str) -> tuple[str, str, str | None]:
        with boot(snapshot, task.id) as base_url:
            exploit_req, benign_req = exploit_and_benign(snapshot.graph, kind)
            control = control_request(snapshot.graph, kind)
            return (
                perform(base_url, exploit_req),
                perform(base_url, benign_req),
                perform(base_url, control.request) if control else None,
            )

    return run_probes


def realize_generated(
    snapshot: Snapshot,
    backend: LLMBackend,
    boot: BootEpisode,
) -> Snapshot:
    """Consume the manifest's ``generate`` knob: route an admitted procedural snapshot
    through generate -> verify -> freeze when it asked for generation, else return it
    unchanged. ``generate: "vuln"`` realizes each vuln's handler behind the verifier
    (#260); ``"service"`` / ``"world"`` are later stages (#212) and raise until wired.
    The host injects ``backend`` (the LLM) and ``boot`` (an episode for a snapshot +
    task); booting an episode stays a host concern, so the pack stays transport-free.
    """
    mode = snapshot.lineage.get("generate", False)
    if not mode:
        return snapshot
    if mode == "vuln":
        return realize_with_backend(snapshot, backend, pentest_probes(snapshot, boot))
    if mode == "novel":
        return realize_novel_with_backend(snapshot, backend, boot)
    raise PackError(f"generate mode {mode!r} is not wired yet (see #212)")


@dataclass(frozen=True, slots=True)
class NovelClass:
    """An LLM-proposed vulnerability class the catalog does not have: a new `kind`, its
    exploit `recipe` (technique + flag location, never the value), a vulnerable
    `handler`, and the (exploit, benign) pair that proves it — verified as one coherent
    unit behind the kind-agnostic gate."""

    kind: str
    recipe: str
    handler: str
    exploit: str
    benign: str


_NOVEL_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["kind", "recipe", "handler", "exploit", "benign"],
    "properties": {
        "kind": {"type": "string"},
        "recipe": {"type": "string"},
        "handler": {"type": "string"},
        "exploit": {"type": "string"},
        "benign": {"type": "string"},
    },
}

_CATALOG_KINDS = (
    "sql_injection, command_injection, path_traversal, xxe, ssti, idor, "
    "broken_authz, weak_credentials, ssrf"
)


def novel_class_request(graph: WorldGraph) -> LLMRequest:
    """The LLM request to propose a NOVEL vulnerability class for a procedural skeleton
    (#261): a class outside the catalog, leaking the same procedurally-planted flag from
    the same db lookup endpoint. The prompt carries the endpoint, the param, the `state`
    interface and the flag's LOCATION — never its value — so the proposal is honest by
    construction (re-seed the flag, re-run the exploit; a genuine one still leaks)."""
    vuln = _novel_target(graph)
    params = vuln.attrs["params"]
    if not isinstance(params, Mapping):
        raise PackError("skeleton vuln has no params mapping")
    param = str(params.get("target_param", ""))
    flag_key = _flag_record_key(graph)
    prompt = (
        "Propose a NOVEL web vulnerability class for a lookup endpoint -- a class "
        f"NOT in this catalog: {_CATALOG_KINDS}. It can be ANY flaw (a logic bug, a "
        "wildcard match, a type or parser quirk) -- it need not be injection.\n"
        "Write the handler against this EXACT runtime contract:\n"
        "- `def handle(query, state):` where `query` is dict[str, list[str]]; read the "
        f"input as `(query.get({param!r}, ['']) or [''])[0]`.\n"
        "- `state['db']` is a sqlite3 connection with `row_factory = sqlite3.Row` (a "
        "row is NOT a dict: index `row['col']` or wrap `dict(row)`, there is no "
        "row.get). `state['schema']` is a dict -- read the REAL identifiers from it: "
        "`tbl = state['schema']['table']`, `kc = state['schema']['key_column']`, "
        "`vc = state['schema']['value_column']`, then build SQL by substituting those "
        "(e.g. f'SELECT {kc}, {vc} FROM {tbl} WHERE ...'). Never write the literal "
        "words 'table'/'key_column'/'value_column' into SQL. The table has exactly "
        f"those two columns; the one row whose key_column value == {flag_key!r} holds "
        "the flag in its value_column.\n"
        "- The handler MUST `return (status: int, headers: dict, body: bytes)` -- "
        "exactly three values. Standard library only.\n"
        "- `exploit` is the value to send that makes YOUR handler return the flag row; "
        "`benign` returns only its own (non-flag) row. Trace the exploit through your "
        "own handler and confirm the flag row is in the response before returning.\n"
        "Return JSON: kind (a short snake_case name), recipe (the technique plus where "
        "the flag is, never its value), handler (the full def handle source), exploit "
        "(the value that leaks the flag), benign (one that does not)."
    )
    return LLMRequest(prompt=prompt, system=_SYSTEM, json_schema=_NOVEL_SCHEMA)


def novel_from_result(parsed_json: Mapping[str, object] | None) -> NovelClass | None:
    """A `NovelClass` from a result's parsed JSON, or None if a field is missing."""
    data = parsed_json or {}
    fields = {
        k: data.get(k) for k in ("kind", "recipe", "handler", "exploit", "benign")
    }
    if not all(isinstance(v, str) and v.strip() for v in fields.values()):
        return None
    return NovelClass(**{k: str(v) for k, v in fields.items()})


def realize_novel(
    snapshot: Snapshot,
    propose: Callable[[WorldGraph], NovelClass | None],
    boot: BootEpisode,
) -> Snapshot:
    """Realize an LLM-PROPOSED vulnerability class the catalog does not have (#261).

    `propose` reads the procedural skeleton and returns a coherent novel class, or None
    to keep the skeleton. The proposal is stamped onto the skeleton's vuln (its `kind`,
    the `realized_handler`, and the recipe on `meta`) and admitted by the SAME
    kind-agnostic gate: the authored exploit must leak the flag and the benign must not,
    and -- the integrity check -- the exploit must recover a FRESHLY re-seeded flag,
    so a memorized value or a handler that hard-codes the flag is rejected. Accepted ->
    re-freeze with the novel kind on the lineage; rejected -> return the skeleton
    unchanged. The host injects `propose` (the LLM) and `boot` (an episode), so the pack
    stays transport-free. Mutates `snapshot.graph` -- use the returned snapshot.
    """
    graph = snapshot.graph
    task = next(t for t in snapshot.tasks if t.meta.get("family") == "webapp.pentest")
    original = _novel_target(graph)
    proposal = propose(graph)
    if proposal is None or not _is_valid_handler(proposal.handler):
        return snapshot
    graph.nodes[original.id] = dataclasses.replace(
        original,
        attrs={
            **original.attrs,
            "kind": proposal.kind,
            "realized_handler": proposal.handler,
        },
        meta={**original.meta, "exploit_recipe": proposal.recipe},
    )
    if _novel_admits(snapshot, task.id, proposal, boot):
        return Snapshot(
            snapshot_id=graph.content_hash(),
            ontology_id=snapshot.ontology_id,
            graph=graph,
            tasks=snapshot.tasks,
            lineage={**dict(snapshot.lineage), "generated_class": proposal.kind},
            history=snapshot.history,
        )
    graph.nodes[original.id] = original  # rejected -> restore the procedural skeleton
    return snapshot


def realize_novel_with_backend(
    snapshot: Snapshot,
    backend: LLMBackend,
    boot: BootEpisode,
) -> Snapshot:
    """`realize_novel` driven by an `LLMBackend` via this pack's novel-class prompt."""

    def propose(graph: WorldGraph) -> NovelClass | None:
        return novel_from_result(
            backend.complete(novel_class_request(graph)).parsed_json
        )

    return realize_novel(snapshot, propose, boot)


def _novel_target(graph: WorldGraph) -> Node:
    vulns = list(graph.by_kind("vulnerability"))
    if len(vulns) != 1:
        raise PackError("realize_novel expects a single-vuln procedural skeleton")
    return vulns[0]


def _novel_admits(
    snapshot: Snapshot, task_id: str, proposal: NovelClass, boot: BootEpisode
) -> bool:
    # Lazy import dodges a sampling import cycle (like _annotate_exploit_recipes).
    from cyber_webapp.reseed import replant_flag
    from cyber_webapp.sampling import generate_flag

    with boot(snapshot, task_id) as base_url:
        verdict = verdict_authored(
            snapshot.graph, base_url, proposal.kind, proposal.exploit, proposal.benign
        )
    if not verdict.accepted:
        return False
    # Integrity: a fresh flag + the SAME exploit must still recover it, so a memorized
    # value or a flag-hard-coding handler (which passed the first gate) is caught.
    fresh = replant_flag(snapshot, generate_flag(random.Random(_RESEED_SEED)))
    with boot(fresh, task_id) as base_url:
        leaked = perform(
            base_url, wrap_payload(fresh.graph, proposal.kind, proposal.exploit)
        )
    return str(fresh.graph.nodes["secret_flag"].attrs["value_ref"]) in leaked


_RESEED_SEED = 0xC0FFEE


_SERVICE_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["endpoints"],
    "properties": {
        "endpoints": {"type": "object", "additionalProperties": {"type": "string"}}
    },
}


def benign_endpoints_of(graph: WorldGraph, service_id: str) -> list[Node]:
    """The service's endpoints with no affecting vuln — its realizable benign surface.

    Empty if a vuln affects the whole service, or for the framework routes / and
    /openapi.json (which the LLM must not author).
    """
    vuln_eps = {
        e.dst
        for v in graph.by_kind("vulnerability")
        for e in graph.out_edges(v.id, "affects")
    }
    if service_id in vuln_eps:  # a service-level vuln owns every endpoint
        return []
    out: list[Node] = []
    for edge in graph.out_edges(service_id, "exposes"):
        ep = graph.nodes.get(edge.dst)
        if ep is None or edge.dst in vuln_eps:
            continue
        if str(ep.attrs.get("path", "")) in ("/", "/openapi.json"):
            continue
        out.append(ep)
    return out


def service_realization_request(graph: WorldGraph, service_id: str) -> LLMRequest:
    """The LLM request to author realistic bodies for a service's benign endpoints.

    Procedural keeps the vuln, flag and routes; the LLM only fills the non-vuln
    endpoints with plausible content. The host runs this and passes the result through
    `realize_service_surface`'s whole-service admission.
    """
    service = graph.nodes[service_id]
    kind = str(service.attrs.get("kind", "service"))
    name = str(service.attrs.get("name", service_id))
    paths = [str(ep.attrs.get("path")) for ep in benign_endpoints_of(graph, service_id)]
    listing = "\n".join(f"  - {p}" for p in paths)
    prompt = (
        f"Author benign endpoint handlers for the {kind} service {name!r} of a "
        "security-training web app.\n"
        "For each path below, write a Python `def handle(query, state):` returning a "
        "tuple (status:int, headers:dict, body:bytes) with realistic, plausible "
        "content for that path — JSON for /api/* paths, simple HTML for pages.\n"
        f"Paths:\n{listing}\n"
        "- query is dict[str, list[str]]; stdlib only; import inside handle.\n"
        "- These are BENIGN: do NOT read state['secrets'], the flag, or a data store "
        "value column. Return only static or schema-shaped placeholders.\n"
        'Return JSON: {"endpoints": {"<path>": "<the full def handle source>"}}.'
    )
    return LLMRequest(prompt=prompt, system=_SYSTEM, json_schema=_SERVICE_SCHEMA)


def service_handlers_from_result(
    parsed_json: Mapping[str, object] | None,
) -> dict[str, str]:
    """The {path: handler source} map out of an LLM result, dropping non-str entries."""
    endpoints = (parsed_json or {}).get("endpoints")
    if not isinstance(endpoints, Mapping):
        return {}
    return {
        str(path): src
        for path, src in endpoints.items()
        if isinstance(src, str) and src.strip()
    }


def realize_service_surface(
    snapshot: Snapshot,
    service_id: str,
    propose_service: Callable[[WorldGraph, str], Mapping[str, str]],
    run_service_probes: Callable[[str], tuple[str, str, Mapping[str, str], bool]],
) -> Snapshot:
    """Generate-verify-freeze for a service's benign surface (DESIGN.md §9, #212).

    The LLM (`propose_service`) authors the non-vuln endpoint bodies of `service_id`;
    the host (`run_service_probes`) boots the realized world once and returns the
    oracle exploit/benign bodies, a benign GET per realized endpoint, and whether `/`
    still serves 200. Admit ALL-OR-NOTHING: keep the bodies only if the oracle still
    fires, no benign endpoint leaks the flag, and the world boots — else fall back to
    the procedural stubs. Re-freezes to a new snapshot recording the realized paths.
    Mutates `snapshot.graph` — use the return.
    """
    graph = snapshot.graph
    benign = {
        str(ep.attrs.get("path")): ep for ep in benign_endpoints_of(graph, service_id)
    }
    applied: list[Node] = []
    for path, src in propose_service(graph, service_id).items():
        ep = benign.get(path)
        if ep is None or not src.strip():
            continue
        try:
            _extract_handle_body(src)  # must be valid Python with a def handle
        except PackError:
            continue
        ep.attrs["realized_handler"] = src
        applied.append(ep)

    realized: tuple[str, ...] = ()
    if applied:
        exploit_body, benign_body, endpoint_bodies, root_ok = run_service_probes(
            service_id
        )
        verdict = classify_service_admission(
            graph,
            oracle_exploit_body=exploit_body,
            oracle_benign_body=benign_body,
            benign_endpoint_bodies=endpoint_bodies,
            root_ok=root_ok,
        )
        if verdict.accepted:
            realized = tuple(str(ep.attrs.get("path")) for ep in applied)
        else:
            for ep in applied:
                del ep.attrs["realized_handler"]
    return Snapshot(
        snapshot_id=graph.content_hash(),
        ontology_id=snapshot.ontology_id,
        graph=graph,
        tasks=snapshot.tasks,
        lineage={**dict(snapshot.lineage), "realized_endpoints": realized},
        history=snapshot.history,
    )
