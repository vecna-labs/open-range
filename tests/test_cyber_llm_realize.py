"""LLM handler realization across classes (DESIGN.md §9, #260).

Two halves, both free of a live LLM: the per-class realization *request* is a pure
function of the world (tested directly), and the dynamic admission gate is exercised by
injecting hand-written handlers — a faithful one is admitted, a trivial one rejected —
proving the gate generalizes past command-injection to the response-leak and file-read
families via the reference solver.
"""

from __future__ import annotations

import os
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path

import pytest
from cyber_webapp import WebappPack
from cyber_webapp.llm_realize import (
    benign_endpoints_of,
    handler_from_result,
    realization_request,
    realize_service_surface,
    realize_with_backend,
    realize_world,
    service_handlers_from_result,
    service_realization_request,
)
from cyber_webapp.ontology import ONTOLOGY_ID
from cyber_webapp.realize_admit import (
    AdmissionVerdict,
    classify_admission,
    classify_admission_with_control,
    classify_service_admission,
)
from cyber_webapp.reference_solver import (
    _flag_record_key,
    _vuln_of_kind,
    control_request,
    exploit_and_benign,
)
from cyber_webapp.verify import perform
from graphschema import Edge, Node, WorldGraph
from openrange_pack_sdk import PackError, Snapshot

from openrange.core.admit import admit
from openrange.core.episode import EpisodeService
from openrange.llm import ClaudeBackend


def _admit(loot: str, kind: str, **pin: object) -> Snapshot:
    snap = admit(
        WebappPack(),
        manifest={
            "pack": {"id": "webapp"},
            "runtime": {"tick": {"mode": "off"}},
            "npc": [],
            "seed": 7,
            "loot_shapes": {loot: 1, "db" if loot == "file" else "file": 0},
            "vuln_kinds": {kind: 1},
        },
        max_repairs=3,
    )
    assert isinstance(snap, Snapshot), snap
    if pin:
        params = _vuln_of_kind(snap.graph, kind).attrs["params"]
        assert isinstance(params, dict)
        params.update(pin)  # pin the context so the exploit + handler agree
    return snap


def _fetch(url: str) -> str:
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            return str(resp.read().decode())
    except urllib.error.HTTPError as exc:
        return exc.read().decode()


def _gate(snap: Snapshot, kind: str, handler: str, workdir: Path) -> AdmissionVerdict:
    graph = snap.graph
    _vuln_of_kind(graph, kind).attrs["realized_handler"] = handler
    exploit_req, benign_req = exploit_and_benign(graph, kind)
    svc = EpisodeService(WebappPack(), workdir)
    try:
        task = next(t for t in snap.tasks if t.meta.get("family") == "webapp.pentest")
        handle = svc.start_episode(snap, task.id)
        base = str(svc.surface(handle)["base_url"])
        exploit_body = perform(base, exploit_req)
        benign_body = perform(base, benign_req)
    finally:
        svc.close()
    return classify_admission(graph, exploit_body, benign_body)


def _faithful_sqli(graph: WorldGraph) -> str:
    p = _vuln_of_kind(graph, "sql_injection").attrs["params"]
    param, table, leak = str(p["target_param"]), str(p["table"]), str(p["leak_column"])
    return "\n".join(
        [
            "def handle(query, state):",
            "    import json",
            f"    val = (query.get({param!r}, ['']) or [''])[0]",
            "    kc = state['schema']['key_column']",
            f'    sql = "SELECT " + kc + ", {leak} FROM {table} WHERE " + kc'
            ' + " = \'" + val + "\'"',
            "    try:",
            "        rows = [dict(r) for r in state['db'].execute(sql).fetchall()]",
            "    except Exception as e:",
            "        return 500, {}, json.dumps({'error': str(e)}).encode()",
            "    return 200, {}, json.dumps({'rows': rows}).encode()",
        ]
    )


def _trivial_sqli(graph: WorldGraph) -> str:
    p = _vuln_of_kind(graph, "sql_injection").attrs["params"]
    table, leak = str(p["table"]), str(p["leak_column"])
    return "\n".join(
        [
            "def handle(query, state):",
            "    import json",
            f"    rows = [dict(r) for r in"
            f" state['db'].execute('SELECT key, {leak} FROM {table}').fetchall()]",
            "    return 200, {}, json.dumps({'rows': rows}).encode()",
        ]
    )


def _faithful_path(graph: WorldGraph) -> str:
    p = _vuln_of_kind(graph, "path_traversal").attrs["params"]
    param, base = str(p["target_param"]), str(p["base_dir"])
    return "\n".join(
        [
            "def handle(query, state):",
            "    import posixpath",
            f"    req = (query.get({param!r}, ['']) or [''])[0]",
            "    while '../' in req:",
            "        req = req.replace('../', '')",
            f"    resolved = posixpath.normpath(posixpath.join({base!r}, req))",
            "    files = state.get('files', {})",
            "    if resolved in files:",
            "        return 200, {}, str(files[resolved]).encode()",
            "    return 404, {}, b'nope'",
        ]
    )


def _faithful_xxe(graph: WorldGraph) -> str:
    p = _vuln_of_kind(graph, "xxe").attrs["params"]
    param = str(p["target_param"])
    return "\n".join(
        [
            "def handle(query, state):",
            "    import io",
            "    import xml.sax",
            "    import xml.sax.handler",
            "    import xml.sax.xmlreader",
            f"    doc = (query.get({param!r}, ['']) or [''])[0]",
            "    files = state.get('files', {})",
            "    class _R(xml.sax.handler.EntityResolver):",
            "        def resolveEntity(self, public_id, system_id):",
            "            src = xml.sax.xmlreader.InputSource()",
            "            text = ''",
            "            if system_id and 'file://' in system_id:",
            "                path = system_id.split('file://', 1)[-1]",
            "                text = files.get(path, '')",
            "            src.setCharacterStream(io.StringIO(text))",
            "            return src",
            "    seen = []",
            "    class _C(xml.sax.handler.ContentHandler):",
            "        def __init__(self):",
            "            self.depth = 0",
            "        def startElement(self, name, attrs):",
            "            self.depth += 1",
            "        def endElement(self, name):",
            "            self.depth -= 1",
            "        def characters(self, content):",
            "            if self.depth == 1:",
            "                seen.append(content)",
            "    try:",
            "        parser = xml.sax.make_parser()",
            "        parser.setFeature(xml.sax.handler.feature_external_ges, True)",
            "        parser.setContentHandler(_C())",
            "        parser.setEntityResolver(_R())",
            "        parser.parse(io.StringIO(doc))",
            "        out = ''.join(seen)",
            "    except Exception as exc:",
            "        out = 'error: ' + type(exc).__name__",
            "    return 200, {'Content-Type': 'application/xml'}, out.encode()",
        ]
    )


def _trivial_xxe(graph: WorldGraph) -> str:
    return "\n".join(
        [
            "def handle(query, state):",
            "    files = state.get('files', {})",
            "    dump = '\\n'.join(str(v) for v in files.values())",
            "    return 200, {'Content-Type': 'application/xml'}, dump.encode()",
        ]
    )


def _faithful_ssti(graph: WorldGraph) -> str:
    p = _vuln_of_kind(graph, "ssti").attrs["params"]
    param = str(p["target_param"])
    sink = str(p.get("render_sink", "attribute"))
    return "\n".join(
        [
            "def handle(query, state):",
            "    from jinja2.sandbox import SandboxedEnvironment",
            f"    expr = (query.get({param!r}, ['']) or [''])[0]",
            "    config = state.get('files', {})",
            f"    sink = {sink!r}",
            "    if sink == 'comment':",
            "        source = '{# ' + expr + ' #}'",
            "    elif sink == 'expr':",
            "        source = \"{{ '[' ~ \" + expr + \" ~ ']' }}\"",
            "    else:",
            "        source = '{\"v\": \"' + expr + '\"}'",
            "    try:",
            "        env = SandboxedEnvironment()",
            "        rendered = env.from_string(source).render(config=config)",
            "    except Exception as exc:",
            "        rendered = 'render error: ' + type(exc).__name__",
            "    return 200, {'Content-Type': 'text/plain'}, rendered.encode()",
        ]
    )


def _trivial_ssti(graph: WorldGraph) -> str:
    flag_key = _flag_record_key(graph)
    return "\n".join(
        [
            "def handle(query, state):",
            "    del query",
            "    config = state.get('files', {})",
            f"    body = str(config.get({flag_key!r}, ''))",
            "    return 200, {'Content-Type': 'text/plain'}, body.encode()",
        ]
    )


def _faithful_idor(graph: WorldGraph) -> str:
    p = _vuln_of_kind(graph, "idor").attrs["params"]
    param = str(p["target_param"])
    ctx = str(p.get("ref_context", "direct"))
    prefix = str(p.get("ref_prefix", "user-"))
    return "\n".join(
        [
            "def handle(query, state):",
            "    import base64",
            "    import json",
            f"    raw = (query.get({param!r}, ['']) or [''])[0]",
            f"    ctx = {ctx!r}",
            f"    prefix = {prefix!r}",
            "    if ctx == 'base64':",
            "        try:",
            "            rid = base64.b64decode(raw).decode('utf-8')",
            "        except Exception:",
            "            rid = ''",
            "    elif ctx == 'prefixed':",
            "        rid = raw[len(prefix):] if raw.startswith(prefix) else ''",
            "    else:",
            "        rid = raw",
            "    schema = state['schema']",
            "    table = schema['table']",
            "    key_col = schema['key_column']",
            "    value_col = schema['value_column']",
            "    cur = state['db'].cursor()",
            "    sql = ("
            '"SELECT " + key_col + ", " + value_col + " FROM " + table'
            ' + " WHERE " + key_col + " = ?"',
            "    )",
            "    rows = [dict(r) for r in cur.execute(sql, (rid,)).fetchall()]",
            "    return 200, {'Content-Type': 'application/json'}, "
            "json.dumps({'record': rows}).encode()",
        ]
    )


def _trivial_idor(graph: WorldGraph) -> str:
    return "\n".join(
        [
            "def handle(query, state):",
            "    import json",
            "    schema = state['schema']",
            "    table = schema['table']",
            "    key_col = schema['key_column']",
            "    value_col = schema['value_column']",
            "    cur = state['db'].cursor()",
            "    sql = 'SELECT ' + key_col + ', ' + value_col + ' FROM ' + table",
            "    rows = [dict(r) for r in cur.execute(sql).fetchall()]",
            "    return 200, {'Content-Type': 'application/json'}, "
            "json.dumps({'record': rows}).encode()",
        ]
    )


def _faithful_broken_authz(graph: WorldGraph) -> str:
    p = _vuln_of_kind(graph, "broken_authz").attrs["params"]
    header, expected = str(p["trust_header"]), str(p["expected_value"])
    field, ctx = str(p["leak_field"]), str(p.get("trust_context", "single_token"))
    confirm, gate_on = str(p["confirm_param"]), str(p["confirm_value"])
    pool = list(p["confirm_pool"])
    lines = [
        "def handle(query, state):",
        "    import json",
        f"    role = (query.get({header!r}, ['']) or [''])[0]",
        f"    present = any(n in query for n in {pool!r})",
    ]
    if ctx == "encoded_token":
        lines.append(f"    ok = role == {expected!r}.encode().hex() and not present")
    elif ctx == "dual_factor":
        lines += [
            f"    gate = (query.get({confirm!r}, ['']) or [''])[0]",
            f"    ok = role == {expected!r} and gate == {gate_on!r}",
        ]
    else:  # single_token
        lines.append(f"    ok = role == {expected!r} and not present")
    lines += [
        "    if not ok:",
        "        return 403, {}, b'forbidden'",
        f"    secret = state.get('secrets', {{}}).get({field!r}, '')",
        f"    body = json.dumps({{{field!r}: secret}}).encode()",
        "    return 200, {'Content-Type': 'application/json'}, body",
    ]
    return "\n".join(lines)


def _trivial_broken_authz(graph: WorldGraph) -> str:
    p = _vuln_of_kind(graph, "broken_authz").attrs["params"]
    field = str(p["leak_field"])
    return "\n".join(
        [
            "def handle(query, state):",
            "    import json",
            f"    secret = state.get('secrets', {{}}).get({field!r}, '')",
            f"    body = json.dumps({{{field!r}: secret}}).encode()",
            "    return 200, {'Content-Type': 'application/json'}, body",
        ]
    )


def _faithful_weak_credentials(graph: WorldGraph) -> str:
    p = _vuln_of_kind(graph, "weak_credentials").attrs["params"]
    up, pp = str(p["user_param"]), str(p["password_param"])
    wu, wp = str(p["weak_user"]), str(p["weak_password"])
    fmt = str(p.get("cred_format", "pair"))
    if fmt == "combined":
        read = [
            f"    raw = (query.get({pp!r}, ['']) or [''])[0]",
            "    user, _, password = raw.partition(':')",
        ]
    elif fmt == "basic":
        read = [
            f"    token = (query.get({pp!r}, ['']) or [''])[0]",
            "    try:",
            "        creds = base64.b64decode(token, validate=True).decode()",
            "    except Exception:",
            "        creds = ''",
            "    user, _, password = creds.partition(':')",
        ]
    else:
        read = [
            f"    user = (query.get({up!r}, ['']) or [''])[0]",
            f"    password = (query.get({pp!r}, ['']) or [''])[0]",
        ]
    return "\n".join(
        [
            "def handle(query, state):",
            "    import base64",
            "    import json",
            "    secrets = state.get('secrets', {})",
            *read,
            f"    if user == {wu!r} and password == {wp!r}:",
            "        token = json.dumps({'token': secrets.get('flag', '')})",
            "        return 200, {}, token.encode()",
            "    return 401, {}, json.dumps({'error': 'unauthorized'}).encode()",
        ]
    )


def _trivial_weak_credentials(graph: WorldGraph) -> str:
    return "\n".join(
        [
            "def handle(query, state):",
            "    import json",
            "    secrets = state.get('secrets', {})",
            "    token = json.dumps({'token': secrets.get('flag', '')})",
            "    return 200, {}, token.encode()",
        ]
    )


def _faithful_ssrf(graph: WorldGraph) -> str:
    p = _vuln_of_kind(graph, "ssrf").attrs["params"]
    param, host = str(p["target_param"]), str(p["internal_host"])
    return "\n".join(
        [
            "def handle(query, state):",
            "    import json",
            "    from urllib.parse import urlparse",
            f"    raw = (query.get({param!r}, ['']) or [''])[0]",
            "    if not raw:",
            "        return 400, {}, b'missing url'",
            "    parsed = urlparse(raw)",
            "    scheme = (parsed.scheme or '').lower()",
            "    host = (parsed.hostname or '').lower()",
            f"    internal_host = {host!r}",
            "    if scheme in ('http', 'https'):",
            "        return 403, {}, b'scheme not allowed'",
            "    if scheme != 'gopher':",
            "        return 502, {}, b'unsupported scheme'",
            "    secret = ''",
            "    if host == internal_host.lower():",
            "        secret = state.get('secrets', {}).get('flag', '')",
            "    return 200, {}, json.dumps({'data': secret}).encode()",
        ]
    )


def _trivial_ssrf(graph: WorldGraph) -> str:
    p = _vuln_of_kind(graph, "ssrf").attrs["params"]
    param = str(p["target_param"])
    return "\n".join(
        [
            "def handle(query, state):",
            "    import json",
            f"    _ = (query.get({param!r}, ['']) or [''])[0]",
            "    secret = state.get('secrets', {}).get('flag', '')",
            "    return 200, {}, json.dumps({'data': secret}).encode()",
        ]
    )


_CONTEXTS = [
    ("command_injection", "file", "inj_context", "separator"),
    ("command_injection", "file", "inj_context", "substitution"),
    ("command_injection", "file", "inj_context", "quoted"),
    ("sql_injection", "db", "context", "single"),
    ("sql_injection", "db", "context", "numeric"),
    ("sql_injection", "db", "context", "double"),
    ("path_traversal", "file", "confinement", "absolute_only"),
    ("path_traversal", "file", "confinement", "relative"),
    ("path_traversal", "file", "confinement", "dotdot_filter"),
    ("xxe", "file", "entity_context", "element_content"),
    ("xxe", "file", "entity_context", "wrapped_root"),
    ("xxe", "file", "entity_context", "scheme_prefix"),
    ("ssti", "file", "render_sink", "attribute"),
    ("ssti", "file", "render_sink", "comment"),
    ("ssti", "file", "render_sink", "expr"),
    ("idor", "db", "ref_context", "direct"),
    ("idor", "db", "ref_context", "base64"),
    ("idor", "db", "ref_context", "prefixed"),
    ("broken_authz", "db", "trust_context", "single_token"),
    ("broken_authz", "db", "trust_context", "encoded_token"),
    ("broken_authz", "db", "trust_context", "dual_factor"),
    ("weak_credentials", "db", "cred_format", "pair"),
    ("weak_credentials", "db", "cred_format", "combined"),
    ("weak_credentials", "db", "cred_format", "basic"),
    ("ssrf", "db", "ssrf_filter", "scheme_block"),
    ("ssrf", "db", "ssrf_filter", "host_allowlist"),
    ("ssrf", "db", "ssrf_filter", "decimal_ip"),
]


@pytest.mark.parametrize(("kind", "loot", "key", "ctx"), _CONTEXTS)
def test_realization_request_per_context(
    kind: str, loot: str, key: str, ctx: str
) -> None:
    # Every class × sampled-context pair yields a handler-authoring prompt that names
    # the flag, so the realized handler matches the exploit the solver will run.
    req = realization_request(_admit(loot, kind, **{key: ctx}).graph, kind)
    assert "def handle" in req.prompt
    assert "flag" in req.prompt.lower()
    assert req.json_schema is not None


def test_realization_request_sqli_names_its_table() -> None:
    graph = _admit("db", "sql_injection").graph
    table = str(_vuln_of_kind(graph, "sql_injection").attrs["params"]["table"])
    assert table in realization_request(graph, "sql_injection").prompt


def test_realization_request_rejects_unrealized_kind() -> None:
    # A real vuln kind with no realization prompt yet hits the no-prompt guard
    # (config_disclosure is an internal chain primitive, not in REALIZABLE_KINDS).
    graph = WorldGraph(ontology=ONTOLOGY_ID)
    graph.add_node(
        Node(
            id="v",
            kind="vulnerability",
            attrs={"kind": "config_disclosure", "params": {}},
        )
    )
    with pytest.raises(PackError):
        realization_request(graph, "config_disclosure")


def test_realization_request_rejects_non_mapping_params() -> None:
    graph = WorldGraph(ontology=ONTOLOGY_ID)
    graph.add_node(
        Node(
            id="v",
            kind="vulnerability",
            attrs={"kind": "command_injection", "params": "not-a-map"},
        )
    )
    with pytest.raises(PackError):
        realization_request(graph, "command_injection")


def test_handler_from_result() -> None:
    assert handler_from_result({"handler": "def handle(): ..."}) == "def handle(): ..."
    assert handler_from_result({}) == ""
    assert handler_from_result(None) == ""
    assert handler_from_result({"handler": 123}) == ""


# kind, loot, context key, pinned context, faithful handler, trivial handler (or None).
_GATE: list[
    tuple[
        str,
        str,
        str,
        str,
        Callable[[WorldGraph], str],
        Callable[[WorldGraph], str] | None,
    ]
] = [
    ("sql_injection", "db", "context", "single", _faithful_sqli, _trivial_sqli),
    ("path_traversal", "file", "confinement", "absolute_only", _faithful_path, None),
    ("xxe", "file", "entity_context", "element_content", _faithful_xxe, _trivial_xxe),
    ("ssti", "file", "render_sink", "attribute", _faithful_ssti, _trivial_ssti),
    ("idor", "db", "ref_context", "base64", _faithful_idor, _trivial_idor),
    (
        "broken_authz",
        "db",
        "trust_context",
        "dual_factor",
        _faithful_broken_authz,
        _trivial_broken_authz,
    ),
    (
        "weak_credentials",
        "db",
        "cred_format",
        "pair",
        _faithful_weak_credentials,
        _trivial_weak_credentials,
    ),
    ("ssrf", "db", "ssrf_filter", "scheme_block", _faithful_ssrf, _trivial_ssrf),
]


@pytest.mark.parametrize(("kind", "loot", "key", "ctx", "faithful", "trivial"), _GATE)
def test_gate_admits_faithful_rejects_trivial(
    kind: str,
    loot: str,
    key: str,
    ctx: str,
    faithful: Callable[[WorldGraph], str],
    trivial: Callable[[WorldGraph], str] | None,
    tmp_path: Path,
) -> None:
    # A hand-written handler vulnerable to the pinned context is admitted; one that
    # leaks on a benign request is rejected — the gate generalizes across all classes.
    snap = _admit(loot, kind, **{key: ctx})
    ok = _gate(snap, kind, faithful(snap.graph), tmp_path / "ok")
    assert ok.accepted, f"{kind}: {ok.reason}"
    if trivial is not None:
        bad = _gate(snap, kind, trivial(snap.graph), tmp_path / "bad")
        assert not bad.accepted and bad.trivial, f"{kind}: trivial not rejected"


def _episode_runner(
    snap: Snapshot, base_dir: Path
) -> Callable[[str], tuple[str, str, str | None]]:
    # The host side realize_world injects: boot the (mutated) world and return the
    # exploit, benign and faithfulness-control response bodies.
    counter = iter(range(1000))
    task = next(t for t in snap.tasks if t.meta.get("family") == "webapp.pentest")

    def run_probes(kind: str) -> tuple[str, str, str | None]:
        svc = EpisodeService(WebappPack(), base_dir / f"e{next(counter)}")
        try:
            handle = svc.start_episode(snap, task.id)
            base = str(svc.surface(handle)["base_url"])
            exploit_req, benign_req = exploit_and_benign(snap.graph, kind)
            control = control_request(snap.graph, kind)
            control_body = perform(base, control.request) if control else None
            return (
                perform(base, exploit_req),
                perform(base, benign_req),
                control_body,
            )
        finally:
            svc.close()

    return run_probes


def test_realize_world_bakes_in_a_faithful_handler(tmp_path: Path) -> None:
    snap = _admit("db", "sql_injection", context="single")
    before = snap.graph.content_hash()  # live hash (the pin mutated the graph)
    out = realize_world(
        snap, lambda g, _k: _faithful_sqli(g), _episode_runner(snap, tmp_path)
    )
    assert "sql_injection" in out.lineage["realized_handlers"]
    assert out.snapshot_id == out.graph.content_hash()  # re-frozen
    assert out.snapshot_id != before  # the realized handler changed the world


def test_realize_world_falls_back_on_a_trivial_handler(tmp_path: Path) -> None:
    snap = _admit("db", "sql_injection", context="single")
    before = snap.graph.content_hash()  # live hash (the pin mutated the graph)
    out = realize_world(
        snap, lambda g, _k: _trivial_sqli(g), _episode_runner(snap, tmp_path)
    )
    assert out.lineage["realized_handlers"] == ()  # rejected
    assert out.snapshot_id == before  # template kept -> world unchanged


def test_realize_world_skips_an_empty_proposal(tmp_path: Path) -> None:
    snap = _admit("db", "sql_injection", context="single")
    before = snap.graph.content_hash()  # live hash (the pin mutated the graph)
    out = realize_world(snap, lambda _g, _k: "", _episode_runner(snap, tmp_path))
    assert out.lineage["realized_handlers"] == ()
    assert out.snapshot_id == before


@pytest.mark.skipif(
    not os.environ.get("OPENRANGE_LIVE_LLM"),
    reason="needs a live LLM CLI (set OPENRANGE_LIVE_LLM=1)",
)
def test_realize_with_backend_drives_a_live_llm(tmp_path: Path) -> None:
    backend = ClaudeBackend()
    backend.preflight()
    snap = _admit("db", "sql_injection", context="single")
    out = realize_with_backend(snap, backend, _episode_runner(snap, tmp_path))
    assert "sql_injection" in out.lineage["realized_handlers"]
    assert out.snapshot_id == out.graph.content_hash()


def _faithful_cmdi(graph: WorldGraph) -> str:
    param = str(
        _vuln_of_kind(graph, "command_injection").attrs["params"]["target_param"]
    )
    return "\n".join(
        [
            "def handle(query, state):",
            f"    inp = (query.get({param!r}, ['']) or [''])[0]",
            "    files = state.get('files', {})",
            "    out = ''",
            "    if '; cat ' in inp:",
            "        path = inp.split('; cat ', 1)[1].split(';')[0].strip()",
            "        out = str(files.get(path, ''))",
            "    return 200, {'Content-Type': 'text/plain'}, out.encode()",
        ]
    )


def _faked(graph: WorldGraph, kind: str, benign_value: str) -> str:
    # A faked engine: hard-codes the flag and returns it for any non-benign input
    # without running the mechanic. Passes solvable+not-trivial; the control catches it.
    flag = str(graph.nodes["secret_flag"].attrs["value_ref"])
    param = str(_vuln_of_kind(graph, kind).attrs["params"]["target_param"])
    return "\n".join(
        [
            "def handle(query, state):",
            f"    inp = (query.get({param!r}, ['']) or [''])[0]",
            f"    if inp == {benign_value!r}:",
            "        return 200, {}, b''",
            f"    return 200, {{}}, {flag!r}.encode()",
        ]
    )


def _control_gate(
    snap: Snapshot, kind: str, handler: str, workdir: Path
) -> AdmissionVerdict:
    graph = snap.graph
    _vuln_of_kind(graph, kind).attrs["realized_handler"] = handler
    exploit_req, benign_req = exploit_and_benign(graph, kind)
    control = control_request(graph, kind)
    svc = EpisodeService(WebappPack(), workdir)
    try:
        task = next(t for t in snap.tasks if t.meta.get("family") == "webapp.pentest")
        handle = svc.start_episode(snap, task.id)
        base = str(svc.surface(handle)["base_url"])
        exploit_body = perform(base, exploit_req)
        benign_body = perform(base, benign_req)
        control_body = perform(base, control.request) if control else None
    finally:
        svc.close()
    return classify_admission_with_control(
        graph,
        exploit_body,
        benign_body,
        control_body,
        control.expected if control else None,
    )


_CONTROL_CASES = [
    (
        "command_injection",
        "file",
        "inj_context",
        "separator",
        "127.0.0.1",
        _faithful_cmdi,
    ),
    ("sql_injection", "db", "context", "single", "1", _faithful_sqli),
    (
        "path_traversal",
        "file",
        "confinement",
        "absolute_only",
        "index.html",
        _faithful_path,
    ),
    ("xxe", "file", "entity_context", "element_content", "<r>ping</r>", _faithful_xxe),
    ("ssti", "file", "render_sink", "attribute", "status", _faithful_ssti),
    ("idor", "db", "ref_context", "base64", "does-not-exist", _faithful_idor),
]


@pytest.mark.parametrize(
    ("kind", "loot", "key", "ctx", "benign", "faithful"), _CONTROL_CASES
)
def test_control_rejects_faked_admits_faithful(
    kind: str,
    loot: str,
    key: str,
    ctx: str,
    benign: str,
    faithful: Callable[[WorldGraph], str],
    tmp_path: Path,
) -> None:
    # A faithful handler computes/fetches the control's non-flag value and is admitted;
    # a faked handler that hard-codes the flag passes solvable+not-trivial but fails the
    # control and is rejected.
    snap = _admit(loot, kind, **{key: ctx})
    good = _control_gate(snap, kind, faithful(snap.graph), tmp_path / "good")
    assert good.accepted and good.faithful, f"{kind} faithful rejected: {good.reason}"
    bad = _control_gate(snap, kind, _faked(snap.graph, kind, benign), tmp_path / "bad")
    assert bad.solvable and not bad.trivial, (
        f"{kind} faked should pass solvable+nontrivial"
    )
    assert not bad.accepted and not bad.faithful, f"{kind} faked admitted: {bad.reason}"


def test_control_request_per_class() -> None:
    # The six compute/decoy-read classes get a control; the gate-y classes + ssrf don't.
    for loot, kind in [
        ("file", "command_injection"),
        ("db", "sql_injection"),
        ("file", "path_traversal"),
        ("file", "xxe"),
        ("file", "ssti"),
        ("db", "idor"),
    ]:
        c = control_request(_admit(loot, kind).graph, kind)
        assert c is not None and c.expected
        # The control carries a payload: a GET query, or a body-shaped POST body.
        assert c.request.body or "?" in c.request.path
    for loot, kind in [
        ("db", "broken_authz"),
        ("db", "weak_credentials"),
        ("db", "ssrf"),
    ]:
        assert control_request(_admit(loot, kind).graph, kind) is None


@pytest.mark.parametrize("sink", ["attribute", "comment", "expr"])
def test_control_request_ssti_each_sink(sink: str) -> None:
    c = control_request(_admit("file", "ssti", render_sink=sink).graph, "ssti")
    assert c is not None and c.expected == "41897569"


def test_control_request_none_without_a_decoy() -> None:
    # A file-loot class with a flag record but no decoy file -> no control.
    graph = WorldGraph(ontology=ONTOLOGY_ID)
    graph.add_node(
        Node(id="rec", kind="record", attrs={"key": "/f", "fields": {"value": "x"}})
    )
    graph.add_node(
        Node(
            id="secret_flag",
            kind="secret",
            attrs={"kind": "flag", "value_ref": "FLAGVALUE"},
        )
    )
    graph.add_edge(Edge(id="h", kind="holds", src="rec", dst="secret_flag", attrs={}))
    graph.add_node(Node(id="ep", kind="endpoint", attrs={"public_url": "/x"}))
    graph.add_node(
        Node(
            id="v",
            kind="vulnerability",
            attrs={
                "kind": "command_injection",
                "params": {"target_param": "q", "inj_context": "separator"},
            },
        )
    )
    graph.add_edge(Edge(id="a", kind="affects", src="v", dst="ep", attrs={}))
    assert control_request(graph, "command_injection") is None


def test_control_request_none_when_params_not_mapping() -> None:
    graph = WorldGraph(ontology=ONTOLOGY_ID)
    graph.add_node(
        Node(
            id="v",
            kind="vulnerability",
            attrs={"kind": "command_injection", "params": "not-a-map"},
        )
    )
    assert control_request(graph, "command_injection") is None


def _service_with_benign_surface(graph: WorldGraph) -> str:
    # The service exposing the most benign (non-vuln) endpoints — the richest surface.
    return max(
        (s.id for s in graph.by_kind("service")),
        key=lambda sid: len(benign_endpoints_of(graph, sid)),
    )


def _benign_body() -> str:
    return "\n".join(
        [
            "def handle(query, state):",
            "    del query, state",
            "    return 200, {'Content-Type': 'application/json'}, b'{\"items\": []}'",
        ]
    )


def _leaky_files_body() -> str:
    # The sharp risk: a 'realistic' benign endpoint that dumps the shared file store,
    # serving the flag with no exploit.
    return "\n".join(
        [
            "def handle(query, state):",
            "    import json",
            "    return 200, {}, json.dumps(state.get('files', {})).encode()",
        ]
    )


def _service_probes(
    snap: Snapshot, oracle_kind: str, base_dir: Path
) -> Callable[[str], tuple[str, str, dict[str, str], bool]]:
    counter = iter(range(1000))
    task = next(t for t in snap.tasks if t.meta.get("family") == "webapp.pentest")

    def run(service_id: str) -> tuple[str, str, dict[str, str], bool]:
        svc = EpisodeService(WebappPack(), base_dir / f"s{next(counter)}")
        try:
            handle = svc.start_episode(snap, task.id)
            base = str(svc.surface(handle)["base_url"])
            ex_req, bn_req = exploit_and_benign(snap.graph, oracle_kind)
            bodies = {
                str(ep.attrs.get("path")): _fetch(base + str(ep.attrs["public_url"]))
                for ep in benign_endpoints_of(snap.graph, service_id)
            }
            try:
                with urllib.request.urlopen(base + "/", timeout=15) as resp:
                    root_ok = resp.status == 200
            except urllib.error.HTTPError:
                root_ok = False
            return perform(base, ex_req), perform(base, bn_req), bodies, root_ok
        finally:
            svc.close()

    return run


def test_realize_service_surface_admits_benign_and_rejects_leaky(
    tmp_path: Path,
) -> None:
    snap = _admit("file", "command_injection")
    sid = _service_with_benign_surface(snap.graph)
    paths = [str(ep.attrs.get("path")) for ep in benign_endpoints_of(snap.graph, sid)]
    assert paths  # the public service has a benign surface to realize
    before = snap.graph.content_hash()
    out = realize_service_surface(
        snap,
        sid,
        lambda _g, _s: {p: _benign_body() for p in paths},
        _service_probes(snap, "command_injection", tmp_path / "ok"),
    )
    assert set(out.lineage["realized_endpoints"]) == set(paths)
    assert out.snapshot_id != before  # re-frozen with the realized benign surface

    # The sharp risk: one benign endpoint dumps the file store -> the whole service
    # realization is rejected, all-or-nothing, and nothing is left realized.
    snap2 = _admit("file", "command_injection")
    sid2 = _service_with_benign_surface(snap2.graph)
    paths2 = [
        str(ep.attrs.get("path")) for ep in benign_endpoints_of(snap2.graph, sid2)
    ]
    before2 = snap2.graph.content_hash()
    leaky = {paths2[0]: _leaky_files_body()}
    leaky.update({p: _benign_body() for p in paths2[1:]})
    out2 = realize_service_surface(
        snap2,
        sid2,
        lambda _g, _s: leaky,
        _service_probes(snap2, "command_injection", tmp_path / "bad"),
    )
    assert out2.lineage["realized_endpoints"] == ()
    assert out2.snapshot_id == before2  # all-or-nothing: graph unchanged
    assert all(
        "realized_handler" not in ep.attrs
        for ep in benign_endpoints_of(snap2.graph, sid2)
    )


def test_classify_service_admission_reasons() -> None:
    graph = _admit("file", "command_injection").graph
    flag = str(graph.nodes["secret_flag"].attrs["value_ref"])
    ok = classify_service_admission(
        graph,
        oracle_exploit_body=flag,
        oracle_benign_body="clean",
        benign_endpoint_bodies={"/a": "clean"},
        root_ok=True,
    )
    assert ok.accepted
    leak = classify_service_admission(
        graph,
        oracle_exploit_body=flag,
        oracle_benign_body="clean",
        benign_endpoint_bodies={"/a": flag},
        root_ok=True,
    )
    assert not leak.accepted and "leak" in leak.reason and "/a" in leak.reason
    unsolv = classify_service_admission(
        graph,
        oracle_exploit_body="clean",
        oracle_benign_body="clean",
        benign_endpoint_bodies={},
        root_ok=True,
    )
    assert not unsolv.accepted and "solvable" in unsolv.reason
    triv = classify_service_admission(
        graph,
        oracle_exploit_body=flag,
        oracle_benign_body=flag,
        benign_endpoint_bodies={},
        root_ok=True,
    )
    assert not triv.accepted and "trivial" in triv.reason
    noboot = classify_service_admission(
        graph,
        oracle_exploit_body=flag,
        oracle_benign_body="clean",
        benign_endpoint_bodies={},
        root_ok=False,
    )
    assert not noboot.accepted and "boot" in noboot.reason


def test_benign_endpoints_of() -> None:
    graph = _admit("db", "sql_injection").graph
    sid = _service_with_benign_surface(graph)
    eps = benign_endpoints_of(graph, sid)
    assert eps  # the public service exposes benign endpoints
    vuln_eps = {
        e.dst
        for v in graph.by_kind("vulnerability")
        for e in graph.out_edges(v.id, "affects")
    }
    assert all(ep.id not in vuln_eps for ep in eps)  # never the vuln endpoint


def test_service_realization_request_and_parse() -> None:
    graph = _admit("db", "sql_injection").graph
    sid = _service_with_benign_surface(graph)
    req = service_realization_request(graph, sid)
    assert "def handle" in req.prompt and "do NOT read" in req.prompt
    parsed = service_handlers_from_result(
        {"endpoints": {"/x": "def handle(): ...", "/y": 5, "/z": "  "}}
    )
    assert parsed == {"/x": "def handle(): ..."}
    assert service_handlers_from_result(None) == {}
    assert service_handlers_from_result({"endpoints": "nope"}) == {}


def test_realize_service_surface_skips_malformed_and_empty(tmp_path: Path) -> None:
    snap = _admit("file", "command_injection")
    sid = _service_with_benign_surface(snap.graph)
    before = snap.graph.content_hash()
    paths = [str(ep.attrs.get("path")) for ep in benign_endpoints_of(snap.graph, sid)]
    # A proposal that is not valid handler source is skipped; nothing is realized.
    out = realize_service_surface(
        snap,
        sid,
        lambda _g, _s: {paths[0]: "not python @@@"},
        _service_probes(snap, "command_injection", tmp_path / "x"),
    )
    assert out.lineage["realized_endpoints"] == ()
    assert out.snapshot_id == before


def test_benign_endpoints_of_edge_cases() -> None:
    # A service-level vuln owns every endpoint -> no benign surface.
    g = WorldGraph(ontology=ONTOLOGY_ID)
    g.add_node(Node(id="s", kind="service", attrs={"name": "s"}))
    g.add_node(
        Node(id="api", kind="endpoint", attrs={"path": "/api", "public_url": "/api"})
    )
    g.add_edge(Edge(id="e", kind="exposes", src="s", dst="api", attrs={}))
    g.add_node(Node(id="v", kind="vulnerability", attrs={"kind": "x", "params": {}}))
    g.add_edge(Edge(id="av", kind="affects", src="v", dst="s", attrs={}))
    assert benign_endpoints_of(g, "s") == []

    # Endpoint-level only: / and /openapi.json are framework routes, excluded.
    g2 = WorldGraph(ontology=ONTOLOGY_ID)
    g2.add_node(Node(id="s", kind="service", attrs={"name": "s"}))
    for eid, p in [("root", "/"), ("oa", "/openapi.json"), ("list", "/list")]:
        g2.add_node(Node(id=eid, kind="endpoint", attrs={"path": p, "public_url": p}))
        g2.add_edge(Edge(id="x" + eid, kind="exposes", src="s", dst=eid, attrs={}))
    assert [str(e.attrs["path"]) for e in benign_endpoints_of(g2, "s")] == ["/list"]


def test_realize_service_surface_ignores_unknown_path(tmp_path: Path) -> None:
    snap = _admit("file", "command_injection")
    sid = _service_with_benign_surface(snap.graph)
    before = snap.graph.content_hash()
    out = realize_service_surface(
        snap,
        sid,
        lambda _g, _s: {"/not-an-endpoint": _benign_body()},
        _service_probes(snap, "command_injection", tmp_path / "u"),
    )
    assert out.lineage["realized_endpoints"] == ()
    assert out.snapshot_id == before


def test_db_decoy_matches_seeding() -> None:
    # The idor control's decoy must stay a real seeded row, or it silently breaks.
    from cyber_webapp.codegen.seeding import _DECOY_ROWS
    from cyber_webapp.reference_solver import _DB_DECOY

    assert _DB_DECOY in _DECOY_ROWS


def test_classify_with_control_distinguishes_reasons() -> None:
    graph = _admit("db", "sql_injection", context="single").graph
    flag = str(graph.nodes["secret_flag"].attrs["value_ref"])
    # exploit leaks, benign clean, control computes -> accepted.
    ok = classify_admission_with_control(graph, flag, "clean", "41897569", "41897569")
    assert ok.accepted and ok.faithful
    # benign also leaks -> trivial (even with a passing control).
    triv = classify_admission_with_control(graph, flag, flag, "41897569", "41897569")
    assert not triv.accepted and triv.trivial and "trivial" in triv.reason
    # exploit leaks, benign clean, but the control did not compute -> not faithful.
    faked = classify_admission_with_control(graph, flag, "clean", "nope", "41897569")
    assert not faked.accepted and not faked.faithful and "control" in faked.reason
    # the exploit itself did not leak -> not solvable.
    unsolv = classify_admission_with_control(
        graph, "clean", "clean", "41897569", "41897569"
    )
    assert not unsolv.accepted and not unsolv.solvable
