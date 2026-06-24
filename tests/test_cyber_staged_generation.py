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
from typing import Any

import pytest
from cyber_webapp import WebappBuilder, WebappPack
from cyber_webapp.codegen import _realize_graph
from cyber_webapp.consequence import LeakVerdict, detect_leak, guarded_values
from cyber_webapp.difficulty import _oracle_vuln
from cyber_webapp.reference_solver import (
    _broken_authz_query,
    _cmdi_payload,
    _flag_record_key,
    _idor_id,
    _path_payload,
    _request,
    _sqli_payload,
    _ssrf_url,
    _ssti_payload,
    _weak_cred_query,
    _xxe_payload,
    exploit_and_benign,
)
from cyber_webapp.verify import perform
from cyber_webapp.vulnerabilities import CATALOG
from graphschema import Node, WorldGraph
from openrange_pack_sdk import PackError, Snapshot

from openrange.core.admit import admit
from openrange.core.episode import EpisodeService


def _manifest(loot: str, seed: int = 7, **extra: object) -> dict[str, object]:
    return {
        "pack": {"id": "webapp"},
        "runtime": {"tick": {"mode": "off"}},
        "npc": [],
        "seed": seed,
        "loot": {loot: 1, "db" if loot == "file" else "file": 0},
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


def _oracle_shape(graph: WorldGraph) -> str:
    # The shape of the flag-READING exploit specifically — not every vuln in the world.
    # A db-loot world may still carry an off-path file_read decoy (it reads the empty
    # file map and leaks nothing); the loot shape only constrains the oracle.
    oracle = _oracle_vuln(graph)
    kind = str(oracle.attrs.get("kind", "")) if oracle is not None else ""
    return CATALOG[kind].shape if kind in CATALOG else ""


def test_file_loot_admits_and_forces_file_read_oracle() -> None:
    snap = _admit("file")
    assert "file" in _store_kinds(snap.graph)
    # File loot forces a file-store exploit (read or exec) as the oracle.
    assert _oracle_shape(snap.graph) in {"file_read", "code_exec"}


def test_db_loot_admits_and_forces_response_leak_oracle() -> None:
    snap = _admit("db")
    assert "kv" in _store_kinds(snap.graph)
    assert "file" not in _store_kinds(snap.graph)
    # No db world has a file store, so the flag-reading oracle can't be a file exploit
    # (an off-path file_read decoy is allowed — it reads the empty file map and leaks
    # nothing).
    assert _oracle_shape(snap.graph) not in {"file_read", "code_exec"}


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


def test_a_malformed_loot_knob_is_rejected() -> None:
    with pytest.raises(PackError):
        WebappBuilder(None)._effective_prior(
            {"pack": {"id": "webapp"}, "npc": [], "loot": "not-a-mapping"}
        )


def test_degenerate_loot_weights_fall_back_to_db() -> None:
    # All-zero weights leave an empty pool, which resolves to db.
    snap = admit(
        WebappPack(),
        manifest={
            "pack": {"id": "webapp"},
            "seed": 7,
            "runtime": {"tick": {"mode": "off"}},
            "npc": [],
            "loot": {"db": 0, "file": 0},
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


def _exploit_body(kind: str, graph: WorldGraph, base: str) -> str:
    # The pack's reference solver owns the recipe; the staged tests prove it leaks.
    exploit, _benign = exploit_and_benign(graph, kind)
    return perform(base, exploit)


def _send(base: str, graph: WorldGraph, vuln: Node, value: str) -> str:
    # Deliver a custom payload to the vuln's endpoint in that endpoint's request shape
    # (a body-shaped class is POST), so a probe reaches the handler, not a 405.
    params = vuln.attrs["params"]
    assert isinstance(params, Mapping)
    endpoint = graph.nodes[next(e.dst for e in graph.out_edges(vuln.id, "affects"))]
    ep = str(endpoint.attrs["public_url"])
    method = str(endpoint.attrs.get("method", "GET"))
    return perform(base, _request(ep, method, str(params["target_param"]), value))


_CLASS_CASES = [
    ("file", "path_traversal"),
    ("file", "command_injection"),
    ("file", "xxe"),
    ("file", "ssti"),
    ("db", "sql_injection"),
    ("db", "ssrf"),
    ("db", "broken_authz"),
    ("db", "idor"),
    ("db", "weak_credentials"),
]


@pytest.mark.parametrize(("loot", "kind"), _CLASS_CASES)
def test_fanout_class_episode_recovers_flag(
    loot: str, kind: str, tmp_path: Path
) -> None:
    # Each class is forced as the oracle and solved by its own
    # context-appropriate exploit; the recovered body must contain the flag.
    snap = _admit(loot, vuln={"pin": [{"kind": kind}]})
    expected = str(snap.graph.nodes["secret_flag"].attrs["value_ref"])
    pentest = next(t for t in snap.tasks if t.meta.get("family") == "webapp.pentest")

    svc = EpisodeService(WebappPack(), tmp_path)
    try:
        handle = svc.start_episode(snap, pentest.id)
        base = svc.surface(handle)["base_url"]
        recovered = _exploit_body(kind, snap.graph, base)
        assert expected in recovered, recovered[:120]
        (svc.solver_root(handle) / "result.json").write_text(
            json.dumps({"flag": expected})
        )
        report = svc.stop_episode(handle)
    finally:
        svc.close()
    assert report.passed, report.episode_result.reason


def _exploit_response_body(snap: Snapshot, kind: str, tmp_path: Path) -> str:
    pentest = next(t for t in snap.tasks if t.meta.get("family") == "webapp.pentest")
    svc = EpisodeService(WebappPack(), tmp_path)
    try:
        handle = svc.start_episode(snap, pentest.id)
        base = svc.surface(handle)["base_url"]
        body: str = _exploit_body(kind, snap.graph, base)
        return body
    finally:
        svc.close()


@pytest.mark.parametrize(("loot", "kind"), _CLASS_CASES)
def test_consequence_verifier_detects_real_flag_leak(
    loot: str, kind: str, tmp_path: Path
) -> None:
    # The independent leak oracle must fire on every faithful exploit and name the
    # flag as the leaked node — agreeing with the planted-flag verdict it generalizes.
    snap = _admit(loot, vuln={"pin": [{"kind": kind}]})
    body = _exploit_response_body(snap, kind, tmp_path)
    verdict = detect_leak(snap.graph, [body])
    assert verdict.occurred
    assert "secret_flag" in verdict.leaked


def test_consequence_verifier_silent_on_benign_response(tmp_path: Path) -> None:
    # A benign request leaks nothing, so the oracle stays silent: no false positive.
    snap = _admit("file", vuln={"pin": [{"kind": "command_injection"}]})
    vuln = next(
        n
        for n in snap.graph.by_kind("vulnerability")
        if n.attrs.get("kind") == "command_injection"
    )
    pentest = next(t for t in snap.tasks if t.meta.get("family") == "webapp.pentest")

    svc = EpisodeService(WebappPack(), tmp_path)
    try:
        handle = svc.start_episode(snap, pentest.id)
        base = svc.surface(handle)["base_url"]
        body = _send(base, snap.graph, vuln, "8.8.8.8")
    finally:
        svc.close()
    assert not detect_leak(snap.graph, [body]).occurred


def test_consequence_verifier_silent_on_neutralized_attack(tmp_path: Path) -> None:
    # An attack-shaped attempt the sampled context neutralizes recovers no flag,
    # so the oracle must stay silent on it too — not only on benign traffic.
    snap = _admit("file", vuln={"pin": [{"kind": "command_injection"}]})
    vuln = next(
        n
        for n in snap.graph.by_kind("vulnerability")
        if n.attrs.get("kind") == "command_injection"
    )
    params = vuln.attrs["params"]
    assert isinstance(params, Mapping)
    flag_path = _flag_record_key(snap.graph)
    # The injection vector the sampled context strips — the wrong technique.
    if params.get("inj_context") == "substitution":
        wrong = f"x; cat {flag_path}"
    else:
        wrong = f"$(cat {flag_path})"
    pentest = next(t for t in snap.tasks if t.meta.get("family") == "webapp.pentest")

    svc = EpisodeService(WebappPack(), tmp_path)
    try:
        handle = svc.start_episode(snap, pentest.id)
        base = svc.surface(handle)["base_url"]
        body = _send(base, snap.graph, vuln, wrong)
    finally:
        svc.close()
    assert not detect_leak(snap.graph, [body]).occurred


def test_guarded_values_selects_only_hidden_value_refs() -> None:
    from cyber_webapp.ontology import ONTOLOGY_ID
    from graphschema import Node, Visibility

    graph = WorldGraph(ontology=ONTOLOGY_ID)
    # Guarded: HIDDEN with a long-enough string value_ref.
    graph.add_node(
        Node(
            id="flag",
            kind="secret",
            attrs={"value_ref": "FLAGVALUE"},
            visibility=Visibility.HIDDEN,
        )
    )
    # Not guarded: PUBLIC is the observable surface, even carrying a value_ref.
    graph.add_node(Node(id="pub", kind="credential", attrs={"value_ref": "PUBVALUE0"}))
    # Not guarded: HIDDEN but no usable value_ref — missing, non-str, or below the
    # length floor (a short value would substring-match benign text).
    graph.add_node(
        Node(
            id="novref",
            kind="vulnerability",
            attrs={"kind": "ssti"},
            visibility=Visibility.HIDDEN,
        )
    )
    graph.add_node(
        Node(
            id="nonstr",
            kind="secret",
            attrs={"value_ref": 123},
            visibility=Visibility.HIDDEN,
        )
    )
    graph.add_node(
        Node(
            id="short",
            kind="secret",
            attrs={"value_ref": "tiny"},
            visibility=Visibility.HIDDEN,
        )
    )
    assert guarded_values(graph) == {"flag": "FLAGVALUE"}


def test_detect_leak_reports_leaked_nodes_and_handles_empties() -> None:
    from cyber_webapp.ontology import ONTOLOGY_ID
    from graphschema import Node, Visibility

    graph = WorldGraph(ontology=ONTOLOGY_ID)
    graph.add_node(
        Node(
            id="flag",
            kind="secret",
            attrs={"value_ref": "FLAGVALUE"},
            visibility=Visibility.HIDDEN,
        )
    )
    hit = detect_leak(graph, ["nothing here", "prefix-FLAGVALUE-suffix"])
    assert hit.occurred
    assert hit.leaked == frozenset({"flag"})
    assert not detect_leak(graph, ["clean", "also clean"]).occurred
    assert not detect_leak(graph, []).occurred
    # No guarded values at all -> silent via the early return.
    assert not detect_leak(WorldGraph(ontology=ONTOLOGY_ID), ["FLAGVALUE"]).occurred
    assert LeakVerdict(frozenset()).occurred is False


def test_detect_leak_catches_encoded_exfil() -> None:
    import base64
    import urllib.parse as _url

    from cyber_webapp.consequence import value_variants
    from cyber_webapp.ontology import ONTOLOGY_ID
    from graphschema import Node, Visibility

    flag = "ORANGE{s3cr3t/v}"  # has {, /, } so the url-encoded form differs from raw
    graph = WorldGraph(ontology=ONTOLOGY_ID)
    graph.add_node(
        Node(
            id="flag",
            kind="secret",
            attrs={"value_ref": flag},
            visibility=Visibility.HIDDEN,
        )
    )
    b64 = base64.b64encode(flag.encode()).decode()
    encoded = (
        f'{{"v":"{b64}"}}',
        f"hex={flag.encode().hex()}",
        _url.quote(flag, safe=""),
    )
    for body in encoded:
        assert detect_leak(graph, [body]).occurred, body
    assert detect_leak(graph, [f"...{flag}..."]).occurred  # literal still caught
    assert not detect_leak(graph, ["unrelated benign text only"]).occurred
    assert {flag, b64}.issubset(value_variants(flag))


def test_detect_leak_drops_contained_values() -> None:
    from cyber_webapp.ontology import ONTOLOGY_ID
    from graphschema import Node, Visibility

    graph = WorldGraph(ontology=ONTOLOGY_ID)
    graph.add_node(
        Node(
            id="short",
            kind="secret",
            attrs={"value_ref": "SHORTSECRET"},
            visibility=Visibility.HIDDEN,
        )
    )
    graph.add_node(
        Node(
            id="long",
            kind="secret",
            attrs={"value_ref": "SHORTSECRET_EXTENDED"},
            visibility=Visibility.HIDDEN,
        )
    )
    # Leaking only the longer value reports just it — the shorter is a substring
    # artifact, not a distinct leak.
    long_only = detect_leak(graph, ["x SHORTSECRET_EXTENDED y"]).leaked
    assert long_only == frozenset({"long"})
    # Leaking the shorter value on its own still reports it.
    assert detect_leak(graph, ["x SHORTSECRET y"]).leaked == frozenset({"short"})


def test_rendered_app_scanner_agrees_with_consequence() -> None:
    import base64
    import urllib.parse as _url

    snap = _admit("db", vuln={"pin": [{"kind": "sql_injection"}]})
    graph = snap.graph
    flag = str(graph.nodes["secret_flag"].attrs["value_ref"])
    namespace: dict[str, Any] = {}
    exec(compile(_realize_graph(graph)["app.py"], "<app>", "exec"), namespace)
    app_scan = namespace["_scan_leaks"]
    guarded = dict(guarded_values(graph))
    b64 = base64.b64encode(flag.encode()).decode()
    bodies = [
        flag,
        f"x{b64}y",
        f"h={flag.encode().hex()}",
        f"q={_url.quote(flag, safe='')}",
        "clean nothing here",
    ]
    for body in bodies:
        assert app_scan(body.encode(), guarded) == sorted(
            detect_leak(graph, [body]).leaked
        ), body


def test_seed_carries_guarded_map_from_graph() -> None:
    # Convergence: the values the runtime watches for are exactly what the offline
    # verifier reads — both come from consequence.guarded_values(graph).
    snap = _admit("db", vuln={"pin": [{"kind": "sql_injection"}]})
    seed = json.loads(_realize_graph(snap.graph)["seed.json"])
    assert seed["guarded"] == dict(guarded_values(snap.graph))
    assert seed["guarded"]["secret_flag"] == str(
        snap.graph.nodes["secret_flag"].attrs["value_ref"]
    )


def test_rendered_app_scans_leaks_to_node_ids_not_values() -> None:
    # Integrity: the app records WHICH guarded node leaked (its id), never the secret
    # value — so the request log can't be grepped for the flag.
    snap = _admit("db", vuln={"pin": [{"kind": "sql_injection"}]})
    namespace: dict[str, Any] = {}
    exec(compile(_realize_graph(snap.graph)["app.py"], "<app>", "exec"), namespace)
    scan = namespace["_scan_leaks"]
    assert scan(b"prefix SECRETVAL suffix", {"secret_flag": "SECRETVAL"}) == [
        "secret_flag"
    ]
    assert scan("plain-str SECRETVAL", {"secret_flag": "SECRETVAL"}) == ["secret_flag"]
    assert scan(b"nothing in here", {"secret_flag": "SECRETVAL"}) == []


def test_live_episode_records_the_flag_leak_as_a_node_id(tmp_path: Path) -> None:
    # End to end: a real exploit makes the flag cross the response boundary, and the
    # runtime records that leak on final_state by node id.
    snap = _admit("db", vuln={"pin": [{"kind": "sql_injection"}]})
    pentest = next(t for t in snap.tasks if t.meta.get("family") == "webapp.pentest")
    svc = EpisodeService(WebappPack(), tmp_path)
    try:
        handle = svc.start_episode(snap, pentest.id)
        base = svc.surface(handle)["base_url"]
        _exploit_body("sql_injection", snap.graph, base)
        report = svc.stop_episode(handle)
    finally:
        svc.close()
    leaked = report.final_state.get("leaked_secret_ids") or []
    assert "secret_flag" in leaked


def test_check_success_surfaces_leak_without_changing_the_reward_rungs() -> None:
    from cyber_webapp.families.pentest import WebappPentest

    snap = _admit("db", vuln={"pin": [{"kind": "sql_injection"}]})
    pentest = next(t for t in snap.tasks if t.meta.get("family") == "webapp.pentest")
    fam = WebappPentest()
    base = {"requests_made": ["/svc"], "flag_from_response": "wrong"}

    leaked = fam.check_success(
        snap.graph, pentest, {**base, "leaked_secret_ids": ["secret_flag"]}
    )
    quiet = fam.check_success(snap.graph, pentest, base)
    assert "leaked" in leaked.reason and "leaked" not in quiet.reason
    assert leaked.success is False and quiet.success is False
    # The averaged subgoal set is unchanged, so the trainer's rungs are intact.
    assert set(leaked.subgoals) == {
        "reached_endpoint",
        "extracted_anything",
        "matched_flag",
    }


def test_flag_path_is_discoverable_via_config(tmp_path: Path) -> None:
    # The flag location is recon, not a blind guess: a config at a conventional
    # path discloses the flag path, which the same vuln then reads.
    import re

    snap = _admit("file", vuln={"pin": [{"kind": "path_traversal"}]})
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
    snap = _admit("file", vuln={"pin": [{"kind": "path_traversal"}]})
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
    snap = _admit("file", vuln={"pin": [{"kind": "command_injection"}]})
    vuln = next(
        n
        for n in snap.graph.by_kind("vulnerability")
        if n.attrs.get("kind") == "command_injection"
    )
    params = vuln.attrs["params"]
    assert isinstance(params, Mapping)
    flag_path = _flag_record_key(snap.graph)
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
        benign = _send(base, snap.graph, vuln, "8.8.8.8")
        rejected = _send(base, snap.graph, vuln, wrong)
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
        snap = _admit(loot, vuln={"pin": [{"kind": kind}]}, instruction_tier="easy")
        task = next(t for t in snap.tasks if t.meta.get("family") == "webapp.pentest")
        assert task.meta.get("tier") == "easy"
        assert label in task.instruction.lower(), (kind, task.instruction)
        assert "`" in task.instruction  # a concrete payload/recipe code span


def test_standard_tier_stays_thin_and_aliases_map_to_easy() -> None:
    std = _admit("db", vuln={"pin": [{"kind": "sql_injection"}]})  # default = standard
    task = next(t for t in std.tasks if t.meta.get("family") == "webapp.pentest")
    assert task.meta.get("tier") == "standard"
    assert "guided" not in task.instruction.lower()
    for alias in ("guided", "bootstrap", "tutorial"):
        snap = _admit(
            "db", vuln={"pin": [{"kind": "sql_injection"}]}, instruction_tier=alias
        )
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
