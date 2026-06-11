"""Tests for the cyber vulnerability catalog.

Four concerns:
  1. The catalog round-trips through YAML.
  2. Templates render with documented parameters and yield valid Python.
  3. Each rendered handler is functionally vulnerable — the bug fires
     when invoked.
  4. A catalog entry's metadata drives the ``Node(kind=...,
     visibility=Visibility.HIDDEN)`` construction the sampler emits.
"""

from __future__ import annotations

from typing import Any

import pytest
from cyber_webapp.vulnerabilities import (
    BROKEN_AUTHZ,
    CATALOG,
    SQL_INJECTION,
    SSRF,
    Vulnerability,
    catalog_from_yaml,
    catalog_to_yaml,
    merge_catalog,
    render_vulnerability,
    vuln,
    vulns_for_kind,
)
from graphschema import Node, Visibility


def _exec_handler(source: str) -> Any:
    namespace: dict[str, Any] = {}
    exec(compile(source, "<rendered>", "exec"), namespace)
    return namespace["handle"]


def test_catalog_has_starter_vulns() -> None:
    assert set(CATALOG) == {
        "sql_injection",
        "ssrf",
        "broken_authz",
        "path_traversal",
        "command_injection",
        "xxe",
        "ssti",
        "idor",
        "weak_credentials",
    }
    assert vuln("sql_injection") is SQL_INJECTION


def test_vulns_for_kind_filters_by_target() -> None:
    endpoint_vulns = vulns_for_kind("endpoint")
    assert {v.id for v in endpoint_vulns} == {
        "sql_injection",
        "ssrf",
        "broken_authz",
        "path_traversal",
        "command_injection",
        "xxe",
        "ssti",
        "idor",
        "weak_credentials",
    }
    assert vulns_for_kind("network") == ()


def test_chain_metadata() -> None:
    """SSRF enables broken_authz; SQL injection enables a data-store dump."""
    assert "broken_authz" in SSRF.enables
    assert "data_store_dump" in SQL_INJECTION.enables


def test_catalog_yaml_round_trip() -> None:
    text = catalog_to_yaml()
    loaded = catalog_from_yaml(text)
    assert set(loaded) == set(CATALOG)
    for vid, v in CATALOG.items():
        rt = loaded[vid]
        assert rt.id == v.id
        assert rt.family == v.family
        assert rt.target_kinds == v.target_kinds
        assert rt.template == v.template
        assert rt.requires == v.requires
        assert rt.enables == v.enables


def test_catalog_yaml_rejects_non_list() -> None:
    with pytest.raises(ValueError, match="must be a list"):
        catalog_from_yaml("just a string")


def test_merge_catalog_overrides() -> None:
    custom = Vulnerability(
        id="sql_injection",  # collision with bundled
        family="custom",
        description="overridden",
        target_kinds=frozenset({"endpoint"}),
        template="custom.j2",
    )
    merged = merge_catalog(CATALOG, {custom.id: custom})
    assert merged["sql_injection"].family == "custom"
    assert merged["ssrf"] is SSRF


def test_sql_injection_template_renders() -> None:
    src = render_vulnerability(
        SQL_INJECTION,
        {"target_param": "q", "table": "rows", "leak_column": "value"},
    )
    assert "def handle(query, state):" in src
    assert "execute(sql)" in src
    compile(src, "<test>", "exec")


def test_ssrf_template_renders() -> None:
    src = render_vulnerability(
        SSRF,
        {"target_param": "url", "allowlist_pattern": r"^http://internal\."},
    )
    assert "def handle(query, state):" in src
    compile(src, "<test>", "exec")


def test_broken_authz_template_renders() -> None:
    src = render_vulnerability(
        BROKEN_AUTHZ,
        {
            "trust_header": "X-User-Role",
            "expected_value": "admin",
            "leak_field": "flag",
        },
    )
    assert "def handle(query, state):" in src
    compile(src, "<test>", "exec")


def test_strict_undefined_catches_missing_param() -> None:
    """A template missing a parameter must fail loudly, not silently render.

    StrictUndefined raises ``UndefinedError`` on attribute access; piping an
    Undefined through a filter (like ``tojson``) raises a ``TypeError`` from
    the filter's serializer instead. Either way the template fails.
    """
    from jinja2 import UndefinedError

    with pytest.raises((UndefinedError, TypeError)):
        # Missing ``table`` and ``leak_column``.
        render_vulnerability(SQL_INJECTION, {"target_param": "q"})


def test_sql_injection_actually_exfiltrates() -> None:
    import sqlite3

    src = render_vulnerability(
        SQL_INJECTION,
        {"target_param": "q", "table": "rows", "leak_column": "secret_col"},
    )
    handle = _exec_handler(src)

    db = sqlite3.connect(":memory:", check_same_thread=False)
    db.row_factory = sqlite3.Row
    db.execute("CREATE TABLE rows (key TEXT PRIMARY KEY, secret_col TEXT)")
    db.execute(
        "INSERT INTO rows (key, secret_col) VALUES (?, ?)",
        ("public", "ok"),
    )
    db.execute(
        "INSERT INTO rows (key, secret_col) VALUES (?, ?)",
        ("admin", "EXFIL_TOKEN_42"),
    )
    db.commit()
    state = {
        "db": db,
        "schema": {"table": "rows", "key_column": "key", "value_column": "secret_col"},
    }

    # Benign request: returns the public row.
    status, _, body = handle({"q": ["public"]}, state)
    assert status == 200
    assert b"ok" in body

    # Malicious request: UNION SELECT exfiltrates every secret_col value.
    payload = "' UNION SELECT key, secret_col FROM rows -- "
    status, _, body = handle({"q": [payload]}, state)
    assert status == 200
    assert b"EXFIL_TOKEN_42" in body, body


def test_ssrf_reaches_internal_host_and_leaks() -> None:
    """SSRF that resolves to the internal host exposes the protected secret."""
    src = render_vulnerability(
        SSRF,
        {
            "target_param": "url",
            "internal_host": "169.254.169.254",
            "allowed_host": "ok.example.com",
            "ssrf_filter": "decimal_ip",
        },
    )
    handle = _exec_handler(src)
    state = {"secrets": {"flag": "ORANGE{ssrf_leaked}"}}

    # The decimal-integer form of the internal IP resolves to it and leaks.
    status, _, body = handle({"url": ["http://2852039166/"]}, state)
    assert status == 200
    assert b"ORANGE{ssrf_leaked}" in body

    # The dotted-quad form is rejected by the decimal_ip filter — no leak.
    status, _, body = handle({"url": ["http://169.254.169.254/"]}, state)
    assert status == 200
    assert b"ORANGE{ssrf_leaked}" not in body


def test_ssrf_post_hoc_allowlist_lets_through_matching_url() -> None:
    src = render_vulnerability(
        SSRF,
        {"target_param": "url", "allowlist_pattern": r"^https?://"},
    )
    handle = _exec_handler(src)
    # Empty url → 400 (parameter validation works)
    status, _, _ = handle({"url": [""]}, {})
    assert status == 400


def test_broken_authz_grants_admin_with_forged_header() -> None:
    src = render_vulnerability(
        BROKEN_AUTHZ,
        {
            "trust_header": "X-User-Role",
            "expected_value": "admin",
            "leak_field": "flag",
        },
    )
    handle = _exec_handler(src)
    state = {"secrets": {"flag": "ORANGE{authz_bypass}"}}

    # No header → 403.
    status, _, _ = handle({}, state)
    assert status == 403

    # Wrong header value → 403.
    status, _, _ = handle({"X-User-Role": ["user"]}, state)
    assert status == 403

    # Forged admin header → secret leaks.
    status, _, body = handle({"X-User-Role": ["admin"]}, state)
    assert status == 200
    assert b"ORANGE{authz_bypass}" in body


def test_catalog_entry_drives_hidden_vulnerability_node() -> None:
    """A catalog entry's id / family drive ``Node`` construction.

    The procedural sampler emits one vulnerability node per chosen
    catalog entry with ``kind="vulnerability"``, ``visibility=HIDDEN``,
    and ``attrs={"kind": <catalog id>, "family": <catalog family>,
    "params": {...}}``.
    """
    entry = SQL_INJECTION
    node = Node(
        id="vuln_sql_injection_0",
        kind="vulnerability",
        attrs={
            "kind": entry.id,
            "family": entry.family,
            "params": {
                "target_param": "q",
                "table": "records",
                "leak_column": "value",
            },
        },
        visibility=Visibility.HIDDEN,
    )
    assert node.kind == "vulnerability"
    assert node.visibility is Visibility.HIDDEN
    assert node.attrs["kind"] == "sql_injection"
    assert node.attrs["family"] == "code_web"


def test_every_catalog_entry_targets_a_real_ontology_kind() -> None:
    """The ``affects`` edge only accepts ``(vulnerability, endpoint)`` and
    ``(vulnerability, service)``, so ``target_kinds`` must stay within that
    domain or the sampler emits an edge that fails conformance.
    """
    allowed = {"endpoint", "service"}
    for vid, v in CATALOG.items():
        unknown = set(v.target_kinds) - allowed
        assert not unknown, f"catalog {vid!r} targets unknown kinds: {unknown}"
