from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path

import yaml
from jinja2 import Environment, FileSystemLoader, StrictUndefined, select_autoescape

VULN_TEMPLATES_DIR = Path(__file__).parent / "templates"


@dataclass(frozen=True, slots=True)
class Vulnerability:
    # ``attrs_schema`` is documentation only — not validated against
    # the graph's vulnerability node attrs.

    id: str
    family: str
    description: str
    target_kinds: frozenset[str]
    template: str
    requires: frozenset[str] = frozenset()
    enables: frozenset[str] = frozenset()
    attrs_schema: Mapping[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "family": self.family,
            "description": self.description,
            "target_kinds": sorted(self.target_kinds),
            "template": self.template,
            "requires": sorted(self.requires),
            "enables": sorted(self.enables),
            "attrs_schema": dict(self.attrs_schema),
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> Vulnerability:
        target_kinds_raw = data.get("target_kinds", ())
        requires_raw = data.get("requires", ())
        enables_raw = data.get("enables", ())
        attrs_raw = data.get("attrs_schema", {})
        if not isinstance(target_kinds_raw, list | tuple | frozenset | set):
            raise ValueError("target_kinds must be a sequence")
        if not isinstance(requires_raw, list | tuple | frozenset | set):
            raise ValueError("requires must be a sequence")
        if not isinstance(enables_raw, list | tuple | frozenset | set):
            raise ValueError("enables must be a sequence")
        if not isinstance(attrs_raw, Mapping):
            raise ValueError("attrs_schema must be a mapping")
        return cls(
            id=str(data["id"]),
            family=str(data["family"]),
            description=str(data.get("description", "")),
            target_kinds=frozenset(str(k) for k in target_kinds_raw),
            template=str(data["template"]),
            requires=frozenset(str(k) for k in requires_raw),
            enables=frozenset(str(k) for k in enables_raw),
            attrs_schema={str(k): str(v) for k, v in attrs_raw.items()},
        )


SQL_INJECTION = Vulnerability(
    id="sql_injection",
    family="code_web",
    description=(
        "Endpoint that interpolates an unparameterized query parameter "
        "into a SQL statement, allowing exfiltration via UNION SELECT."
    ),
    target_kinds=frozenset({"endpoint"}),
    template="sql_injection.py.j2",
    enables=frozenset({"data_store_dump"}),
    attrs_schema={
        "target_param": "name of the query parameter that flows into SQL",
        "table": "table the vulnerable query reads from",
        "leak_column": "column to leak via UNION SELECT",
    },
)

SSRF = Vulnerability(
    id="ssrf",
    family="code_web",
    description=(
        "Endpoint that fetches a URL supplied by the client without "
        "filtering destination — agent can reach internal services."
    ),
    target_kinds=frozenset({"endpoint"}),
    template="ssrf.py.j2",
    enables=frozenset({"broken_authz", "metadata_credential_leak"}),
    attrs_schema={
        "target_param": "name of the query parameter holding the URL",
        "allowlist_pattern": "regex for allowed hosts (the bug is that it's "
        "checked AFTER the fetch, or not at all)",
    },
)

BROKEN_AUTHZ = Vulnerability(
    id="broken_authz",
    family="code_web",
    description=(
        "Endpoint trusts a client-controlled header for the user's role "
        "without verifying provenance — agent can forge admin access."
    ),
    target_kinds=frozenset({"endpoint"}),
    template="broken_authz.py.j2",
    requires=frozenset(),  # primary; chains often start here or via SSRF
    attrs_schema={
        "trust_header": "HTTP header name the endpoint trusts (e.g. X-User-Role)",
        "expected_value": "value that grants admin access (e.g. 'admin')",
        "leak_field": "field of the response that exposes the secret",
    },
)


CATALOG: Mapping[str, Vulnerability] = {
    SQL_INJECTION.id: SQL_INJECTION,
    SSRF.id: SSRF,
    BROKEN_AUTHZ.id: BROKEN_AUTHZ,
}


def vuln(id_: str) -> Vulnerability:
    """Look up a vuln by id; raises KeyError on miss."""
    return CATALOG[id_]


def vulns_for_kind(kind: str) -> tuple[Vulnerability, ...]:
    """Return all catalog entries that target the given graph node kind."""
    return tuple(v for v in CATALOG.values() if kind in v.target_kinds)


def _jinja_env() -> Environment:
    """Build a Jinja2 environment scoped to the bundled templates dir.

    ``StrictUndefined`` raises on missing variables — callers must supply
    every parameter the template references. Autoescape is disabled because
    we're rendering Python source, not HTML.
    """
    return Environment(
        loader=FileSystemLoader(str(VULN_TEMPLATES_DIR)),
        undefined=StrictUndefined,
        autoescape=select_autoescape(disabled_extensions=("py",), default=False),
        keep_trailing_newline=True,
    )


def render_vulnerability(
    vulnerability: Vulnerability,
    params: Mapping[str, object],
) -> str:
    """Render the vulnerability's template with the given parameters.

    Returns a Python source string ready for the realizer to drop into a
    service module. Strict-undefined variables: a missing param fails fast.
    """
    template = _jinja_env().get_template(vulnerability.template)
    return template.render(vuln=vulnerability, **params)


def catalog_to_yaml(catalog: Mapping[str, Vulnerability] = CATALOG) -> str:
    """Serialize the catalog to a YAML string."""
    payload = [v.as_dict() for v in catalog.values()]
    return str(yaml.safe_dump(payload, sort_keys=False))


def catalog_from_yaml(text: str) -> dict[str, Vulnerability]:
    """Parse a YAML catalog into a Vulnerability dict.

    ``id`` collisions overwrite the bundled entry.
    """
    data = yaml.safe_load(text)
    if not isinstance(data, list):
        raise ValueError("catalog YAML must be a list of vulnerability mappings")
    result: dict[str, Vulnerability] = {}
    for entry in data:
        if not isinstance(entry, Mapping):
            raise ValueError("catalog entries must be mappings")
        v = Vulnerability.from_mapping(entry)
        result[v.id] = v
    return result


def merge_catalog(
    base: Mapping[str, Vulnerability],
    override: Mapping[str, Vulnerability],
) -> dict[str, Vulnerability]:
    """Return a new catalog with override entries taking precedence."""
    return {**base, **override}


__all__ = [
    "BROKEN_AUTHZ",
    "CATALOG",
    "SQL_INJECTION",
    "SSRF",
    "VULN_TEMPLATES_DIR",
    "Vulnerability",
    "catalog_from_yaml",
    "catalog_to_yaml",
    "merge_catalog",
    "render_vulnerability",
    "vuln",
    "vulns_for_kind",
]


# Avoid unused-import noise
_ = replace
