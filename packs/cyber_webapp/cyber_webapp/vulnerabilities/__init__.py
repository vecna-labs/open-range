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
    # How the exploit reaches the flag; the loot stage picks a vuln whose
    # shape matches the placed loot.
    shape: str = "response_leak"
    # CVSS-style exploit difficulty (0 trivial .. 1 hard) the difficulty metric
    # reads to weight this class when it is the flag-reading exploit.
    exploit_complexity: float = 0.5
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
            "shape": self.shape,
            "exploit_complexity": self.exploit_complexity,
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
        complexity_raw = data.get("exploit_complexity", 0.5)
        if not isinstance(target_kinds_raw, list | tuple | frozenset | set):
            raise ValueError("target_kinds must be a sequence")
        if not isinstance(requires_raw, list | tuple | frozenset | set):
            raise ValueError("requires must be a sequence")
        if not isinstance(enables_raw, list | tuple | frozenset | set):
            raise ValueError("enables must be a sequence")
        if not isinstance(attrs_raw, Mapping):
            raise ValueError("attrs_schema must be a mapping")
        if not isinstance(complexity_raw, int | float):
            raise ValueError("exploit_complexity must be a number")
        return cls(
            id=str(data["id"]),
            family=str(data["family"]),
            description=str(data.get("description", "")),
            target_kinds=frozenset(str(k) for k in target_kinds_raw),
            template=str(data["template"]),
            shape=str(data.get("shape", "response_leak")),
            exploit_complexity=float(complexity_raw),
            requires=frozenset(str(k) for k in requires_raw),
            enables=frozenset(str(k) for k in enables_raw),
            attrs_schema={str(k): str(v) for k, v in attrs_raw.items()},
        )


SQL_INJECTION = Vulnerability(
    id="sql_injection",
    exploit_complexity=0.7,
    family="code_web",
    description=(
        "Endpoint that interpolates an unparameterized query parameter "
        "into a SQL statement, allowing exfiltration via UNION SELECT."
    ),
    target_kinds=frozenset({"endpoint"}),
    template="sql_injection.py.j2",
    shape="response_leak",
    enables=frozenset({"data_store_dump"}),
    attrs_schema={
        "target_param": "name of the query parameter that flows into SQL",
        "table": "table the vulnerable query reads from",
        "leak_column": "column to leak via UNION SELECT",
    },
)

SSRF = Vulnerability(
    id="ssrf",
    exploit_complexity=0.5,
    family="code_web",
    description=(
        "Endpoint that fetches a URL supplied by the client without "
        "filtering destination — agent can reach internal services."
    ),
    target_kinds=frozenset({"endpoint"}),
    template="ssrf.py.j2",
    shape="response_leak",
    enables=frozenset({"broken_authz", "metadata_credential_leak"}),
    attrs_schema={
        "target_param": "name of the query parameter holding the URL",
        "allowlist_pattern": "regex for allowed hosts (the bug is that it's "
        "checked AFTER the fetch, or not at all)",
    },
)

BROKEN_AUTHZ = Vulnerability(
    id="broken_authz",
    exploit_complexity=0.3,
    family="code_web",
    description=(
        "Endpoint trusts a client-controlled header for the user's role "
        "without verifying provenance — agent can forge admin access."
    ),
    target_kinds=frozenset({"endpoint"}),
    template="broken_authz.py.j2",
    shape="response_leak",
    requires=frozenset(),
    attrs_schema={
        "trust_header": "HTTP header name the endpoint trusts (e.g. X-User-Role)",
        "expected_value": "value that grants admin access (e.g. 'admin')",
        "leak_field": "field of the response that exposes the secret",
    },
)


PATH_TRAVERSAL = Vulnerability(
    id="path_traversal",
    exploit_complexity=0.5,
    family="code_web",
    description=(
        "Endpoint serves a file named by a client parameter, joining it onto "
        "a base directory without confining the result — '../' or an absolute "
        "path escapes to read any file in the store, including the flag."
    ),
    target_kinds=frozenset({"endpoint"}),
    template="path_traversal.py.j2",
    shape="file_read",
    attrs_schema={
        "target_param": "query parameter holding the requested file path",
        "base_dir": "directory the path is joined onto (the unenforced confinement)",
    },
)


COMMAND_INJECTION = Vulnerability(
    id="command_injection",
    exploit_complexity=0.8,
    family="code_web",
    description=(
        "Diagnostic endpoint concatenates a client parameter into a shell "
        "command without sanitizing it — shell metacharacters (';', '|') "
        "inject an extra command that reads the flag file."
    ),
    target_kinds=frozenset({"endpoint"}),
    template="command_injection.py.j2",
    shape="code_exec",
    attrs_schema={
        "target_param": "query parameter concatenated into the command",
        "base_command": "the diagnostic command the input is appended to",
    },
)


XXE = Vulnerability(
    id="xxe",
    exploit_complexity=0.7,
    family="code_web",
    description=(
        "Endpoint parses client XML with external entities enabled, so a "
        "SYSTEM entity referencing a file path is resolved and reflected — "
        "reading any file in the store, including the flag."
    ),
    target_kinds=frozenset({"endpoint"}),
    template="xxe.py.j2",
    shape="file_read",
    attrs_schema={
        "target_param": "query parameter holding the XML document",
    },
)


SSTI = Vulnerability(
    id="ssti",
    exploit_complexity=0.7,
    family="code_web",
    description=(
        "Endpoint renders a client parameter as a template, so a template "
        "expression like '{{ read(\"/path\") }}' is evaluated and reads the "
        "flag file from the store."
    ),
    target_kinds=frozenset({"endpoint"}),
    template="ssti.py.j2",
    shape="code_exec",
    attrs_schema={
        "target_param": "query parameter rendered as a template",
    },
)


IDOR = Vulnerability(
    id="idor",
    exploit_complexity=0.2,
    family="code_web",
    description=(
        "Endpoint returns a record by a client-supplied id with no ownership "
        "or authorization check — referencing the flag record's id leaks it."
    ),
    target_kinds=frozenset({"endpoint"}),
    template="idor.py.j2",
    shape="response_leak",
    attrs_schema={
        "target_param": "query parameter holding the object id",
    },
)


WEAK_CREDENTIALS = Vulnerability(
    id="weak_credentials",
    exploit_complexity=0.3,
    family="code_web",
    description=(
        "Authentication endpoint accepts a default/weak credential pair, so "
        "guessing it (e.g. admin/admin) returns the protected secret."
    ),
    target_kinds=frozenset({"endpoint"}),
    template="weak_credentials.py.j2",
    shape="response_leak",
    attrs_schema={
        "user_param": "query parameter holding the username",
        "password_param": "query parameter holding the password",
        "weak_user": "the accepted default username",
        "weak_password": "the accepted default password",
    },
)


METADATA_CREDENTIAL_LEAK = Vulnerability(
    id="metadata_credential_leak",
    exploit_complexity=0.1,
    family="code_web",
    description=(
        "An unauthenticated internal endpoint (cloud-metadata / admin style) returns a "
        "secret on a plain GET. Not reachable from outside; it is the resource an SSRF "
        "on a public service pivots to — the internal half of the networked SSRF chain."
    ),
    target_kinds=frozenset({"endpoint"}),
    template="metadata_credential_leak.py.j2",
    shape="response_leak",
    attrs_schema={},
)


CONFIG_DISCLOSURE = Vulnerability(
    id="config_disclosure",
    exploit_complexity=0.0,
    family="code_web",
    description=(
        "A public status/config endpoint over-shares internal infrastructure — it "
        "names the internal hosts the app can reach. Recon for the SSRF pivot, not a "
        "flag leak: it discloses the candidate internal targets, never the secret. "
        "Placed only on a company world's public service, never by general sampling."
    ),
    target_kinds=frozenset({"endpoint"}),
    template="config_disclosure.py.j2",
    shape="response_leak",
    attrs_schema={
        "internal_services": "internal hostnames disclosed (the SSRF's targets)",
        "internal_path": "the internal metadata path the disclosure names",
    },
)


CREDENTIAL_LEAK = Vulnerability(
    id="credential_leak",
    exploit_complexity=0.1,
    family="code_web",
    description=(
        "An unauthenticated internal endpoint hands out a service credential (a db "
        "token) on a plain GET. Reachable only by pivoting; the leaked token is the "
        "intermediate loot the agent reuses to reach the internal DB. The internal "
        "half of a lateral-movement chain — never placed by general sampling."
    ),
    target_kinds=frozenset({"endpoint"}),
    template="credential_leak.py.j2",
    shape="response_leak",
    attrs_schema={
        "credential": "the db token this endpoint discloses",
        "vault_host": "internal host the token authenticates to",
        "vault_path": "path on that host the token opens",
        "token_param": "query parameter the token must be presented in",
    },
)


CREDENTIAL_GATED_FLAG = Vulnerability(
    id="credential_gated_flag",
    exploit_complexity=0.1,
    family="code_web",
    description=(
        "An internal data endpoint that serves the flag only to a caller presenting "
        "the db token leaked elsewhere — the lateral-movement target. Solvable only by "
        "reusing the credential moved over from the metadata host. Never placed by "
        "general sampling."
    ),
    target_kinds=frozenset({"endpoint"}),
    template="credential_gated_flag.py.j2",
    shape="response_leak",
    attrs_schema={
        "credential": "the db token a caller must present",
        "token_param": "query parameter the token is read from",
    },
)


CREDENTIAL_GATED_RELAY = Vulnerability(
    id="credential_gated_relay",
    exploit_complexity=0.1,
    family="code_web",
    description=(
        "An intermediate internal relay: validates the reused db token, then hands "
        "over the NEXT host's credential and how to reach it. One composable hop of an "
        "arbitrary-depth lateral chain. Never placed by general sampling."
    ),
    target_kinds=frozenset({"endpoint"}),
    template="credential_gated_relay.py.j2",
    shape="response_leak",
    attrs_schema={
        "credential": "the db token a caller must present to pass this hop",
        "token_param": "query parameter the token is read from",
        "next_credential": "the credential handed over for the next host",
        "next_vault_host": "the next host on the chain",
        "next_vault_path": "the next host's gated path",
        "next_token_param": "the query parameter the next host reads its token from",
    },
)


CATALOG: Mapping[str, Vulnerability] = {
    SQL_INJECTION.id: SQL_INJECTION,
    SSRF.id: SSRF,
    BROKEN_AUTHZ.id: BROKEN_AUTHZ,
    PATH_TRAVERSAL.id: PATH_TRAVERSAL,
    COMMAND_INJECTION.id: COMMAND_INJECTION,
    XXE.id: XXE,
    SSTI.id: SSTI,
    IDOR.id: IDOR,
    WEAK_CREDENTIALS.id: WEAK_CREDENTIALS,
    METADATA_CREDENTIAL_LEAK.id: METADATA_CREDENTIAL_LEAK,
    CONFIG_DISCLOSURE.id: CONFIG_DISCLOSURE,
    CREDENTIAL_LEAK.id: CREDENTIAL_LEAK,
    CREDENTIAL_GATED_FLAG.id: CREDENTIAL_GATED_FLAG,
    CREDENTIAL_GATED_RELAY.id: CREDENTIAL_GATED_RELAY,
}


def vuln(id_: str) -> Vulnerability:
    """Look up a vuln by id; raises KeyError on miss."""
    return CATALOG[id_]


def vulns_for_kind(kind: str) -> tuple[Vulnerability, ...]:
    """Return all catalog entries that target the given graph node kind."""
    return tuple(v for v in CATALOG.values() if kind in v.target_kinds)


def _jinja_env() -> Environment:
    # StrictUndefined makes a missing template parameter fail fast; autoescape
    # is off because the output is Python source, not HTML.
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
    "COMMAND_INJECTION",
    "CONFIG_DISCLOSURE",
    "CREDENTIAL_GATED_FLAG",
    "CREDENTIAL_GATED_RELAY",
    "CREDENTIAL_LEAK",
    "IDOR",
    "METADATA_CREDENTIAL_LEAK",
    "PATH_TRAVERSAL",
    "SQL_INJECTION",
    "SSRF",
    "SSTI",
    "WEAK_CREDENTIALS",
    "XXE",
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
