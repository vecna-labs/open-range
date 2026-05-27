"""Ontology contract for the cyber webapp pack.

The realizer is a pure projection of this graph — it never decides URLs,
table names, or other agent-observable strings of its own.
"""

from __future__ import annotations

from graphschema import AttrSpec, AttrType, EdgeKind, NodeKind, Ontology

ONTOLOGY_ID = "cyber.webapp@v2"


def webapp_ontology() -> Ontology:
    # fresh instance per call so callers can mutate without leaking into other consumers
    s = AttrSpec
    return Ontology(
        id=ONTOLOGY_ID,
        node_kinds={
            "host": NodeKind(
                "host",
                attrs={
                    "hostname": s(AttrType.STRING, required=True),
                    "os": s(
                        AttrType.ENUM,
                        required=True,
                        enum=["linux", "windows", "container"],
                    ),
                    "zone": s(
                        AttrType.ENUM,
                        required=True,
                        enum=["dmz", "corp", "data", "management", "external"],
                    ),
                },
                description="a runtime host (VM / container / bare-metal)",
            ),
            "service": NodeKind(
                "service",
                attrs={
                    "name": s(AttrType.STRING, required=True),
                    "kind": s(
                        AttrType.ENUM,
                        required=True,
                        enum=["web", "api", "auth", "db", "queue", "mail", "fileshare"],
                    ),
                    "language": s(
                        AttrType.ENUM,
                        enum=["python", "node", "go", "ruby", "java"],
                        default="python",
                    ),
                    "exposure": s(
                        AttrType.ENUM,
                        required=True,
                        enum=["public", "internal", "management"],
                    ),
                },
                description="a running service in the realized webapp",
            ),
            "endpoint": NodeKind(
                "endpoint",
                attrs={
                    "path": s(
                        AttrType.STRING,
                        required=True,
                        description="logical path inside the application",
                    ),
                    "public_url": s(
                        AttrType.STRING,
                        required=True,
                        description=(
                            "agent-facing URL the realizer mounts this endpoint at; "
                            "the realizer is a pure projection — it never invents URLs"
                        ),
                    ),
                    "method": s(
                        AttrType.ENUM,
                        required=True,
                        enum=["GET", "POST", "PUT", "DELETE", "PATCH"],
                    ),
                    "auth_required": s(AttrType.BOOL, default=False),
                    "behavior_ref": s(
                        AttrType.STRING,
                        description="template ref the realizer renders",
                    ),
                },
                description="one HTTP endpoint exposed by a service",
            ),
            "account": NodeKind(
                "account",
                attrs={
                    "username": s(AttrType.STRING, required=True),
                    "role": s(
                        AttrType.ENUM,
                        required=True,
                        enum=["user", "admin", "service"],
                    ),
                    "active": s(AttrType.BOOL, default=True),
                },
                description="a user / admin / service account",
            ),
            "credential": NodeKind(
                "credential",
                attrs={
                    "kind": s(
                        AttrType.ENUM,
                        required=True,
                        enum=["password", "api_key", "session", "token"],
                    ),
                    "value_ref": s(
                        AttrType.STRING,
                        required=True,
                        description="opaque ref the realizer resolves",
                    ),
                },
                description="an authenticator (something the account 'has')",
            ),
            "secret": NodeKind(
                "secret",
                attrs={
                    "kind": s(
                        AttrType.ENUM,
                        required=True,
                        enum=["flag", "api_key", "password", "private_key"],
                    ),
                    "value_ref": s(
                        AttrType.STRING,
                        required=True,
                        description="opaque ref the realizer resolves",
                    ),
                    "description": s(AttrType.STRING),
                },
                description="a hidden value the agent may need to discover; "
                "always Visibility.HIDDEN at construction",
            ),
            "vulnerability": NodeKind(
                "vulnerability",
                attrs={
                    "kind": s(
                        AttrType.STRING,
                        required=True,
                        description="catalog id from cyber_webapp.vulnerabilities",
                    ),
                    "family": s(
                        AttrType.ENUM,
                        enum=[
                            "code_web",
                            "config_identity",
                            "secret_exposure",
                            "logic_flaw",
                            "supply_chain",
                        ],
                    ),
                    "params": s(
                        AttrType.JSON,
                        description="vuln-specific tuning",
                    ),
                },
                description="an exploitable defect; always Visibility.HIDDEN at "
                "construction",
            ),
            "network": NodeKind(
                "network",
                attrs={
                    "name": s(AttrType.STRING, required=True),
                    "isolation": s(
                        AttrType.ENUM,
                        required=True,
                        enum=["bridge", "host", "isolated"],
                    ),
                    "zone": s(AttrType.STRING),
                },
                description="a network segment connecting services",
            ),
            "data_store": NodeKind(
                "data_store",
                attrs={
                    "name": s(AttrType.STRING, required=True),
                    "kind": s(
                        AttrType.ENUM,
                        required=True,
                        enum=["sql", "kv", "file", "object"],
                    ),
                    "engine": s(
                        AttrType.ENUM,
                        required=True,
                        enum=["sqlite", "postgres", "mysql", "redis", "fs", "s3"],
                    ),
                },
                description="a backing data store",
            ),
            "record": NodeKind(
                "record",
                attrs={
                    "key": s(AttrType.STRING, required=True),
                    "fields": s(
                        AttrType.JSON,
                        description=(
                            "column name -> seed value (or value_ref for secrets)"
                        ),
                    ),
                },
                description="one row / document in a data_store",
            ),
        },
        edge_kinds={
            "exposes": EdgeKind(
                "exposes",
                endpoints=[("service", "endpoint")],
                description="this service serves this endpoint",
            ),
            "backed_by": EdgeKind(
                "backed_by",
                endpoints=[("service", "data_store")],
                attrs={
                    "mode": s(
                        AttrType.ENUM,
                        enum=["read", "write", "readwrite"],
                        default="readwrite",
                    ),
                },
                description="this service reads/writes this store",
            ),
            "contains": EdgeKind(
                "contains",
                endpoints=[("data_store", "record")],
                description="this record lives in this store",
            ),
            "holds": EdgeKind(
                "holds",
                endpoints=[("record", "secret")],
                attrs={
                    "field": s(
                        AttrType.STRING,
                        description="column / attribute name the secret occupies",
                    ),
                },
                description="this record holds this secret in the given field",
            ),
            "has_credential": EdgeKind(
                "has_credential",
                endpoints=[("account", "credential")],
                description="this account authenticates with this credential",
            ),
            "can_access": EdgeKind(
                "can_access",
                endpoints=[("account", "endpoint")],
                attrs={
                    "auth_method": s(
                        AttrType.STRING,
                        description="how the account authenticates to this endpoint",
                    ),
                },
                description="legitimate access — the agent's negative space",
            ),
            "runs_on": EdgeKind(
                "runs_on",
                endpoints=[("service", "host")],
                description="this service runs on this host",
            ),
            "connected_to": EdgeKind(
                "connected_to",
                endpoints=[("service", "network")],
                description="this service is wired to this network segment",
            ),
            "affects": EdgeKind(
                "affects",
                endpoints=[
                    ("vulnerability", "endpoint"),
                    ("vulnerability", "service"),
                ],
                attrs={
                    "injection_site": s(
                        AttrType.STRING,
                        description="where the vuln is reachable",
                    ),
                },
                description=(
                    "this weakness is reachable through this endpoint / service"
                ),
            ),
            "enables": EdgeKind(
                "enables",
                endpoints=[("vulnerability", "vulnerability")],
                description="a vuln chain: A enables exploitation of B",
            ),
            "derives": EdgeKind(
                "derives",
                endpoints=[("credential", "secret")],
                description="a credential whose value comes from this secret",
            ),
        },
    )
