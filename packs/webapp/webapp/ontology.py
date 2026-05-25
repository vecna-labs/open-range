"""The `webapp@0.1.0` ontology — declarative node and edge kinds.

Scope: HTTP-shaped business webapps at small-to-medium scale. A world
graph in this ontology describes a system that can be built (write a
feature endpoint) or attacked (exploit an existing endpoint). The same
graph serves both task families.

Out of scope for v1: container/k8s primitives, cloud-IAM permission
graphs, queue-based workflows. Those are different world-families and
should be their own pack.
"""

from __future__ import annotations

from openrange import AttrSpec, AttrType, EdgeKind, NodeKind, Ontology

ONTOLOGY_ID = "webapp@0.1.0"


def webapp_ontology() -> Ontology:
    """Return the `webapp@0.1.0` ontology value."""
    s = AttrSpec
    return Ontology(
        id=ONTOLOGY_ID,
        node_kinds={
            "repo": NodeKind(
                "repo",
                attrs={
                    "name": s(AttrType.STRING, required=True),
                    "language": s(
                        AttrType.ENUM, enum=["python", "node", "go"], default="python"
                    ),
                },
                description="a source repository holding the app's code",
            ),
            "service": NodeKind(
                "service",
                attrs={
                    "name": s(AttrType.STRING, required=True),
                    "kind": s(
                        AttrType.ENUM,
                        required=True,
                        enum=["web", "api", "auth", "db_proxy"],
                    ),
                    "exposure": s(
                        AttrType.ENUM, required=True, enum=["public", "internal"]
                    ),
                },
                description="a running service in the realized webapp",
            ),
            "endpoint": NodeKind(
                "endpoint",
                attrs={
                    "path": s(AttrType.STRING, required=True),
                    "method": s(
                        AttrType.ENUM,
                        required=True,
                        enum=["GET", "POST", "PUT", "DELETE"],
                    ),
                    "auth_required": s(AttrType.BOOL, default=False),
                },
                description="an HTTP endpoint exposed by a service",
            ),
            "datastore": NodeKind(
                "datastore",
                attrs={
                    "name": s(AttrType.STRING, required=True),
                    "engine": s(
                        AttrType.ENUM,
                        required=True,
                        enum=["sqlite", "postgres", "redis"],
                    ),
                },
                description="a backing data store",
            ),
            "record": NodeKind(
                "record",
                attrs={
                    "key": s(AttrType.STRING, required=True),
                },
                description="one row/document in a datastore",
            ),
            "secret": NodeKind(
                "secret",
                attrs={
                    "name": s(AttrType.STRING, required=True),
                    "kind": s(
                        AttrType.ENUM,
                        required=True,
                        enum=["flag", "api_key", "password"],
                    ),
                    "value_ref": s(
                        AttrType.STRING,
                        required=True,
                        description="opaque ref the realizer resolves",
                    ),
                },
                description="a hidden value the agent may need to discover; "
                "always Visibility.HIDDEN",
            ),
            "weakness": NodeKind(
                "weakness",
                attrs={
                    "name": s(AttrType.STRING, required=True),
                    "kind": s(
                        AttrType.ENUM,
                        required=True,
                        enum=[
                            "sql_injection",
                            "ssrf",
                            "broken_authz",
                            "path_traversal",
                        ],
                    ),
                },
                description="an exploitable defect in a service or endpoint; "
                "always Visibility.HIDDEN",
            ),
        },
        edge_kinds={
            "owned_by": EdgeKind(
                "owned_by",
                endpoints=[("service", "repo")],
                description="the service's code lives in this repo",
            ),
            "exposes": EdgeKind(
                "exposes",
                endpoints=[("service", "endpoint")],
                description="this service serves this endpoint",
            ),
            "backed_by": EdgeKind(
                "backed_by",
                endpoints=[("service", "datastore")],
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
                endpoints=[("datastore", "record")],
                description="this record lives in this store",
            ),
            "holds": EdgeKind(
                "holds",
                endpoints=[("record", "secret")],
                attrs={
                    "field": s(AttrType.STRING, description="column or attribute name"),
                },
                description="this record holds this secret",
            ),
            "affects": EdgeKind(
                "affects",
                endpoints=[
                    ("weakness", "endpoint"),
                    ("weakness", "service"),
                ],
                description="this weakness can be exploited "
                "via this endpoint or service",
            ),
        },
    )
