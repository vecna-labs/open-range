"""Pack-level invariants for the `webapp@0.1.0` ontology.

Each invariant is a plain function `(graph) -> list[Issue]` returned by
`WebappPack.invariants()`. They run as the Tier-3 layer of OpenRange's
admission loop — *after* structural + conformance + invariants of the
generic validator, *before* task feasibility.

These invariants are graph-wide (true for any task posed against this
world). Task-relative checks live on the TaskFamilies as
`check_feasibility`.

Rule of thumb for whether a constraint belongs here:
  yes — every webapp world MUST satisfy this regardless of which task
        family runs (e.g. every secret needs a record holding it; an
        orphan account is a sampling bug)
  no  — it depends on whether the task is build vs. pentest (e.g.
        "there exists a vuln-chain to a flag" is only relevant when a
        pentest task exists; that check belongs to WebappPentest)
"""

from __future__ import annotations

from openrange import Issue, WorldGraph

# Exempt kinds from the no-orphan check: repos and datastores legitimately
# stand alone in some valid graphs (a repo with no service yet; a fresh
# datastore awaiting its service).
_ORPHAN_EXEMPT: frozenset[str] = frozenset({"repo", "datastore"})


def no_orphan_nodes(graph: WorldGraph) -> list[Issue]:
    """Every non-exempt node must be touched by at least one edge.

    Catches sampling bugs — an account with no credential, a weakness
    affecting nothing, a secret in no record.
    """
    referenced: set[str] = set()
    for e in graph.edges.values():
        referenced.add(e.src)
        referenced.add(e.dst)
    issues: list[Issue] = []
    for n in graph.nodes.values():
        if n.kind in _ORPHAN_EXEMPT:
            continue
        if n.id not in referenced:
            issues.append(
                Issue(
                    "error",
                    "orphan_node",
                    f"node {n.id!r} of kind {n.kind!r} has no incident edges",
                    n.id,
                )
            )
    return issues


def secret_must_be_held(graph: WorldGraph) -> list[Issue]:
    """Every `secret` must be the destination of a `holds` edge.

    A secret floating with no record holding it is a sampling bug — the
    realizer would not know where to seed the value.
    """
    held: set[str] = set()
    for e in graph.edges.values():
        if e.kind == "holds":
            held.add(e.dst)
    issues: list[Issue] = []
    for n in graph.by_kind("secret"):
        if n.id not in held:
            issues.append(
                Issue(
                    "error",
                    "secret_not_held",
                    f"secret {n.id!r} is not held by any record",
                    n.id,
                )
            )
    return issues


def service_must_own_repo_and_expose_endpoint(graph: WorldGraph) -> list[Issue]:
    """Every `service` must be owned by a repo and expose at least one endpoint.

    A service with no repo cannot be built; a service exposing no
    endpoint is not reachable by an agent.
    """
    issues: list[Issue] = []
    for svc in graph.by_kind("service"):
        if not graph.out_edges(svc.id, "owned_by"):
            issues.append(
                Issue(
                    "error",
                    "service_no_repo",
                    f"service {svc.id!r} has no owned_by edge to any repo",
                    svc.id,
                )
            )
        if not graph.out_edges(svc.id, "exposes"):
            issues.append(
                Issue(
                    "error",
                    "service_no_endpoint",
                    f"service {svc.id!r} has no exposes edge to any endpoint",
                    svc.id,
                )
            )
    return issues
