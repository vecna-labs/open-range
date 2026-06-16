from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from graphschema import Issue, Ontology, WorldGraph
from openrange_pack_sdk import (
    Backing,
    Builder,
    Pack,
    PackPrior,
    RuntimeHandle,
    TaskFamily,
)

from cyber_webapp.builder import WebappBuilder
from cyber_webapp.container import minimum_backing
from cyber_webapp.families import WebappBuild, WebappPentest
from cyber_webapp.invariants import (
    credential_reuse_binding,
    no_orphan_nodes,
    oracle_path_exists,
    secret_must_be_held,
    sqli_targets_db_backed_service,
)
from cyber_webapp.ontology import ONTOLOGY_ID, webapp_ontology
from cyber_webapp.realize import (
    ContainerWebappRuntime,
    NetworkedContainerWebappRuntime,
    WebappRuntime,
    WebappRuntimeError,
)


class WebappPack(Pack):
    id = "webapp"
    version = "v2"

    def __init__(self, dir: Path | None = None) -> None:
        # accepted for parity with path-loaded packs; nothing on disk to load
        del dir
        self.dir = None

    def ontology(self) -> Ontology:
        return webapp_ontology()

    def invariants(self) -> list[Callable[[WorldGraph], list[Issue]]]:
        return [
            no_orphan_nodes,
            secret_must_be_held,
            oracle_path_exists,
            sqli_targets_db_backed_service,
            credential_reuse_binding,
        ]

    def make_builder(self, prior: PackPrior | None) -> Builder:
        return WebappBuilder(prior)

    def realize(
        self,
        graph: WorldGraph,
        backing: Backing,
    ) -> RuntimeHandle:
        if backing is Backing.CONTAINER:
            # A *networked* world — one whose flag is reachable only by pivoting from
            # the public service to an internal one — runs as one container per service
            # on a network. Single-host worlds stay one container.
            if _is_networked(graph):
                return NetworkedContainerWebappRuntime(graph, backing)
            return ContainerWebappRuntime(graph, backing)
        return WebappRuntime(graph, backing)

    def task_families(self) -> list[TaskFamily]:
        return [WebappBuild(), WebappPentest()]


def _is_networked(graph: WorldGraph) -> bool:
    # Networked = the flag is reachable only by pivoting: an SSRF on a PUBLIC service
    # reaches an internal service that holds the flag. A vuln co-located with the flag
    # on one service is not networked — it stays single-container.
    public_services = {
        n.id for n in graph.by_kind("service") if n.attrs.get("exposure") == "public"
    }
    service_of_endpoint = {
        e.dst: e.src for e in graph.edges.values() if e.kind == "exposes"
    }
    return any(
        service_of_endpoint.get(edge.dst) in public_services
        for vuln in graph.by_kind("vulnerability")
        if vuln.attrs.get("kind") == "ssrf"
        for edge in graph.out_edges(vuln.id, "affects")
    )


__all__ = [
    "ONTOLOGY_ID",
    "ContainerWebappRuntime",
    "NetworkedContainerWebappRuntime",
    "WebappBuild",
    "WebappBuilder",
    "WebappPack",
    "WebappPentest",
    "WebappRuntimeError",
    "WebappRuntime",
    "credential_reuse_binding",
    "minimum_backing",
    "no_orphan_nodes",
    "oracle_path_exists",
    "secret_must_be_held",
    "sqli_targets_db_backed_service",
    "webapp_ontology",
]
