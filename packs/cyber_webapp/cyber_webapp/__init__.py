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
    credential_value_binding,
    flag_confined_to_gate,
    no_orphan_nodes,
    oracle_path_exists,
    secret_must_be_held,
    sqli_targets_db_backed_service,
    unique_vuln_per_endpoint,
)
from cyber_webapp.mutation import monotone_chain_gate
from cyber_webapp.ontology import ONTOLOGY_ID, webapp_ontology
from cyber_webapp.realize import (
    ContainerWebappRuntime,
    NetworkedContainerWebappRuntime,
    WebappRuntime,
    WebappRuntimeError,
)
from cyber_webapp.sampling import _is_networked as _is_networked  # re-export


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
            credential_value_binding,
            flag_confined_to_gate,
            unique_vuln_per_endpoint,
        ]

    def make_builder(self, prior: PackPrior | None) -> Builder:
        return WebappBuilder(prior)

    def realize(
        self,
        graph: WorldGraph,
        backing: Backing,
    ) -> RuntimeHandle:
        if backing is Backing.CONTAINER:
            if _is_networked(graph):
                return NetworkedContainerWebappRuntime(graph)
            return ContainerWebappRuntime(graph)
        if backing is Backing.PROCESS:
            return WebappRuntime(graph)
        raise NotImplementedError(f"webapp pack does not support backing={backing!r}")

    def task_families(self) -> list[TaskFamily]:
        return [WebappBuild(), WebappPentest()]


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
    "credential_value_binding",
    "flag_confined_to_gate",
    "minimum_backing",
    "monotone_chain_gate",
    "no_orphan_nodes",
    "oracle_path_exists",
    "secret_must_be_held",
    "sqli_targets_db_backed_service",
    "webapp_ontology",
]
