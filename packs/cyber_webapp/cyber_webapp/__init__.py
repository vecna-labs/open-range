from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from graphschema import Issue, Ontology, WorldGraph

from cyber_webapp.builder import WebappBuilder
from cyber_webapp.families import WebappBuild, WebappPentest
from cyber_webapp.invariants import (
    no_orphan_nodes,
    oracle_path_exists,
    secret_must_be_held,
    sqli_targets_db_backed_service,
)
from cyber_webapp.ontology import ONTOLOGY_ID, webapp_ontology
from cyber_webapp.realize import WebappRuntimeError, WebappRuntimeHandle
from openrange.core.pack import (
    Backing,
    Builder,
    Pack,
    PackPrior,
    RuntimeHandle,
    TaskFamily,
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
        ]

    def make_builder(self, prior: PackPrior | None) -> Builder:
        return WebappBuilder(prior)

    def realize(
        self,
        graph: WorldGraph,
        backing: Backing,
    ) -> RuntimeHandle:
        return WebappRuntimeHandle(graph, backing)

    def task_families(self) -> list[TaskFamily]:
        return [WebappBuild(), WebappPentest()]


__all__ = [
    "ONTOLOGY_ID",
    "WebappBuild",
    "WebappBuilder",
    "WebappPack",
    "WebappPentest",
    "WebappRuntimeError",
    "WebappRuntimeHandle",
    "no_orphan_nodes",
    "oracle_path_exists",
    "secret_must_be_held",
    "sqli_targets_db_backed_service",
    "webapp_ontology",
]
