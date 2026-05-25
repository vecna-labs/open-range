"""WebappPack — the Pack implementation wiring up the webapp world-family.

One Pack, two TaskFamilies. Same world graph admits both
`webapp.build` (entrypoint=repo) and `webapp.pentest`
(entrypoint=endpoint) tasks — that cross-domain story is the
load-bearing demonstration of the new shape.

This v1 ships a no-op `realize()`. The runtime side (HTTPBacking,
Flask code generation, NPC threads) was removed during the OpenRange
core refactor and will be re-wired against the new RuntimeHandle
contract in a follow-up PR. Admission and the cross-family test work
today without a realizer.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from openrange import (
    Backing,
    Builder,
    Issue,
    Ontology,
    Pack,
    PackPrior,
    RuntimeHandle,
    TaskFamily,
    WorldGraph,
)
from webapp.builder import WebappBuilder
from webapp.families import WebappBuild, WebappPentest
from webapp.invariants import (
    no_orphan_nodes,
    secret_must_be_held,
    service_must_own_repo_and_expose_endpoint,
)
from webapp.ontology import webapp_ontology


class _NoopRuntimeHandle:
    """v1 placeholder. Real realizer (Flask app, HTTP backing) lands later."""

    def reset(self) -> None: ...
    def surface(self) -> Mapping[str, Any]:
        return {}

    def collect(self) -> Mapping[str, Any]:
        return {}

    def stop(self) -> None: ...


class WebappPack(Pack):
    id = "webapp"
    version = "0.1.0"

    def ontology(self) -> Ontology:
        return webapp_ontology()

    def invariants(self) -> list[Callable[[WorldGraph], list[Issue]]]:
        return [
            no_orphan_nodes,
            secret_must_be_held,
            service_must_own_repo_and_expose_endpoint,
        ]

    def make_builder(self, prior: PackPrior | None) -> Builder:
        return WebappBuilder(prior)

    def realize(self, graph: WorldGraph, backing: Backing) -> RuntimeHandle:
        """v1 returns a no-op handle. The runtime side is being re-wired
        against the new contract; until then, episodes against this pack
        will collect an empty final_state and the TaskFamily success
        checks will read the empty mapping (and report success=False).
        """
        del graph, backing
        return _NoopRuntimeHandle()

    def task_families(self) -> list[TaskFamily]:
        return [WebappBuild(), WebappPentest()]
