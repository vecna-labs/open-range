"""Cyber webapp pack — procedural builder + Flask realizer.

This package IS the pack. One Pack, two TaskFamilies, one webapp
world-family. The same world graph admits both `webapp.build` (the
agent implements / repairs a feature endpoint) and `webapp.pentest`
(the agent discovers and exploits a vulnerability chain to recover a
hidden flag). That cross-family-on-one-world story is the load-bearing
demonstration that "domain" lives on the TaskFamily, not on the Pack.

Module map:

  - `ontology.py`      the declarative `Ontology` (cyber.webapp@v1)
                       with 10 node kinds + 11 edge kinds and rich
                       AttrSpec (enums, refs, required flags)
  - `invariants.py`    pack-level invariants: no orphan nodes, every
                       secret held by a record, an oracle exploitation
                       path exists from public surface to a flag
  - `families/`        WebappBuild + WebappPentest TaskFamilies
  - `priors.py`        hand-authored `default_prior() -> PackPrior` and
                       the pack-private `_CYBER_GENERATION_CONFIG`
                       sampler knobs
  - `sampling.py`      procedural graph sampler — emits new-shape
                       WorldGraph against the ontology
  - `mutation.py`      curriculum mutation enumerator — emits
                       `Mutation` carrying `GraphPatch` per direction
                       (harden / soften / diversify)
  - `llm_generation.py`  optional LLM enrichment for task instructions
                         and curriculum mutation relevance scoring
  - `builder.py`       `WebappBuilder` orchestrating sampling +
                       family.generate(), with repair-via-resample
  - `realize.py`       `WebappRuntimeHandle` — the realizer, implements
                       the `RuntimeHandle` Protocol (surface, poll_events,
                       terminal, collect, checkpoint, restore, stop)
  - `codegen/`         Flask app + seed.json source generation,
                       consumed by WebappRuntimeHandle
  - `npcs/`            per-pack NPC factories (browsing_user, admin_audit,
                       office_persona, etc.) — bind to nodes with
                       role=NPC in the world graph

The pack registers as `openrange.packs` entry-point `webapp` via this
package's `pyproject.toml`. The id and the entry-point name share the
same string — `webapp`.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from cyber_webapp.builder import WebappBuilder
from cyber_webapp.families import WebappBuild, WebappPentest
from cyber_webapp.invariants import (
    no_orphan_nodes,
    oracle_path_exists,
    secret_must_be_held,
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
from openrange.world_ir import Issue, Ontology, WorldGraph


class WebappPack(Pack):
    """The cyber webapp pack.

    Concrete Pack wiring together the modules in this package. Ships
    no on-disk source — everything is generated at build/realize time
    from the sampled graph. `dir` is therefore `None`.

    Construct with no args:

        from cyber_webapp import WebappPack
        pack = WebappPack()
        snapshot = admit(pack, manifest={"seed": 0})

    Or load via the entry-point registry:

        from openrange.core.registry import resolve_pack  # future
        pack = resolve_pack({"id": "webapp"})
    """

    id = "webapp"
    version = "v2"

    def __init__(self, dir: Path | None = None) -> None:
        # `dir` is reserved for filesystem-backed packs; this pack
        # generates everything at build time, so there's no on-disk
        # source to point at. Accept the arg for parity with
        # path-loaded pack instantiation.
        del dir
        self.dir = None

    def ontology(self) -> Ontology:
        return webapp_ontology()

    def invariants(self) -> list[Callable[[WorldGraph], list[Issue]]]:
        return [
            no_orphan_nodes,
            secret_must_be_held,
            oracle_path_exists,
        ]

    def make_builder(self, prior: PackPrior | None) -> Builder:
        # `prior=None` is the boot path — `WebappBuilder` falls back to
        # the hand-authored `default_prior()` shipped in priors.py.
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
    "webapp_ontology",
]
