"""Procedural Builder for the cyber webapp pack."""

from __future__ import annotations

import random

from openrange_pack_sdk import (
    BuildResult,
    Manifest,
    PackPrior,
    ProceduralBuilder,
    TaskSpec,
)

from cyber_webapp.families import WebappBuild, WebappPentest
from cyber_webapp.priors import default_prior
from cyber_webapp.sampling import sample_graph


class WebappBuilder(ProceduralBuilder):
    def __init__(self, prior: PackPrior | None = None) -> None:
        super().__init__(prior if prior is not None else default_prior())

    def sample(self, rng: random.Random, manifest: Manifest) -> BuildResult:
        graph = sample_graph(rng, self.prior)
        tasks: list[TaskSpec] = []
        tasks.extend(WebappBuild().generate(graph, manifest, self.prior))
        tasks.extend(WebappPentest().generate(graph, manifest, self.prior))
        return BuildResult(
            graph=graph,
            tasks=tasks,
            admission_meta={
                "builder": "cyber.webapp.v2",
                "seed": self.current_seed,
                "prior_source": (self.prior.source if self.prior is not None else None),
                "manifest_keys": sorted(manifest.keys()),
            },
        )
