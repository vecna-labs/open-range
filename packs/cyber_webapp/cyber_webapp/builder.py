"""Procedural Builder for the cyber webapp pack."""

from __future__ import annotations

import dataclasses
import random
from collections.abc import Mapping

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
        prior = self._effective_prior(manifest)
        graph = sample_graph(rng, prior)
        tasks: list[TaskSpec] = []
        tasks.extend(WebappBuild().generate(graph, manifest, prior))
        tasks.extend(WebappPentest().generate(graph, manifest, prior))
        return BuildResult(
            graph=graph,
            tasks=tasks,
            admission_meta={
                "builder": "cyber.webapp.v2",
                "seed": self.current_seed,
                "prior_source": prior.source,
                "manifest_keys": sorted(manifest.keys()),
            },
        )

    def _effective_prior(self, manifest: Manifest) -> PackPrior:
        # manifest["scale"] overrides the prior's count_ranges so a world
        # scales without a hand-built PackPrior. Determinism is unchanged:
        # the seed still selects within the (possibly widened) ranges.
        base = self.prior if self.prior is not None else default_prior()
        overrides = manifest.get("scale")
        if not isinstance(overrides, Mapping):
            return base
        topology = dict(base.topology)
        count_ranges = dict(topology.get("count_ranges") or {})
        for key, spec in overrides.items():
            if isinstance(spec, Mapping):
                count_ranges[str(key)] = dict(spec)
        topology["count_ranges"] = count_ranges
        return dataclasses.replace(base, topology=topology)
