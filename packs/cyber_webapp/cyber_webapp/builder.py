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
        # Manifest knobs override the prior's topology so a world is
        # configurable without a hand-built PackPrior, while staying
        # deterministic: the seed still selects within the overridden ranges
        # and weights. ``scale`` feeds count_ranges; ``loot_shapes`` /
        # ``vuln_kinds`` feed kind_weights.
        base = self.prior if self.prior is not None else default_prior()
        scale = manifest.get("scale")
        weight_keys = [k for k in ("loot_shapes", "vuln_kinds") if k in manifest]
        if not isinstance(scale, Mapping) and not weight_keys:
            return base
        topology = dict(base.topology)
        if isinstance(scale, Mapping):
            count_ranges = dict(topology.get("count_ranges") or {})
            for key, spec in scale.items():
                if isinstance(spec, Mapping):
                    count_ranges[str(key)] = dict(spec)
            topology["count_ranges"] = count_ranges
        if weight_keys:
            kind_weights = dict(topology.get("kind_weights") or {})
            for key in weight_keys:
                spec = manifest[key]
                if isinstance(spec, Mapping):
                    kind_weights[key] = {str(k): v for k, v in spec.items()}
            topology["kind_weights"] = kind_weights
        return dataclasses.replace(base, topology=topology)
