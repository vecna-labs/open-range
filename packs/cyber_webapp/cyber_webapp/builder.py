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
        # ``vuln_kinds`` feed kind_weights; ``company`` seeds a believable
        # multi-service estate, itself still overridable.
        base = self.prior if self.prior is not None else default_prior()
        lateral = bool(manifest.get("lateral_movement"))
        company = bool(manifest.get("company")) or lateral
        scale = manifest.get("scale")
        weight_keys = [k for k in ("loot_shapes", "vuln_kinds") if k in manifest]
        if not company and not isinstance(scale, Mapping) and not weight_keys:
            return base
        topology = dict(base.topology)
        count_ranges = dict(topology.get("count_ranges") or {})
        kind_weights = dict(topology.get("kind_weights") or {})
        if company:
            # A medium-company estate the agent recons and pivots through: more
            # services. ``service_count`` / ``vuln_count`` are tunable (a manifest
            # ``scale`` still wins below); the recon disclosure and the internal/dmz
            # segmentation are added by the sampler off ``preset``. ``lateral_movement``
            # swaps the direct pivot for a credential-reuse chain (the SSRF proxy +
            # an internal credential leak gating the flag on a separate internal db).
            topology["preset"] = "company"
            if lateral:
                topology["lateral"] = True
            count_ranges.setdefault("service_count", {"min": 6, "max": 8})
            count_ranges.setdefault("vuln_count", {"min": 3, "max": 6})
        if isinstance(scale, Mapping):
            for key, spec in scale.items():
                if isinstance(spec, Mapping):
                    count_ranges[str(key)] = dict(spec)
        if weight_keys:
            for key in weight_keys:
                spec = manifest[key]
                if isinstance(spec, Mapping):
                    kind_weights[key] = {str(k): v for k, v in spec.items()}
        if company:
            # The networked SSRF pivot to a db-backed internal flag IS the company
            # world's shape, not a tunable weight: force it last so a stray
            # ``vuln_kinds`` / ``loot_shapes`` override can't quietly yield a
            # non-networked or unsolvable "company" world. ssrf wins the oracle slot
            # (the only response-leak match for db loot, whatever its weight); the
            # file-read decoys are noise.
            kind_weights["loot_shapes"] = {"db": 1, "file": 0}
            kind_weights["vuln_kinds"] = {"ssrf": 1, "path_traversal": 3, "xxe": 2}
        if count_ranges:
            topology["count_ranges"] = count_ranges
        if kind_weights:
            topology["kind_weights"] = kind_weights
        return dataclasses.replace(base, topology=topology)
