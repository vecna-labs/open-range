"""Procedural Builder for the cyber webapp pack."""

from __future__ import annotations

import random
from collections.abc import Mapping
from typing import Any

from graphschema import Issue

from cyber_webapp.families import WebappBuild, WebappPentest
from cyber_webapp.priors import default_prior
from cyber_webapp.sampling import sample_graph
from openrange.core.pack import (
    Builder,
    BuildResult,
    Manifest,
    PackPrior,
    TaskSpec,
)


def _seed_from_manifest(manifest: Manifest) -> int:
    raw = manifest.get("seed", 0)
    if isinstance(raw, bool):
        return 0
    if isinstance(raw, int):
        return raw
    return 0


class WebappBuilder(Builder):
    def __init__(self, prior: PackPrior | None) -> None:
        self._prior = prior if prior is not None else default_prior()
        self._attempt = 0
        self._last_manifest: Manifest = {}

    def build(self, manifest: Manifest) -> BuildResult:
        self._last_manifest = manifest

        seed = _seed_from_manifest(manifest) + self._attempt
        rng = random.Random(seed)
        graph = sample_graph(rng, self._prior)

        tasks: list[TaskSpec] = []
        tasks.extend(WebappBuild().generate(graph, manifest, self._prior))
        tasks.extend(WebappPentest().generate(graph, manifest, self._prior))

        return BuildResult(
            graph=graph,
            tasks=tasks,
            admission_meta=_admission_meta(seed, self._prior, manifest),
        )

    def repair(
        self,
        prev: BuildResult,
        errors: list[Issue],
        infeasible: list[str],
    ) -> BuildResult:
        del prev, errors, infeasible
        self._attempt += 1
        return self.build(self._last_manifest)


def _admission_meta(
    seed: int,
    prior: PackPrior,
    manifest: Manifest,
) -> Mapping[str, Any]:
    return {
        "builder": "cyber.webapp.v2",
        "seed": seed,
        "prior_source": prior.source,
        "manifest_keys": sorted(manifest.keys()),
    }
