"""WebappBuilder — the cyber pack's procedural Builder (new shape).

One Builder per pack. Constructed by `WebappPack.make_builder(prior)`;
called by core's `admit()` loop:

  - `build(manifest)`  draws a fresh world via `sampling.sample_graph`
                       and asks every TaskFamily on the pack to
                       contribute its tasks against that graph
  - `repair(prev, errors, infeasible)`
                       resamples with a perturbed seed when admission
                       rejected the previous candidate. v1 is a
                       fresh-resample policy; future versions may
                       patch the offending bit instead.
  - `evolve(snapshot, mutation)`
                       returns the mutation's `GraphPatch` verbatim
                       (the default Builder behavior is correct for
                       this pack — the mutation already encodes the
                       full patch).

The builder is deterministic in `(manifest, prior)` modulo
`manifest["seed"]`. Same seed + same prior → same world. Different
seeds → different worlds. That's what makes "sweep seeds for distinct
snapshots" the right pattern.
"""

from __future__ import annotations

import random
from collections.abc import Mapping
from typing import Any

from cyber_webapp.families import WebappBuild, WebappPentest
from cyber_webapp.priors import default_prior
from cyber_webapp.sampling import sample_graph
from openrange.core.contracts import (
    Builder,
    BuildResult,
    Manifest,
    PackPrior,
    TaskSpec,
)
from openrange.world_ir import Issue


def _seed_from_manifest(manifest: Manifest) -> int:
    """Derive the rng seed from the manifest.

    Reads `manifest["seed"]` if present (a curriculum-aware caller
    passes a fresh seed per build to get distinct snapshots); falls
    back to 0 for a fully deterministic default.
    """
    raw = manifest.get("seed", 0)
    if isinstance(raw, bool):
        return 0
    if isinstance(raw, int):
        return raw
    return 0


class WebappBuilder(Builder):
    """Procedural cyber-webapp Builder.

    Wires `sampling.sample_graph` (the world sampler) together with the
    two TaskFamilies (`WebappBuild`, `WebappPentest`) into one
    `BuildResult`. Repair perturbs the seed; evolve is the default
    pass-through.
    """

    def __init__(self, prior: PackPrior | None) -> None:
        # `prior=None` lands the hand-authored default (priors.default_prior),
        # so the builder always has SOMETHING to read from. The builder has
        # one code path — it never knows whether the prior was distilled or
        # authored.
        self._prior = prior if prior is not None else default_prior()
        self._attempt = 0
        self._last_manifest: Manifest = {}

    def build(self, manifest: Manifest) -> BuildResult:
        # Remember the manifest so `repair()` (which doesn't receive one)
        # can re-build against the same input with a perturbed seed.
        self._last_manifest = manifest

        seed = _seed_from_manifest(manifest) + self._attempt
        rng = random.Random(seed)
        graph = sample_graph(rng, self._prior)

        # Each family contributes its tasks against the freshly-sampled
        # graph. Admission validates them; families decide what counts
        # as a feasible task in their domain.
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
        """Resample with a perturbed seed against the same manifest.

        v1's repair policy is "the rejected world's seed is unlucky;
        try a different one." We bump an internal counter so the next
        `build(manifest)` call samples with `seed + attempt`. This
        keeps `manifest["seed"]` reproducible at the original seed when
        the first attempt succeeds, and produces a deterministic
        sequence of follow-up worlds when it doesn't.

        Future versions can be smarter — patch the offending region
        rather than resample wholesale — by inspecting `errors` and
        `infeasible` and applying a targeted `GraphPatch` to
        `prev.graph`.
        """
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
