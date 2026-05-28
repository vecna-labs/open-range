"""Procedural Builder for the trading pack.

The pack's weight lives here: ``sample`` fetches a real historical window
(seeded, so reproducible) and embeds it as the world graph. ``repair``'s
reseed (from ``ProceduralBuilder``) draws a fresh window when admission
rejects one.
"""

from __future__ import annotations

import random
from pathlib import Path

from openrange_pack_sdk import (
    BuildResult,
    Manifest,
    PackPrior,
    ProceduralBuilder,
    TaskSpec,
    manifest_str,
)

from trading.families import TradePnl
from trading.priors import default_prior
from trading.sampling import sample_graph

_PACK_DIR = Path(__file__).resolve().parent
_FIXTURE = _PACK_DIR / "fixtures" / "synthetic.json"
# Gitignored perf cache for fetched windows; never committed (the pack ships
# code that fetches, not redistributed market data).
_DEFAULT_CACHE = _PACK_DIR.parent / ".cache"


class TradingBuilder(ProceduralBuilder):
    def __init__(self, prior: PackPrior | None = None) -> None:
        super().__init__(prior if prior is not None else default_prior())

    def sample(self, rng: random.Random, manifest: Manifest) -> BuildResult:
        cache_dir = _DEFAULT_CACHE
        override = manifest_str(manifest, "cache_dir")
        if override:
            cache_dir = Path(override)
        graph = sample_graph(
            rng, manifest, self.prior, cache_dir=cache_dir, fixture=_FIXTURE
        )
        tasks: list[TaskSpec] = TradePnl().generate(graph, manifest, self.prior)
        return BuildResult(
            graph=graph,
            tasks=tasks,
            admission_meta={
                "builder": "trading.v1",
                "seed": self.current_seed,
                "data_source": graph.meta.get("data_source"),
                "regime_intensity": graph.meta.get("regime_intensity"),
                "prior_source": self.prior.source if self.prior is not None else None,
            },
        )
