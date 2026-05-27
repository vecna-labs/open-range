"""Reusable Builder base classes for common pack-author patterns.

These are optional. Packs can implement ``Builder`` directly. Use one of
these when your builder fits the pattern — typically you'll cut 50+ LOC
of boilerplate.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod

from graphschema import Issue

from openrange_pack_sdk._helpers import manifest_int
from openrange_pack_sdk._protocols import Builder
from openrange_pack_sdk._types import BuildResult, Manifest, PackPrior


class ProceduralBuilder(Builder, ABC):
    """Builder base for packs that procedurally sample worlds from a
    seeded RNG. Handles seed extraction from manifest and the
    repair-by-reseed loop.

    Packs override ``sample(rng, manifest)`` to produce a ``BuildResult``.
    The base maintains an attempt counter; each ``repair()`` increments
    it and re-runs ``sample()`` with the next seed. This is the typical
    "rejection sampling" pattern: admission rejects a world; the builder
    retries with a different seed; repeat until admission accepts or the
    max-repair budget runs out.

    Subclasses access their pack-specific prior via ``self.prior``.
    """

    def __init__(
        self,
        prior: PackPrior | None = None,
        *,
        seed_key: str = "seed",
    ) -> None:
        self._prior = prior
        self._seed_key = seed_key
        self._attempt = 0
        self._current_seed = 0
        self._last_manifest: Manifest = {}

    @abstractmethod
    def sample(self, rng: random.Random, manifest: Manifest) -> BuildResult:
        """Produce a BuildResult from a seeded RNG + the manifest.

        ``rng`` is freshly seeded for each ``build()`` / ``repair()`` call;
        callers should treat it as the sole source of randomness so the
        outcome is deterministic in ``(seed, attempt)``.
        """

    @property
    def prior(self) -> PackPrior | None:
        return self._prior

    @property
    def current_seed(self) -> int:
        """The seed used for the most recent ``sample()`` call.

        Useful when the subclass wants to surface it in
        ``BuildResult.admission_meta`` for snapshot lineage.
        """
        return self._current_seed

    def build(self, manifest: Manifest) -> BuildResult:
        self._last_manifest = manifest
        self._current_seed = (
            manifest_int(manifest, self._seed_key, default=0) + self._attempt
        )
        return self.sample(random.Random(self._current_seed), manifest)

    def repair(
        self,
        prev: BuildResult,
        errors: list[Issue],
        infeasible: list[str],
    ) -> BuildResult:
        del prev, errors, infeasible
        self._attempt += 1
        return self.build(self._last_manifest)
