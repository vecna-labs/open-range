"""Deterministic loader Builder for the SWE pack.

The imported (SWE-bench-passthrough) source: the instance recipe *is* the world,
so there is nothing to sample. ``build`` resolves a recipe two ways behind the
same ontology + grader:

- a bundled **fixture** (``manifest["instance"]``) — the offline default, and
- a SWE-bench **row** (``manifest["swebench"]``) — clone the referenced
  repo@base_commit and recover the held-out tests + gold fix from its diffs.
  This is the "pull a world from GitHub" path made reachable from the manifest;
  ``manifest["source"]`` optionally overrides where to clone from (a local path
  in tests, so the whole path stays offline).

(The injected and authored sources, per the design doc, slot in here as further
recipe producers behind the same ontology + grader.)
"""

from __future__ import annotations

import tempfile
from collections.abc import Mapping
from pathlib import Path

from openrange_pack_sdk import (
    Builder,
    BuildResult,
    Manifest,
    PackPrior,
    TaskSpec,
    manifest_str,
)

from swe.families import SweBuild, SweFix
from swe.instances import SweInstance, load_instance, to_graph
from swe.swebench import instance_from_row

_DEFAULT_INSTANCE = "calc_sum"


class SweBuilder(Builder):
    def __init__(self, prior: PackPrior | None = None) -> None:
        self._prior = prior

    def build(self, manifest: Manifest) -> BuildResult:
        instance, source = self._resolve(manifest)
        graph = to_graph(instance)
        # Each family self-selects on the suite shape: SweFix claims worlds with
        # fail_to_pass, SweBuild claims worlds with integration_tests. The shipped
        # fixtures are one shape or the other, so one family emits; a suite that
        # declared both tiers would emit a task from each.
        tasks: list[TaskSpec] = [
            task
            for family in (SweFix(), SweBuild())
            for task in family.generate(graph, manifest, self._prior)
        ]
        return BuildResult(
            graph=graph,
            tasks=tasks,
            admission_meta={
                "builder": "swe.v1",
                "source": source,
                "instance": instance.instance_id,
                "language": instance.language,
            },
        )

    def _resolve(self, manifest: Manifest) -> tuple[SweInstance, str]:
        """Build the instance from a SWE-bench row (cloned) or a fixture.

        The clone happens in a throwaway temp dir whose tree is inlined into the
        graph; nothing on disk outlives the build (the scale ceiling for big
        repos is the lazy-realize milestone, #212).
        """
        row = manifest.get("swebench")
        if isinstance(row, Mapping):
            override = manifest_str(manifest, "source") or None
            with tempfile.TemporaryDirectory(prefix="swe-build-") as tmp:
                instance = instance_from_row(row, workdir=Path(tmp), source=override)
            return instance, "github"
        name = manifest_str(manifest, "instance", default=_DEFAULT_INSTANCE)
        return load_instance(name), "imported"
