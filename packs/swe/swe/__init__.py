from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from graphschema import Issue, Ontology, WorldGraph
from openrange_pack_sdk import (
    Backing,
    Builder,
    Pack,
    PackPrior,
    RuntimeHandle,
    TaskFamily,
)

from swe.builder import SweBuilder
from swe.families import SweBuild, SweFix
from swe.invariants import repo_has_base_files, solution_present, suite_well_formed
from swe.ontology import ONTOLOGY_ID, repo_ontology
from swe.realize import SweRuntime


class SwePack(Pack):
    id = "swe"
    version = "v1"

    def __init__(self, dir: Path | None = None) -> None:
        # accepted for parity with path-loaded packs; nothing on disk to load
        del dir
        self.dir = None

    def ontology(self) -> Ontology:
        return repo_ontology()

    def invariants(self) -> list[Callable[[WorldGraph], list[Issue]]]:
        return [repo_has_base_files, suite_well_formed, solution_present]

    def make_builder(self, prior: PackPrior | None) -> Builder:
        return SweBuilder(prior)

    def realize(self, graph: WorldGraph, backing: Backing) -> RuntimeHandle:
        return SweRuntime(graph, backing)

    def task_families(self) -> list[TaskFamily]:
        return [SweFix(), SweBuild()]


__all__ = [
    "ONTOLOGY_ID",
    "SweBuild",
    "SweBuilder",
    "SweFix",
    "SwePack",
    "SweRuntime",
    "repo_has_base_files",
    "repo_ontology",
    "solution_present",
    "suite_well_formed",
]
