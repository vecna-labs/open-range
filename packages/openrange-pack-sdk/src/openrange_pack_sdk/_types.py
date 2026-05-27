"""Value types that cross the pack ↔ runtime boundary."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from graphschema import GraphPatch, Ontology, WorldGraph

from openrange_pack_sdk._errors import LLMRequestError

Manifest = Mapping[str, Any]
"""The harness-supplied build request. OpenRange treats it as an opaque
mapping — only ``pack.id`` is read by core. Every other key is the pack's
contract: validate fields you depend on, ignore the rest. There is no
TypedDict here because manifests are free-form; pack authors are expected
to publish their own schema in pack docs."""


class Backing(StrEnum):
    """Runtime substrate the realizer maps the graph onto.

    A Pack's ``realize(graph, backing)`` branches over this value; packs
    raise ``NotImplementedError`` for backings they do not support (the
    cyber webapp realizer supports only ``PROCESS``).
    """

    PROCESS = "process"
    CONTAINER = "container"
    SIMULATOR = "simulator"
    HYBRID = "hybrid"


@dataclass(frozen=True)
class TaskSpec:
    """Goal nodes may be HIDDEN; entrypoints may not."""

    id: str
    instruction: str
    entrypoints: tuple[str, ...]
    goal_nodes: tuple[str, ...]
    feasibility_check: str
    success_check: str
    meta: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class TaskSeed:
    """A hint a TaskFamily's `generate()` may consult. Mutable so callers
    can re-tag `family` after the seed is produced."""

    theme: str
    anchor_kinds: list[str]
    suggested_goal_kinds: list[str]
    difficulty: float
    evidence: int = 1
    family: str | None = None


@dataclass
class PackPrior:
    """Generic graph statistics the Builder INTERPRETS; never dictates outputs."""

    source: str
    ontology: Ontology
    topology: Mapping[str, Any]
    task_seeds: list[TaskSeed] = field(default_factory=list)
    difficulty: Mapping[str, float] = field(default_factory=dict)
    coverage: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class BuildResult:
    """The candidate world + tasks from `Builder.build()`. `admission_meta`
    rides into `Snapshot.lineage`; opaque to core."""

    graph: WorldGraph
    tasks: list[TaskSpec]
    admission_meta: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FeasibilityVerdict:
    feasible: bool
    reason: str = ""


@dataclass(frozen=True)
class EpisodeResult:
    """Structured outcome — never a scalar reward. Harness-side
    training adapters do the shaping."""

    success: bool
    subgoals: Mapping[str, bool] = field(default_factory=dict)
    reason: str = ""


@dataclass(frozen=True)
class Mutation:
    """One curriculum move. `direction` ∈ {"harden","soften","diversify"};
    `relevance` ∈ [0,1]."""

    patch: GraphPatch
    direction: str
    relevance: float
    family: str
    note: str = ""


@dataclass(frozen=True)
class BuildEvent:
    """One entry in `Snapshot.history`.

    `phase` ∈ {"build", "validate", "feasibility", "repair", "freeze", "evolve"}.
    """

    seq: int
    phase: str
    detail: str
    refs: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "seq": self.seq,
            "phase": self.phase,
            "detail": self.detail,
        }
        if self.refs:
            d["refs"] = list(self.refs)
        return d


@dataclass(frozen=True)
class Snapshot:
    """An admitted, frozen world. `snapshot_id == graph.content_hash()`."""

    snapshot_id: str
    ontology_id: str
    graph: WorldGraph
    tasks: tuple[TaskSpec, ...]
    lineage: Mapping[str, Any]
    history: tuple[BuildEvent, ...] = ()


@dataclass(frozen=True, slots=True)
class LLMRequest:
    prompt: str
    system: str | None = None
    json_schema: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        if self.json_schema is None:
            return
        try:
            json.dumps(self.json_schema)
        except TypeError as exc:
            raise LLMRequestError("json_schema must be JSON serializable") from exc

    def as_prompt(self) -> str:
        if self.system is None:
            return self.prompt
        return f"{self.system}\n\n{self.prompt}"


@dataclass(frozen=True, slots=True)
class LLMResult:
    text: str
    parsed_json: Mapping[str, object] | None = None
