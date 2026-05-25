"""OpenRange public API — the new pack/admission/distill shape.

OpenRange turns a manifest into a content-addressed `Snapshot` through a
layered admission loop, then runs agent episodes against admitted
snapshots. Domain lives in Packs (one per world-family) and TaskFamilies
(one per domain of tasks against that world).

The seam to long-horizon agent memory (BBG / Wayfinder runtimes) is
`distill()`: it consumes a graph + status-event log and produces a
`PackPrior` that any builder can use to bias generation.

This module re-exports the most useful types. The full surface is in
`openrange.core` and `openrange.world_ir`. The BBG ontology is in
`openrange.ontologies.bbg`.
"""

from openrange.core import (
    AdmissionError,
    AdmissionFailure,
    Backing,
    Builder,
    BuildEvent,
    BuildResult,
    EpisodeResult,
    FeasibilityVerdict,
    Manifest,
    ManifestError,
    Mutation,
    OpenRangeError,
    Pack,
    PackError,
    PackPrior,
    RuntimeHandle,
    Snapshot,
    StatusEvent,
    StoreError,
    TaskFamily,
    TaskSeed,
    TaskSpec,
    admit,
    distill,
    snapshot_to_dict,
    validate_task_bindings,
)
from openrange.ontologies.bbg import BBG_ONTOLOGY_ID, wayfinder_ontology
from openrange.world_ir import (
    AttrSpec,
    AttrType,
    Edge,
    EdgeKind,
    GraphPatch,
    Issue,
    Node,
    NodeKind,
    Ontology,
    Role,
    Visibility,
    WorldGraph,
    apply_patch,
    validate,
)

__all__ = [
    # meta-model
    "AttrSpec",
    "AttrType",
    "Edge",
    "EdgeKind",
    "GraphPatch",
    "Issue",
    "Node",
    "NodeKind",
    "Ontology",
    "Role",
    "Visibility",
    "WorldGraph",
    "apply_patch",
    "validate",
    # BBG ontology data
    "BBG_ONTOLOGY_ID",
    "wayfinder_ontology",
    # pack protocols + wire shapes
    "Backing",
    "BuildResult",
    "Builder",
    "EpisodeResult",
    "FeasibilityVerdict",
    "Manifest",
    "Mutation",
    "Pack",
    "PackPrior",
    "RuntimeHandle",
    "TaskFamily",
    "TaskSeed",
    "TaskSpec",
    # admission
    "AdmissionFailure",
    "BuildEvent",
    "Snapshot",
    "admit",
    "snapshot_to_dict",
    "validate_task_bindings",
    # distill
    "StatusEvent",
    "distill",
    # errors
    "AdmissionError",
    "ManifestError",
    "OpenRangeError",
    "PackError",
    "StoreError",
]
