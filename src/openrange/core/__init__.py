"""OpenRange runtime-side core API.

Pack-author types (Pack, TaskFamily, TaskSpec, etc.) live in
``openrange_pack_sdk``. This module re-exports only runtime-side helpers:
admission, curriculum, registries, store, turns.
"""

from openrange.core.admit import (
    AdmissionFailure,
    admit,
    snapshot_to_dict,
    validate_task_bindings,
)
from openrange.core.curriculum import (
    CurriculumPolicy,
    Direction,
    auto_evolve,
    direction_from_reports,
)
from openrange.core.errors import (
    AdmissionError,
    EpisodeRuntimeError,
    StoreError,
)
from openrange.core.pack import (
    PACKS,
    PackRegistry,
)
from openrange.core.store import SnapshotStore
from openrange.core.turn import ActorTurn

__all__ = [
    "PACKS",
    "ActorTurn",
    "AdmissionError",
    "AdmissionFailure",
    "CurriculumPolicy",
    "Direction",
    "EpisodeRuntimeError",
    "PackRegistry",
    "SnapshotStore",
    "StoreError",
    "admit",
    "auto_evolve",
    "direction_from_reports",
    "snapshot_to_dict",
    "validate_task_bindings",
]
