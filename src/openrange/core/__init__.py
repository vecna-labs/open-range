"""OpenRange core — public surface for the new pack/admission/distill shape.

The exports here form the typed binding between OpenRange core and Packs.
Everything is domain-free: no `host`, `endpoint`, `vuln`, `trading`, etc.
That property is enforced by CI (see `.github/workflows/` or
`scripts/check_boundary.sh`).

The runtime side of OpenRange (episode service, dashboard, NPCs, agent
backends, LLM backends) is being re-wired against the new types
incrementally. Until that lands, the runtime modules are not auto-imported
from this package — import them directly (`openrange.core.episode`, etc.)
once they are migrated.
"""

from openrange.core.admit import (
    AdmissionFailure,
    BuildEvent,
    Snapshot,
    admit,
    snapshot_to_dict,
    validate_task_bindings,
)
from openrange.core.distill import (
    StatusEvent,
    distill,
)
from openrange.core.errors import (
    AdmissionError,
    ManifestError,
    OpenRangeError,
    PackError,
    StoreError,
)
from openrange.core.pack import (
    Backing,
    Builder,
    BuildResult,
    EpisodeResult,
    FeasibilityVerdict,
    Manifest,
    Mutation,
    Pack,
    PackPrior,
    RuntimeHandle,
    TaskFamily,
    TaskSeed,
    TaskSpec,
)

__all__ = [
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
    # errors
    "AdmissionError",
    "ManifestError",
    "OpenRangeError",
    "PackError",
    "StoreError",
]
