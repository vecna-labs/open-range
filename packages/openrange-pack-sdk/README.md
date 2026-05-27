# openrange-pack-sdk

The pack-author SDK for [OpenRange](https://github.com/vecna-labs/open-range).

A pack defines a *world type* (ontology + builder + realizer) and one or
more *task families* (how to grade an agent against that world). The SDK
exposes the Protocols, value types, and base errors a pack needs to
implement those contracts — nothing else.

**Zero runtime dependency on OpenRange.** Packs that depend only on
`openrange-pack-sdk` (and `graphschema`) can be authored, versioned, and
published independently of any OpenRange release. The OpenRange runtime
imports this same SDK; both sides agree on the same contract surface.

## Contents

- `Pack`, `Builder`, `TaskFamily`, `RuntimeHandle` — the pack-side
  Protocols / ABCs OpenRange's admission and runtime consume.
- `NPC`, `AgentNPC`, `AgentBackend` — the NPC contract for in-world
  background actors.
- `TaskSpec`, `BuildResult`, `Mutation`, `Snapshot`, `BuildEvent`,
  `FeasibilityVerdict`, `EpisodeResult`, `PackPrior`, `TaskSeed`,
  `LLMRequest`, `LLMResult`, `Backing` — the value types that cross the
  pack ↔ runtime boundary.
- `LLMBackend`, `EpisodeReportLike` — Protocols the family may consume.
- `OpenRangeError`, `PackError`, `ManifestError`, `LLMError`,
  `NPCError`, `AgentBackendError` — the error hierarchy.

## Install

```bash
pip install openrange-pack-sdk
```

## Versioning

Semver. Breaking changes to any Protocol shape, dataclass field, or
exported name require a major version bump.

## Migrating off `openrange.core.pack`

If you authored a pack against the old `openrange.core.pack` /
`openrange.core.errors` / `openrange.llm` / `openrange.npc` /
`openrange.agent_backend` surface, the diff is mostly a sed:

| Was | Now |
|---|---|
| `from openrange.core.pack import Pack, TaskFamily, ...` | `from openrange_pack_sdk import Pack, TaskFamily, ...` |
| `from openrange.core.errors import OpenRangeError, PackError, ManifestError` | `from openrange_pack_sdk import OpenRangeError, PackError, ManifestError` |
| `from openrange.core.admit import Snapshot, BuildEvent` | `from openrange_pack_sdk import Snapshot, BuildEvent` |
| `from openrange.llm import LLMBackend, LLMRequest, LLMResult, LLMError, LLMRequestError, LLMBackendError` | `from openrange_pack_sdk import LLMBackend, LLMRequest, LLMResult, LLMError, LLMRequestError, LLMBackendError` |
| `from openrange.npc import NPC, AgentNPC, NPCError` | `from openrange_pack_sdk import NPC, AgentNPC, NPCError` |
| `from openrange.agent_backend import AgentBackend, AgentBackendError` | `from openrange_pack_sdk import AgentBackend, AgentBackendError` |

Two real semantic changes to know about:

- **`LLMBackendLike` collapsed into `LLMBackend`.** The old loose
  `complete(request: Any) -> Any` Protocol is gone. The single `LLMBackend`
  takes `LLMRequest` and returns `LLMResult`. Custom fakes must update
  their signatures.
- **`LLMBackend` no longer requires `preflight()`.** Concrete impls MAY
  declare one (`CodexBackend` does); callers check via `hasattr`. Minimal
  in-process fakes no longer need a no-op `preflight`.

Concrete classes (`CodexBackend`, `StrandsAgentBackend`,
`CodexAgentBackend`, the `admit()` function, `AdmissionFailure`,
`PackRegistry`, `NPCRegistry`, the `EpisodeService` runtime) STAY in
`openrange`. Packs should never need them — the runtime supplies any
concrete backend via `RunConfig.npc_agent_backend` and the harness drives
admission.
