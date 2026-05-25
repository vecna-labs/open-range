# OpenRange Roadmap

> [!NOTE]
> Direction, not a release schedule.

OpenRange just landed a substantial rewrite around a new pack /
admission / distill shape. See [DESIGN.md](DESIGN.md) for the
architecture and [CONTRACTS.md](CONTRACTS.md) for the wire formats.
This roadmap is what comes next.

## What's shipped (v0.1.0)

- ✅ Typed-property-graph meta-model (`Node`, `Edge`, `WorldGraph`,
  `Ontology`, three-tier `validate`, `GraphPatch`, `Role`,
  `Visibility`).
- ✅ The `bbg@0.1.0` ontology as built-in declarative data.
- ✅ `Pack` / `Builder` / `TaskFamily` protocols + `PackPrior` /
  `TaskSeed` / `TaskSpec` / `BuildResult` / `Mutation` /
  `RuntimeHandle`.
- ✅ Layered admission loop (`admit()`): structural + conformance +
  pack invariants + task bindings + per-task feasibility, with a
  repair budget.
- ✅ Content-addressed `Snapshot` with build history (`BuildEvent`s
  living beside the timeless graph).
- ✅ `distill(graph, status_log) → PackPrior` — the seam any
  BBG-emitting harness can drive through.
- ✅ Reference Pack: `webapp` — one world-family, two TaskFamilies
  (`webapp.build` + `webapp.pentest`), with three pack invariants
  and a deterministic v1 builder.
- ✅ CI boundary checks: core is domain-free; no harness imports.

## What's next

### Runtime layer (re-wiring against the new shape)

The runtime side (episode service, HTTP backing, NPC threads, agent
backends, dashboard) was removed during the refactor and needs to come
back against the new `Pack.realize(graph, backing) → RuntimeHandle`
contract.

- 🚧 **`run_episode()` re-wire** — replace the removed
  `EpisodeService` with one that consumes a `RuntimeHandle` from
  `Pack.realize`, runs the agent against `surface()`, reads
  `collect()` at the end, dispatches to the TaskFamily's
  `check_success`.
- 🚧 **`RuntimeBacking` re-wire** — HTTPBacking against the new node
  shape. Per-node-kind dispatch from inside `Pack.realize`.
- 🚧 **AgentBackend re-wire** — `StrandsAgentBackend` and
  `CodexAgentBackend` reconnected to `run_episode`.
- 🚧 **NPCs re-wire** — `NPC` / `AgentNPC` bound to nodes with
  `role=NPC` in the world graph.
- 🚧 **Dashboard re-wire** — topology view, build-event timeline,
  episode-event stream against the new `Snapshot` shape.

### webapp pack v2

The v1 webapp pack is hand-authored — same world every seed. v2 lifts
that to a richer demo:

- 🟢 **Procedural sampler** consuming `PackPrior.topology` and
  `PackPrior.task_seeds` to bias generation.
- 🟢 **LLM-driven instruction generation** in TaskFamily.generate().
- 🟢 **`available_mutations`** for curriculum evolution (harden /
  soften / diversify directives carrying `GraphPatch`).
- 🟢 **Flask realizer** + AST splice of weakness templates into the
  generated service code. The old `cyber_webapp` pack had this for
  the v0 shape; it needs porting against the new `RuntimeHandle`.
- 🟢 **Smoke-test collection** in `RuntimeHandle.collect()` so
  `webapp.build.check_success` can actually verify the agent's edits.

### The flywheel

- 🟡 **BBG harness adapter** — formalize the JSON wire format for
  what any harness emits. The shape lives in
  [CONTRACTS.md](CONTRACTS.md) §6; we want at least one reference
  decoder + a fixture file under `tests/fixtures/bbg/`.
- 🟡 **Multi-trajectory `distill`** — today `distill` takes a single
  graph; the flywheel needs to merge evidence across many runs.
- 🔮 **Pack-induction path** — `distill(graph, into=None)` proposes
  node kinds but not edge kinds. Closing that gap would let a fresh
  domain bootstrap a pack from purely observed structure.

### New packs

The library is already domain-free; the test of that is more packs
that exercise the shape from different angles.

- 🟢 **Trading pack** — order books, venues, strategies; goals like
  "discover an arbitrage" or "audit a counterparty's risk".
- 🟢 **Compliance pack** — controls, evidence, regulations;
  TaskFamilies for audit and remediation.
- 🟢 **SWE pack** — repos, files, tests; TaskFamilies for refactor,
  bug-fix, feature-add. Probably the closest analog to the agent-SDK
  use case.

## Out of scope (for now)

- A managed cloud, hosted runtimes, billing.
- A specific RL/SFT training loop.
- A specific BBG runtime implementation. OpenRange consumes the JSON
  wire format; harnesses (vecna's wayfinder; others) produce it.
