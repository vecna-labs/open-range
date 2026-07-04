# OpenRange

OpenRange is a domain-agnostic environment platform for training and evaluating agents.

The core idea is simple: bring your own agent harness, and run it against generated worlds with a stable episode lifecycle. OpenRange owns world construction, task admission, runtime coordination, episode checking, and observability. It does not own the model, agent framework, tool harness, training loop, or reward policy.

This document is a high-level overview. See [CONTRACTS.md](../CONTRACTS.md) for the wire shapes and cross-pack invariants, and [DESIGN.md](../DESIGN.md) for the rationale behind the split.

## What OpenRange does

OpenRange turns a user request into an admitted world that an agent can act inside.

```text
manifest + pack + builder
        ↓
world graph + tasks
        ↓
admission (structural / ontology / pack invariants / task bindings / feasibility)
        ↓
frozen world snapshot
        ↓
agent episode
        ↓
structured EpisodeResult
```

The goal is not to force every domain into one Gym-style API. The goal is to let different worlds — cyber ranges, trading environments, robotics tasks, enterprise simulations — fit into the same build, admission, runtime, and evaluation flow.

The agent interacts with whatever surface the world exposes: HTTP endpoints, files, shells, MCP tools, simulator APIs, browser sessions, or custom interfaces.

## Core objects

### Manifest

The manifest describes what the user wants built.

It is a free-form `Mapping[str, Any]`. The only key core ever reads is `pack.id` (the registered pack to admit against). Every other key is the pack's contract; core never branches on a manifest field.

See [manifest.md](manifest.md) for the cross-pack invariant and how pack-specific keys work.

### Pack

A pack is the reusable starting point for a **family of worlds**.

A Pack owns:

- an `Ontology` (node kinds, edge kinds, attribute schemas)
- pack invariants (graph-wide callables the validator runs)
- a `Builder` factory (`make_builder(prior)`)
- a realizer (`realize(graph, backing) -> RuntimeHandle`)
- one or more `TaskFamily` classes

A TaskFamily owns **one domain of tasks** against that pack's world. The same pack can ship multiple families. The built-in `webapp` pack ships `webapp.build` and `webapp.pentest` against the same world graph: same nodes, different entrypoints, different goals, different success criteria. This split is where the word "domain" lives — on the TaskFamily, not on the Pack.

### Builder

The builder turns a manifest and a pack into a candidate `BuildResult`.

A builder may be handwritten Python, procedural generation, an LLM pipeline, search/sampling, a domain-specific generator, or a hybrid. OpenRange does not require the builder to be an LLM.

The builder protocol has three operations:

```text
build(manifest)                       -> BuildResult (graph + tasks)
repair(prev, errors, infeasible)      -> BuildResult (when admission rejects)
evolve(snapshot, mutation)            -> GraphPatch (apply a curriculum move)
```

`build` is required. `repair` is optional — without it, admission won't retry. `evolve` defaults to returning the mutation's patch verbatim; packs that want to refine the patch override it.

### World graph

OpenRange represents each generated world as a graph before turning it into runnable artifacts.

The graph is content-addressed and **timeless** — the graph IS its content, no timestamps inside. Two identical builds (same builder, same manifest, same seed → same graph) share a snapshot id and are interchangeable for every downstream purpose.

The graph answers two questions:

```text
What exists?
How is it connected?
```

The graph is the build plan OpenRange uses to produce the runtime; it is not the runtime itself.

### Task

A task is what the agent is asked to do inside an admitted world.

`TaskSpec` has seven fields:

```text
id                  - unique within the snapshot
instruction         - what the agent sees
entrypoints         - tuple of node-ids in the world graph where the agent starts acting
goal_nodes          - tuple of node-ids that count as completion (may be HIDDEN)
feasibility_check   - TaskFamily id whose check_feasibility decides "solvable here?"
success_check       - TaskFamily id whose check_success reads the realizer's final state
meta                - free-form mapping the pack and harness agree on; serialized only when non-empty
```

Entrypoints and goal_nodes live on the task, never as node roles — two tasks against the same world may entrypoint different nodes. `feasibility_check` and `success_check` are **handles**, not exec'd source; the pack's `task_family(id)` resolves them to a class whose methods run the check.

### Episode

An episode runs an agent against a realized snapshot. The realizer (`RuntimeHandle`) exposes eight methods:

```text
reset()             - prepare a clean run state
surface()           - agent-facing IO surface (URLs, file roots, MCP endpoints)
poll_events()       - drain side-effect events since the last poll
terminal()          - has the agent finished? -> (done, reason)
checkpoint()        - capture an opaque state snapshot
restore(state)      - restore from a checkpoint payload
collect()           - structured final state at episode end
stop()              - tear down running processes / services
```

The episode loop calls `collect()` at the end; the TaskFamily's `check_success(graph, task, final_state)` reads that mapping and returns an `EpisodeResult`.

## Admission

Admission is a **layered gate** between generation and execution:

```text
1. structural       : id formats, edge endpoints reference real nodes
2. ontology         : required attrs, enums/REFs, kind agreement
3. pack invariants  : Tier-3 callables the pack ships (Pack.invariants())
4. task bindings    : entrypoints/goal_nodes exist; entrypoints not HIDDEN
5. task feasibility : each TaskFamily's check_feasibility(graph, task)
```

Layers 1+2 catch malformed worlds. Layer 3 catches structurally-valid but semantically-broken worlds. Layer 4 catches mis-bound tasks. Layer 5 catches well-formed worlds no one can actually solve. Each layer catches a different bug; all are required.

If admission rejects a candidate, core calls `builder.repair(prev, errors, infeasible)` (up to a configured budget). A task is never accepted just because the builder generated it.

Admission output is a `Snapshot`:

```text
snapshot_id      = graph.content_hash()
ontology_id      = pack.ontology().id
graph            = WorldGraph (timeless)
tasks            = tuple[TaskSpec, ...]
lineage          = flat dict (manifest, pack id+version, attempts, builder meta)
history          = tuple[BuildEvent, ...] (build/validate/feasibility/repair/freeze)
```

The world graph is timeless on purpose — content-addressed reproducibility. The build PROCESS still has a story worth keeping (which pass ran, what a repair changed); that story lives in `history`, BESIDE the graph, never inside it.

## Runtime backing

A world can be backed by real systems, synthetic systems, or a mix.

A **real backing** runs the actual thing or a close stand-in:

```text
real container
real web service
real binary
real shell
sandboxed broker API
MuJoCo simulator
```

A **synthetic backing** imitates the thing with cheaper code:

```text
Python state machine
scripted fake service
in-memory order book
symbolic network state
mock endpoint backed by generated state
```

A **hybrid backing** combines both. For example, a trading world might expose a broker-like HTTP API, but the API is backed by a Python state machine instead of a real broker. A cyber world might run the vulnerable web app as a real container, but simulate background employees and external systems.

The manifest can request the desired backing. The pack decides what it can support.

## NPCs and multi-actor worlds

A world can include non-player characters: scripted actors, LLM-driven personas, other agents, background users, defenders, attackers, counterparties, or external systems.

NPCs live inside the world runtime. Their actions can affect the state the agent observes and the final state the TaskFamily's success check inspects.

Examples:

```text
a defender rotating credentials
a user responding to phishing email
a trading counterparty placing orders
a human-like persona answering questions
a background process writing logs
```

### Adding a persona NPC (no per-pack code)

`PersonaAgent` (in `openrange_pack_sdk.npcs.persona_agent`) is a generic,
config-driven NPC that works in any pack: it wraps whatever callables your pack
surfaces into the per-tick interface, so you define a believable inhabitant with
data, not code. Register it once and declare instances in the manifest:

```toml
# your pack's pyproject.toml
[project.entry-points."openrange.npcs"]
"mypack.persona" = "openrange_pack_sdk.npcs.persona_agent:factory"
```

```json
{"npc": [{"type": "mypack.persona", "count": 3, "config": {
  "name": "Dana", "role": "accountant",
  "goal": "reconcile invoices with a colleague",
  "tools": ["mail_send", "mail_read"],
  "channels": ["finance"], "cadence_ticks": 5}}]}
```

Every persona entry needs a non-empty `name` — the dashboard seats and animates
personas by that name (an anonymous persona still runs but isn't seated).

On a **graded** run, give personas only the world-mediated comms tools
(`mail_send`/`mail_read`/`chat_post`/`chat_read`), not `http_get`: comms are
attributed to the sender and drained separately, but persona `http_get` traffic
lands in the same request log as the system-under-test — unattributed — so it
pollutes `requests_made`/`leaked_secret_ids` and the pentest `reached_endpoint`
subgoal (per-actor request tagging is deferred; see the PR). It's fine for a
non-graded/demo world.

Config keys:

| key | meaning |
|-----|---------|
| `name`, `role`, `backstory`, `tone` | who the persona is (rendered to the system prompt) |
| `goal` | its own independent objective (what stops assistant-like helpfulness) |
| `traits` | how it comes across — flags `{"terse": true}` or numeric intensities `{"curious": 0.9, "patient": 0.3}` |
| `channels` | chat channels it uses (grounds `chat_post`/`chat_read`) |
| `tools` | which surface keys it may use; only the ones your pack actually provides are bound |
| `cadence_ticks` | acts once every N ticks (auto phase-staggered across a population) |
| `long_term_memory` | opt-in scoped note store (default off) |

For a **diverse population** instead of `count=n` clones, generate configs with
`sample_persona` (deterministic, no model) and pass your pack's own role
vocabulary:

```python
from openrange_pack_sdk import sample_persona
roster = [sample_persona(i, roles=["accountant", "it admin", "exec"]) for i in range(50)]
# each gets a distinct name, role, and a numeric trait vector
```

Comms are through the world: your pack surfaces `mail_send`/`mail_read`/
`chat_post`/`chat_read` from `surface_extras()` (see
`openrange_pack_sdk.comms.surface_mailbox`/`surface_chat`) and drains the stores
in `collect_extras()` for grading — the persona injects its own identity as the
sender, so cover traffic is attributable and unspoofable.

**Model:** a persona needs an `AgentBackend`. `RunConfig.npc_llm_model` (a
provider string) drives the default Bedrock backend; to run a **local** model,
pass a custom `RunConfig.npc_agent_backend` wrapping e.g.
`strands.models.OllamaModel` — see `examples/persona_eval.py` for the shape.

The cyber webapp registers both `cyber.persona` (this generic class) and the
older bespoke `cyber.office_persona`; prefer `cyber.persona` for new worlds.

Deterministic properties are CI-gated (population diversity via role entropy,
comms sender is never forgeable, the persona prompt doesn't read like an
assistant — see `openrange_pack_sdk.npcs.metrics`). Live believability against a
real model, a non-degeneracy NPC-on/off A/B, and per-actor HTTP tagging are
deferred to a follow-up.

## Episode checks and rewards

OpenRange checks what happened. It does not define the training reward.

After an episode, the TaskFamily's `check_success(graph, task, final_state)` reads the realizer's collected final state against the task and returns an `EpisodeResult`:

```json
{"success": true}
```

```json
{"success": false, "reason": "admin credential was not recovered"}
```

```json
{"success": true, "subgoals": {"found_login": true, "exploited_sqli": true, "exfiltrated_secret": true}}
```

A training adapter can map this result into scalar rewards, dense rewards, preference data, SFT traces, GRPO/PPO signals, or evaluation metrics.

## World evolution

OpenRange does not define a curriculum algorithm.

Each `TaskFamily` may implement `available_mutations(snapshot, reports)`, returning structured `Mutation` objects — each a `GraphPatch` tagged with a direction (`harden` / `soften` / `diversify`), a relevance score, and the family that proposed it.

A curriculum policy selects mutations; `Builder.evolve(snapshot, mutation)` applies them as graph patches. Proposed edits go through the same admission gate as the initial build.

## Observability and lineage

Every admitted world is inspectable and reproducible.

OpenRange tracks:

```text
manifest
pack id + version
build history (BuildEvent stream)
world graph (content-addressed)
tasks
admission verdicts
runtime events
episode results
```

The dashboard shows what was generated, why a task was admitted or rejected, which world snapshot an episode used, and how the world changed over time.

See [dashboard.md](dashboard.md).

## Design boundaries

OpenRange owns:

```text
world construction
pack contracts
builder interface
admission
runtime coordination
task feasibility checks
episode checks
structured results
world lineage
observability
```

OpenRange does not own:

```text
the agent implementation
the model
the tool harness
the training algorithm
reward shaping policy
rollout infrastructure
```

This boundary is intentional. It lets OpenRange support many domains and training setups without becoming a full agent framework.

## Further reading

- [CONTRACTS.md](../CONTRACTS.md) — wire shapes and cross-pack invariants
- [DESIGN.md](../DESIGN.md) — rationale behind the pack / admission split
- [api.md](api.md) — the lifecycle the harness sees
- [dashboard.md](dashboard.md) — inspection surface
- [manifest.md](manifest.md) — the one key core reads, and a pointer to pack-specific keys
