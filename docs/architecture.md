# OpenRange architecture

How the pieces fit together. Three diagrams from zoom-out to zoom-in.

## Package dependencies

```mermaid
flowchart BT
    G["graphschema<br/><i>WorldGraph, Node, Edge,<br/>Ontology, GraphPatch</i>"]
    S["openrange-pack-sdk<br/><b>THE CONTRACT</b><br/><i>Pack, Builder, TaskFamily,<br/>RuntimeHandle, NPC, AgentBackend,<br/>TaskSpec, Snapshot, Mutation, ...</i>"]
    R["openrange<br/><b>THE RUNTIME</b><br/><i>admit, EpisodeService, PackRegistry,<br/>dashboard, CodexBackend</i>"]
    P["packs/cyber_webapp<br/><b>A PACK</b><br/><i>WebappPack, WebappBuilder,<br/>WebappBuild, WebappPentest,<br/>WebappRuntimeHandle, NPCs</i>"]

    S --> G
    R --> S
    P --> S
    P --> G
```

Arrow = "depends on". The runtime and the pack each depend on the SDK; **they never import each other**. The boundary `scripts/check_boundary.sh` enforces. A pack can live in its own repo, be versioned independently, and only pin `openrange-pack-sdk` + `graphschema`.

## Type hierarchy

The SDK owns the contracts; concrete cyber_webapp classes implement them.

```mermaid
classDiagram
    direction LR

    class Pack {
        <<ABC>>
        id: str
        version: str
        ontology() Ontology
        invariants() list
        make_builder(prior) Builder
        realize(graph, backing) RuntimeHandle
        task_families() list[TaskFamily]
    }
    class Builder {
        <<ABC>>
        build(manifest) BuildResult
        repair(prev, errors, infeasible) BuildResult
        evolve(snapshot, mutation) GraphPatch
    }
    class TaskFamily {
        <<ABC>>
        id: str
        pack_id: str
        generate(graph, manifest, prior) list[TaskSpec]
        check_feasibility(graph, task) FeasibilityVerdict
        check_success(graph, task, final_state) EpisodeResult
        available_mutations(snap, reports) tuple[Mutation]
    }
    class RuntimeHandle {
        <<Protocol>>
        reset()
        surface() Mapping
        poll_events() tuple
        terminal() tuple
        checkpoint() Any
        restore(state)
        collect() Mapping
        stop()
    }
    class NPC {
        <<ABC>>
        actor_id: str
        step(interface)
        start(context)
        stop()
    }
    class AgentBackend {
        <<Protocol>>
        preflight()
        build_agent(...) AgentSession
    }

    class WebappPack { id = "webapp" }
    class WebappBuilder { procedural sampler }
    class WebappBuild { id = "webapp.build" }
    class WebappPentest { id = "webapp.pentest" }
    class WebappRuntimeHandle { spawns HTTP subprocess }
    class OfficePersona { LLM-backed office worker }
    class StrandsAgentBackend { wraps strands.Agent }

    Pack <|.. WebappPack
    Builder <|.. WebappBuilder
    TaskFamily <|.. WebappBuild
    TaskFamily <|.. WebappPentest
    RuntimeHandle <|.. WebappRuntimeHandle
    NPC <|-- AgentNPC
    AgentNPC <|.. OfficePersona
    AgentBackend <|.. StrandsAgentBackend

    WebappPack ..> WebappBuilder : make_builder()
    WebappPack ..> WebappRuntimeHandle : realize()
    WebappPack ..> WebappBuild : task_families()
    WebappPack ..> WebappPentest : task_families()
```

Solid arrows (`<|--`) are inheritance. Dashed arrows (`<|..`) are Protocol implementation. Open dashed (`..>`) are "produces/uses." The SDK side (top) declares the contract; the cyber_webapp side (bottom) is one implementation. A trading pack would add `TradingPack`, `TradingBuilder`, etc. as siblings.

## Build → Episode flow

```mermaid
sequenceDiagram
    autonumber
    participant H as Harness
    participant OR as openrange Runtime
    participant Reg as PackRegistry
    participant P as WebappPack
    participant B as WebappBuilder
    participant F as TaskFamily (build/pentest)
    participant RT as WebappRuntimeHandle

    Note over H,RT: Build phase
    H->>OR: OpenRangeRun.build(manifest)
    OR->>Reg: resolve("webapp")
    Reg-->>OR: WebappPack instance
    OR->>P: make_builder(prior)
    P-->>OR: WebappBuilder
    OR->>B: build(manifest)
    B-->>OR: BuildResult(graph, tasks)
    OR->>OR: admit: validate ontology + invariants
    OR->>F: check_feasibility(graph, task) × all tasks
    F-->>OR: FeasibilityVerdict
    OR-->>H: Snapshot (frozen, content-addressed)

    Note over H,RT: Episode phase
    H->>OR: episode_service(snap).start_episode(snap, task.id)
    OR->>P: realize(graph, Backing.PROCESS)
    P-->>OR: WebappRuntimeHandle
    OR->>RT: reset()
    RT->>RT: spawn subprocess, serve HTTP
    OR-->>H: EpisodeHandle (with base_url)

    Note over H,RT: ... agent acts: HTTP, file writes, etc ...

    H->>OR: stop_episode(handle)
    OR->>RT: collect()
    RT-->>OR: final_state
    OR->>F: check_success(graph, task, final_state)
    F-->>OR: EpisodeResult(success, subgoals)
    OR-->>H: EpisodeReport
```

The harness only ever talks to `openrange`. Everything pack-specific flows through the SDK's Pack / Builder / TaskFamily / RuntimeHandle methods. The pack's WebappPack / Builder / etc. never call into `openrange` — they're called BY it.

## Plain-English glossary

- **Pack** — a class that defines a *kind of world* (its ontology, how to sample one, how to realize it, which task families it ships). One Pack = one world type; each `build()` against it produces a fresh concrete world.
- **Builder** — the Pack's hand for *producing* concrete world graphs from a manifest. Procedural sampler, LLM pipeline, hand-coded — anything that returns a `BuildResult`.
- **Snapshot** — one concrete admitted world. Frozen, content-addressed (`snapshot_id == graph.content_hash()`).
- **TaskFamily** — a *category of tasks* the agent can be given inside any snapshot of this pack's world type. Different families exercise different agent skills against the same world.
- **RuntimeHandle** — the running realization of a snapshot. The actual process serving HTTP / files / whatever the agent acts against.
- **NPC** — a non-player actor inside the world's runtime, optionally LLM-backed (`AgentNPC`).
- **AgentBackend** — the SDK Protocol any LLM agent loop (Strands, Codex, custom) implements. The harness wires a concrete backend into `RunConfig`; NPCs receive it via context.

## Related docs

- [start_here.md](start_here.md) — overview + vocabulary in prose
- [../CONTRACTS.md](../CONTRACTS.md) — wire shapes and cross-pack invariants
- [../DESIGN.md](../DESIGN.md) — rationale behind the pack / admission split
