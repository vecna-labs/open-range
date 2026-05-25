# OpenRange — design

Status: design landed; pack/admission/distill foundation is shipped and
green; runtime side (episode service, dashboard, NPCs, agent backends)
is being re-wired against this shape.

Audience: contributors to OpenRange, to packs built on it, or to agent
harnesses that want to drive its flywheel.

This document describes *what* and *why*. The *how* lives in the modules
(`world_ir.py`, `ontologies/bbg.py`, `core/pack.py`, `core/admit.py`,
`core/distill.py`). For frozen wire formats, see `CONTRACTS.md`.

---

## 1. The one-paragraph summary

**OpenRange** turns a manifest into a frozen, content-addressed `Snapshot`
through a layered admission loop, then runs agent episodes against
admitted snapshots. Domain lives entirely in `Pack`s (one per
world-family — e.g. `webapp`) and `TaskFamily`s (one per *domain of tasks*
against that world — e.g. `webapp.build`, `webapp.pentest`). The library
is domain-free at its core: nothing in `src/openrange/world_ir.py` or
`src/openrange/core/` names a `host`, `vuln`, `endpoint`, or any other
domain word.

The seam to long-horizon agent memory (e.g. the BBG/Wayfinder graphs an
agent runtime maintains) is `distill(graph, status_log) → PackPrior`. The
prior is JSON; any agent harness emitting BBG-shaped JSON (see
`CONTRACTS.md` §6) can drive the flywheel. OpenRange does not import any
specific harness; the harness does not import OpenRange. The connection
is the wire format.

---

## 2. The shared foundation: a typed graph

Everything in OpenRange is a **typed property graph**: nodes and edges,
each with a `kind` and an attribute bag. The type system has two tiers,
and the split is the most important idea in the whole design:

- **The meta-model** — `Node`, `Edge`, `WorldGraph`, and the validator —
  is fixed. It lives in `openrange.world_ir` and never changes per use
  case.
- **An ontology** — the set of allowed node kinds, edge kinds, their
  attributes and constraints — is declared per use case. It is plain
  data; one generic validator can check *any* graph against *any*
  ontology. Nothing hard-codes what a `service` or a `thought` means.

OpenRange ships one ontology as built-in data: `bbg@0.1.0` (from
`openrange.ontologies.bbg`). Packs declare their own ontologies.

### Kinds vs. roles

A node carries a `kind` (domain vocabulary — `service`, `thing`) and a
set of `roles` (a small, fixed, cross-cutting vocabulary). Generic code
reads `roles`; it never branches on `kind`.

Roles are **world-absolute**: true regardless of what task is being run.
The current set is `ACTOR`, `NPC`, `EXTERNAL`.

What is *not* a role: anything **task-relative**. "Entrypoint" and "goal"
are not roles, because the same world serves multiple tasks that
entrypoint different nodes. They are declared per task (see §3).

Rule of thumb: world-absolute facts are roles or node attributes;
task-relative facts belong to a task; pack- or graph-specific state goes
in the attribute bag, never in the role enum.

---

## 3. The OpenRange flywheel

OpenRange turns a request into a frozen, runnable world that an agent
can be trained or evaluated against.

### 3.1 Objects

**Pack** — the reusable starting point for one *world-family* (e.g.
`webapp`). A pack owns the ontology, the builder, and the realizer. A
pack is **not** per-domain; a webapp is a webapp whether you build it or
attack it.

**TaskFamily** — a "domain" of tasks posed against a world (e.g.
`webapp.build`, `webapp.pentest`). A task family owns task generation,
entrypoint selection, feasibility checking, and success checking. One
pack offers one or more task families. **This is where "domain" actually
lives.**

**Builder** — turns a manifest into a candidate world + tasks. May be
procedural, search-based, an LLM pipeline, or a hybrid. OpenRange does
not care which.

**World graph** — the concrete world: entities, connections, hidden
state. Carries no task markers — a world is task-neutral.

**TaskSpec** — what an agent is asked to do *in* a world. Three parts:
an instruction, entrypoints (where it acts), and goal_nodes (what counts
as completion). Entrypoints and goal nodes are declared here, per task,
because they are task-relative.

**Snapshot** — an admitted, frozen, content-addressed world. Episodes
run against snapshots.

**PackPrior** — the BBG → Builder seam; a generic graph-statistics shape
distilled from agent memory.

### 3.2 The build-and-admission loop

A world is not trusted because a builder produced it. It must pass
admission, which is **layered on purpose**:

```
  request
     |
  builder.build(manifest)   -> candidate world graph + tasks
     |
  1. structural validation  -> ids, edge-to-node shape
  2. ontology conformance   -> kinds, required attrs, enum/REF
  3. pack invariants        -> Tier-3 callables the pack ships
  4. task bindings          -> entrypoints/goals exist; entrypoints not HIDDEN
  5. task feasibility       -> each TaskFamily's check_feasibility(graph, task)
     |
  pass?  -- no -->  builder.repair(prev, errors, infeasible), loop up to budget
     |
    yes
     |
  freeze -> Snapshot        -> content-hashed, immutable
```

Layers 1+2 catch malformed worlds. Layer 3 catches structurally-valid
but semantically-broken worlds (a webapp with no endpoints). Layer 4
catches mis-bound tasks (a pentest task entrypointing a hidden secret).
Layer 5 catches well-formed worlds that no one can actually solve.

Each layer catches a different bug; all of them are required.

The world graph itself is **timeless** — content-addressed, no
timestamps inside it, so two builds that produce the same world share
one snapshot id. The build *process* still has a story worth keeping
(which pass ran, what a repair changed, why an attempt was rejected);
that story is recorded as an ordered list of `BuildEvent`s in the
snapshot's `history`, *beside* the graph, never inside it.

### 3.3 The boundary

OpenRange **core** owns *types and generic algorithms*: the meta-model,
the validator, the admission loop, the snapshot/episode lifecycle, the
distill seam. It never names a domain concept.

A **pack** owns *values and domain functions*: the ontology value, the
builder, the realizer, the task families, the checks.

The `Pack` / `Builder` / `TaskFamily` protocols (`core/pack.py`) are the
binding surface. The enforceable test: `grep` any core file for any
domain word (`host`, `endpoint`, `vuln`, `trading`); zero hits. The day
one appears, the boundary has leaked.

---

## 4. The distill seam — connecting agent memory to world generation

### 4.1 BBG is not the world graph

It is tempting to feed a BBG straight into OpenRange as a world. This is
wrong, for four reasons:

- A world needs **hidden state** for the agent to discover. A BBG
  contains only what an agent already saw — replayed as a world, nothing
  is left to find.
- `thought` nodes are **agent inferences**, not facts. Baking them into
  a world trains the next agent against the last agent's conclusions.
- A BBG is **shaped like past trajectories** — survivorship-biased
  toward what the agent could already do.
- A world must be **closed** for feasibility checking; a BBG is open (a
  missing node means "not yet seen", not "does not exist").

The two graphs share a meta-model and play opposite roles: the world
graph is *ontic* (what is), the BBG is *epistemic* (what was seen).

### 4.2 BBG distills into a PackPrior

The correct connection: a BBG is distilled into a **`PackPrior`** — an
optional generation prior that informs a builder. It is a prior, not a
pack, because a prior carries only *generative knowledge*; a pack also
needs a realizer and check code, which cannot be synthesized from
observation.

`distill()` maps the thing/thought seam onto OpenRange:

```
  things    -- structure -->  world node-kind frequencies, topology
                              (salient things weighted far above trajectory)
  thoughts  -- difficulty -->  task seeds; clusters of refuted/open thoughts
                              mark where the world is hard; confirmed thoughts
                              mark where hidden state should be placed
  coverage  -- extrapolation -> per-region observation density, so the builder
                              generates at the edges of what was seen rather
                              than replaying it
  goal     --  heuristic  -->  kinds of things at the ends of productive
                              paths become candidate goal kinds
```

`distill()` emits only **generic graph statistics**. The builder
interprets those into domain decisions. This is what keeps `distill()`
domain-agnostic — it never emits a pack-specific key.

### 4.3 The flywheel

```
        +---------------------------------------------------+
        |                                                   |
        v                                                   |
   agent runs real task                                      |
        |                                                    |
   (some harness) grows a BBG-shaped graph                    |
        |                                                    |
   serialize BBG state -> JSON                                |
        |                                                    |
   openrange.distill(graph, status_log) -> PackPrior          |
        |                                                    |
   Pack.make_builder(prior)                                   |
        |                                                    |
   builder samples a world  -> admission  -> Snapshot         |
        |                                                    |
   episode runs an agent                                      |
        |                                                    |
   trajectory ----------------------------------------------+
```

Each turn, episodes produce trajectories; trajectories grow BBGs; BBGs
redistill into sharper priors; priors yield better worlds. The BBG is
the memory that connects real-world experience to synthetic training.

### 4.4 Optional by construction

The bootstrap is one optional argument: `Pack.make_builder(prior=None)`
falls back to a hand-authored default prior of the *same type* a BBG
distills into. The builder has exactly one code path; it never knows
whether its prior was learned or authored.

OpenRange never imports any specific BBG runtime. A BBG reaches
OpenRange only as a JSON state dump that `distill()` parses. "OpenRange
works without a BBG" is therefore not a maintained feature — it is a
property of the dependency graph.

---

## 5. Ownership boundaries

OpenRange (this repo, open-source) **owns**: the graph meta-model
(`world_ir.py`), the `bbg@0.1.0` ontology data (`ontologies/bbg.py`),
pack/builder/family contracts (`core/pack.py`), admission (`core/admit.py`),
the distill seam (`core/distill.py`), the snapshot/episode lifecycle,
observability, and ships built-in packs.

OpenRange **does not own**: the agent, the model, the tool harness, the
training algorithm, reward shaping, the wayfinder runtime, persistent
BBG storage, agent-SDK adapters.

A BBG-producing harness (e.g. vecna's wayfinder) is **harness-side**. It
depends on `openrange.world_ir` and `openrange.ontologies.bbg` *only* if
it wants to share Python types; otherwise it implements those shapes
itself and communicates with OpenRange purely through the JSON wire
formats in `CONTRACTS.md`.

### 5.1 Time: where it lives, and where it does not

The two graph shapes treat time differently, on purpose:

- **World graph — timeless.** Content-addressed; the graph *is* its
  content. No timestamps inside it, or two identical builds would get
  different ids and reproducibility would break. Build history lives
  *beside* it, in the snapshot's `history` (an ordered list of build
  events).
- **BBG — mono-temporal.** A growing log with no frozen identity to
  protect, so its transaction time goes *inside*, as the status-event
  log.

The rule: **transaction time goes inside a graph only when the graph
has no other identity to protect.** Neither graph is bitemporal — valid
time (a world's own clock) belongs to the OpenRange runtime, not to
either graph.

---

## 6. Glossary

- **Meta-model** — the fixed `Node`/`Edge`/`WorldGraph` types and validator.
- **Ontology** — a declared set of node kinds, edge kinds, and constraints.
- **Kind** — domain vocabulary for a node or edge; opaque to generic code.
- **Role** — small fixed cross-cutting, world-absolute vocabulary.
- **Pack** — reusable starting point for one world-family.
- **TaskFamily** — a domain of tasks against a world; owns entrypoints,
  feasibility, success.
- **Builder** — turns a manifest into a world graph + tasks.
- **World graph** — a concrete, task-neutral, timeless world.
- **Snapshot** — an admitted, frozen, content-addressed world, plus its
  build history.
- **BuildEvent** — one entry in a snapshot's build history (which pass
  ran, what a repair changed); lives beside the timeless graph.
- **Admission** — the layered gate (validation + feasibility) before
  freezing.
- **TaskSpec** — what an agent is asked to do; carries entrypoints +
  goal_nodes + family handles.
- **BBG** — an agent's spatial-memory graph; things and thoughts on the
  `bbg@0.1.0` ontology. Maintained by a harness, not OpenRange.
- **PackPrior** — generic graph statistics distilled from a BBG;
  optional input to a Builder.
- **Distill** — `openrange.distill(graph, status_log)` — turn a
  BBG-shaped graph into a `PackPrior`.
- **Flywheel** — episodes → trajectories → BBGs → priors → worlds.
