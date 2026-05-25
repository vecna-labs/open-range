# OpenRange contracts

Frozen wire formats for the data structures that cross a boundary — between
core and a pack, between OpenRange and a harness emitting BBG state, between
distill and a builder, or between language runtimes.

`DESIGN.md` explains *why* these shapes exist; this file pins down *what*
they are. Where the two disagree, the reference implementation
(`src/openrange/world_ir.py`, `src/openrange/core/pack.py`,
`src/openrange/core/admit.py`, `src/openrange/core/distill.py`) wins.

Conventions: all formats are JSON. Optional fields are omitted when empty,
not sent as `null`. Enums are exact lowercase strings.

---

## 1. Graph — `Node`, `Edge`, `WorldGraph`

The shared meta-model. Every typed-property-graph in OpenRange — world
graphs produced by Packs, BBG state dumps arriving from a harness —
serializes this way. They differ only in which `ontology` they name and
which `kind`s appear.

### Node

```json
{
  "id": "thing.x",
  "kind": "thing",
  "attrs": { "label": "x", "provenance": "trajectory", "status": "incidental" },
  "roles": ["actor"],
  "visibility": "hidden",
  "runtime": { "...": "opaque, pack-owned" },
  "meta": { "...": "opaque provenance" }
}
```

- `id`, `kind` — required strings.
- `attrs` — object; keys and value types are constrained by the ontology.
- `roles` — array; subset of `["actor", "npc", "external"]`. Omitted if empty.
- `visibility` — `"public"` (default, omitted) or `"hidden"`.
- `runtime`, `meta` — opaque objects; the core never reads inside them.

### Edge

```json
{
  "id": "e1",
  "kind": "traversed",
  "src": "thing.x",
  "dst": "thing.y",
  "attrs": { "outcome": "productive", "count": 1 }
}
```

All of `id`, `kind`, `src`, `dst` required. `src`/`dst` must be node ids
in the same graph.

### WorldGraph

```json
{
  "ontology": "webapp@0.1.0",
  "nodes": [ ...Node... ],
  "edges": [ ...Edge... ],
  "meta": { "...": "manifest ref, pack version, ..." }
}
```

`content_hash()` is `sha256:<hex>` over `ontology + nodes + edges` only —
**`meta` is excluded**, plus each node's `runtime` and `meta`, so two
graphs that differ only in provenance noise share the same content-addressed
identity.

---

## 2. Ontology

A declared schema, also plain data.

```json
{
  "id": "webapp@0.1.0",
  "node_kinds": {
    "service": { "parent": "component",
                 "attrs": { "port": { "type": "int", "required": true } } }
  },
  "edge_kinds": {
    "runs_on": { "endpoints": [["component", "host"]],
                 "src_max": null, "dst_max": null, "attrs": {} }
  }
}
```

`AttrSpec` fields: `type` (one of `string int float bool enum ref json`),
`required`, `enum`, `ref_kinds`, `default`, `description`.

OpenRange ships one ontology as built-in data: `bbg@0.1.0` (returned by
`openrange.ontologies.bbg.bbg_ontology()`). Any harness recording
agent memory in the BBG shape declares this id; OpenRange's `distill()`
recognises it.

---

## 3. PackPrior — the distill → builder contract

What `distill()` emits and `Pack.make_builder(prior=...)` consumes.
Carries only generic graph statistics — never pack-specific config keys.

```json
{
  "source": "bbg@0.1.0 :: sha256:...",
  "ontology": { ...Ontology... },
  "topology": {
    "node_kind_freq":    { "endpoint": 7, "db": 1 },
    "salient_kind_freq": { "endpoint": 1 },
    "dead_end_ratio":    0.25,
    "hidden_signal":     { "endpoint": 2 }
  },
  "task_seeds": [
    { "theme": "cluster-0", "anchor_kinds": ["endpoint", "db"],
      "suggested_goal_kinds": ["cred"], "difficulty": 0.7,
      "evidence": 1, "family": "webapp.pentest" }
  ],
  "difficulty": { "cluster-0": 0.7 },
  "coverage":   { "endpoint": 1.0, "db": 1.0 }
}
```

- `topology` keys are a fixed generic set. A builder *interprets* them;
  the prior never tells the builder what to do.
- `task_seeds[].family` may be omitted when `distill()` could not classify
  the cluster — TaskFamilies then self-select by `anchor_kinds`.
  **`distill()` itself never sets `family`** — a harness with that
  knowledge may attach one.

---

## 4. Snapshot — the admission output

```json
{
  "snapshot_id": "sha256:...",
  "ontology_id": "webapp@0.1.0",
  "graph": { ...WorldGraph... },
  "tasks": [ ...TaskSpec... ],
  "lineage": { "manifest": {...}, "pack": "webapp",
               "pack_version": "0.1.0", "attempts": 1 },
  "history": [
    { "seq": 0, "phase": "build",
      "detail": "builder produced 10 nodes, 2 tasks",
      "refs": ["webapp.build.0", "webapp.pentest.0"] },
    { "seq": 1, "phase": "validate",    "detail": "attempt 1: 0 error(s)" },
    { "seq": 2, "phase": "feasibility", "detail": "attempt 1: 0 infeasible task(s)" },
    { "seq": 3, "phase": "freeze",      "detail": "world admitted and frozen" }
  ]
}
```

- `graph` is timeless — no timestamps inside it.
- `history` is the build story, beside the graph. `phase` is one of
  `build`, `validate`, `feasibility`, `repair`, `freeze`, `evolve`.
- `snapshot_id` equals `graph.content_hash()`.

### TaskSpec

```json
{
  "id": "webapp.pentest.0",
  "instruction": "Recover the hidden admin credential.",
  "entrypoints": ["ep4"],
  "goal_nodes": ["cred17"],
  "feasibility_check": "webapp.pentest",
  "success_check": "webapp.pentest",
  "meta": { "family": "webapp.pentest", "difficulty": 0.7 }
}
```

`entrypoints` / `goal_nodes` are node ids in the snapshot's graph. They
are declared here, per task — **not** as node roles, because they are
task-relative.

`feasibility_check` and `success_check` are TaskFamily *handles* (e.g.
`"webapp.pentest"`). The Pack resolves the handle to a class; the class's
`check_feasibility(graph, task)` and `check_success(graph, task,
final_state)` methods run when needed. There is no exec'd Python source
in this format.

---

## 5. EpisodeResult

What an episode's success-check returns. Structured — **not** a scalar
reward; a harness-side training adapter maps it to whatever signal is
needed.

```json
{ "success": true,
  "subgoals": { "found_login": true, "exploited_sqli": true },
  "reason": "" }
```

Only `success` (boolean) is required. `subgoals` and `reason` are
optional.

---

## 6. BBG state dump — the JSON OpenRange ingests via distill()

Any agent harness recording agent memory in the BBG shape MAY emit this
JSON to drive OpenRange's flywheel. OpenRange's `distill()` parses it
into a `WorldGraph` (already covered by §1) plus an optional
`status_events` log (the BBG's transaction-time changelog) plus the
trajectory.

This is the only wire format OpenRange *consumes* from another system.
The shape lives in this document, not in any specific harness, so any
harness emitting it can drive distillation.

```json
{
  "ontology": "bbg@0.1.0",
  "nodes": [ ...Node — with thing/thought kinds... ],
  "edges": [ ...Edge — with traversed/part_of/anchored_to/revises kinds... ],
  "status_events": [ ...StatusEvent... ],
  "trail": ["thing.a", "thing.b", "thing.x"],
  "fetched_at": "2026-05-25T01:00:00Z"
}
```

- `ontology` MUST be `"bbg@0.1.0"` (or a future bump like `bbg@0.2.0`,
  which OpenRange may or may not support depending on its version).
- `status_events` is the BBG's transaction-time record (see below).
  Omitted means "no status changes recorded"; the current `status` attr
  on each node is taken as the latest state.
- `trail` is the ordered list of thing ids the agent visited (optional;
  used only for dashboard scrubbing today).
- `fetched_at` is informational; not consumed by `distill()`.

### StatusEvent

One entry per status transition. Ordered; each event also carries a
monotonic `seq` for stable wire-format ordering.

```json
{ "seq": 7, "node_id": "thought.0", "status": "refuted",
  "at_step": 4, "cause": "revised-by:thought.1" }
```

- `node_id`, `status`, `at_step` — required.
- `seq` — optional on input; OpenRange may treat array order as canonical.
- `cause` — string, omitted if empty.
- `status` values: thoughts `open|confirmed|refuted`; things
  `incidental|salient`.

---

## 7. Stability

These shapes are settled enough to build against. Stability rules:

- **Adding optional fields** to any record is non-breaking. Receivers
  ignore unknown keys.
- **Adding `topology` keys** to `PackPrior` is non-breaking. Builders
  read keys they recognize.
- **Adding `phase` values** to `BuildEvent` is non-breaking. Dashboards
  render unknown phases generically.
- **Changing required fields, removing keys, or changing enum values**
  is a *breaking* change. Such a change bumps the `ontology` id (for
  graph/ontology shapes) or the corresponding contract section's version
  (we'll start versioning when it becomes necessary).

The `bbg@0.1.0` ontology id is the version handle for the BBG state
shape. Any breaking change to the BBG node/edge schema bumps that id;
consumers should branch on the prefix (`bbg@`) and the version
(`0.1.0`) if they need to support multiple versions.
