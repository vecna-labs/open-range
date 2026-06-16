# OpenRange Contracts

Wire formats, dataclass shapes, and Protocol signatures that cross the
boundary between OpenRange core and the rest of the world — packs,
agent harnesses, snapshot consumers. This is the document you implement
against if you are writing a new Pack, building an agent harness, or
reading a Snapshot file off disk.

The companion document is [DESIGN.md](DESIGN.md) — it explains *why*
each shape looks the way it does. Where the two disagree, the
reference implementations in the `graphschema` package and
`src/openrange/core/` are authoritative.

Conventions throughout:

- Field names are lifted verbatim from the Python dataclasses. The doc
  and the code must agree letter-for-letter.
- JSON wire shapes use lowercase string tokens for enum values
  (`"public"`, `"hidden"`, `"harden"`). Optional fields are omitted when
  empty rather than serialized as `null`.
- File paths in this doc are repo-relative
  (`src/openrange/core/admit.py`); meta-model types are imported from
  the `graphschema` package (e.g. `from graphschema import WorldGraph`).
  The package's source structure (`graphschema._ir`, etc.) is an
  implementation detail.

---

## 1. Scope

The typed-property-graph meta-model (`Node`, `Edge`, `WorldGraph`,
`Ontology`, `validate`, `apply_patch`) lives in the shared
`graphschema` package (an OpenRange dependency). This document focuses
on the OpenRange-specific contracts that consume those types. See the
`graphschema` package's README for the meta-model wire format.

This document specifies:

- How OpenRange uses the typed-property-graph meta-model imported from
  `graphschema` (`Node`, `Edge`, `WorldGraph`, `Ontology`).
- The on-the-wire shape of an admitted, frozen world: `Snapshot` and
  its JSON projection `snapshot_to_dict`.
- The Python `Pack` / `Builder` / `TaskFamily` / `RuntimeHandle`
  Protocols a pack must satisfy.
- The admission layered gate.
- The `PackPrior` shape an external producer hands to a Builder.
- The `Mutation` + `GraphPatch` shape the curriculum seam uses.
- How a pack registers via Python entry points.

Not in scope here (see DESIGN.md, the `graphschema` package, or
external docs):

- Agent loops, reward shaping, training adapters — harness-side.
- Dashboard internals, run-directory layout, CLI ergonomics.
- The implementation of any specific Pack (`cyber_webapp` ships as
  the reference; its module docstrings document its own internals).

---

## 2. Ontology — the typed-property-graph schema

An `Ontology` is **plain data**, declared per-pack and validated at
admission. The same generic validator (`graphschema.validate`) checks
any graph against any ontology — no domain word ever lands in core
code.

Reference: the `graphschema` package (`graphschema._ir`).

### 2.1 `Ontology`

```python
@dataclass
class Ontology:
    id: str
    node_kinds: dict[str, NodeKind] = field(default_factory=dict)
    edge_kinds: dict[str, EdgeKind] = field(default_factory=dict)
```

`id` is a versioned string (the cyber pack ships `"cyber.webapp@v2"`).
Two graphs share an ontology id iff they conform to the same declared
schema; a schema revision means a new id.

Graph-wide invariants beyond node/edge shape do **not** live on the
Ontology — they live on the Pack as `invariants() -> list[Callable]`.
The Ontology is data; invariants are functions, and they depend on
pack-specific reasoning the schema deliberately cannot express.

### 2.2 `NodeKind`

```python
@dataclass
class NodeKind:
    id: str
    parent: str | None = None
    attrs: dict[str, AttrSpec] = field(default_factory=dict)
    description: str = ""
```

`parent` names another NodeKind in the same Ontology whose `attrs` this
kind inherits. The child's `attrs` override the parent's by key; required
parent attrs are still enforced via `graphschema._ir::_compose_node_attrs`.
Cycles in the parent chain are broken silently — cycle detection is the
job of a separate ontology-validator pass, not the runtime validator.

### 2.3 `EdgeKind`

```python
@dataclass
class EdgeKind:
    id: str
    endpoints: list[tuple[str, str]] = field(default_factory=list)
    src_max: int | None = None
    dst_max: int | None = None
    attrs: dict[str, AttrSpec] = field(default_factory=dict)
    description: str = ""
```

`endpoints` lists allowed `(src_kind, dst_kind)` pairs; an empty list
allows any node-kind pair. `src_max` / `dst_max` cap how many edges of
this kind may originate / terminate at a single node (`None` = unbounded).

### 2.4 `AttrSpec`

```python
@dataclass
class AttrSpec:
    type: AttrType
    required: bool = False
    enum: list[str] | None = None
    ref_kinds: list[str] | None = None
    default: Any = None
    description: str = ""
```

`type` is one of the seven primitives in `AttrType`:

| `AttrType.*` value | Wire token | Validator rule |
|--------------------|-----------|----------------|
| `STRING`           | `"string"` | `isinstance(v, str)` |
| `INT`              | `"int"`    | `isinstance(v, int) and not isinstance(v, bool)` — bool is excluded so `True` does not pass as `1` |
| `FLOAT`            | `"float"`  | `isinstance(v, (int, float)) and not isinstance(v, bool)` — ints are legal where floats are expected |
| `BOOL`             | `"bool"`   | `isinstance(v, bool)` |
| `ENUM`             | `"enum"`   | value in `spec.enum` (must be a non-empty list) |
| `REF`              | `"ref"`    | value is a string id, resolves to a node, target kind in `spec.ref_kinds` if set |
| `JSON`             | `"json"`   | any value — opaque blob |

`required=False` attrs may be omitted from a node/edge's `attrs` bag;
required attrs missing produce a `missing_required_attr` Issue.
`default` is **informational** — the validator does not apply it; the
builder is responsible for materializing defaults.

### 2.5 `Role`

```python
class Role(StrEnum):
    ACTOR = "actor"
    NPC = "npc"
    EXTERNAL = "external"
```

Roles are the fixed, world-absolute role vocabulary the generic
machinery reads (defined in `graphschema._ir`). The set is deliberately
closed and small so core code can branch on it without knowing any
pack's domain vocabulary.

World-absolute means: a role is true regardless of what task is being
run against this world. Task-relative facts like "entrypoint" or
"goal" never appear as roles — they live on `TaskSpec` (§5).

Today `ACTOR`/`NPC`/`EXTERNAL` is enough for the cyber webapp pack;
adding a role is a coordinated edit to the enum and any generic code
that branches on it.

### 2.6 `Visibility`

```python
class Visibility(StrEnum):
    PUBLIC = "public"
    HIDDEN = "hidden"
```

- `PUBLIC` is the default; it is **omitted on the wire** when a node
  is public (see `graphschema._ir::_node_data` and `admit::_node_dict`).
- `HIDDEN` nodes still exist in the graph (an internal secret, an
  undisclosed asset) but a task's `entrypoints` cannot reference them
  — discovering hidden state is often the point.
- A task's `goal_nodes` **may** point at a HIDDEN node. Discovery is
  often the goal.

Admission enforces this in `validate_task_bindings` (§10).

---

## 3. WorldGraph

```python
@dataclass
class WorldGraph:
    ontology: str
    nodes: dict[str, Node] = field(default_factory=dict)
    edges: dict[str, Edge] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)
```

Reference: `graphschema._ir.WorldGraph` in the shared `graphschema`
package.

`ontology` is the ontology **id string** — the graph carries the
reference, not the schema itself. `validate()` accepts the Ontology as
a separate argument so the same graph can be checked against multiple
revisions.

`nodes` and `edges` are keyed by id; dict-key versus `.id` desync is
caught structurally (`graphschema._ir::_validate_structural`).

### 3.1 `Node`

```python
@dataclass
class Node:
    id: str
    kind: str
    attrs: dict[str, Any] = field(default_factory=dict)
    roles: set[Role] = field(default_factory=set)
    visibility: Visibility = Visibility.PUBLIC
    runtime: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)
```

`runtime` and `meta` are opaque to the meta-model and the core — the
validator never reads them and they are **excluded from `content_hash()`**
(only `id`, `kind`, `attrs`, `roles`, `visibility` contribute).
This lets packs stash provenance, instrumentation, or temporary
realizer state in `runtime` / `meta` without changing a graph's
content-addressed identity.

### 3.2 `Edge`

```python
@dataclass
class Edge:
    id: str
    kind: str
    src: str
    dst: str
    attrs: dict[str, Any] = field(default_factory=dict)
```

Both endpoint ids must reference nodes in the same graph; dangling
endpoints are an `edge_dangling_src` / `edge_dangling_dst` Issue.

### 3.3 Why graphs are TIMELESS

A `WorldGraph` carries no timestamps. `content_hash()` returns
`sha256:<hex>` over the deterministic serialization of `ontology` +
`nodes` + `edges` only (see `graphschema._ir::WorldGraph.content_hash`).
Two builds that produce the same world share one content hash regardless
of when they ran, what manifest produced them, or what was logged during
the build process.

That identity is what makes a `Snapshot` content-addressed (§5) and
what makes reproducibility a structural property rather than a
documentation hope. Build-process facts (which pass ran, what a
repair changed, why an attempt was rejected) live in
`Snapshot.history` **beside** the graph, never inside it. See
DESIGN.md §8 for the rationale.

### 3.4 `GraphPatch` and `apply_patch`

```python
@dataclass
class GraphPatch:
    nodes_added: list[Node] = field(default_factory=list)
    nodes_updated: list[Node] = field(default_factory=list)
    nodes_removed: list[str] = field(default_factory=list)
    edges_added: list[Edge] = field(default_factory=list)
    edges_updated: list[Edge] = field(default_factory=list)
    edges_removed: list[str] = field(default_factory=list)
```

The universal diff type for any graph mutation — used by `Mutation`
(§12), by `Builder.evolve`, and anywhere else a partial graph update
crosses a boundary.

`apply_patch(graph, patch)` mutates `graph` in place
(see `graphschema._ir::apply_patch`). Order:
removals → updates → additions, so an update sharing an id with a
removal in the same patch keeps the new value, and an addition cannot
collide with something the patch itself just removed. Removing a node
also drops any dangling edges that reference it.

Updates are **full replacements** — there is no per-attribute merge.
The patch carries the new shape; the caller decides what it should be.

---

## 4. Validation — three tiers

`validate(graph, ontology, invariants=None) -> list[Issue]` (in the
shared `graphschema` package, `graphschema._ir`) is the generic
checker. It runs three tiers and concatenates findings:

1. **Structural** (`_validate_structural`): ids are non-empty strings,
   dict-key/`.id` consistency holds, edge endpoints point at nodes
   that exist.
2. **Conformance** (`_validate_conformance`): every node / edge kind
   is declared in the ontology, required attrs are present, attr
   values match `AttrSpec` (incl. enum membership, REF resolution,
   `ref_kinds` checks), endpoint pairs match a declared
   `(src_kind, dst_kind)`, and `src_max` / `dst_max` are respected.
   Parent `NodeKind` chains contribute required attrs.
3. **Pack invariants**: each callable in `invariants` is run with the
   graph and its `list[Issue]` concatenated. Packs use this tier for
   domain-specific structural rules the generic schema cannot express
   (the cyber pack ships `no_orphan_nodes`, `secret_must_be_held`,
   `oracle_path_exists` — see `packs/cyber_webapp/cyber_webapp/invariants.py`).

```python
@dataclass
class Issue:
    severity: str  # "error" or "warning"
    code: str      # stable machine-readable tag
    message: str
    where: str     # node id, edge id, task id, or pseudo-path
```

`code` is the tag callers group / suppress on (the validator emits
~15 codes — `unknown_node_kind`, `missing_required_attr`,
`edge_endpoint_mismatch`, `ref_dangling`, etc.). Only `severity ==
"error"` causes admission to reject; warnings pass through.

---

## 5. Snapshot — the admission output

```python
@dataclass(frozen=True)
class Snapshot:
    snapshot_id: str
    ontology_id: str
    graph: WorldGraph
    tasks: tuple[TaskSpec, ...]
    lineage: Mapping[str, Any]
    history: tuple[BuildEvent, ...] = ()
```

Reference: `src/openrange/core/admit.py::Snapshot`.

A `Snapshot` is the **only** thing an episode runs against. It is
content-addressed (`snapshot_id == graph.content_hash()`, set in
`admit::admit` on the freeze branch), frozen, and serializes through
`snapshot_to_dict` to exactly the JSON wire shape below.

### 5.1 Wire shape

`snapshot_to_dict(snap)` (see `admit::snapshot_to_dict`) emits:

```json
{
  "snapshot_id": "sha256:b3a1...",
  "ontology_id": "cyber.webapp@v2",
  "graph": {
    "ontology": "cyber.webapp@v2",
    "nodes": [
      {
        "id": "svc.auth",
        "kind": "service",
        "attrs": { "name": "auth", "kind": "auth", "exposure": "public" }
      },
      {
        "id": "secret.flag",
        "kind": "secret",
        "attrs": { "value_ref": "FLAG{...}" },
        "visibility": "hidden"
      }
    ],
    "edges": [
      {
        "id": "e1",
        "kind": "exposes",
        "src": "svc.auth",
        "dst": "ep.login"
      }
    ]
  },
  "tasks": [
    {
      "id": "webapp.pentest.0",
      "instruction": "Recover the hidden admin credential.",
      "entrypoints": ["ep.login"],
      "goal_nodes": ["secret.flag"],
      "feasibility_check": "webapp.pentest",
      "success_check": "webapp.pentest",
      "meta": { "family": "webapp.pentest", "difficulty": 0.7 }
    }
  ],
  "lineage": {
    "manifest": { "seed": 0 },
    "pack": "webapp",
    "pack_version": "v2",
    "attempts": 1
  },
  "history": [
    { "seq": 0, "phase": "build",
      "detail": "builder produced 11 nodes, 2 tasks",
      "refs": ["webapp.build.0", "webapp.pentest.0"] },
    { "seq": 1, "phase": "validate",
      "detail": "attempt 1: 0 error(s)" },
    { "seq": 2, "phase": "feasibility",
      "detail": "attempt 1: 0 infeasible task(s)" },
    { "seq": 3, "phase": "freeze",
      "detail": "world admitted and frozen" }
  ]
}
```

Top-level keys (all required):

| Key            | Type           | Meaning |
|----------------|---------------|---------|
| `snapshot_id`  | string         | `sha256:<hex>` from `graph.content_hash()`. Identifies the snapshot. |
| `ontology_id`  | string         | The Ontology id the graph conforms to (mirrors `graph.ontology`). |
| `graph`        | object         | `{ontology, nodes, edges}` — the timeless world. |
| `tasks`        | array          | The TaskSpecs admission verified. |
| `lineage`      | object         | Flat provenance dict — see §5.3. |
| `history`      | array          | Ordered `BuildEvent` records — see §5.4. |

### 5.2 `graph` sub-block

- `ontology` is the id string (mirrors `Snapshot.ontology_id`).
- `nodes` is a list, sorted by `id`. Each entry:
  `{id, kind, attrs}` plus optional `roles` (when non-empty;
  lowercase strings from `Role`) and optional `visibility`
  (only when `HIDDEN` — `PUBLIC` is omitted).
  `runtime` and `meta` are **not** serialized — they are opaque
  pack-owned scratch.
- `edges` is a list, sorted by `id`. Each entry:
  `{id, kind, src, dst}` plus optional `attrs` (when non-empty).

Both lists are sorted deterministically so two identical worlds emit
byte-identical JSON.

### 5.3 `lineage` sub-block

A flat `Mapping[str, Any]` carrying provenance and build metadata.
`admit()` always populates at least:

- `manifest` — the manifest dict admission was driven from (verbatim
  copy).
- `pack` — `pack.id`.
- `pack_version` — `pack.version`.
- `attempts` — how many `build()` + `repair()` rounds were needed
  before admission froze (1 = first build was clean).

Anything the builder put in `BuildResult.admission_meta` is also
merged in (see `admit::admit`, freeze branch). Builders use this slot
for LLM prompt records, sampling seeds, prior summaries.

### 5.4 `history` sub-block

```python
@dataclass(frozen=True)
class BuildEvent:
    seq: int
    phase: str
    detail: str
    refs: tuple[str, ...] = ()
```

`phase` is one of the closed vocabulary:

| `phase`        | Emitted by | What it records |
|----------------|-----------|-----------------|
| `build`        | every initial `build()` | "builder produced N nodes, M tasks". `refs` carries the task ids. |
| `validate`     | every admission attempt | "attempt K: N error(s)". `refs` carries the `Issue.where` ids. |
| `feasibility`  | every admission attempt | "attempt K: N infeasible task(s)". `refs` carries the task ids that failed. |
| `repair`       | between failed attempts | "builder regenerated after attempt K". |
| `freeze`       | exactly once, on success | "world admitted and frozen". |
| `evolve`       | curriculum (`auto_evolve`) | "evolved from `<parent>` via `<family>/<direction>`". Appended once to the evolved snapshot's history; `refs` carries the parent `snapshot_id`. The evolved snapshot's `lineage` also carries the structured `_evolve` block (parent id, direction, family, note). |

`history` is the build **story**, not part of the graph's identity.
Two builds that produce the same graph but took different repair
paths share one `snapshot_id` and differ only in `history`.

### 5.5 `TaskSpec` shape

```python
@dataclass(frozen=True)
class TaskSpec:
    id: str
    instruction: str
    entrypoints: tuple[str, ...]
    goal_nodes: tuple[str, ...]
    feasibility_check: str
    success_check: str
    meta: Mapping[str, Any] = field(default_factory=dict)
```

Reference: `src/openrange/core/pack.py::TaskSpec`.

- `id` — task identifier, unique within the snapshot.
- `instruction` — the natural-language statement of what the agent must do.
- `entrypoints` — tuple of node ids in `snapshot.graph` where the agent
  starts acting. Admission rejects empty tuples (the episode loop also
  enforces this in `_resolve_task`).
- `goal_nodes` — tuple of node ids that represent completion. May be
  HIDDEN — discovering hidden state is often the point.
- `feasibility_check` — the **id of a TaskFamily** the pack ships
  (e.g. `"webapp.pentest"`). Admission dispatches into
  `pack.task_family(feasibility_check).check_feasibility(graph, task)`.
- `success_check` — likewise a TaskFamily id. The episode loop dispatches
  into `pack.task_family(success_check).check_success(graph, task,
  final_state)`. Usually equal to `feasibility_check`, but a pack may
  split them.
- `meta` — free-form mapping; the pack and the harness agree on its
  keys (the cyber pack puts `family`, `difficulty`, `target_path`, etc.).
  Serialized only when non-empty.

Why entrypoints and goal_nodes live on the **task**, not the **node**:
they are task-relative. The same world graph can serve `webapp.build`
(entrypoint: a `service` node; goal: an `endpoint`) and
`webapp.pentest` (entrypoint: an `endpoint`; goal: a `secret`). A
node has world-absolute roles (`actor`, `npc`, `external`); a task
declares its own bindings into those nodes. See DESIGN.md §5.

### 5.6 Round-trip — `snapshot_from_dict`

`openrange.core.store.snapshot_from_dict` is the exact inverse of
`snapshot_to_dict`. Feeding the output of one into the other
reproduces the original Snapshot. `SnapshotStore.save` / `load`
(`src/openrange/core/store.py`) ferry the JSON to disk at
`<root>/<snapshot_id>.json`; load verifies the file's stored id
matches the filename and raises `StoreError` on any mismatch.

---

## 6. Pack — the binding surface

```python
class Pack(ABC):
    id: str = ""
    version: str = ""

    @abstractmethod
    def ontology(self) -> Ontology: ...

    def invariants(self) -> list[Callable[[WorldGraph], list[Issue]]]:
        return []

    @abstractmethod
    def make_builder(self, prior: PackPrior | None) -> Builder: ...

    @abstractmethod
    def realize(self, graph: WorldGraph, backing: Backing) -> RuntimeHandle: ...

    def task_families(self) -> list[TaskFamily]:
        return []

    def task_family(self, family_id: str) -> TaskFamily | None: ...
```

Reference: `src/openrange/core/pack.py::Pack`.

Five mandatory pieces a pack ships:

1. **`ontology()` → `Ontology`.** The declarative schema. Called once
   per admission, once per evolved re-admission, and once on each
   episode start (to drive validation). Returning a **fresh** Ontology
   on each call is a convention (so callers can mutate the result
   safely); the cyber pack does this.

2. **`invariants()` → `list[Callable[[WorldGraph], list[Issue]]]`.**
   Tier-3 functions the validator runs. Default is empty. Each
   callable takes the graph and returns a list of `Issue`s; an empty
   list means the invariant passed. These are the domain-specific
   structural rules the generic schema can't express (the cyber pack
   ships three; see `packs/cyber_webapp/cyber_webapp/invariants.py`).

3. **`make_builder(prior)` → `Builder`.** Constructs a fresh Builder
   each call. `prior=None` is the boot path — the pack falls back to
   a hand-authored default (the cyber pack uses
   `priors.default_prior()`). The builder has one code path; it never
   knows whether the prior came from an external producer or the
   pack's own default.

4. **`realize(graph, backing)` → `RuntimeHandle`.** Turns an admitted
   graph into a runnable handle. The `backing` argument is one of:

   ```python
   class Backing(StrEnum):
       PROCESS = "process"      # in-process simulation
       CONTAINER = "container"  # docker / podman / k8s per service
       SIMULATOR = "simulator"  # pack-provided simulator
       HYBRID = "hybrid"        # mix
   ```

   A pack may support one backing or several; today the cyber pack
   supports only `PROCESS` and raises `NotImplementedError` for the
   others (see `packs/cyber_webapp/cyber_webapp/realize.py::WebappRuntimeHandle.__init__`).
   Choosing a backing is a runtime decision (laptop / container farm)
   that does not change graph identity.

5. **`task_families()` → `list[TaskFamily]`.** Default is empty. A
   pack with no families won't admit anything (admission requires at
   least one task), so packs that admit at all must return at least
   one. `task_family(family_id)` is the dispatch lookup; the default
   is a linear scan by `id`.

`id` and `version` are class attributes — the pack registers under
its `id` (`"webapp"`), and `version` (`"v2"`) shows up in
`Snapshot.lineage`.

---

## 7. Builder

```python
class Builder(ABC):
    @abstractmethod
    def build(self, manifest: Manifest) -> BuildResult: ...

    def repair(
        self,
        prev: BuildResult,
        errors: list[Issue],
        infeasible: list[str],
    ) -> BuildResult: ...  # default raises

    def evolve(
        self,
        snapshot: Snapshot,
        mutation: Mutation,
    ) -> GraphPatch:  # default returns mutation.patch
        return mutation.patch
```

Reference: `src/openrange/core/pack.py::Builder`.

### 7.1 `build(manifest) -> BuildResult`

The first candidate world. `Manifest` is just `Mapping[str, Any]` —
free-form; the pack documents which keys it reads. Returns:

```python
@dataclass(frozen=True)
class BuildResult:
    graph: WorldGraph
    tasks: list[TaskSpec]
    admission_meta: Mapping[str, Any] = field(default_factory=dict)
```

`admission_meta` is the builder's provenance — sampling seed, LLM
prompts, prior summary, whatever. It rides into `Snapshot.lineage`.

Builders should be deterministic in `(manifest, prior)` modulo any
builder-internal seed. The cyber pack hashes `manifest["seed"]` into
its RNG and increments it on `repair` (so retries land different
worlds).

### 7.2 `repair(prev, errors, infeasible) -> BuildResult`

Optional. Default raises `NotImplementedError` with a message that
explains how to opt in. A builder that wants to participate in the
admission repair loop overrides this; admission calls it up to
`max_repairs` times when validation or feasibility fails.

- `prev` — the rejected `BuildResult` (graph + tasks + meta).
- `errors` — the `Issue`s that have `severity == "error"`.
- `infeasible` — task ids whose `check_feasibility` returned `False`.

The builder decides how to respond. The cyber pack's procedural
builder simply resamples with a perturbed seed. An LLM builder might
patch the offending bit. A search-based builder might back up and try
a different branch.

### 7.3 `evolve(snapshot, mutation) -> GraphPatch`

Optional. Default returns `mutation.patch` verbatim — a pack-side
refinement step the curriculum loop (§12, `auto_evolve`) goes through
on its way to applying a `Mutation`. A pack that wants to refine the
mutation (e.g. mint deterministic ids that fit the existing world)
overrides this.

---

## 8. TaskFamily

```python
class TaskFamily(ABC):
    id: str = ""
    pack_id: str = ""

    @abstractmethod
    def generate(
        self,
        graph: WorldGraph,
        manifest: Manifest,
        prior: PackPrior | None,
    ) -> list[TaskSpec]: ...

    @abstractmethod
    def check_feasibility(
        self,
        graph: WorldGraph,
        task: TaskSpec,
    ) -> FeasibilityVerdict: ...

    @abstractmethod
    def check_success(
        self,
        graph: WorldGraph,
        task: TaskSpec,
        final_state: Mapping[str, Any],
    ) -> EpisodeResult: ...

    def available_mutations(
        self,
        snapshot: Snapshot,
        reports: Sequence[EpisodeReportLike],
        *,
        llm: LLMBackendLike | None = None,
    ) -> tuple[Mutation, ...]:
        return ()
```

Reference: `src/openrange/core/pack.py::TaskFamily`.

A `TaskFamily` owns one **domain** of tasks against a Pack's world.
The cyber pack ships two: `webapp.build` (build / repair an endpoint)
and `webapp.pentest` (discover and exploit a vulnerability chain).
Both live against the same world graph in any given snapshot —
the same world serves multiple families with different entrypoints,
different goals, different success criteria.

### 8.1 `generate(graph, manifest, prior) -> list[TaskSpec]`

Build one or more TaskSpecs against this world. The family chooses
entrypoints (which node-ids the agent acts from), goal-nodes (which
node-ids count as completion), and composes the instruction string.
`prior.task_seeds` is a hint — the family may use it or ignore it.

Returning an empty list is legal (the family had no candidate task
for this world); the builder aggregates across all families, and
admission requires at least one task total.

### 8.2 `check_feasibility(graph, task) -> FeasibilityVerdict`

```python
@dataclass(frozen=True)
class FeasibilityVerdict:
    feasible: bool
    reason: str = ""
```

Pure-graph reasoning — no realizer, no runtime. The family walks the
graph and decides whether the task is **actually solvable** in this
world. Schema correctness is already covered by admission's
structural + conformance tiers; this is the domain-meaning check that
only the family knows how to do (the cyber pentest family checks
that there's a path from an exposed endpoint to a hidden secret via
`enables`/`affects` edges).

### 8.3 `check_success(graph, task, final_state) -> EpisodeResult`

```python
@dataclass(frozen=True)
class EpisodeResult:
    success: bool
    subgoals: Mapping[str, bool] = field(default_factory=dict)
    reason: str = ""
```

`final_state` is the dict `RuntimeHandle.collect()` returned at
episode end. Its keys are a pack/family convention (the cyber pack
collects `flag_from_response`, `requests_made`, `endpoint_serves_200`).
The family reads it against the graph + task and returns a
**structured** result — never a scalar reward. Reward shaping is
harness-side (see DESIGN.md §13).

### 8.4 `available_mutations(snapshot, reports, *, llm=None) -> tuple[Mutation, ...]`

Default returns `()`. A family that participates in curriculum
evolution overrides this to enumerate candidate `Mutation`s. The
`llm` argument is offered so a family can re-score relevance with a
semantic pass; families that don't use LLMs ignore it. See §12 for
the `Mutation` shape and how `auto_evolve` consumes these.

---

## 9. RuntimeHandle — the realized-world Protocol

```python
@runtime_checkable
class RuntimeHandle(Protocol):
    def reset(self) -> None: ...
    def surface(self) -> Mapping[str, Any]: ...
    def poll_events(self) -> tuple[Mapping[str, Any], ...]: ...
    def terminal(self) -> tuple[bool, str | None]: ...
    def checkpoint(self) -> Any: ...
    def restore(self, state: Any) -> None: ...
    def collect(self) -> Mapping[str, Any]: ...
    def stop(self) -> None: ...
```

Reference: `src/openrange/core/pack.py::RuntimeHandle`.

Eight methods. `Pack.realize(graph, backing)` returns one; the
`EpisodeService` (`src/openrange/core/episode.py`) drives it through
the episode lifecycle.

| Method            | Returns                       | When called | Purpose |
|-------------------|-------------------------------|-------------|---------|
| `reset()`         | None                          | At episode start, after `realize()` | Boot the world — start subprocesses, materialize artifact files, seed databases. |
| `surface()`       | `Mapping[str, Any]`           | After `reset()`, cached for the episode | The agent-facing IO bundle. Pack-defined keys: HTTP base URL, file roots, MCP endpoints, NPC adapter dicts. The harness binds against the keys it expects. |
| `poll_events()`   | `tuple[Mapping[str, Any], ...]` | Each tick | Drain side-effect events (HTTP requests, file writes, log entries) the world produced since the last poll. Forwarded to the dashboard. |
| `terminal()`      | `(bool, str \| None)`         | Each tick | Has the solver finished? `(True, reason)` ends the episode; `(False, None)` continues. |
| `checkpoint()`    | `Any` (opaque)                | On `EpisodeService.checkpoint` / `fork` | Capture an opaque pack-defined state snapshot for counterfactual replay. |
| `restore(state)`  | None                          | On `EpisodeService.restore` / `fork` | Replay an opaque payload. Process-state semantics are the pack's call. |
| `collect()`       | `Mapping[str, Any]`           | At episode stop, before `stop()` | Structured final state. The family's `check_success` reads this. |
| `stop()`          | None                          | At episode stop | Tear down running processes / services. Must be idempotent. |

`surface()` is **not** just `base_url` — it is the full IO bundle.
Some keys are stringly-typed (`base_url`, `solver_root`); others may be
callables (`http_get`, `http_get_json`). The episode layer's
`_observation_metadata` is what selects the stringly-typed slice for
the dashboard's JSON serializer.

Per-pack contract: the cyber pack's `WebappRuntimeHandle.surface()`
returns `{base_url, http_get, http_get_json, solver_root}`. The
harness and NPCs that expect those keys are written against that
shape.

---

## 10. Admission — the layered gate

`admit(pack, manifest, prior=None, max_repairs=2) -> Snapshot | AdmissionFailure`

Reference: `src/openrange/core/admit.py::admit`.

A candidate world produced by a builder is **not trusted** because a
builder produced it. It must pass five layers:

1. **Structural** — `_validate_structural` in `graphschema._ir`. IDs are
   non-empty strings, dict-key/`.id` consistency holds, edge endpoints
   are non-dangling.
2. **Ontology conformance** — `_validate_conformance`. Every node /
   edge kind is declared; required attrs are present; attr value
   types match `AttrSpec`; enum values are in range; REF attrs
   resolve to a node of an allowed kind; endpoint pairs match a
   declared `(src_kind, dst_kind)`; degree caps respected.
3. **Pack invariants** — each callable in `pack.invariants()` is
   run; their `Issue`s are concatenated.
4. **Task bindings** — `admit::validate_task_bindings`: every task's
   `entrypoints` and `goal_nodes` reference real nodes; no entrypoint
   is HIDDEN (a hidden node can be a goal, never a starting surface).
5. **Task feasibility** — for each task, dispatch into
   `pack.task_family(task.feasibility_check).check_feasibility(graph,
   task)`. A task naming a family the pack does not declare is
   treated as infeasible.

```python
@dataclass
class AdmissionFailure:
    issues: list[Issue]
    infeasible_tasks: list[str]
    attempts: int
    history: tuple[BuildEvent, ...] = ()
```

If any layer fails, admission calls `builder.repair(prev, errors,
infeasible)` and loops, up to `max_repairs` times (default 2 = 3 total
attempts). When the budget is exhausted, `admit` returns an
`AdmissionFailure` carrying the final error / infeasible-task lists
plus the full `history`. On success it returns a `Snapshot`.

`BuildEvent.phase` values emitted during the loop:
`"build"` (initial), `"validate"` (each attempt),
`"feasibility"` (each attempt), `"repair"` (between attempts),
`"freeze"` (exactly once, on success).

Each layer catches a different bug. Layers 1+2 catch malformed worlds.
Layer 3 catches structurally-valid but semantically-broken worlds (an
oracle-path invariant violation that can only be expressed by walking
the graph). Layer 4 catches mis-bound tasks. Layer 5 catches
well-formed, well-bound, schema-correct worlds that nobody can
actually solve. All five are required because each one's bug class
slips past the others. See DESIGN.md §6.

---

## 11. PackPrior — the generation prior

```python
@dataclass
class PackPrior:
    source: str
    ontology: Ontology
    topology: Mapping[str, Any]
    task_seeds: list[TaskSeed] = field(default_factory=list)
    difficulty: Mapping[str, float] = field(default_factory=dict)
    coverage: Mapping[str, float] = field(default_factory=dict)
```

Reference: `src/openrange/core/pack.py::PackPrior`.

`PackPrior` is the input shape a Builder consults; producing one is
the caller's concern. A pack may ship a hand-authored default (the
boot path: `make_builder(prior=None)`), or a harness may hand one in
from some external source.

The one rule that keeps `PackPrior` reusable across packs: the prior
carries **only generic graph statistics**. The builder INTERPRETS
those statistics into domain decisions; the prior never tells the
builder what to do.

### 11.1 Fields

- `source` — opaque identifier of where this prior came from. The
  convention is a short tag like `"<pack-id> :: hand-authored"` for a
  built-in default; external producers pick their own tag.
- `ontology` — the target Ontology the prior is keyed against,
  typically the target pack's own ontology.
- `topology` — generic graph statistics. Keys are a fixed closed set
  (§11.2).
- `task_seeds` — list of `TaskSeed` (§11.3). Mutable (a list, not a
  tuple) deliberately, so a harness with extra knowledge can re-tag
  seeds with a `family` before passing them to a builder.
- `difficulty` — per-seed difficulty scores (`theme -> [0..1]`).
- `coverage` — per-kind explored ratio (`kind -> [0..1]`).

### 11.2 `topology` keys

Closed vocabulary, all four keys conventionally emitted:

| Key | Type | What it carries |
|-----|------|-----------------|
| `node_kind_freq`    | `dict[str, int]` | Expected count of each node-kind in a typical world. |
| `salient_kind_freq` | `dict[str, int]` | Same restricted to nodes that matter (the agent reasoned about them, repeated visits, explicit observation). A builder weights this far higher than `node_kind_freq` — it's signal where the other is mostly background. |
| `dead_end_ratio`    | `float` (rounded to 3 dp) | Expected fraction of paths that look productive and aren't. Signals how easily agents get fooled in this domain. |
| `hidden_signal`     | `dict[str, int]` | Per-kind count of confirmed hidden-state anchors. Signals where the world has discoverable hidden state. |

The "builder interprets" rule means: a builder reads
`hidden_signal["secret"] = 2` and decides to put two `secret` nodes
behind `Visibility.HIDDEN`. The prior doesn't say "make secrets
hidden"; it says "expect two anchors of kind 'secret' to be hidden."

### 11.3 `TaskSeed`

```python
@dataclass
class TaskSeed:
    theme: str
    anchor_kinds: list[str]
    suggested_goal_kinds: list[str]
    difficulty: float
    evidence: int = 1
    family: str | None = None
```

One per task-cluster the seed-producer identifies. Each seed carries:

- `theme` — opaque cluster id (`"cluster-0"`, `"cluster-1"`, ...).
- `anchor_kinds` — kinds of things the cluster anchored on.
- `suggested_goal_kinds` — kinds at the sinks of productive paths.
- `difficulty` — `[0..1]`.
- `evidence` — how many independent observations support this seed.
- `family` — optional `TaskFamily.id` tag. A producer without family
  knowledge leaves this unset and lets TaskFamilies self-select by
  `anchor_kinds`. Hand-authored defaults can tag `family` directly
  (the cyber pack does).

### 11.4 Wire shape

```json
{
  "source": "cyber.webapp@v2 :: hand-authored",
  "ontology": { "id": "cyber.webapp@v2", "node_kinds": {...}, "edge_kinds": {...} },
  "topology": {
    "node_kind_freq":    { "service": 3, "endpoint": 5, "secret": 1 },
    "salient_kind_freq": { "endpoint": 1, "secret": 1 },
    "dead_end_ratio": 0.25,
    "hidden_signal": { "secret": 2 }
  },
  "task_seeds": [
    { "theme": "cluster-0",
      "anchor_kinds": ["endpoint", "vulnerability"],
      "suggested_goal_kinds": ["secret"],
      "difficulty": 0.7,
      "evidence": 1,
      "family": "webapp.pentest" }
  ],
  "difficulty": { "cluster-0": 0.7 },
  "coverage": { "service": 0.85, "endpoint": 0.9, "secret": 0.5 }
}
```

`task_seeds[].family` is omitted when unset.

---

## 12. Mutation + GraphPatch — the curriculum seam

```python
@dataclass(frozen=True)
class Mutation:
    patch: GraphPatch
    direction: str
    relevance: float
    family: str
    note: str = ""
```

Reference: `src/openrange/core/pack.py::Mutation`.

One curriculum move proposed by a TaskFamily. Aggregated by
`auto_evolve` (`src/openrange/core/curriculum.py`), applied via
`Builder.evolve`, then re-admitted through the full gate.

- `patch` — the `GraphPatch` (§3.4) that describes the world change.
  `apply_patch(graph_copy, patch)` is what actually mutates the world.
- `direction` — one of three string tags:

  | Tag           | Meaning |
  |---------------|---------|
  | `"harden"`    | Make the task harder (introduce a new defensive surface, raise a difficulty knob). |
  | `"soften"`    | Make the task easier (remove an attack vector, expose a hint). |
  | `"diversify"` | Keep the difficulty roughly equal but rotate which thing is tested. |

  These are the `Direction` `Literal` values
  (`src/openrange/core/curriculum.py`); `direction_from_reports`
  picks one based on the recent pass-rate.

- `relevance` — `[0..1]`. How well the move responds to the recent
  episode reports. The curriculum policy picks the highest-relevance
  candidate in the chosen direction.
- `family` — the id of the TaskFamily that proposed the move (the
  cyber pack tags each Mutation with `"webapp.build"` or
  `"webapp.pentest"` so aggregation can route back).
- `note` — optional human-readable context.

`Mutation` is **frozen** — the patch + tags are an immutable proposal.
A pack that wants to refine the patch overrides `Builder.evolve` and
returns a fresh `GraphPatch`; the default returns `mutation.patch`
verbatim.

---

## 13. Pack registration — entry points

A pack ships as a Python package that registers itself via the
`openrange.packs` entry-point group. The pack's `pyproject.toml`
declares:

```toml
[project.entry-points."openrange.packs"]
"webapp" = "cyber_webapp:WebappPack"
```

The **key** (`"webapp"`) is the pack id — it must equal the Pack
class's `id` attribute. The **value** is a dotted import path that
resolves to a Pack class (or any callable returning a Pack instance
with a parameterless `__init__`).

Reference: `src/openrange/core/pack.py::PackRegistry`.

`PackRegistry` consumes the entry points. The canonical instance is
`PACKS = PackRegistry(autodiscover=True)` (module-level in
`src/openrange/core/pack.py`) — on first `resolve()` call, it iterates
the entry-point group and instantiates each pack. Tests construct their
own `PackRegistry(autodiscover=False)` and `.register(pack)` instances
explicitly to avoid pulling in installed packs.

Errors during discovery (an entry point that fails to import, a
class that doesn't subclass `Pack`, a class whose `.id` doesn't
match the entry-point name) raise `PackError`.

NPC registration follows the same pattern via the `openrange.npcs`
entry-point group (the cyber pack registers five NPC factories;
see its `pyproject.toml`).

---

## 14. Stability

These shapes are settled enough to build against. The most likely
additive changes — none breaking:

- More `topology` keys in `PackPrior` (statistics that any builder
  could read, none mandatory).
- A `valid` summary on `EpisodeResult`.
- Additional `BuildEvent.phase` values (the existing six are stable;
  `evolve` is currently sparse).
- More `Direction` values in curriculum mutations.

Existing required fields won't change meaning without a version bump
in the relevant ontology id. The `Ontology` id format
(`"<name>@<version>"`) is the version-bump channel — a pack that
changes its node/edge kinds publishes a new id and the old snapshots
still load against the old id.

See [DESIGN.md](DESIGN.md) for the reasoning behind these shapes.
