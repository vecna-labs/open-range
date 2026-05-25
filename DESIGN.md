# OpenRange Design Rationale

Where [CONTRACTS.md](CONTRACTS.md) pins down *what* the shapes are,
this document explains *why* — what alternatives we considered, what
each lost on, what the design buys you, and where the boundaries are
deliberate rather than coincidental.

Audience: a senior engineer evaluating whether to bet on OpenRange,
or a contributor about to propose a change that would touch a load-
bearing seam.

No marketing. The interesting content is the alternatives that lost.

---

## 1. The bet

OpenRange exists to test a falsifiable claim:

> Agents that train on **fresh, runnable, admission-checked worlds**
> generalize better than agents that overfit static benchmarks.

A static benchmark is a fixed graph. Every training run sees the same
fixed graph. Whatever the agent learns is exactly the right amount of
information to solve that fixed graph — no more. Performance numbers
go up; transfer to anything off the benchmark goes down.

A *fresh, runnable, admission-checked world* is the opposite:

- **Fresh** — sampled new for each build. The agent never sees the
  exact same graph twice. Memorizing structure is no longer a
  shortcut.
- **Runnable** — the world isn't a static configuration; it actually
  exists. Services answer real HTTP requests, files exist on disk,
  exploits land or don't based on what the agent does.
- **Admission-checked** — every task is verified to be solvable in
  the world it was generated against, before the episode starts. A
  broken task doesn't pollute the training signal.

If the bet is right, the same training setup that trains an agent
against the cyber webapp pack should generalize to a different pack
(say, a trading environment) without rewriting the harness or the
reward policy.

The whole rest of this document is choices made in service of that
bet. Each seam below names what would have been easier and why we
chose the harder thing instead.

---

## 2. CORE vs. PACK — the seam that everything else depends on

The single most important boundary in OpenRange is between
**core** (`src/openrange/world_ir.py`, `src/openrange/core/`,
`src/openrange/ontologies/`) and **packs**
(`packs/cyber_webapp/...`, future `packs/*/`):

| Layer | Owns | Knows |
|-------|------|-------|
| CORE  | Types and generic algorithms. The admission loop, the graph validator, the episode lifecycle, `distill`, `auto_evolve`, `SnapshotStore`. | Nothing about any specific domain. No `service`, no `vulnerability`, no `endpoint`, no `order`, no `flag`. |
| PACK  | Values and domain functions. The `Ontology` value, the `Builder`, the realizer, the `TaskFamily` classes, the success-check code. | Everything about its domain. Free to name `service`, `vulnerability`, `endpoint`. |

This split is enforceable as a one-liner: `grep` every file under
`src/openrange/world_ir.py`, `src/openrange/core/`, and
`src/openrange/ontologies/` for the domain vocabulary list. If a
domain word shows up in core, the boundary leaked.

### Why this matters

The obvious alternative is to put domain affordances directly in
core. "Just have core know about `service` nodes — it'd be simpler."
The trap: every time you add a pack, you grow core. Every pack pays
for every other pack's complexity. Cross-domain transfer (the bet)
becomes impossible because the core is fused to one domain.

The non-obvious alternative is to make core *parametric* on a domain
config — pass in dict-shaped descriptors of what `service` means and
let core branch on them. This is the bug: the moment core branches
on a domain config key, it's domain code wearing a generic costume.
The branch table is a tiny ontology, every config key is a domain
word, and you've reinvented the problem.

The chosen design: an **Ontology is a value**. A pack declares one;
the same generic validator checks any graph against any Ontology.
Core code never branches on an ontology — it iterates over it.

The cost: the pack has more responsibility. The pack must write its
Ontology, its invariants, its builder, its task families. The pay-off
is that everything specific to a pack lives in *one place* (under
`packs/<pack-name>/`) and core doesn't grow when a new pack ships.

### Where this rule slips and you have to be vigilant

Two places. First: `Snapshot.lineage` is `Mapping[str, Any]`. A pack
could put a domain key in there and the wire format wouldn't notice.
The rule is: lineage is provenance, not behavior. Anything core
*branches on* must come from the ontology or the protocols, never
from lineage.

Second: the `Manifest` is also `Mapping[str, Any]`. Same rule —
core never reads a manifest key. The episode layer reads two
runtime-knob keys (`runtime.tick.mode`, `runtime.tick.rate_hz`); those
are generic infrastructure knobs, not domain knowledge.

---

## 3. Why a typed property graph, not a flat dict

The world is a typed property graph: `Node`, `Edge`, an ontology over
both. Three choices were live when the meta-model landed:

1. **A flat dict per pack.** Each pack defines its own dataclass for
   its world (`Webapp`, `TradingFloor`, ...). Core takes `Any` and
   trusts the pack.
2. **JSON schema.** Each pack ships a JSON schema; core validates
   incoming worlds against it.
3. **A typed property graph + an ontology-as-data.**

The flat-dict approach loses on transfer: the snapshot serializer,
the dashboard, the curriculum, the snapshot store, the distill
function — every generic algorithm has to grow a `match pack:`
statement. Or be forked per pack. Either is the failure mode §2 warns
about.

JSON schema covers the shape but not the *operations*. Apply a patch
to a graph? Walk the graph's in-edges? Content-address it? None of
those are JSON-schema things; we'd reinvent them per pack.

The typed property graph wins for three reasons:

1. **Diffability.** `GraphPatch` is the universal diff type. Any
   structural change to any world — a builder's repair, a
   curriculum mutation, a distill-induced refinement — is a
   `GraphPatch`. One `apply_patch` function works for every pack.
2. **Content addressing.** `WorldGraph.content_hash()` is `sha256:`
   over `(ontology, nodes, edges)` only. `meta` and per-node
   `runtime`/`meta` are excluded, so two builds that produce the
   same world share one snapshot id regardless of manifest noise
   or provenance fields. Reproducibility becomes a structural
   property, not a documentation hope.
3. **Generic walks.** `graph.by_kind("service")`, `graph.in_edges`,
   `graph.out_edges` — every pack gets these for free because every
   pack speaks the same graph type.

The cost is that the pack must phrase its world as nodes and edges
even when a dict would be more natural. We think the
forced-uniformity is the feature, not the cost. (The cyber pack
phrases a Flask app as 11 node kinds, and the dashboard, the
curriculum, and the snapshot store don't have to know that.)

### Ontology is a value, not a type

Within the typed-property-graph choice, there's still a question:
is an `Ontology` *code* (a Python class hierarchy) or *data* (a
dataclass instance)?

If it's code, every pack imports `Node` subclasses (`Service`,
`Endpoint`, `Vulnerability`), and validation is `isinstance`
checks. Faster, more familiar — but core can no longer iterate
generically over the ontology to do anything (validate against it,
serialize it, distill into it). Cross-pack tooling has to
introspect Python types. That doesn't compose.

If it's data, the Ontology is an `Ontology(id=..., node_kinds=...,
edge_kinds=...)` dataclass that *describes* the schema. Core does
one generic walk: "for each node, look up its kind in the ontology,
check its attrs against the AttrSpec." One validator works for every
pack and is exactly as fast as the kind table is big.

We chose data. The cost is that you can't do `isinstance(node,
Service)` — the pack helpers that need that have to filter by
`node.kind == "service"`. In return, every generic algorithm
(`distill`, `validate`, `snapshot_to_dict`, dashboard rendering, the
boundary script) is the same code for every pack.

---

## 4. Why one Pack, many TaskFamilies

The cyber pack ships *one* Pack (`WebappPack`) and *two* TaskFamilies
(`WebappBuild`, `WebappPentest`). The same world graph admits both:
an agent doing `webapp.build` is implementing an endpoint; an agent
doing `webapp.pentest` is exploiting one. Different entrypoints,
different goals, different success criteria, same world.

The wrong shape would have been *one Pack per task type*: a
`WebappBuildPack` and a `WebappPentestPack` that each ship their own
ontology and builder. That looks tidy until you ask: what makes them
the same world? Nothing — there's no shared graph value, just two
unrelated packs that happen to be named similarly.

So we drew the line differently. **A Pack is a world-family.** A
TaskFamily is a *role* against that world.

Why this matters:

- **One ontology, one builder, one realizer.** The same procedural
  sampler emits worlds that serve both families. The same Flask app
  is the runtime substrate. No duplication, no drift.
- **Cross-family transfer is free.** A trained pentest agent can be
  evaluated on build tasks (or vice versa) without rebuilding the
  world.
- **Curriculum across families.** `auto_evolve` aggregates
  mutations across `pack.task_families()`. A move that hardens the
  pentest family (introduce a new vuln) and a move that hardens the
  build family (require an auth step) are both candidates from one
  pool, picked by relevance.

"Domain" lives on the **TaskFamily**, not the Pack. Two domains
against one world is the load-bearing cyber demo (`webapp.build`,
`webapp.pentest`).

---

## 5. Why entrypoints and goals live on the task, not on the node

The alternative was tempting: bake "this is an entrypoint" and "this
is a goal" into the node, as roles in the world graph.

It breaks for the same reason §4 went the way it did. The same world
graph serves two families. `webapp.build`'s entrypoint is a
*service* node (the agent opens its source); `webapp.pentest`'s
entrypoint is an *endpoint* node (the agent hits a URL). The build
family's goal is an endpoint; the pentest family's goal is a hidden
secret. **The same node** is the build family's goal and not the
pentest family's goal.

If "entrypoint" and "goal" were node roles, the graph would have to
carry "is build-family's entrypoint, is pentest-family's entrypoint,
is build-family's goal..." — a quadratic explosion of role
combinations, each one specific to a task family core was supposed
to know nothing about.

So we drew the line: **world-absolute** facts live on the node;
**task-relative** facts live on the task. The node's `Role` enum
(`ACTOR`, `NPC`, `EXTERNAL`) is world-absolute — true regardless of
what task is running. The `TaskSpec.entrypoints` and
`TaskSpec.goal_nodes` are task-relative — they're the binding from a
specific task into specific nodes.

The validator enforces the binding (`validate_task_bindings` in
`src/openrange/core/admit.py`): every entrypoint and goal must
resolve to a real node, and no entrypoint may be `HIDDEN` (you
cannot start the agent at a node it isn't supposed to be able to
see). Goals may be hidden — discovery is often the point.

---

## 6. Why admission has five layers

The admission gate is layered, on purpose. Each layer catches a
different bug class.

```
1. structural        : ids / edges / dangling endpoints
2. ontology          : kinds, required attrs, REFs, endpoints, degree caps
3. pack invariants   : domain-specific structural rules
4. task bindings     : entrypoints/goals exist; entrypoints not HIDDEN
5. task feasibility  : each TaskFamily.check_feasibility(graph, task)
```

The naive alternative is a single "is this world valid" check. It
fails because the bugs aren't the same shape:

- Layer 1 (structural) catches a malformed builder — duplicate
  ids, a stray edge with no destination. Cheap to write, cheap to
  run.
- Layer 2 (conformance) catches a builder that doesn't respect its
  own ontology. Required `name` attr missing, `kind=service` with
  no allowed parent. Schema-level wrongness.
- Layer 3 (pack invariants) catches structurally-valid but
  *semantically* broken worlds — for the cyber pack, "every secret
  has a record that holds it" and "every flag is reachable from a
  public surface." These can't be encoded in `AttrSpec`s; they need
  to walk the graph and reason.
- Layer 4 (task bindings) catches tasks pointing at the wrong
  nodes. A pentest task whose `entrypoint` is a HIDDEN secret —
  schema-valid, but unusable.
- Layer 5 (feasibility) catches well-formed, well-bound,
  schema-correct worlds that *no agent can actually solve*. The
  cyber pentest family walks `endpoint -- routes_to --> service
  -- enables --> vulnerability -- affects --> secret` and refuses
  to admit a task whose secret is unreachable. This is the layer
  the bet most depends on: a broken task is invisible until layer
  5 unless you specifically look.

Each layer's bugs slip past every other layer's check. You don't get
to skip one because "the others probably catch it." They don't.

The cost is five passes per admission attempt. The pay-off is that
no broken world ever runs. The training signal stays clean.

---

## 7. Why repair, not validate-only

Once admission detects a problem, it has two options: tell the
caller "rejected" and stop, or hand the problem back to the builder
and ask for a fix.

The validate-only option treats `admit()` like a unit test. Fine for
debugging, useless in a training loop — a single rejected build kills
the run.

The repair option recognizes that **only the builder knows how to
fix it**. Core knows the world is wrong; core doesn't know *which
sampling decision* produced the wrong world or how to roll it back.
A procedural builder resamples with a perturbed seed. An LLM
builder might rewrite the offending bit. A search-based builder
might backtrack.

So `admit()` calls `builder.repair(prev, errors, infeasible)` up to
`max_repairs` times (default 2 = 3 total attempts). The builder
returns a new `BuildResult`. The same five layers run on it. If it
passes, freeze; if not, retry.

The default `Builder.repair` raises `NotImplementedError` with a
message that explains how to opt in. Packs that ship a procedural
builder must override this; the LLM-driven cyber pack ships a
"perturb seed and resample" implementation.

What we considered and rejected: handing the builder *partial* fixes
("here's a patch that would close issue X"). That puts core in the
position of inventing repair patches, which means core has to know
the domain. Wrong layer. Core hands the builder the *failures*; the
builder hands back a new candidate.

---

## 8. Snapshot = timeless graph + build history beside it

The world graph carries no timestamps. `content_hash()` excludes
`meta`, excludes `Node.runtime`, excludes `Node.meta`. Two identical
builds (same builder, same seed, same prior) produce the same
content hash. The `Snapshot.snapshot_id` is literally
`graph.content_hash()`. That's reproducibility as a structural
property: two snapshots with the same id are interchangeable for
every downstream purpose.

But the build *process* has a story worth keeping. Which pass ran?
What did a repair change? Why was an attempt rejected? Where did the
manifest come from? If we recorded that *inside* the graph, every
build would have a different timestamp and the content hash would
change. Reproducibility would die.

So the rule is: **transaction time goes inside a graph only when the
graph has no other identity to protect.** The world graph has an
identity to protect (content hash); its build history lives
*beside* it, as `Snapshot.history: tuple[BuildEvent, ...]`. Each
event has a `seq`, a `phase` (`build` / `validate` / `feasibility` /
`repair` / `freeze` / `evolve`), a `detail` string, and optional
`refs`. Provenance, attempts, lineage all live in `Snapshot.lineage`,
also beside the graph.

This is a deliberate asymmetry with the BBG. A BBG has no other
identity to protect — it's a growing log of an agent's beliefs,
not a content-addressed thing. So a BBG's transaction time goes
*inside*, as a status-event log: every node's `status` transitions
append an event, and the node's current status is the latest event
for it. Bitemporal would have been the other obvious choice (valid
time and transaction time both inside the graph); we explicitly
rejected it because *valid time belongs to the OpenRange runtime,
not to the memory*. The wayfinder's only clock is the agent's
knowledge advancing.

The general rule, restated:

| Graph         | Identity | Transaction time |
|---------------|----------|------------------|
| World graph   | Content-addressed | Beside (Snapshot.history) |
| BBG           | None protected     | Inside (status-event log) |

---

## 9. PackPrior — generic statistics, not domain decisions

`PackPrior` is the seam between observed agent experience (a BBG)
and a Builder. The temptation here is enormous: emit *specific
generation directives* in the prior so the builder doesn't have to
think. "Use this service mix." "Place a SQL injection on `/login`."
"Make the flag a JSON field."

This is the bug. The moment `PackPrior` carries a pack-specific
directive, `distill()` becomes pack-specific. You write a separate
`distill_webapp()`, `distill_trading()`, `distill_robotics()`, and
the flywheel is no longer reusable. Worse, the prior shape no longer
matches across packs, so a hand-authored default for pack X is not
shaped like a distilled prior for pack X — and the builder has two
code paths instead of one.

The chosen rule: **the prior carries only generic graph statistics.
The builder INTERPRETS those into domain decisions.**

Generic statistics are things any builder could read regardless of
domain:

- `node_kind_freq` — how many of each kind appeared.
- `salient_kind_freq` — how many of each kind were judged to matter
  (anchored by a thought, repeatedly visited, or `observe()`'d).
- `dead_end_ratio` — what fraction of traversals went nowhere.
- `hidden_signal` — per-kind density of confirmed-thought anchors.
- `task_seeds` — clusters of related thoughts, with anchor kinds
  and difficulty scores.
- `coverage` — per-kind explored ratio.

A cyber builder reads `salient_kind_freq["endpoint"] = 5` and decides
"sample more endpoints." A trading builder reads
`salient_kind_freq["order"] = 5` and decides "sample more orders." A
robotics builder reads `salient_kind_freq["waypoint"] = 5` and
decides "sample more waypoints." Same prior shape, three packs, no
cross-pack leak.

Why this is the bootstrap-to-flywheel seam: the cyber pack ships a
**hand-authored** `default_prior()`. That hand-authored prior has
*exactly the same shape* as a `distill()` output. When the flywheel
kicks in — episodes produce trajectories, trajectories grow BBGs,
BBGs distill into priors — the builder switches from reading the
hand-authored prior to reading the distilled prior and **doesn't
change its code**. The transition is seamless because both priors
have the same shape and both carry only generic statistics.

There is still a pack-specific generation config (the cyber pack
ships `_CYBER_GENERATION_CONFIG` next to `default_prior()` — vuln
weights, service-kind weights, chain depth caps). This lives
*inside* the pack, not in the prior. The pack's sampler reads it
when it lowers generic frequencies into concrete service / vuln
picks. The prior is generic; the lowering is domain-specific; the
lowering lives where the domain lives.

---

## 10. Why TaskFamily owns success_check, not core

`EpisodeService.stop_episode` collects `final_state =
RuntimeHandle.collect()` and asks the pack:
`pack.task_family(task.success_check).check_success(graph, task,
final_state)`. The family returns an `EpisodeResult(success: bool,
subgoals: Mapping[str, bool], reason: str)`.

The naive alternative is to bake success into the world graph — "a
goal is reached when the agent visits the goal node." This fails for
two reasons. First, "reached" is task-relative — visiting a node
doesn't mean the agent has succeeded at *this task*. The pentest
agent might land on the secret node and still not have submitted
the flag; the build agent might have the right repo open and never
make the endpoint serve 200. Second, success criteria are
domain-shaped — "endpoint serves 200 to a smoke test", "submitted
flag matches `value_ref`", "all subgoals satisfied" — and putting
them in core re-creates the §2 leak.

So success-check is a TaskFamily method. The family already knows
what its task looks like; it's the natural place to decide what
counts as completion.

Core's contract is narrow on purpose:

- It hands the family `(graph, task, final_state)`.
- It accepts back an `EpisodeResult`.
- It **never inspects the structured fields**. `EpisodeReport`
  carries the result through to whoever wants it (curriculum,
  training adapter, tests, dashboard).

Crucially: `EpisodeResult` is **structured**, not a scalar reward.
No `EpisodeResult.score: float`. The reason is the bet (§1): if the
same training setup must work across packs, the reward shape can't
be baked into the env. Reward shaping is the harness's job. The env
returns a structured pass/fail + subgoals + reason, and the harness
maps it.

A nice side-effect: `EpisodeResult` is human-readable. A training
log of "agent passed, subgoals={found_login: true, exploited_sqli:
true}" tells you exactly what happened. A training log of `0.83`
tells you nothing.

---

## 11. Why RuntimeHandle has eight methods, not five

The `RuntimeHandle` Protocol has eight methods: `reset`, `surface`,
`poll_events`, `terminal`, `checkpoint`, `restore`, `collect`,
`stop`. The temptation is to collapse some of them. Why not just
`(start, observe, end)`?

Each one names a thing the episode loop genuinely needs from the
substrate:

- `reset()` — boot the world. Two-phase initialization is essential
  because a pack may construct a handle (deciding the backing,
  rendering source) without paying the I/O cost of starting a
  subprocess. The episode loop calls `reset()` when it actually
  wants the world running.
- `surface()` — the agent-facing IO bundle. Critically, this is not
  just `base_url`. The cyber pack returns
  `{base_url, http_get, http_get_json, agent_root}`. NPCs need the
  callables; the agent needs the URL and the working dir; the
  dashboard wants the strings. One method returns the whole bundle;
  the caller picks what it needs.
- `poll_events()` — drain side-effect events. The episode loop
  calls this each tick. Forwarded to the dashboard.
- `terminal()` — has the agent finished? Returning `(done, reason)`
  (not just a bool) is so the episode loop can record *why* it
  ended. The cyber pack returns `(True, "result.json written")`.
- `checkpoint()` / `restore(state)` — counterfactual replay. The
  payload is opaque to core. A pack that backs onto a stateful
  subprocess can stash a state-machine snapshot; a pack that backs
  onto a filesystem can stash a directory snapshot. Both replay
  the same way from core's perspective.
- `collect()` — structured final state. The dict the family's
  `check_success` reads. Separate from `stop()` because the family
  may need the state and `stop()` is destructive.
- `stop()` — tear it down. Idempotent.

Collapsing `surface()` and `observe()` into one was tempting until
we noticed the cache pattern: the harness reads `base_url` ten
times in a tick, and `surface()` is called once per `reset()` and
cached. They're different lifetimes.

Collapsing `checkpoint`/`restore` was tempting too — until you
recognize that *forking* is `checkpoint + restore` on the same
service, and you want that to be one cheap operation
(`EpisodeService.fork`). Splitting them keeps both available.

---

## 12. BBG sharing — declaration, not import

OpenRange ships an `Ontology` for `bbg.wayfinder@0.1.0`
(`src/openrange/ontologies/bbg.py`). The harness that *maintains* a
BBG — perception, status-event log, NAV-pack generation — lives
outside OpenRange. Neither side imports the other.

The shared thing is the wire format: a BBG-shaped `WorldGraph`. The
harness emits one when it wants to ship its BBG out. OpenRange's
`distill()` reads one and emits a `PackPrior`. The contract is the
ontology id (`"bbg.wayfinder@0.1.0"`) and the JSON shape.

The alternative was the harness depending on OpenRange (then the
harness can't ship without OpenRange) or OpenRange depending on the
harness (then OpenRange can't ship without the harness). Both fail
the "OpenRange works standalone" requirement.

The chosen design lets each side reference the other only by id:

- The harness writes `WorldGraph(ontology="bbg.wayfinder@0.1.0",
  nodes=..., edges=...)` and serializes.
- OpenRange's `distill()` accepts any `WorldGraph` whose
  `ontology == "bbg.wayfinder@0.1.0"`.

If a fork of the harness wants its own meta-model types, it can
vendor `world_ir.py` — the comment at the top of that file says so
explicitly. The contract is the JSON wire format, not the Python
classes.

This is also why `distill()` lives in `openrange.core.distill` and
not in the harness:

- It *consumes* a `WorldGraph` (an OpenRange meta-model type).
- It *produces* a `PackPrior` (an OpenRange consumer concept).

Putting it anywhere else would invert one of those dependency
arrows.

---

## 13. What's deliberately not in core

The core does **not** own:

- **Reward shaping.** No `EpisodeResult.score`. No reward function
  anywhere. A harness-side training adapter maps the structured
  result into whatever signal the training setup needs. Cross-pack
  transfer requires this — see §10.
- **The LLM.** Core does not import any LLM library. The `LLMBackendLike`
  Protocol is offered to TaskFamilies that want optional enrichment
  (the cyber pack uses it for task-instruction templating and
  curriculum relevance scoring). The Protocol takes `Any` as the
  request type to avoid importing the concrete LLM module from core
  (the concrete `LLMRequest`/`LLMResult` types live in
  `openrange.llm`, not in `openrange.core`).
- **The agent loop.** Core ships `EpisodeService` (start, observe,
  tick, advance, stop). It does **not** ship the agent's strategy
  loop. The harness owns the model, the tool definitions, the
  rollout policy.
- **Training-step bookkeeping.** No notion of an "epoch", a "step",
  a "batch". OpenRange runs episodes; the training harness aggregates
  episodes.

These omissions are the price of cross-pack transfer. If the env
prescribed a reward function, a model, an agent loop, or a training
algorithm, swapping packs would require swapping those too — defeating
the bet.

What this means in practice: an agent harness that wants to train
against OpenRange writes:

1. A function that maps `EpisodeResult` to its training signal
   (reward, preference pair, SFT target, whatever).
2. A rollout loop that calls `EpisodeService.start_episode` /
   `observe` / `advance` / `stop_episode` against a `Snapshot`.
3. Whatever model / tools / training algorithm it already has.

OpenRange is invisible to all three. That's the point.

---

## 14. The flywheel — bootstrap to learned, no code change

The endgame is a closed loop:

```
   agent runs real task
      |
   wayfinder grows a BBG
      |
   distill()   ->  PackPrior  (statistical, generic)
      |
   Pack.make_builder(prior)
      |
   builder samples a world  ->  admission  ->  Snapshot
      |
   episode runs an agent
      |
   trajectory grows the next BBG
```

Each turn: real-world experience grows BBGs, BBGs distill into
sharper priors, priors yield better worlds, worlds train better
agents.

The hard part is the **transition from hand-authored to learned**.
At bootstrap, no BBG exists; the builder reads a hand-authored
`PackPrior` shipped by the pack. After a few flywheel turns, a
distilled prior is available — but the builder has no idea which it's
reading.

The whole architecture in this document is designed so that
transition is **seamless**:

- The hand-authored default and the distill output have the **same
  shape** (`PackPrior` with the same generic-statistics fields).
- The builder reads `PackPrior` regardless of provenance.
- `Pack.make_builder(prior=None)` lands the hand-authored default
  via the pack-private `default_prior()` helper; everything else is
  one code path.

This is why §9 insists on generic statistics. If the prior carried
domain-specific directives, the hand-authored version would look
nothing like the distilled version, and the builder would need two
modes. The generic-only rule keeps it one mode.

---

## 15. The shipped omissions — what we've left for later

This document describes what shipped, not what's planned. A handful
of seams have a fixed contract and a placeholder body:

- **Pack-specific edge kind names in `distill(into=None)`.** The
  `into=None` path induces a single generic `transitions` edge kind
  carrying the observed kind-pair endpoints. A bootstrap caller can
  build against this, but pack authors typically refine the names
  during ontology authoring (e.g. `transitions` → `runs_on` +
  `exposes`).
- **Multi-trajectory `distill`.** Today `distill` reads one BBG.
  Real flywheel use will merge evidence across many BBGs, with
  `TaskSeed.evidence` actually accumulating. The shape is right;
  the implementation is one-shot.
- **`status_log` mining.** `distill` accepts a `status_log`
  argument and currently ignores it. The parameter is kept in the
  signature so the wire format stays stable; v2 starts reading
  "agents believed X for N steps before discovering not-X".
- **Container / simulator backings.** The cyber pack's
  `WebappRuntimeHandle` raises `NotImplementedError` for backings
  other than `Backing.PROCESS`. The Protocol shape supports all
  four; the wiring is partial.

Each is an additive widening of a stable contract. None requires a
breaking change to anything in [CONTRACTS.md](CONTRACTS.md).

---

## 16. Reading order

If you're new to OpenRange, the order that works:

1. This document for the *why* of the boundaries.
2. [CONTRACTS.md](CONTRACTS.md) for the wire formats and Protocol
   signatures.
3. `src/openrange/world_ir.py` for the meta-model (300 lines).
4. `src/openrange/core/pack.py` for the Protocols (600 lines).
5. `src/openrange/core/admit.py` for the admission loop (400 lines).
6. `packs/cyber_webapp/cyber_webapp/__init__.py` for what a Pack
   wires up.
7. `packs/cyber_webapp/cyber_webapp/families/build.py` and
   `families/pentest.py` for what a TaskFamily looks like.

By that point you'll have a working mental model of how a manifest
becomes a Snapshot becomes an episode.
