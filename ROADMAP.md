# OpenRange Roadmap

> [!NOTE]
> Direction, not a release schedule. Each item links to its tracking
> issue — that's where status, scope, and discussion live.

The bet behind OpenRange: agents trained against fresh, runnable,
admission-checked worlds will generalize better than agents that
overfit to static benches. Making that real means (a) good packs
across many domains, (b) a clean way to plug a training loop into
them, and (c) a core that scales from "single process" to "production
training fleet."

Status tags on each item:

- 🚧 **in progress** — actively being worked on
- 🟢 **help wanted** — well-scoped, contributor-friendly
- 🟡 **design needed** — design doc PR before code
- 🔮 **research** — exploratory, no near-term plan

Browse all roadmap issues: [`label:roadmap`](https://github.com/vecna-labs/open-range/issues?q=is%3Aissue+is%3Aopen+label%3Aroadmap).

## Packs

Packs are the reusable starting points for a family of worlds —
ontology, realizer, defaults. See
[docs/start_here.md](docs/start_here.md) for the contract.

### Cyber (priority)

Cyber is the first-class pack. Both **offense and defense** are
first-class task families on the same pack — same world, different
role, different success check. Long-term goal: dual-mode scenarios where
one realized world hosts an offense agent and a defense agent in
parallel.

- 🚧 **`webapp`** — procedural builder + AST-spliced
  vuln injection + NPCs + per-task feasibility/success checks via
  TaskFamily. Already shipping. LLM enrichment is wired only on the
  pentest family's curriculum (`available_mutations`); task-instruction
  enrichment is unwired.
- 🚧 **`webapp.build` rigorous grader — staged rollout.** Stage 1 has
  landed: agent submits a handler source string into `result.json`;
  `check_success` runs it through a sandboxed subprocess against a
  held-out behavioral contract. Admission validates the contract is
  well-posed (clean reference passes; bug-injecting mutation breaks).
  Only the `api` service kind is wired; other kinds emit no build task.
  Stages 2–4 expand the mutation/contract library, add curriculum
  mutations on build, and plumb live realizer reload + an agent
  test-runner tool so the loop is multi-turn instead of single-shot.
- 🟢 **Kind / Kubernetes runtime backing**
  ([#189](https://github.com/vecna-labs/open-range/issues/189)) —
  unblocks real cross-service exploit chains.
- 🟢 **Expand the vulnerability catalog**
  ([#190](https://github.com/vecna-labs/open-range/issues/190)) —
  command injection, path traversal, deserialization, weak creds, etc.
- 🟢 **Defense-cyber task family**
  ([#191](https://github.com/vecna-labs/open-range/issues/191)) — same
  pack, detection / mitigation / patching tasks instead of flag
  retrieval. Enables dual-mode adversarial scenarios.
- 🟡 **LLM-driven naming realism**
  ([#192](https://github.com/vecna-labs/open-range/issues/192)) —
  `svc_web` → `customer-portal`, `acct_0` → `alice@corp.example`.
- 🔮 **MCTS-based world generator**
  ([#193](https://github.com/vecna-labs/open-range/issues/193)) —
  search over graph mutations instead of rejection sampling.

### Other domains

We want at least one non-cyber pack so the Pack ABC isn't accidentally
cyber-shaped.

- 🟢 **Trading starter pack**
  ([#194](https://github.com/vecna-labs/open-range/issues/194)) —
  order book + P&L success check. First non-HTTP runtime backing.
- 🔮 **Social / negotiation pack**
  ([#195](https://github.com/vecna-labs/open-range/issues/195)) —
  multi-turn dialogue with NPC counterparties.
- 🔮 **Robotics pack**
  ([#196](https://github.com/vecna-labs/open-range/issues/196)) —
  MuJoCo / Isaac integration via non-HTTP backing.
- 🟢 **Pack author guide**
  ([#197](https://github.com/vecna-labs/open-range/issues/197)) —
  step-by-step doc lowering the bar for first-time pack authors.

## Training integration

Today the examples are *eval* loops. Closing them into actual
*training* is the next big piece.

- 🟡 **Training-loop integration via open-trajectory-gym**
  ([#198](https://github.com/vecna-labs/open-range/issues/198)) —
  pair OpenRange with
  [open-trajectory-gym](https://github.com/vecna-labs/open-trajectory-gym).
  Headline ask. Needs a design doc on the `EpisodeReport` ↔ trajectory
  format seam.
- 🟢 **Per-domain reward adapters**
  ([#199](https://github.com/vecna-labs/open-range/issues/199)) —
  structured `EpisodeResult` → scalar / vector reward signal.
- 🟢 **Curriculum-driven training demo**
  ([#200](https://github.com/vecna-labs/open-range/issues/200)) —
  notebook showing success-rate curves as `evolve(...)` hardens worlds.

## Core

Primitives every pack and harness depends on.

- 🚧 **Per-NPC LLM backend selection** — extracting
  `openrange-pack-sdk` removed the `model:` config key from the cyber
  webapp NPC factories (it used to construct a concrete
  `StrandsAgentBackend` and so couldn't survive the strict
  pack-cannot-import-openrange boundary). Every NPC in an episode now
  shares the single `RunConfig.npc_agent_backend`. To restore per-NPC
  model selection cleanly, add a `backend_id:` config key on NPC factories
  + a `RunConfig.npc_backend_factory: Callable[[str], AgentBackend] | None`
  the runtime resolves at NPC start. Mirror of the entry-point pattern;
  no openrange import in pack code.
- 🟢 **Restore 100% test coverage**
  ([#201](https://github.com/vecna-labs/open-range/issues/201)) —
  coverage dropped to 80% during the typed-property-graph + pack /
  admission refactor; bring it back.
- 🟢 **Second non-cyber pack to validate domain-agnostic shape** —
  today only `webapp` ships, so the design's domain-agnosticism is
  unverified end-to-end. A second pack against a non-HTTP backing is
  the load-bearing proof.
- 🟡 **TaskFamily check sandboxing**
  ([#202](https://github.com/vecna-labs/open-range/issues/202)) —
  the pack/admission refactor replaced exec'd verifier source with
  regular Python methods on TaskFamily; revisit isolation/timeouts at
  training scale. The cyber `webapp.build` grader exec's untrusted
  agent source as a subprocess with wall-clock + best-effort RLIMIT_CPU
  only — no filesystem, network, or syscall isolation. Production use
  needs firejail / bwrap / seccomp / container.
- 🟡 **Lineage as a pool, not a chain**
  ([#203](https://github.com/vecna-labs/open-range/issues/203)) —
  multi-parent reference graph for curriculum / multi-snapshot training.
- 🟡 **Multi-host EpisodeService**
  ([#204](https://github.com/vecna-labs/open-range/issues/204)) —
  distributed training fleets.
- 🔮 **Container snapshotting for stateful runtimes**
  ([#205](https://github.com/vecna-labs/open-range/issues/205)) —
  CRIU / overlayfs / `docker commit` for fork/restore.

## Dashboard

The inspection surface for builders, snapshots, and live episodes.
See [docs/dashboard.md](docs/dashboard.md).

- 🟢 **Snapshot inspector improvements**
  ([#206](https://github.com/vecna-labs/open-range/issues/206)) —
  better UX for large world graphs, oracle-path highlighting.
- 🟢 **Live training view**
  ([#207](https://github.com/vecna-labs/open-range/issues/207)) —
  curriculum evolution + per-rollout request log when training lands.
- 🟡 **Pluggable per-pack dashboard widgets**
  ([#208](https://github.com/vecna-labs/open-range/issues/208)) —
  cyber's natural view is service topology; trading's is order book.

## Docs

- 🟢 **Tutorial: build → eval → curriculum**
  ([#209](https://github.com/vecna-labs/open-range/issues/209)) —
  `uv sync` to working eval to curriculum walk in one notebook.
- 🟢 **Architecture decisions log**
  ([#210](https://github.com/vecna-labs/open-range/issues/210)) —
  short notes on the *why* of big shape choices.

## How to help

1. Pick something tagged 🟢 → comment on the issue saying you want it.
2. For new packs (trading, social, robotics, anything else), open the
   issue with a sketch of the ontology + TaskFamily shape; we'll work
   through the contract with you.
3. 🟡 items want a design doc PR before code — saves cycles.
4. Friday community call — link in
   [README.md](README.md#-community-call).

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup and workflow.
