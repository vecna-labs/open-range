# The curriculum is an evolving pool of worlds-and-tasks

OpenRange's bet is that agents trained on fresh, admission-checked worlds
generalize better than agents that overfit a fixed benchmark. That bet is
stated today at eval time ("the agent never sees the same graph twice"). This
document states it at training time and gives it a shape: what the training loop
trains on, how it changes over time, and why that is stable for the kind of
online RL (GRPO) we use.

It is grounded in the published work on automatic curricula and environment
design (POET, PAIRED, Prioritized Level Replay, ACCEL) and on online RL for
language models (GRPO, WebRL, Absolute Zero); the appendix lists the sources.
Where the shipped code already provides a seam, it is named; where the design
needs something that does not exist yet, it says so plainly.

## 1. The shape of the loop

Worked example first. You want to train a pentest agent. The simplest loop
trains it on one company world, watches it learn to breach that world, mutates
the world a little harder, and trains on the harder one — a chain, one world
live at a time. That is what ships today. It has two holes: the agent sees only
one world per round, so it has no spread to generalize from; and each round
throws the previous world away, so nothing stops it forgetting what it learned
two worlds back.

The fix changes *what the loop trains on*. Instead of one current world, it
trains on a **pool**: many admitted `(world, task)` pairs, sampled per episode.
The pool is not the opposite of evolution — the pool is the thing that evolves.
Generalization comes from the *spread* of worlds the agent sees within a round.
Curriculum comes from a *slow shift* of the pool's makeup between rounds.

This document is about evolving the *worlds*. Growing the *generator itself* — an
external knowledge graph that expands as the agent trains and later bootstraps a
fresh generation pass — is a level up from this and out of scope here.

## 2. Three units: group, pool, round

Named by what each holds fixed.

- **Group** — `G` rollouts on one `(world, task)`. This is the GRPO group, and
  it is the unit that must stay fixed (§5).
- **Pool** — the set of admitted `(world, task)` pairs a round samples from.
  Different groups in a round land on different pairs. Keyed by
  `(snapshot_id, task_id)`, *not* by world content-hash alone: two tasks on one
  graph share a content-hash but are different problems, so content-hash keying
  would under-count.
- **Round** — one pass of GRPO over the groups drawn from the current pool, then
  one pool-update that may shift the pool for the next round.

The pool widens along **two axes that cost very differently**:

- **Tasks per world** (cheap). Same realized world, different goal / role /
  entrypoint — the `TaskFamily`/`TaskSpec` split (entrypoints and goal nodes are
  task-relative). Reuses a warm world, no boot. It widens *task coverage*, not
  the world diversity the generalization argument needs.
- **Worlds in the pool** (a boot each). Different topology, chain depth, flag
  placement. This is the spread that generalizes.

A round needs a floor of *distinct worlds*, set independently of the warm-cache
budget, or the cheap task axis silently collapses the spread the bet relies on.

## 3. Ways of evolution

Today's evolution vocabulary is three operators on the vulnerability *set* — add
a vuln, remove a vuln, swap a vuln's kind — on a fixed topology with a fixed
solve path. That changes recon difficulty and which exploit to find, but never
the required skill: the agent solves the same pivot with more or fewer decoys
around it. That is why the chain felt like a toy — it only ever made the world
busier, not harder.

It also confines evolution to graph-patches on a fixed estate. But building and
evolving are the same act — produce an admission-checked world, from scratch (a
seed) or from a parent (a step) — so an evolution operator can just as well
**build new structure**: an LLM-realized service or tier, exactly as the builder
does, under the same gate (§2 of the cyber design: procedural owns correctness,
the LLM owns variety, behind admission). Enterprise worlds cannot be generated
in one shot, so this is how they grow — build, train, build, train, each round
adding to the world, not only mutating what is there.

A real curriculum needs operators that change the *problem*. Classify them by
what they do to the agent's required skill, because the type decides where the
operator is allowed to act:

1. **Deepen — append a hop** *(extends the skill)*. Extend the credential chain
   by one hop: the agent loots one more credential and reuses it one more time.
   The N-hop solution is a *prefix* of the (N+1)-hop solution, so the policy's
   learned behaviour stays a valid partial solution. This is the one operator
   safe to use as a frontier step. The hop can be a graph-patch (the company
   pack's 1/2/3-hop credential chains already exist) or a whole **LLM-built
   service** appended to the path — same skill effect, richer surface. The patch
   form ships (`auto_evolve`'s append-a-hop); the LLM-built form is new work.
2. **Add a required recon step** *(extends the skill)*. Stop disclosing a host's
   address so the agent must enumerate to find it. Adds a discovery step the
   path now requires.
3. **Plant a decoy vuln** *(leaves the skill unchanged)* — today's `harden`.
   Adds an off-path vuln the agent routes around. Changes recon difficulty, not
   the required path. Useful for diversity, wrong for the difficulty frontier —
   calling it "harden" oversells it.
4. **Swap the required exploit class** *(replaces the skill)*. Change the vuln on
   the solve path (e.g. SSRF → a different primitive). The agent must learn a
   different move.
5. **Relocate the flag / restructure topology** *(replaces the skill)*. Move the
   flag to another tier, add or drop a service. The solution changes wholesale.
6. **Re-skin** *(skill unchanged)*. Rename services and hosts, shuffle layout,
   vary which host holds the loot. Pure robustness.

The classification *is* the rule:

- The **frontier** — the curriculum's difficulty step between rounds — moves only
  via **skill-extending** operators (deepen, add-required-recon). They are small
  and monotone, so the policy climbs a ramp instead of falling off a cliff.
- **Diversity** — the pool's cross-world spread — comes from **skill-replacing**
  and **decoy** operators applied at **seeding** (building fresh root worlds),
  not as small in-place steps. A wholesale path change is fine as a *fresh
  sample*; it is unsafe as a *step* under a live learning policy (§5).
- **Re-skinning** is free robustness, applied any time.

**Two axes, one gate.** Skill-effect (above) decides *where* an operator may act.
A second axis — *how* the change is generated — decides its cost and reach: a
cheap procedural graph-patch (add a vuln, bump a level) or a richer **LLM-built**
increment (a new service, a new tier). The two are orthogonal — you can build a
hop (extends), build a relocated flag (replaces), or build a decoy service
(decoy) — and both run behind the **same admission gate**, so the LLM never
ships the part that must be correct and neither can outrun the verifier ceiling
(§9): the LLM proposes, admission certifies the world is still solvable. One
discipline keeps it RL-safe: a *large* built increment lands as a **fresh seed
world** (diversity), never an in-place rebuild of a world a policy is mid-
training on; the in-place frontier step stays small and skill-extending, patch or
built hop alike.

So the add/remove/swap-vuln operators are the decoy/replacing kind: fine for
seeding diversity, wrong for the frontier. The skill-extending operator —
append-a-hop as a graph-patch — and the check that a "frontier" step actually
extends rather than replaces (§6) both now ship; the LLM-built-service form of
the hop is what remains.

## 4. The two-timescale picture

Putting §2 and §3 together: within a round the pool is fixed and sampled wide
(spread → generalization); between rounds the frontier takes one skill-extending
step and the seed set may gain a fresh, differently-shaped world (diversity).
Re-skinning rides underneath both, for free. Stable within a round, gradual
between rounds, diverse across worlds.

## 5. Why it is stable for GRPO

Three levels of stability. Only the first is free; the other two are constraints
to enforce and measure.

**Within a group — required, and guaranteed by construction.** GRPO's baseline
is the average reward of `G` outputs sampled for the *same* problem, and the
advantage is `(reward − mean) / std` over the group. If the `(world, task)`
varied inside a group, the mean would no longer be a per-problem baseline and the
gradient would be polluted by between-problem reward-scale differences. So the
invariant is firm: **a group is the rollouts sharing one `(snapshot_id,
task_id)`.** The cyber pack already feeds it — the pentest verdict grades three
rungs (reach → extract → match) so a group has reward spread to learn from; a
binary leak/no-leak signal would collapse the group and kill the gradient.

**Within a batch / round — not automatically safe.** Sampling i.i.d. from a
fixed pool is a stationary objective, and the spread is the generalization
mechanism — this is ordinary multi-task RL / domain randomization. But
stationarity at the distribution level does not discharge the batch-level
estimator: per-group whitening zeroes each group's advantage mean, yet the loss
sums over groups whose advantage *magnitudes* are not difficulty-invariant — an
easy world and a hard world do not contribute symmetric gradients in one batch.
Mitigation: cap the difficulty spread *within a single batch* even when the pool
is wide across the round, and measure the residual cross-group weighting rather
than assume it away.

**Between rounds — a bounded heuristic with an explicit anchor.** A slow shift of
the pool's makeup is sensible, but it carries no automatic guarantee: a bounded
*data* shift toward the regions where the policy is most wrong — exactly where a
curriculum pushes — can still produce a large *policy* step (this is why the
analogy to PPO's policy-space trust region does not hold). So the shift rate is a
hyperparameter to tune, and it must be anchored by an explicit KL-to-reference
(§9), not by the implicit KL-bias on-policy RL has on a fixed task.

Bottom line: within-group stability is structural; batch- and round-level
stability are things the harness enforces and the dashboard measures.

## 6. How the pool evolves: signal, mix, frontier

**Move where there is learning signal.** A world the agent always or never solves
gives no gradient; signal lives where it sometimes-but-not-always succeeds. The
shipped `reward_variance_policy` gestures at this — hold a world while its group's
reward spread is alive, nudge when the spread collapses (harden if the mean is
high, soften if low). Two honest limits: with graded rungs the spread rarely
collapses exactly, so the gate mostly *holds* (it stalls rather than
over-hardens); and it is a lagging, whole-pool trigger that only fires once the
gradient is already dead. The published curriculum work uses a *leading,
per-level* score instead — value-loss (PLR) or regret (ACCEL) — to rank which
levels to resample and to edit. GRPO has no critic, so it cannot compute those
directly; the GRPO-native substitute is a per-`(world, task)` priority that scores
the learning signal where it lives. The shipped priority is a *learnability* term
(`1 − |2·solve_rate − 1|`, peaking where the agent sometimes-but-not-always wins),
plus the regret proxy below, plus a staleness term so nothing starves. That
priority — not the lagging collapse gate — is the spine.

**Reward-std is not regret.** A world where every rollout reaches rung 2 and none
reach rung 3 has *low* std (no gradient) yet *high* regret (far from the solvable
optimum) — the frontier world a regret method would prioritize, and the one a
variance gate would wrongly retire. The verifier already knows each world's
solvable ceiling, so a critic-free regret proxy is in reach:
`ceiling-rung − mean-achieved-rung`, and the shipped priority uses it. (Reward-std
is the natural refinement — it separates a live-gradient world from a dead-gradient
one at the same solve-rate — still to add; §10.)

**Keep a mix.** Catastrophic forgetting is the documented failure of
non-stationary RL, and revisiting easy tasks is the fix. So the pool reserves a
floor — about a quarter to a third of each round's groups — for easy-tier members,
enforced when the pool's rows are composed, not hoped to emerge. (The shipped floor
is global by difficulty; whether it should be per-skill, so one skill's easy tail
is not crowded out by another's, is open — §13.) A forgetting metric
(periodically re-run retired members; alarm if their solve-rate drops) is what
makes the floor falsifiable. This requires replacing the shipped discard-the-
parent behaviour with admit-the-child-and-retain-the-parent.

**Move the frontier monotonically.** New frontier worlds extend the required
skill (§3): append a hop, add a required recon step. A *structural* check
confirms the parent's solution is still a sub-path of the child's, so "frontier"
means "genuinely harder," not "more decoys." That check is not the solvability
proof — it reads the chain's shape, not whether the new hop actually leaks; the
independent anchor that the deeper world is still winnable is admission (below)
and the runtime leak oracle, not the gate itself.

The anchor that makes open-ended evolution safe: every world entering the pool —
seeded or evolved — passes the same admission gate, which proves the world is
*graph-reachable-solvable*. That is weaker than *agent-solvable*: admission
rejects impossible worlds, and the soften-when-stuck rule retires
too-hard-for-now ones. The two together keep the pool winnable.

## 7. Pool lifecycle: seed, grow, bound

- **Seed** — build `K` independent root worlds (K admissions over sampled
  manifests, deduped by `(snapshot_id, task_id)`) so the pool starts wide. The
  round-one spread comes from seeding, not from evolution.
- **Grow** — each round, evolve the top-`M` priority members into children, admit
  the children *into* the pool, and down-weight or retire parents under the mix
  floor.
- **Bound** — a maximum pool size `P` with an eviction rule (drop the
  lowest-priority non-floor members), so the pool is a bounded sampled
  population, not an ever-growing set.

`G` and pool width trade off at a fixed rollout budget: `G` must stay large
enough for a low-variance group baseline even as the pool widens, so widening is
not free. State a floor on `G` and a floor on distinct-worlds-per-round; they
compete for the same budget.

## 8. Measuring the bet

The bet is a generalization claim, so it needs a held-out split. Reserve an
**eval pool**: admission-checked but fenced off from training — never sampled
into a group, never evolved — spanning the difficulty range, including a band
beyond the current frontier to test extrapolation. The metric is held-out
solve-rate versus training solve-rate over rounds; the gap between them is the
bet, made measurable. Without this split, the only available signal is training
success-rate — exactly the overfit-to-benchmark number the bet warns against.

## 9. Anchors and safety

Two anchors keep open-ended evolution from drifting:

- **The verifier / admission.** Every member, seeded or evolved, is admission-
  checked, so the pool can never gain an impossible world. Admission proves
  graph-reachability, not agent-solvability (§6) — name it for what it is.
- **A KL-to-reference, required once evolving.** Bounding how far the policy
  drifts from a reference is the load-bearing anchor for the *policy* under a
  shifting pool. The shipped training config disables it (`beta = 0`), which is
  fine for a single static world but not for an open-ended evolving pool: with
  `beta = 0` the verifier is the *only* remaining anchor, so a verifier blind
  spot becomes a single point of failure. `beta > 0` is a prerequisite for the
  evolving regime.

## 10. How it maps to the code

**Already there (plumbing):**
- `make_environment_factory` accepts `Sequence[Snapshot]`, builds a snapshot map,
  and routes `reset()` by `snapshot_id` — the factory can already host many
  worlds.
- `build_grpo_dataset(snapshot)` emits `(snapshot_id, task_id)`-tagged rows for
  one snapshot.
- `reward_variance_policy` — the variance-collapse gate, shipped as whole-list.
- `auto_evolve` — one snapshot to one re-admitted child, with an evolution gate.
- The cyber mutation vocab — re-admitted and deterministic: the add/remove/swap
  operators are decoy/replacing (§3); the skill-extending append-a-hop is below.
- The warm-world pool — booted worlds reused across episodes, now an LRU bounded
  by capacity.
- The pool itself — `WorldPool` seeds wide, composes each round's rows under the
  mix floor, and between rounds re-prioritises members (learnability + regret +
  staleness), evolves the top-`M` into admitted children retaining parents, and
  bounds the size. Pack-agnostic (difficulty injected) and runner-agnostic (the
  caller runs a round), so a scripted solver drives it today and a trainer later.
  It lives in `openrange.pool` — trainer-agnostic, depending only on core — so any
  adapter drives it; it is *not* part of the TRL adapter. The TRL seam is concrete:
  `run_round` builds a `datasets.Dataset` from `round_rows` and runs one short
  GRPO pass per round (TRL reads `train_dataset` fresh each `train()`, so the
  evolving set just gets reassigned between rounds), and the `snapshot_id`/`task_id`
  columns flow into TRL's per-rollout `environment.reset()`. TRL has no
  world-distribution curriculum of its own; its replay buffer and GFPO are
  complementary *within-batch* signal boosters, a layer below this.
- The skill-extending frontier operator — the cyber vocab now includes
  append-a-hop (deepen the credential chain by one hop), and `monotone_chain_gate`
  admits a frontier step only when the parent's solve walk is a prefix of the
  child's (§3, §6). The pool threads it as a per-parent gate.
- The held-out eval pool — `EvalPool` is admitted alongside training but fenced
  from it; `run_pool_curriculum` measures train-vs-held-out solve-rate each round
  (`RoundMetrics`), so the generalization gap (§8) is observed, not assumed.

**To build (where the work and the risk live):**
- A replayable weighted sampler. The shipped round composes deterministic
  top-priority rows under the mix floor; a seed-contract weighted draw (frozen
  *within* a round, updated *between*) is the refinement that makes within-round
  sampling i.i.d.
- The reward-std term in the priority, and a per-skill (not global) mix floor.
- A behavioural difficulty term layered on the static metric, and persisting the
  metric onto the lineage so the dashboard can read it (§11).
- The LLM-built variant of append-a-hop (§3's second axis), behind the same gate.

The cost model: admission is static graph-reachability — cheap and content-
addressable (cache verdicts by content-hash so unchanged members are not
re-checked). The real per-round cost is world boots and per-episode grading, not
admission.

## 11. The dashboard

The shipped view tells a chain story ("the company hardens back"). The pool needs
a different one. First it needs a *measured* difficulty axis — the generation
knobs the lineage carries today are inputs, not an observed property. A graph-
derived metric (chain depth dominating, vuln count secondary) supplies it:
`world_difficulty` scores each pool member today; persisting it onto the lineage
so the dashboard can read it is part of this section's work.

Then: a difficulty band (the pool's members binned by difficulty) sliding toward
harder over rounds while keeping a visible easy tail; the two axes legible (tasks
fanned under one warm world vs distinct worlds entering and leaving); per-member
learnability (alive / mastered / stuck) instead of one global success curve; the
train-versus-held-out gap; and a signal that distinguishes a *verifier-ceiling*
stall (no admissible harder world for this backing) from genuine mastery, so the
two are not confused.

## 12. Prerequisites

1. **The bounded warm cache.** The warm pool ships as an LRU; a pooled round
   needs its capacity set to at least the distinct-worlds-per-round count, or it
   thrashes. Exec-surface worlds (command injection, SSTI) never warm and always
   re-boot — carve them out of throughput claims.
2. **One canonical lineage `_evolve` block.** Every evolution path writes the
   same top-level schema (`parent`, `direction`, `kind`, `relevance`, `family`,
   `note`), so pool/retention tooling reads provenance without special-casing
   which path produced a snapshot.
3. **A measured difficulty metric** — the band and the mix floor are undefined
   without it.

## 13. Open questions

- Is `ceiling-rung − achieved-rung` the right critic-free regret proxy, or does
  it need a finer estimate? This is the central technical question.
- The mix ratio and its granularity — a global fraction, or a per-skill floor?
- Pool size, tasks per world, groups per round, and `G` — and the minimum
  distinct-worlds-per-round below which the generalization argument stops holding.
- Offense versus defense. The design assumes the agent is an attacker and the
  world is read-only (which is why warm reuse is correct). A defense task mutates
  the world, so it cannot ride the warm cache and may be a different problem
  entirely. **Scope v1 to offense;** specify the defense problem separately.
- Live pool versus a multi-parent lineage graph — is the sampled set enough, with
  lineage kept as a provenance chain?

## 14. Risks

- **Loss of plasticity.** A slowly-shifting pool over many rounds is the regime
  where networks can lose the ability to learn at all — and the replay that fixes
  forgetting does not fix this. It needs its own mitigation (periodic resets /
  continual backprop) and its own dashboard signal; do not mistake a plasticity
  stall for a curriculum stall.
- **The cheap axis narrows the round.** A scheduler that maximizes warm-world
  reuse minimizes cross-world spread — undermining generalization. The round must
  spread across worlds regardless of what is cheapest to warm.
- **A soft mix floor reintroduces forgetting.** The floor must be enforced when
  the pool's rows are composed, not left as a preference.
- **The verifier caps the frontier.** The frontier can stop advancing because the
  verifier ran out of instrumented consequences for a backing, not because the
  agent mastered everything. Surface it, or it reads as mastery.

## Appendix — grounding

Automatic curricula / environment design:
- Domain randomization as a wide, fixed, stationary distribution that transfers —
  Tobin et al. 2017 (arXiv:1703.06907); difficulty expansion by a success-rate
  boundary — OpenAI 2019, Automatic Domain Randomization (arXiv:1910.07113).
- Curriculum framed as a sequence of MDPs, fixed within a step — Narvekar et al.
  2020 (JMLR 21(181), arXiv:2003.04960).
- A population of *levels* with a too-easy/too-hard criterion — Wang et al. 2020,
  Enhanced POET (arXiv:2003.08536).
- Regret as the curriculum signal; unsolvable levels score zero regret, so
  solvability admission is curriculum-correct — Dennis et al. 2020, PAIRED
  (arXiv:2012.02096).
- Leading per-level score plus a staleness term — Jiang et al. 2021, Prioritized
  Level Replay (arXiv:2010.03934); gradients only on curated levels, evaluate
  fresh ones to score — Jiang et al. 2021, Robust PLR (arXiv:2110.02439).
- Evolve by small edits to high-regret levels, single agent, level buffer —
  Parker-Holder et al. 2022, ACCEL (arXiv:2203.01302), the closest analogue.

Online RL for language models:
- Group-relative advantage and the fixed-group invariant — Shao et al. 2024,
  DeepSeekMath (arXiv:2402.03300); verifiable rule-based reward + reference —
  Guo et al. 2025, DeepSeek-R1 (arXiv:2501.12948).
- PPO's clip is a policy-space trust region (so it is not a data-shift analogue) —
  Schulman et al. 2017 (arXiv:1707.06347). On-policy RL forgets less than SFT on
  a fixed task — Shenfeld et al. 2025, "RL's Razor" (arXiv:2509.04259).
- KL-to-reference — Ouyang et al. 2022, InstructGPT (arXiv:2203.02155).
- Mix in easy tasks to avoid forgetting — select tasks by learning-curve slope
  and revisit ones that regress — Matiisen et al. 2020 (IEEE TNNLS,
  arXiv:1707.00183); replay for continual learning — Rolnick et al. 2019, CLEAR
  (arXiv:1811.11682).
- Learnability `1 − solve_rate`, signal at intermediate difficulty — Zhao et al.
  2025, Absolute Zero (arXiv:2505.03335); the shipped priority uses a peaked
  variant `1 − |2·solve_rate − 1|` of that monotone form.
- An evolving online curriculum for an LLM web agent — replay + KL-to-reference +
  a complexifying task stream — Qi et al. 2025, WebRL (ICLR, arXiv:2411.02337).
- Loss of plasticity in long non-stationary training — Dohare et al. 2024
  (Nature 632).
