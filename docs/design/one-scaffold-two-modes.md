# One agent loop, two modes: the training and evaluation harness

OpenRange is a gym: it owns worlds and the grade. To *use* a gym you need an
agent loop — the thing that turns "a world exists" into "the agent produced a
trajectory" (read the world, call the model, parse an action, run it, observe,
repeat, grade). Today OpenRange has two such loops that drift apart: training
goes through TRL's built-in `environment_factory`, evaluation goes through
`run_episode` plus a backend. The policy sees a different harness in each, so a
transfer number conflates a real change in the agent with an accidental change
in the harness.

This document fixes that with one rule: **the agent loop is a single, shared,
framework-free component, and training and evaluation differ only in which
*sampler* drives it.** It states where that loop lives, what contract it emits,
how the per-trainer adapters stay thin, how the evolving world pool feeds it,
and what has to change in the code before any of it runs at training speed.

It is grounded in the 2026 agentic-RL stack (TRL `rollout_func`, rLLM's
gateway, SkyRL-Agent, verl `AgentLoop`) and in the cyber-RL work that already
walked this path (PrivEsc-LLM, Pentest-R1); the appendix lists sources. Where
the shipped code already provides a seam it is named with a file reference;
where the design needs something that does not exist yet it says so plainly.

## 1. The shape of it

Worked example first. You have a generated company world and a task ("read the
billing DB password"). You want two things from the *same* agent: a training
signal, and an honest eval score on a held-out world.

One loop does both:

```
run_agent(episode, sampler) -> rollout
  obs = briefing(episode)                      # turn 0
  loop until solved / max_iters:
      action = sampler.complete(obs, want_logprobs=?)   # the only mode switch
      result = sandbox.run(action.command)     # the agent's single primitive
      obs = obs + mask(result)                 # observation tokens get masked
  reward = grade(episode)                      # independent verifier, never the model
  return {prompt_ids, completion_ids, logprobs, response_mask, reward}
```

- **Training** injects a sampler backed by the trainer's vLLM server and asks
  for logprobs; the returned trace feeds GRPO.
- **Evaluation** injects a sampler backed by any endpoint (your trained
  weights, GPT, Claude, Codex, opencode) and skips logprobs; the same loop, the
  same briefing, the same parsing, the same grade.

What the policy *perceives* — the briefing, the tool, the result formatting, the
masking, the turn limit — is identical across modes. The decoding differs (a
trainer-served sampler vs. an endpoint, possibly different temperature). That is
the parity that matters: **identical contract, not byte-identical bytes.**

## 2. Layers and ownership

```
core / gym (openrange)     world + grade + episode + sandbox primitive + protocols
packs (cyber_webapp)       worlds + grade + verifier + (optional) tools
harness (shared, new)      the one agent loop + standard tools + masking + trace
shims (openrange-<x>)      logic-free translation of the trace into one trainer
examples/                  one notebook per pack, wiring only
```

Rules that fall out of this and must hold:

- **The gym owns no agent runtime.** `openrange-trl`'s own docstring already
  says it: *"OpenRange owns the world + the grade; it owns nothing of the agent
  runtime; tools are brought by the caller."* So the loop is not the gym's.
- **The harness is generic; only world, grade, and tools change per pack.** So
  the loop is not the pack's either — it is a shared component sibling to the
  gym. Packs *inject* tools; they do not re-implement the loop.
- **A shim is translation, never logic.** No loop, no masking, no grading, no
  reward in a shim. It maps the harness's trace onto one trainer's seam.
- **`examples/` is illustration only.** No reusable machinery there; one
  notebook wires the shared pieces and runs.

## 3. Why we own the loop instead of borrowing one

We verified the three candidate harnesses against our constraints (sources in
the appendix). None can *be* our harness without breaking a rule:

- **TRL has no trainer-free turn loop.** The loop lives inside `GRPOTrainer`
  (`environment_factory`) or inside `rollout_func(prompts, trainer)`, which
  takes the trainer as an argument. You cannot run it for eval without standing
  up a trainer. Borrowing it makes "the harness" a trainer instance — which
  kills trainer-agnostic eval and the parity goal.
- **rLLM and SkyRL-Agent do ship reusable, trainer-free harness objects** (the
  right *shape*: `rllm eval`; SkyRL's `AutoAgentRunner.from_task(infer_engine=
  None)`). But they are framework-owned classes. Adopting one as canonical
  couples the gym to that framework's release churn (rLLM v0.3 is pre-release;
  SkyRL is research code), and "swap trainers" becomes "rewrite the harness."

So we write a thin, framework-free loop once. This is **not** a reversal of the
earlier "don't build a rollout engine — TRL ships it" decision: the rollout
*engine* (batching, weight sync, the GRPO loss/loop) still belongs to the
trainer. We own only the ~100-line per-episode *turn loop*. It does go one step
past "just use `environment_factory`" — that step buys trainer-agnostic eval,
parity, and the ability to ship more than one shim.

## 4. The contract between harness and shim

The harness emits one **rollout trace** per episode:

```
{ prompt_ids, completion_ids, logprobs, response_mask, reward }
```

This is structurally one-to-one with every trainer's bring-your-own-loop seam;
the field names differ and renaming them is the shim's entire job:

- **TRL shim** — a `rollout_func` that calls the harness and returns the trace
  as TRL's dict (`logprobs`, `env_mask`).
- **rLLM shim** — wrap the harness as an `AgentFlow` / `@rllm.rollout` returning
  an `Episode`.
- **SkyRL shim** — wrap behind `BaseTrajectory.generate_trajectory`
  (`input_tokens` / `response_tokens`).

No shim contains a loop, masking, or grading.

## 5. Logprobs, mismatch, and why capture happens in the harness

The defining 2026 correctness bug is the **rollout-vs-training logprob
mismatch**: the sampler (vLLM/SGLang) and the trainer (FSDP/Megatron) disagree
on token logprobs *even at identical weights*, so "on-policy" silently becomes
off-policy. The fix is to capture the rollout logprobs at generation time and
apply Truncated Importance Sampling (TIS) or a decoupled-PPO ratio in the loss.
The serving detail that bites: vLLM V1 returns *raw* logprobs by default; RL
needs `--logprobs-mode processed_logprobs` and an fp32 LM head.

We capture logprobs **in the harness** (the rLLM "gateway" pattern), not by
letting each shim delegate to its trainer, because delegation silently loses the
on-policy signal on some backends (SkyRL's non-`tinker`/non-`verl` paths
hardcode `None`). Capturing in one place keeps every shim correct and also saves
the trainer a recompute forward pass.

The correction itself (TIS / decoupled-PPO) is applied by the trainer; the
harness only makes the signal available. FP16 is an emerging alternative that
sidesteps the mismatch but is far less battle-tested than TIS / decoupled-PPO.

## 6. Train efficiency: upheld by design, but it is work we build

Owning the loop is **throughput-neutral** — it is the converged 2026
architecture, and peers measure 1.55×–2.72× *gains* from the async rollout it
unlocks (gains the native synchronous TRL loop cannot reach). The
efficiency-critical machinery stays in the engine and trainer; owning the loop
moves none of it:

- **continuous batching** — in the shared server-mode engine; the loop only
  submits requests;
- **prefix caching** — automatic in the engine (vLLM V1 APC / SGLang
  RadixAttention), and high-value for our growing multi-turn transcripts —
  *survives only if the loop keeps the transcript append-only with a stable
  session id*;
- **weight sync** — trainer↔sampler over NCCL/CUDA-IPC; the seam only stamps a
  `weight_version` on each trace and never touches weights.

The catch: owning the loop **shifts the burden of concurrency onto OpenRange**.
TRL's `environment_factory` batches every turn for free; `rollout_func` is
called once per batch and does **not** fan out. So throughput is neutral **only
after** we make the loop concurrent. Until then the seam regresses to today's
serial ceiling of one episode at a time.

OpenRange is serial on every axis today. The gaps and their fixes:

| Gap | Where | Fix |
|---|---|---|
| **G1** `AgentSandbox.run` blocks (`subprocess.run(["docker","exec",…])`) | `src/openrange/core/sandbox.py:199` | run via `asyncio.to_thread` / async docker client |
| **G2** infra is threaded, not async (tick threads; sync `record_turn`) | `src/openrange/core/episode.py:707` | async `run_episode`, one coroutine per episode |
| **G3** `warm_capacity=1`, and each rollout slot builds its own `EpisodeService` | `src/openrange/core/episode.py:200`, `packages/openrange-trl/src/openrange_trl/__init__.py` | hoist `EpisodeService` to a shared singleton; `warm_capacity ≥ concurrency` (worst case today re-realizes one snapshot N×; inferred from code, not measured) |
| **G4** concurrency ceiling = 1 episode, serial | TRL factory loop | the thin loop replaces it with N concurrent coroutines on one server-mode engine |
| **G5** eager blocking `Runtime.realize` (secondary) | `src/openrange/core/episode.py:287` | pre-heat the shared warm pool; defer lazy-realize |

**Minimal critical path to saturation: G1 + G2 + G3 + a persistent server-mode
engine target.** That converts docker-exec wait from GPU-idle time into
yield-able time and lets concurrent episodes overlap.

Things that *do* regress throughput, to avoid:

- a **serializing logprob gateway** (rLLM explicitly rejected LiteLLM for a
  transparent httpx tap): the tap must inject `logprobs`+token-ids and read
  straight off the vLLM response, never buffer or reorder across sessions;
- a **blocking/serial generate per turn** (kills continuous batching);
- **re-templating or reordering** the transcript per turn (kills prefix-cache
  hits);
- **pipelining without IS correction** (async staleness stacks on the mismatch
  and training collapses — ship TIS / decoupled-PPO *with* the async work).

## 7. The evolving pool feeds the harness; the verifier travels with the world

Every trainer assumes a static dataset and ships no performance-driven
curriculum hook (TRL only offers an `IterableDataset`; rLLM/SkyRL regenerate a
batch per round). TRL adds a sharper limit: env instances are built once at
trainer init and only `reset()` between episodes, so world evolution cannot be
object swaps.

The routing is the one we already hold: **OpenRange owns the evolving-pool
controller outside the harness and trainer**, and the pool surfaces to the
harness as a stream of `(world, task)` rows (for TRL, a self-updating
`IterableDataset` reading a scoreboard we write from the reward path). The
**consequence verifier travels with each row**, so a newly-evolved world carries
its own self-verifying grader and there is no global checker to keep in sync.
The harness only ever sees one realized world and its grader, which sidesteps
TRL's fixed-env limit. This is additive — it is the dynamic-env standard
OpenRange already set out to define, not a fork of anyone's loop. (See
[the evolving-pool design](evolving-pool-curriculum.md).)

## 8. Shims to ship — and what is deliberately not one

- **`openrange-trl`** (have it) — the first *stable* shim; mature, matches the
  "TRL ships the engine" decision. Re-pointed onto the `rollout_func` seam.
- **`openrange-rllm`** — the design *north-star* for the trace shape (its
  gateway already does on-policy logprobs and trainer-free eval); ship after
  TRL, pin the pre-release version.
- **`openrange-verl` / `-skyrl`** — later, only if a single-node loop is
  outgrown.
- **Not OpenEnv.** OpenEnv is a static, hub-posted, reproducible-artifact env
  spec; OpenRange has no static artifact — worlds are generated and the pool
  evolves during training. Conforming would either freeze us to a static gym
  (losing the thesis) or break OpenEnv's reproducibility assumptions for zero
  gain. The standard we pursue is one for *dynamic* envs, not conformance to a
  static one.

Eval runners and baselines sit *outside* the loop: a CAGE-style fan-out runner
(mirror the pattern, do not take the dependency) drives held-out evals, and
Codex / opencode are baseline agents plugged into the **eval sampler slot** —
they drive the same loop, they do not replace it.

## 9. How it maps to the code

What exists and what each piece becomes:

- `packages/openrange-trl/src/openrange_trl/__init__.py:427` `make_grpo_rounds`
  → today wires TRL's `environment_factory`
  (`make_environment_factory:371`, `environment_factory=factory:484`). Re-point
  onto a new `make_rollout_func` (translation only) that calls the shared
  harness, returns the trace, and passes DAPO/TIS knobs through `GRPOConfig`.
  Reward continues to defer to the pack grade (`make_reward_func` →
  `episode_reward`), never reinvented in the shim.
- `src/openrange/core/sandbox.py:199` `AgentSandbox.run`
  → relocate to **core** (it is the env's shell-exec-against-world primitive,
  domain-agnostic and host-needing); make it non-blocking (G1).
- `examples/_briefing.py:14` `agent_briefing` → it is functional glue, so it
  moves out of `examples/` into the shared harness.
- `src/openrange/core/episode.py:183` `EpisodeService` (warm pool at `:200`,
  `:221`; tick thread at `:707`) → gains an async path and a shared,
  appropriately-sized warm pool (G2, G3, G5).
- `src/openrange/training.py:98` `episode_reward`, `packs/cyber_webapp/
  cyber_webapp/consequence.py:95` `detect_leak`, `packs/cyber_webapp/
  cyber_webapp/verify.py:87` `accepts` → unchanged in spirit; the pack's grade,
  which the reward path consumes; shaped reward extends it pack-side.
- `src/openrange/llm.py` backends (`CodexBackend:31`,
  `OpenAICompatibleBackend:238`, `LiteLLMBackend:392`) → wrapped as the **eval
  sampler**; the **train sampler** is the trainer's vLLM client behind a
  logprob-capturing tap.
- Confirmed gap: there is no `logprob` / importance-sampling / mismatch handling
  anywhere in `src/openrange` or `packages/openrange-trl` today.

New, shared, framework-free (sibling to the gym, not in a pack, not in a shim):
the `Sampler` protocol, the `run_agent` loop, observation masking, the
rollout-trace type, the standard `run(command)` + `finish` tools, and a
trainer-free runner so eval and train drive the identical loop.

## 10. Sequence of work

1. **Foundation (the unlock).** Relocate `AgentSandbox` → core and make it
   non-blocking (G1); add an async episode path and a shared, sized warm pool
   (G2, G3, G5); define the `Sampler` protocol and the `run_agent` loop +
   masking + trace in the shared harness; move `agent_briefing` in. This is
   where latent efficiency becomes real throughput.
2. **Re-point `openrange-trl`** onto `make_rollout_func` (the `rollout_func`
   seam): emit the trace, capture processed logprobs via a transparent tap, pass
   DAPO knobs (clip-higher, dynamic sampling, β=0) and a TIS flag through
   `GRPOConfig`. Reward still defers to the pack grade.
3. **The one example** — fold into `examples/trl_grpo_cyber.ipynb`: drive the
   *same* `run_agent` with a train sampler and an eval sampler, run one GRPO
   round through the shim, then a trainer-free held-out eval; Codex/opencode as
   the eval baseline. Delete the duplicated loop glue.
4. **Next shim** — `openrange-rllm`, reusing the same trace.

Each step holds the invariants: packs→core only, `examples/` illustration only,
shims logic-free, and the strict repo `.rules` (no comments, integration tests,
full branch coverage).

## 11. Decisions and open questions

Settled:

- One shared, framework-free harness; train/eval differ only by sampler.
- Identical *contract* parity (not byte-identical decoding).
- In-harness (gateway-style) logprob capture; TIS / decoupled-PPO applied by the
  trainer; off by default until validated.
- TRL `rollout_func` as the first stable shim; rLLM as the trace-shape
  north-star; not OpenEnv.
- Efficiency is upheld only with the async foundation; that work is in scope.
- Colocated, synchronous sampler first; disaggregated async later if idle time
  hurts.

Open:

- The exact home of the shared harness: its own small package vs. a clearly
  separated runtime module beside the gym (leaning: a separate module so the gym
  stays pure world+grade).
- Whether `openrange-trl` should grow the `make_rollout_func` path beside
  `make_environment_factory` or replace it (leaning: beside, then deprecate).
- Reward shaping depth for the first cut (terminal leak only vs. +recon-diversity
  / anti-hallucination), and whether to add an SFT cold-start (Pentest-R1 shows
  it matters) now or later.

## Appendix — grounding

Verified against primary sources (mid-2026):

- **Rollout/training mismatch + TIS, processed logprobs + fp32 head** — verl
  PR #2953; ServiceNow, "Correctness Before Corrections" (2026-05).
- **GRPO successors** — DAPO (clip-higher, dynamic sampling, token-level loss,
  β=0); GSPO (2025-07); CISPO.
- **Multi-turn credit assignment** — GiGPO (anchor-state, critic-free).
- **TRL seams** — `environment_factory` vs `rollout_func`; `AsyncGRPOTrainer`;
  TRL OpenEnv docs (2026-05).
- **rLLM** — "any harness, any backend; same code for eval and training";
  transparent httpx gateway (not LiteLLM) for on-policy logprob capture.
- **SkyRL-Agent** — `val_mode` parity, `infer_engine=None` trainer-free eval,
  automatic `response_mask`; minimal-tools-stall-convergence finding
  (arXiv:2511.16108).
- **Thin harness** — mini-swe-agent (~100 lines, bash-only).
- **Async throughput** — SkyRL-Agent 1.55× (2025-11-20), ROLL Flash 2.72×
  (2025-10-14), AReaL 2.57×, APRIL +24–44%.
- **Prefix caching** — vLLM V1 automatic prefix caching; SGLang RadixAttention.
- **Cyber RL precedent** — PrivEsc-LLM (arXiv:2603.17673): SFT→GRPO/RLVR on
  procedural, leakage-controlled worlds with an automatic verifier, 95.8% vs
  Claude Opus 4.6's 97.5% at ~100× lower cost; Pentest-R1: observation-token
  masking as a correctness requirement; AgentCyberRange/CAGE (arXiv:2606.14295):
  eval-only; ExploitBench forbids RL on the benchmark (train/eval-pool
  separation).

Confidence notes: "throughput-neutral" is conditional on building the async
path; the rLLM gateway is the same architectural shape but a real integration is
unproven (not drop-in); the G3 N× realization cost is inferred from code
structure, not measured; absolute weight-sync cost depends on colocated vs
disaggregated deployment, not yet pinned for OpenRange.
