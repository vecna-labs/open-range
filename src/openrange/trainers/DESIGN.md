# OpenRange × TRL — design

> The "written shape" for the HuggingFace TRL trainer adapter
> ([#245](https://github.com/vecna-labs/open-range/issues/245)) — the first real
> consumer of the training-integration standard
> ([#243](https://github.com/vecna-labs/open-range/issues/243) /
> [#199](https://github.com/vecna-labs/open-range/issues/199) /
> [#200](https://github.com/vecna-labs/open-range/issues/200)).
> [`training.py`](../training.py) is the pack-agnostic seam (`EpisodeReport →
> (trajectory, reward)`); this doc is the *shape* of the trainer side that sits on
> it, and what "done" means.

## The bet, restated for training

OpenRange's thesis is that agents trained against fresh, runnable,
admission-checked worlds that *evolve* generalize better than agents overfit to a
static, hub-posted gym. The training-integration standard is deliberately **not**
OpenEnv: OpenEnv optimizes for posting a frozen environment to a hub and pulling
it back; OpenRange's worlds are generated, content-addressed, and
*curriculum-evolvable*. A trainer adapter that drops the evolve dimension has
thrown away the whole point.

TRL's `GRPOTrainer` is the sharpest possible first consumer. GRPO is
*group-relative*: it samples a group of rollouts per prompt and learns from the
**spread** of their rewards. That makes it a probe of the two things OpenRange has
to get right —

- a **dense reward**, because a group with zero reward variance has zero gradient;
- a **moving frontier**, because as the policy improves a *fixed* world's variance
  collapses, so the world must harden to keep the signal alive.

The end-state below is the demonstration that OpenRange supplies both, live.

## The end-state (north star)

One training run: a small policy model, trained by `trl.GRPOTrainer`, against a
**live, self-curricularizing** OpenRange SWE world — agentic and multi-turn. Per
rollout:

1. **Realize.** The env realizes a SWE episode and roots the policy in
   `solver_root` with only the problem statement + base tree. The held-out suite
   and gold overlay stay in the graph, never on disk — the policy solves *blind*.
2. **Act through tools.** The policy edits the workspace and runs tests through
   tools — `read_file` / `write_file` / `list_dir` / `apply_patch` + `run_tests` —
   the same filesystem patch-loop a human SWE agent uses.
3. **Grade the state.** On rollout end the held-out suite grades the *final
   workspace*, and `episode_reward` shapes the structured `EpisodeResult` (boolean
   success + per-subgoal vector) into a dense scalar + component vector.
4. **Learn.** GRPO turns the group's reward spread into advantages and steps the
   policy.
5. **Evolve.** Between rounds, `reward_variance_policy` reads the round's reports
   and re-admits a harder/softer world — keeping reward variance alive as the model
   improves. The generic path is `auto_evolve` (in-place mutation, for packs that
   wire the hooks); for SWE the same policy climbs/descends an instance ladder of
   distinct worlds (see "Curriculum in the loop").
6. **Attribute.** Every trajectory is tagged with the `snapshot_id` it ran
   against, so a curriculum's worth of trajectories stays attributable to the exact
   (evolved) world that produced it.

Nothing here is SWE-specific except the choice of pack: the seam is
`episode_reward` + `EpisodeService` + `auto_evolve`, all pack-agnostic.

## The lifecycle mapping (the crux)

TRL's agentic GRPO (the `environment_factory` path, `transformers>=5.2`) gives each
rollout its own environment instance, exposes the env's public methods as tools,
and reads a reward back off the env after the rollout. That maps cleanly onto
`EpisodeService`:

| TRL env hook | OpenRange counterpart |
| --- | --- |
| `reset()` → text appended to the prompt | `EpisodeService.start_episode(snapshot, task_id)` → realizes the world, writes the base tree, returns `task.instruction` |
| public methods → agent tools | the file actuators (below) + the callables in `runtime.surface()` (for SWE, `run_tests`) |
| reward read after the rollout | lazy: the first read of `env.reward` runs `stop_episode` → `EpisodeReport` → `episode_reward(report).scalar` |
| — | `env.turns` records each tool call as an `AgentTurn`, so `episode_trajectory` exports a replayable, `snapshot_id`-tagged `Trajectory` |

Two fits worth calling out, both already true in core (verified from source, not
assumed):

- **The reward read is the done-signal.** TRL reads `env.reward` only after the
  model stops calling tools; that read is exactly when we stop and grade.
  `stop_episode` caches its report (`core/episode.py`), so a double read is safe.
- **A group of N rollouts is isolated.** Each `start_episode` does a fresh
  `realize()` → `reset()` → `mkdtemp` (`openrange_pack_sdk/_runtime.py`), so the N
  concurrent envs in a GRPO group never collide on `solver_root`.

## Gap C: the surface had no actuators

The SWE *contract* is "read / edit files under `solver_root`, run tests" — but the
SWE *surface* is only `{solver_root, run_tests}`. `examples/swe_eval.py`'s Codex
agent gets away with this because the Codex CLI edits `solver_root` itself as a
subprocess (`sandbox="workspace-write"`); the surface never had to expose file
edits. A TRL policy emitting tool calls has no such side channel: given only
`run_tests` it can *observe* but never *change* the graded state, so reward is
constant-zero and GRPO has no variance to learn from. **This is the one real gap
the trainer forces us to close.**

The fix is a reusable, sandboxed **`FileWorkspaceTools`** rooted at `solver_root`:

- `read_file(path)`, `write_file(path, content)`, `list_dir(path=".")`,
  `apply_patch(...)` — the minimal patch-loop surface the contract already names.
- **Path-traversal guarded**: every path is resolved and asserted to stay under
  `solver_root`; a `write_file("../../etc/...")` is rejected. Writing *inside* a
  throwaway temp `solver_root` is safe (grading already runs untrusted code
  sandboxed); escaping it is not.
- **Fail-soft**: a bad call returns an error string, never raises — a malformed
  tool call from a weak model costs reward, not the run.

**Decision — actuators are a *consumer-side* primitive, in `openrange.trainers`,
synthesized from the `solver_root` the surface already exposes. No pack or SDK
change.** This is the cleaner separation: a pack's surface exposes what is
*special* about its world (`run_tests`, `solver_root`); the harness synthesizes the
*generic* file-IO loop every code world shares. `FileWorkspaceTools` and
`OpenRangeEnv` are harness-neutral (no TRL import) and can move up out of
`trainers/` when a second adapter (SkyRL,
[#244](https://github.com/vecna-labs/open-range/issues/244)) arrives to share them.

## Module layout & the torch-free seam

The decisive structural choice: **almost the entire adapter is torch-free.**

- **`src/openrange/trainers/trl.py`** — `build_grpo_dataset`, `make_reward_func`,
  `reward_variance_policy`, `OpenRangeEnv`, `FileWorkspaceTools`. Imports only
  `openrange` + stdlib. `import openrange.trainers.trl` works with no torch
  installed, and every piece is deterministically unit-testable without a model.
- **`tests/test_trl_live.py`** + **the notebook tutorial** — the only places that
  touch `trl` / `torch` / `peft`. Both build a real `GRPOTrainer` over
  `OpenRangeEnv` and run live; both are **gated** (the test skips unless
  `OPENRANGE_LIVE_TRL=1` and pulls the stack via `pytest.importorskip`; the
  notebook is run by hand), so the default suite never imports torch.
- **`pyproject.toml`** — `[project.optional-dependencies] trl = ["torch",
  "transformers>=5.2", "trl", "datasets", "accelerate", "peft"]`, plus mypy
  `ignore_missing_imports` overrides, mirroring the existing `strands` extra.
  `import openrange` never requires the extra.

Why it matters: it keeps the merge-ready bar reachable without a GPU — the
deterministic suite proves the seam is *correct* and that the reward surface spans
0.0 → 0.5 → 1.0 (the spread GRPO consumes); the gated live test proves a real
`GRPOTrainer` *runs*; the notebook makes that reward surface concrete and runs the
loop end-to-end.

## The reward bridge

`make_reward_func(...)` returns a TRL-shaped `reward_func(prompts, completions,
**kwargs) -> list[float]`. In the agentic path it reads the `environments` kwarg
and returns `[episode_reward(env.report).scalar for env in environments]` — it
defers entirely to the pack's structured grade through the standard seam. No reward
logic is reinvented trainer-side; the trainer only *shapes*, exactly as the
gym-not-trainer contract intends. `episode_reward`'s dense default (1.0 solved,
else the fraction of subgoals passed) is what makes a long agentic episode
learnable — and is why `swe.build`'s unit/integration split exists: a weak policy
that gets some pieces right earns partial credit, so the group has variance, so
GRPO has a gradient.

## Curriculum in the loop

`reward_variance_policy` is a `CurriculumPolicy` keyed on the signal GRPO actually
consumes: when a round's group reward variance is ≈ 0 (every rollout failed, or
every rollout solved), there is no gradient — time to evolve. It refines the
existing `direction_from_reports` (pass-rate extremes → harden / soften) to read
`episode_reward` spread directly. The training loop calls
`auto_evolve(snapshot, *reports, pack=pack, policy=reward_variance_policy)` between
rounds; the returned snapshot re-roots `build_grpo_dataset` for the next round, and
its `snapshot_id` tags those trajectories. This is the anti-OpenEnv beat made
concrete: the gym tracks the policy.

**The SWE pack opts out of *in-place* mutation.** `auto_evolve` patches a world
through a pack's `available_mutations` / `default_prior` hooks; the SWE pack
implements neither (the imported SWE-bench source has no parametric difficulty to
step — that is the injected/authored-source curriculum tracked in
[#248](https://github.com/vecna-labs/open-range/issues/248)), so `auto_evolve`
returns `None` for it today. The *standard* — the variance policy, the
re-root-the-dataset-on-the-new-world loop, the `snapshot_id` attribution — is the
part #245 owns, and it is pack-agnostic. So the standard's curriculum beat runs over the
dimension SWE *does* expose: an **instance ladder** of three real, distinct,
content-addressed worlds (`calc_sum` → `shapes_area` → `notes_app`, easy→hard), with
the *same* `reward_variance_policy` deciding each round whether to climb (harden),
descend (soften), or hold, calling `auto_evolve` first (the generic seam) and using
whatever world a pack returns. The mechanism is exercised deterministically —
`reward_variance_policy`'s direction and the `auto_evolve` trigger are asserted in
`tests/test_trl_adapter.py` over real snapshots — and the notebook's closing section
shows where the new world re-roots the next round's dataset. A pack that wires the
mutation hooks gets in-place evolution through the same loop for free.

## Validation plan (two layers, like the SWE pack)

1. **Deterministic correctness — no LLM, no torch.** pytest drives every seam at
   the seam itself: `build_grpo_dataset` row shape + `snapshot_id` / `task_id`
   tagging; `make_reward_func` over *real* episodes (no mocks — drive
   `OpenRangeEnv` with direct tool calls, per `.rules`); the `OpenRangeEnv`
   lifecycle (reset → tool calls mutate `solver_root` → reward grades the tree);
   the actuators including the traversal guard; `reward_variance_policy` direction
   + the `auto_evolve` trigger. This proves the integration is *right*; it does not
   measure a model.
2. **Live capability — real model, real GRPO.** Two gated artifacts cover the live
   path. `tests/test_trl_live.py` runs one GRPO step on a tiny model (Qwen2.5-0.5B)
   with LoRA and asserts the *mechanics* — rollouts grade, reward maps through, a
   `snapshot_id`-tagged trajectory comes back. The **notebook tutorial**
   (`examples/trl_grpo_lora.ipynb`, also Qwen2.5-0.5B + LoRA) first makes the reward
   surface concrete — driving the env tools by hand to show no-op → 0.5, targeted
   fix → 1.0, the naive both-line patch → 0.5, break → 0.0 (the same surface
   `TestRewardSpread` asserts) — then runs one real GRPO step end-to-end. This
   proves the integration *runs live* and that the gradient it would climb is real;
   see "Hardware reality" for why *mastery* needs more than a laptop.

## Hardware reality & honest scope

The reference machine is Python 3.14, arm64 macOS, 24 GB, MPS, **no CUDA / no
vLLM**. The stack resolves (`torch 2.12`, `transformers 5.10`, `trl 1.5`,
`datasets`, `accelerate`) and a sub-1B model generates on MPS, so the loop is
genuinely runnable here — the notebook executes a real GRPO step end-to-end.

What a laptop-scale model *cannot* do is master `calc_sum`. Measured directly (0.5B
and 1.5B, 8 rollouts, temperatures 1.0–1.3): every rollout floors at 0.5. The cause
is specific, and it is the same reason the reward is a good probe — `return a - b`
appears in *both* `add` and `subtract`, so the naive replace-all a weak model
reaches for fixes `add` but breaks `subtract`, netting the same 0.5 an untouched
tree earns; only a *targeted* edit escapes that local optimum. So the honest split
is: the **gradient is real and its optimum reachable** — proven deterministically
in `TestRewardSpread` (0.0 / 0.5 / 1.0 by construction) and made concrete in the
notebook — while a **non-zero live spread needs scale** (a stronger policy, more
samples) beyond a laptop budget. The live artifacts prove *mechanics*; the
deterministic suite proves *signal*. Same honest framing `swe_eval.py` uses for
blind-solve capability.

## The pinned trl 1.5.1 agentic contract (verified against installed source)

The agentic path lives in the *main* `trl.trainer.grpo_trainer` (not experimental).
Pinned from source (trl 1.5.1, transformers 5.10.2, torch 2.12, MPS) so the example
matches reality, not a guess:

- **Factory + tools.** `GRPOTrainer(..., environment_factory=...)` builds one env
  per rollout — `[factory() for _ in range(generation_batch_size)]` — and exposes
  each env's methods as tools via `inspect.getmembers(env, predicate=ismethod)`:
  `reset` is required, underscore methods are skipped, the rest become tools.
  `OpenRangeEnv`'s public surface is exactly `reset` + the five actuators.
- **Reset.** Before each generation batch TRL calls `env.reset(**row)` (the *whole*
  dataset row — `prompt`, `snapshot_id`, `task_id`) and appends the returned string
  to the prompt. `reset(*, snapshot_id=None, task_id=None, **_)` absorbs `prompt`.
- **Reward read-back.** The sync reward func is called
  `reward_func(prompts=…, completions=…, completion_ids=…, environments=self.environments, **cols)`,
  and `zip(prompts, self.environments, inputs, strict=True)` guarantees
  env *i* ↔ prompt *i*. `make_reward_func` returns `[env.reward for env in
  environments]` in order; the first read finalizes (stops + grades) each episode.
- **Tool loop bound.** `max_tool_calling_iterations` (defaults to `sys.maxsize`)
  caps the edit/test turns per rollout — the example sets it small.
- **vLLM unused — confirmed.** `use_vllm` defaults to `False` and the agentic path
  runs through transformers generation on MPS; no GPU/vLLM dependency. `beta=0`
  skips the reference model (lighter on a 24 GB laptop).
- **Sequential episodes — confirmed.** TRL reuses the env objects across rounds, so
  each `reset` starts a *fresh* `EpisodeService.start_episode` (new uuid + own
  `solver_root`); `stop_episode` caches + evicts. One service, many episodes.

## Risks / open questions

- **Tiny-model tool-call validity.** A 135M–0.5B model may emit malformed tool
  calls; `OpenRangeEnv` tools fail soft (return an error string) so a bad call
  costs reward, not the run.
- **Trajectory richness.** TRL owns the completion text; the env sees tool args.
  `env.turns` captures tool calls + results; the model's between-call reasoning
  lives in TRL's completion. Enough for a replayable, graded trajectory; full
  reasoning capture is a later refinement.
- **Curriculum carries weights, not optimizer state.** The example runs one
  `GRPOTrainer` per ladder rung over a *shared* model object, so policy weights
  carry across worlds but Adam state resets at each rung — fine for a mechanics
  demo; a single-trainer dataset-swap is a later refinement.

## Milestones

- **M1** — `trl` extra + mypy overrides; torch-free adapter (dataset, reward
  bridge, variance policy) + deterministic tests.
- **M2** — `OpenRangeEnv` + `FileWorkspaceTools` actuators + lifecycle tests.
- **M3** — gated live test (`tests/test_trl_live.py`) + the notebook tutorial;
  install the extra; run live end-to-end.
- **M4** — evolve-in-the-loop wired (`auto_evolve` + `reward_variance_policy`) and
  exercised deterministically over real snapshots (`test_trl_adapter.py`); the
  instance ladder (`calc_sum → shapes_area → notes_app`) is the pack-agnostic live
  shape it re-roots, shown in the notebook's closing section (SWE opts out of
  in-place mutation — see above).
- **Gate** — ruff / mypy (strict) / full pytest green; docs updated; PR referencing
  #245 (touches #199 / #200).
