# openrange-trl

The **optional** [HuggingFace TRL](https://github.com/huggingface/trl) GRPO
integration for OpenRange.

OpenRange is an **environment** platform: it owns world construction, admission,
the episode lifecycle, grading, and the framework-neutral
`EpisodeReport → (reward, trajectory)` seam (`openrange.training`). It does **not**
own the training loop, and ships no trainer code. This package is the opt-in
adapter that wires an OpenRange episode into `trl.GRPOTrainer`'s agentic
`environment_factory` path — so the env never depends on a trainer, and swapping
trainers means swapping this package, not touching OpenRange.

```bash
uv pip install "openrange-trl[train]"   # base is torch-free; [train] adds the trl stack
```

The adapter is **torch-free** (`import openrange_trl` works with no `torch`); only
constructing a real `GRPOTrainer` needs the `train` extra. End-to-end tutorial:
`examples/trl_grpo_cyber.ipynb` (cyber over HTTP — train an agent to breach a
multi-service company). The same adapter trains file-editing (SWE) worlds too,
exercised by `tests/test_trl_adapter.py` + the gated `tests/test_trl_live.py`.

## Surface

- `EpisodeEnv` — one rollout's env over an `EpisodeService` episode. The policy's
  tools are **brought by the caller** (the user's harness), bound to the live world
  surface, and reflected to TRL as the action surface — OpenRange owns the bridge
  and ships **no** tools. (A sandboxed agent runs its own; the in-process policy
  has no shell, so `examples/tools.py` carries minimal reference shims to copy or
  replace.)
- `build_grpo_dataset`, `make_reward_func`, `make_environment_factory(..., tools=)`,
  `env_trajectory` — the TRL-shaped dataset, reward bridge, per-rollout factory (you
  pass the `tools` the policy gets), and trajectory export.
- `reward_variance_policy` — a curriculum policy keyed on the reward spread GRPO consumes.

A tool is a plain callable taking the live `surface` first, then the model's kwargs
(see `examples/tools.py`). All reward/trajectory logic defers to OpenRange's
pack-agnostic `episode_reward` / `episode_trajectory`; none is reinvented here.

## Sandbox cleanup

With `sandbox=True` each episode runs the agent's tools in a throwaway container on a
private per-episode network, both torn down when the episode ends. A best-effort
`atexit` sweep also reclaims this process's own sandboxes if a run unwinds past teardown
(an unhandled exception or Ctrl-C) — but `atexit` can't fire on `SIGKILL`. So every
container and network is labelled `openrange.sandbox=1`; if a hard-killed run leaks one,
reclaim it with:

```bash
docker ps   -aq --filter label=openrange.sandbox=1 | xargs -r docker rm -f
docker network ls -q --filter label=openrange.sandbox=1 | xargs -r docker network rm
```
