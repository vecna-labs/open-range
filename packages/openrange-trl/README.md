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
constructing a real `GRPOTrainer` needs the `train` extra. End-to-end tutorials:
`examples/trl_grpo_cyber.ipynb` (cyber over HTTP — the priority surface) and
`examples/trl_grpo_lora.ipynb` (SWE — the simplest file-editing intro).

## Surface

- `EpisodeEnv` — one rollout's env over an `EpisodeService` episode. The policy's
  tools are **brought by the caller** (the user's harness), bound to the live world
  surface, and reflected to TRL as the action surface — OpenRange owns the bridge,
  not the tools. `openrange_trl.tools` ships a reference set (`WEB_TOOLS`,
  `FILE_TOOLS`) you can use, extend, or replace.
- `build_grpo_dataset`, `make_reward_func`, `make_environment_factory(..., tools=)`,
  `env_trajectory` — the TRL-shaped dataset, reward bridge, per-rollout factory (you
  pass the `tools` the policy gets), and trajectory export.
- `reward_variance_policy` — a curriculum policy keyed on the reward spread GRPO consumes.

A tool is a plain callable taking the live `surface` first, then the model's kwargs
(see `openrange_trl/tools.py`). All reward/trajectory logic defers to OpenRange's
pack-agnostic `episode_reward` / `episode_trajectory`; none is reinvented here.
