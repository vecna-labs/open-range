# openrange-rllm

Optional [rLLM](https://github.com/rllm-org/rllm) `AgentTrainer` integration for
OpenRange. OpenRange owns the world and the grade; rLLM owns the RL training
loop. This adapter is the thin seam between them:

- **`agent_rollout_to_episode`** — maps one OpenRange agent rollout onto rLLM's
  `Episode` / `Trajectory` / `Step`, one step per harness turn in call order.
- **`make_rollout`** — wraps the harness as an `@rllm.rollout` flow `(task,
  config) -> Episode`; it runs one real episode on a shared `EpisodeService`.
- **`make_evaluator`** — surfaces the verifier's grade as an `@rllm.evaluator`.
- **`GatewaySampler`** — a `Sampler` that calls the policy at `config.base_url`
  through OpenRange's own OpenAI-compatible backend. rLLM's gateway records token
  ids and logprobs, so the rollout leaves those fields empty and rLLM's trace
  enrichment fills them.

`import openrange_rllm` pulls **no** rLLM — every rLLM import is local to the
function that needs it. To run the real trainer, install the `train` extra and
construct the trainer in your own script/notebook:

```python
from rllm.trainer import AgentTrainer
from openrange_rllm import make_rollout, make_evaluator

trainer = AgentTrainer(
    config=config,                       # a Hydra DictConfig (backend: verl|tinker)
    agent_flow=make_rollout(service, resolve, bind_run=bind_run),
    evaluator=make_evaluator(),
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    backend="verl",
)
trainer.train()
```

A complete, runnable example — building a world pool, registering the dataset,
and the validated single-GPU run command — is in
[`examples/rllm_grpo_cyber.py`](../../examples/rllm_grpo_cyber.py).

rLLM is installed from source (`rllm-org/rllm`); the GPU backend (`rllm[verl]`)
needs CUDA. The adapter itself, and its tests, run on CPU against rLLM's
pydantic-only core types.
