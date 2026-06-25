"""Train a cyber agent on an OpenRange world pool with rLLM's ``AgentTrainer``.

This is the rLLM half of "one scaffold, two modes": the *same* agent loop that
``examples/codex_eval.py`` evaluates with is trained here, swapping only the
sampler. ``openrange_rllm`` maps each OpenRange episode onto rLLM's
``Episode``/``Step`` and exposes the policy as an ``@rllm.rollout`` flow; rLLM's
gateway captures token ids and logprobs, GRPO does the rest. The reward is the
pack's own dense subgoal ladder (no reward logic here).

A pool of command-injection "company" worlds becomes an rLLM dataset (one row per
pentest task, carrying its ``snapshot_id``/``task_id``); ``snapshot_resolver``
maps each sampled rLLM task back to its world. The agent reaches the live webapp
over HTTP from a host shell (PROCESS backing) and composes ``curl`` itself.

Run on one CUDA GPU through rLLM's verl backend. Validated end to end on an
A100-40GB inside the maintainers' ``verlai/verl:vllm011.latest`` image (torch 2.8
/ vLLM 0.11 / flash-attn)::

    python -m examples.rllm_grpo_cyber \
        rllm/backend=verl algorithm.adv_estimator=grpo \
        +model.name=Qwen/Qwen2.5-7B-Instruct \
        actor_rollout_ref.model.path=Qwen/Qwen2.5-7B-Instruct \
        actor_rollout_ref.model.lora_rank=32 \
        actor_rollout_ref.model.lora_alpha=32 \
        actor_rollout_ref.actor.use_dynamic_bsz=True \
        actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384 \
        actor_rollout_ref.actor.use_kl_loss=False \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.mode=async \
        actor_rollout_ref.rollout.enforce_eager=True \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
        actor_rollout_ref.rollout.n=4 \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
        trainer.n_gpus_per_node=1 data.train_batch_size=2 \
        rllm.trainer.total_batches=1

Gotchas (both cost real debugging):

- LoRA uses the **flat** keys ``lora_rank`` / ``lora_alpha``. The nested
  ``lora.rank`` is silently ignored, which means full fine-tuning — a 7B then
  OOMs a 40GB card, whereas with LoRA on it fits comfortably.
- OpenRange currently requires Python **3.14** (PEP 758 ``except`` syntax) but the
  verl GPU stack ships only Python **3.12** wheels, so they cannot share a
  process. Running this needs OpenRange made 3.12-compatible (a 3-line change) or
  3.14 wheels for torch/vLLM/flash-attn (not available yet).

Building the pool + registering the dataset is CPU-only; ``trainer.train()`` is
the CUDA boundary (Ray + vLLM + FSDP under the verl backend).
"""

from __future__ import annotations

import subprocess
from collections.abc import Mapping
from typing import Any

from cyber_webapp import WebappPack
from cyber_webapp.difficulty import world_difficulty
from openrange_pack_sdk import Snapshot
from openrange_rllm import (
    build_rllm_dataset_rows,
    make_evaluator,
    make_rollout,
    snapshot_resolver,
)

from openrange import EpisodeService, EvalPool, WorldPool
from openrange.core.sandbox import CommandResult

_DATASET = "openrange_cyber"
_FAMILY = "webapp.pentest"
_RUN_ROOT = "or-runs/cyber-rllm"


def _company(seed: int) -> dict[str, Any]:
    return {
        "world": {"goal": "recover the hidden flag from the internal estate"},
        "pack": {"id": "webapp"},
        "runtime": {"tick": {"mode": "off"}},
        "npc": [],
        "seed": seed,
        "topology": "chain",
    }


def _difficulty(snapshot: Snapshot) -> float:
    return float(world_difficulty(snapshot.graph))


class _HostRun:
    def run(self, command: str, *, timeout: float = 120.0) -> CommandResult:
        done = subprocess.run(
            ["bash", "-lc", command],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return CommandResult(done.returncode, done.stdout + done.stderr)

    def close(self) -> None:
        return None


def _host_bind(_surface: Mapping[str, Any]) -> _HostRun:
    return _HostRun()


def main() -> None:
    import hydra
    from omegaconf import DictConfig
    from rllm.data.dataset import DatasetRegistry
    from rllm.trainer import AgentTrainer

    @hydra.main(  # type: ignore[untyped-decorator]
        config_path="pkg://rllm.trainer.config",
        config_name="unified",
        version_base=None,
    )
    def _train(config: DictConfig) -> None:
        pack = WebappPack()
        train_pool = WorldPool.seed(
            pack,
            [_company(seed) for seed in range(4)],
            difficulty_fn=_difficulty,
            family=_FAMILY,
            max_size=8,
        )
        val_pool = EvalPool.seed(
            pack,
            [_company(seed) for seed in (7, 8)],
            difficulty_fn=_difficulty,
            family=_FAMILY,
        )
        DatasetRegistry.register_dataset(
            _DATASET,
            build_rllm_dataset_rows(train_pool.snapshots(), family=_FAMILY),
            "train",
        )
        DatasetRegistry.register_dataset(
            _DATASET,
            build_rllm_dataset_rows(val_pool.snapshots(), family=_FAMILY),
            "test",
        )
        resolve = snapshot_resolver([*train_pool.snapshots(), *val_pool.snapshots()])
        service = EpisodeService(pack, _RUN_ROOT)
        trainer = AgentTrainer(
            backend=config.rllm.get("backend", "verl"),
            agent_flow=make_rollout(service, resolve, bind_run=_host_bind),
            evaluator=make_evaluator(),
            config=config,
            train_dataset=DatasetRegistry.load_dataset(_DATASET, "train"),
            val_dataset=DatasetRegistry.load_dataset(_DATASET, "test"),
        )
        trainer.train()

    _train()


if __name__ == "__main__":  # pragma: no cover
    main()
