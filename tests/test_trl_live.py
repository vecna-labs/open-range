"""Gated, opt-in live test: the adapter drives a real ``trl.GRPOTrainer``.

This is the live half of the training seam whose deterministic, torch-free tests
live in ``test_trl_adapter.py`` (SWE) and ``test_trl_cyber.py`` (cyber). It runs
one real GRPO step (tiny model, LoRA) against a live OpenRange world — once over a
SWE workspace (file tools), once over a cyber webapp (HTTP tools) — and asserts the
*mechanics* end to end: rollouts reach grading, the structured reward maps through,
and a ``snapshot_id``-tagged trajectory comes back.

It is skipped unless ``OPENRANGE_LIVE_TRL=1`` and the ``trl`` extra is installed::

    OPENRANGE_LIVE_TRL=1 uv run --extra trl pytest tests/test_trl_live.py

so the default CI suite stays GPU-free. The bigger-model demonstration of *learning*
(non-zero reward spread, solved rollouts) lives in the notebook tutorial.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest
from openrange_pack_sdk import Backing
from openrange_trl import (
    WEB_TOOL_GUIDE,
    EpisodeEnv,
    build_grpo_dataset,
    env_trajectory,
    make_environment_factory,
    make_reward_func,
    make_web_environment_factory,
)

from openrange.core.admit import AdmissionFailure, admit


def _docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        probe = subprocess.run(
            ["docker", "info"], capture_output=True, timeout=10, check=False
        )
    except Exception:  # noqa: BLE001 - any failure means "no"
        return False
    return probe.returncode == 0


pytestmark = pytest.mark.skipif(
    os.environ.get("OPENRANGE_LIVE_TRL") != "1",
    reason="live TRL GRPO test; set OPENRANGE_LIVE_TRL=1 and install openrange[trl]",
)

_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def test_live_grpo_one_step(tmp_path: Path) -> None:
    """One real GRPO step over calc_sum: rollouts grade, reward + trajectory flow."""
    # Heavy, optional deps are imported here (not at module top) so collection
    # stays clean in the GPU-free dev env; importorskip turns a missing extra
    # into a skip rather than a collection error.
    datasets = pytest.importorskip("datasets")
    peft = pytest.importorskip("peft")
    transformers = pytest.importorskip("transformers")
    trl = pytest.importorskip("trl")

    from swe import SwePack

    pack = SwePack()
    snapshot = admit(pack, manifest={"instance": "calc_sum"}, max_repairs=0)
    assert not isinstance(snapshot, AdmissionFailure)

    model = transformers.AutoModelForCausalLM.from_pretrained(_MODEL)
    tokenizer = transformers.AutoTokenizer.from_pretrained(_MODEL)
    lora = peft.LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )

    num_generations = 2
    dataset = datasets.Dataset.from_list(build_grpo_dataset(snapshot, repeat=2))
    factory = make_environment_factory(pack, [snapshot], tmp_path / "envs")
    config = trl.GRPOConfig(
        output_dir=str(tmp_path / "trainer"),
        per_device_train_batch_size=num_generations,
        num_generations=num_generations,
        steps_per_generation=1,
        gradient_accumulation_steps=1,
        max_steps=1,
        beta=0.0,
        temperature=1.0,
        max_completion_length=128,
        max_tool_calling_iterations=2,
        use_vllm=False,
        log_completions=False,
        report_to="none",
        save_strategy="no",
        bf16=False,
        fp16=False,
    )
    trainer = trl.GRPOTrainer(
        model=model,
        reward_funcs=[make_reward_func()],
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
        environment_factory=factory,
        peft_config=lora,
    )

    envs: list[EpisodeEnv] = list(trainer.environments or ())
    try:
        trainer.train()

        graded = [env for env in envs if env.report is not None]
        assert graded, "no rollout reached grading"
        for env in graded:
            trajectory = env_trajectory(env)
            assert trajectory.snapshot_id == snapshot.snapshot_id
            assert 0.0 <= trajectory.reward.scalar <= 1.0
    finally:
        for env in envs:
            env.service.close()


def test_live_grpo_one_step_cyber(tmp_path: Path) -> None:
    """One real GRPO step over a live cyber webapp: the policy issues HTTP tool
    calls against the booted server, rollouts grade, reward + trajectory flow."""
    datasets = pytest.importorskip("datasets")
    peft = pytest.importorskip("peft")
    transformers = pytest.importorskip("transformers")
    trl = pytest.importorskip("trl")

    from cyber_webapp import WebappPack

    pack = WebappPack()
    snapshot = admit(
        pack,
        manifest={
            "world": {"goal": "recover the flag"},
            "pack": {"id": "webapp"},
            "runtime": {"tick": {"mode": "off"}},
            "npc": [],
            "seed": 0,
        },
    )
    assert not isinstance(snapshot, AdmissionFailure)

    model = transformers.AutoModelForCausalLM.from_pretrained(_MODEL)
    tokenizer = transformers.AutoTokenizer.from_pretrained(_MODEL)
    lora = peft.LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )

    num_generations = 2
    # Only the pentest task suits the HTTP action surface (build is code-gen).
    rows = [
        row
        for row in build_grpo_dataset(snapshot, repeat=2, tool_guide=WEB_TOOL_GUIDE)
        if "pentest" in row["task_id"]
    ]
    dataset = datasets.Dataset.from_list(rows)
    factory = make_web_environment_factory(pack, [snapshot], tmp_path / "envs")
    config = trl.GRPOConfig(
        output_dir=str(tmp_path / "trainer"),
        per_device_train_batch_size=num_generations,
        num_generations=num_generations,
        steps_per_generation=1,
        gradient_accumulation_steps=1,
        max_steps=1,
        beta=0.0,
        temperature=1.0,
        max_completion_length=128,
        max_tool_calling_iterations=3,
        use_vllm=False,
        log_completions=False,
        report_to="none",
        save_strategy="no",
        bf16=False,
        fp16=False,
    )
    trainer = trl.GRPOTrainer(
        model=model,
        reward_funcs=[make_reward_func()],
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
        environment_factory=factory,
        peft_config=lora,
    )

    envs: list[EpisodeEnv] = list(trainer.environments or ())
    try:
        trainer.train()

        graded = [env for env in envs if env.report is not None]
        assert graded, "no rollout reached grading"
        for env in graded:
            trajectory = env_trajectory(env)
            assert trajectory.snapshot_id == snapshot.snapshot_id
            assert 0.0 <= trajectory.reward.scalar <= 1.0
    finally:
        for env in envs:
            env.service.close()


@pytest.mark.skipif(not _docker_available(), reason="docker engine not reachable")
def test_live_grpo_one_step_cyber_container(tmp_path: Path) -> None:
    """One real GRPO step over a *networked* cyber world on the CONTAINER backing:
    each rollout boots per-service containers, the policy acts over HTTP across the
    docker network, rollouts grade, reward + trajectory flow. Proves the networked
    multi-service runtime is a real training environment, not only a runnable one."""
    datasets = pytest.importorskip("datasets")
    peft = pytest.importorskip("peft")
    transformers = pytest.importorskip("transformers")
    trl = pytest.importorskip("trl")

    from cyber_webapp import WebappPack, _is_networked

    pack = WebappPack()
    snapshot = admit(
        pack,
        manifest={
            "world": {"goal": "recover the flag"},
            "pack": {"id": "webapp"},
            "runtime": {"tick": {"mode": "off"}},
            "npc": [],
            "seed": 3,
            "vuln_kinds": {"ssrf": 1},
            "difficulty": "easy",
        },
    )
    assert not isinstance(snapshot, AdmissionFailure)
    # The world must be networked so the CONTAINER backing routes to the multi-service
    # runtime — otherwise this would only exercise the single-container path.
    assert _is_networked(snapshot.graph)

    model = transformers.AutoModelForCausalLM.from_pretrained(_MODEL)
    tokenizer = transformers.AutoTokenizer.from_pretrained(_MODEL)
    lora = peft.LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )

    num_generations = 2
    rows = [
        row
        for row in build_grpo_dataset(snapshot, repeat=2, tool_guide=WEB_TOOL_GUIDE)
        if "pentest" in row["task_id"]
    ]
    dataset = datasets.Dataset.from_list(rows)
    factory = make_web_environment_factory(
        pack, [snapshot], tmp_path / "envs", backing=Backing.CONTAINER
    )
    config = trl.GRPOConfig(
        output_dir=str(tmp_path / "trainer"),
        per_device_train_batch_size=num_generations,
        num_generations=num_generations,
        steps_per_generation=1,
        gradient_accumulation_steps=1,
        max_steps=1,
        beta=0.0,
        temperature=1.0,
        max_completion_length=128,
        max_tool_calling_iterations=3,
        use_vllm=False,
        log_completions=False,
        report_to="none",
        save_strategy="no",
        bf16=False,
        fp16=False,
    )
    trainer = trl.GRPOTrainer(
        model=model,
        reward_funcs=[make_reward_func()],
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
        environment_factory=factory,
        peft_config=lora,
    )

    envs: list[EpisodeEnv] = list(trainer.environments or ())
    try:
        trainer.train()

        graded = [env for env in envs if env.report is not None]
        assert graded, "no rollout reached grading on the CONTAINER backing"
        for env in graded:
            assert env.service.backing is Backing.CONTAINER
            trajectory = env_trajectory(env)
            assert trajectory.snapshot_id == snapshot.snapshot_id
            assert 0.0 <= trajectory.reward.scalar <= 1.0
    finally:
        for env in envs:
            env.service.close()
