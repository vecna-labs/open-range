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
from collections.abc import Sequence
from pathlib import Path

import pytest
from openrange_pack_sdk import Backing, EpisodeReportLike
from openrange_trl import (
    EpisodeEnv,
    build_grpo_dataset,
    env_trajectory,
    make_environment_factory,
    make_reward_func,
)

from examples.tools import FILE_TOOLS, WEB_TOOLS
from openrange.core.admit import AdmissionFailure, admit
from openrange.core.curriculum import Direction, auto_evolve


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
    factory = make_environment_factory(
        pack, [snapshot], tmp_path / "envs", tools=FILE_TOOLS
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
        for row in build_grpo_dataset(snapshot, repeat=2)
        if "pentest" in row["task_id"]
    ]
    dataset = datasets.Dataset.from_list(rows)
    factory = make_environment_factory(
        pack, [snapshot], tmp_path / "envs", tools=WEB_TOOLS
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
        for row in build_grpo_dataset(snapshot, repeat=2)
        if "pentest" in row["task_id"]
    ]
    dataset = datasets.Dataset.from_list(rows)
    factory = make_environment_factory(
        pack, [snapshot], tmp_path / "envs", tools=WEB_TOOLS, backing=Backing.CONTAINER
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


@pytest.mark.skipif(not _docker_available(), reason="docker engine not reachable")
def test_live_grpo_one_step_cyber_container_file_loot(tmp_path: Path) -> None:
    """One real GRPO step over a FILE-LOOT cyber world on the CONTAINER backing.

    command_injection is a ``code_exec`` shape: under PROCESS its loot file sits at a
    randomized path in an in-memory dict with no listing primitive, so a blackbox agent
    can't discover it (untrainable). ``minimum_backing`` therefore routes it to
    CONTAINER, where a real ``sh -c`` restores enumeration. This asserts the
    auto-selected backing and that the GRPO loop runs end-to-end against the real
    container — the trainability the file-loot family only has under CONTAINER."""
    datasets = pytest.importorskip("datasets")
    peft = pytest.importorskip("peft")
    transformers = pytest.importorskip("transformers")
    trl = pytest.importorskip("trl")

    from cyber_webapp import WebappPack, minimum_backing

    pack = WebappPack()
    snapshot = admit(
        pack,
        manifest={
            "world": {"goal": "recover the flag"},
            "pack": {"id": "webapp"},
            "runtime": {"tick": {"mode": "off"}},
            "npc": [],
            "seed": 7,
            "loot_shapes": {"file": 1, "db": 0},
            "vuln_kinds": {"command_injection": 1},
        },
        max_repairs=3,
    )
    assert not isinstance(snapshot, AdmissionFailure)
    # The harness picks the cheapest backing that leaves the world winnable; a
    # file-loot world must come back CONTAINER (PROCESS can't be solved blackbox).
    backing = minimum_backing(snapshot.graph)
    assert backing is Backing.CONTAINER

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
        for row in build_grpo_dataset(snapshot, repeat=2)
        if "pentest" in row["task_id"]
    ]
    dataset = datasets.Dataset.from_list(rows)
    factory = make_environment_factory(
        pack, [snapshot], tmp_path / "envs", tools=WEB_TOOLS, backing=backing
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


def _always_harden(reports: Sequence[EpisodeReportLike]) -> Direction:
    # A tiny laptop model won't drive a reward spread, so force the direction: this
    # proves the *loop* (re-root + carry the model across rounds), not learning.
    del reports
    return "harden"


def test_live_grpo_curriculum_evolves_between_rounds(tmp_path: Path) -> None:
    """Two real GRPO rounds with the world evolving between them.

    Each round trains a real ``GRPOTrainer`` on the current world, grades it, and
    ``auto_evolve`` forks a child snapshot the next round re-roots onto — carrying
    the LoRA model forward. This is the curriculum loop the notebook teaches, here
    against a real trainer; *learning* the exploit is the GPU-scale step.
    """
    datasets = pytest.importorskip("datasets")
    peft = pytest.importorskip("peft")
    transformers = pytest.importorskip("transformers")
    trl = pytest.importorskip("trl")

    from cyber_webapp import WebappPack

    pack = WebappPack()
    snapshot = admit(
        pack,
        manifest={
            "world": {"goal": "recover the hidden flag"},
            "pack": {"id": "webapp"},
            "runtime": {"tick": {"mode": "off"}},
            "npc": [],
            "seed": 0,
            "loot_shapes": {"db": 1, "file": 0},
            "vuln_kinds": {"sql_injection": 1},
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
    config = trl.GRPOConfig(
        output_dir=str(tmp_path / "trainer"),
        per_device_train_batch_size=2,
        num_generations=2,
        steps_per_generation=1,
        max_steps=1,
        beta=0.0,
        max_completion_length=128,
        max_tool_calling_iterations=3,
        use_vllm=False,
        report_to="none",
        save_strategy="no",
        bf16=False,
        fp16=False,
    )

    # The curriculum loop, inline: train a round -> grade -> evolve -> re-root, with
    # the first round wrapping the model in LoRA and later rounds carrying it.
    lineage = [snapshot]
    snap, wrapped = snapshot, False
    for _ in range(2):
        rows = [
            r for r in build_grpo_dataset(snap, repeat=2) if "pentest" in r["task_id"]
        ]
        factory = make_environment_factory(
            pack, [snap], tmp_path / snap.snapshot_id[-12:], tools=WEB_TOOLS
        )
        trainer = trl.GRPOTrainer(
            model=model,
            reward_funcs=[make_reward_func()],
            args=config,
            train_dataset=datasets.Dataset.from_list(rows),
            processing_class=tokenizer,
            environment_factory=factory,
            peft_config=None if wrapped else lora,
        )
        trainer.train()
        model, wrapped = trainer.model, True
        reports = [
            e.report for e in (trainer.environments or ()) if e.report is not None
        ]
        assert reports, "a real GRPO round produced no graded rollout"
        evolved = auto_evolve(snap, *reports, pack=pack, policy=_always_harden)
        assert evolved is not None and evolved.snapshot_id != snap.snapshot_id
        snap = evolved
        lineage.append(snap)

    # Three distinct worlds: the curriculum advanced as a chain across the rounds.
    assert len({s.snapshot_id for s in lineage}) == 3


def test_live_grpo_pool_curriculum(tmp_path: Path) -> None:
    """Two real GRPO rounds driven by the world POOL (not a single chain).

    The pool samples a round's rows; ``make_grpo_run_round`` trains one GRPO pass
    and returns the graded reports keyed by ``(snapshot_id, task_id)``; the pool
    re-prioritises and evolves between rounds — the build-train-build-train loop on
    a real trainer. A laptop model won't drive a spread, so the direction is forced;
    learning the exploit is the GPU-scale step. This proves the pool↔GRPO seam.
    """
    datasets = pytest.importorskip("datasets")
    peft = pytest.importorskip("peft")
    transformers = pytest.importorskip("transformers")
    trl = pytest.importorskip("trl")
    del datasets

    from cyber_webapp import WebappPack
    from cyber_webapp.difficulty import world_difficulty
    from openrange_trl import make_grpo_run_round

    from openrange.pool import WorldPool, run_pool_curriculum

    pack = WebappPack()
    manifest = {
        "world": {"goal": "recover the hidden flag"},
        "pack": {"id": "webapp"},
        "runtime": {"tick": {"mode": "off"}},
        "npc": [],
        "seed": 0,
        "loot_shapes": {"db": 1, "file": 0},
        "vuln_kinds": {"sql_injection": 1},
    }
    pool = WorldPool.seed(
        pack,
        [manifest],
        difficulty_fn=lambda s: float(world_difficulty(s.graph)),
        family="webapp.pentest",
        max_size=4,
    )
    assert len(pool) == 1

    model = transformers.AutoModelForCausalLM.from_pretrained(_MODEL)
    tokenizer = transformers.AutoTokenizer.from_pretrained(_MODEL)
    lora = peft.LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    config = trl.GRPOConfig(
        output_dir=str(tmp_path / "trainer"),
        per_device_train_batch_size=2,
        num_generations=2,
        steps_per_generation=1,
        max_steps=1,
        beta=0.0,
        max_completion_length=128,
        max_tool_calling_iterations=3,
        use_vllm=False,
        report_to="none",
        save_strategy="no",
        bf16=False,
        fp16=False,
    )
    run_round = make_grpo_run_round(
        pack,
        model=model,
        args=config,
        tools=WEB_TOOLS,
        run_root=tmp_path / "envs",
        processing_class=tokenizer,
        peft_config=lora,
    )
    metrics = run_pool_curriculum(
        pool,
        run_round,
        rounds=2,
        pack=pack,
        groups=1,
        num_generations=2,
        policy=_always_harden,
        gate=lambda _snap, mut: mut.family == "webapp.pentest",
    )
    assert len(metrics) == 2
    assert all(0.0 <= m.train_solve_rate <= 1.0 for m in metrics)
    assert len(pool) > 1


def test_live_grpo_held_out_eval(tmp_path: Path) -> None:
    """A real GRPO round trains the pool while a fenced held-out pool is measured by
    a FROZEN round (learning_rate 0, no weight update) — the train-vs-held-out
    generalization gap on a real trainer, the eval pool never trained on.
    """
    datasets = pytest.importorskip("datasets")
    peft = pytest.importorskip("peft")
    transformers = pytest.importorskip("transformers")
    trl = pytest.importorskip("trl")
    del datasets

    from cyber_webapp import WebappPack
    from cyber_webapp.difficulty import world_difficulty
    from openrange_trl import make_grpo_rounds

    from openrange.pool import EvalPool, WorldPool, run_pool_curriculum

    pack = WebappPack()

    def world(seed: int) -> dict[str, object]:
        return {
            "world": {"goal": "recover the hidden flag"},
            "pack": {"id": "webapp"},
            "runtime": {"tick": {"mode": "off"}},
            "npc": [],
            "seed": seed,
            "loot_shapes": {"db": 1, "file": 0},
            "vuln_kinds": {"sql_injection": 1},
        }

    train = WorldPool.seed(
        pack,
        [world(0)],
        difficulty_fn=lambda s: float(world_difficulty(s.graph)),
        family="webapp.pentest",
        max_size=4,
    )
    held_out = EvalPool.seed(
        pack,
        [world(1)],
        difficulty_fn=lambda s: float(world_difficulty(s.graph)),
        family="webapp.pentest",
    )
    assert held_out.keys().isdisjoint(train.keys())

    model = transformers.AutoModelForCausalLM.from_pretrained(_MODEL)
    tokenizer = transformers.AutoTokenizer.from_pretrained(_MODEL)
    lora = peft.LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    config = trl.GRPOConfig(
        output_dir=str(tmp_path / "trainer"),
        per_device_train_batch_size=2,
        num_generations=2,
        steps_per_generation=1,
        max_steps=1,
        beta=0.0,
        max_completion_length=128,
        max_tool_calling_iterations=3,
        use_vllm=False,
        report_to="none",
        save_strategy="no",
        bf16=False,
        fp16=False,
    )
    train_round, eval_round = make_grpo_rounds(
        pack,
        model=model,
        args=config,
        tools=WEB_TOOLS,
        run_root=tmp_path / "envs",
        processing_class=tokenizer,
        peft_config=lora,
    )
    metrics = run_pool_curriculum(
        train,
        train_round,
        rounds=1,
        pack=pack,
        groups=1,
        num_generations=2,
        policy=_always_harden,
        gate=lambda _snap, mut: mut.family == "webapp.pentest",
        eval_pool=held_out,
        eval_round=eval_round,
    )
    assert len(metrics) == 1
    m = metrics[0]
    assert m.held_out_solve_rate is not None and 0.0 <= m.held_out_solve_rate <= 1.0
    assert m.generalization_gap is not None
    assert held_out.keys().isdisjoint(train.keys())
