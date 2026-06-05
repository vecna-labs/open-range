"""The TRL pack's end-to-end example: a small policy model trained by
``trl.GRPOTrainer`` against a **live, self-curricularizing** OpenRange SWE world —
agentic and multi-turn, on MPS/CPU, no GPU required.

This is the live half of [#245](https://github.com/vecna-labs/open-range/issues/245)
(the torch-free seam + its deterministic tests are ``openrange.trainers.trl`` and
``tests/test_trl_adapter.py``). Per rollout the loop is the north star from
``src/openrange/trainers/DESIGN.md``:

1. **Realize + solve blind.** Each rollout's ``OpenRangeEnv.reset`` starts a fresh
   ``EpisodeService`` episode and roots the policy in ``solver_root`` with only the
   problem statement + base tree. The held-out suite and gold overlay stay in the
   graph, never on disk.
2. **Act through tools.** TRL exposes the env's five actuators
   (``read_file`` / ``write_file`` / ``list_dir`` / ``apply_patch`` / ``run_tests``)
   as tools; the model edits the workspace over several turns.
3. **Grade the state.** On rollout end the held-out suite grades the *final*
   workspace and ``episode_reward`` shapes it into a dense scalar — ``1.0`` solved,
   else the fraction of subgoals passed.
4. **Learn.** GRPO turns the group's reward *spread* into advantages and steps the
   shared policy.
5. **Evolve.** Between rounds ``reward_variance_policy`` reads the round's reports
   and ``auto_evolve`` is called as the generic in-place-mutation seam. The SWE pack
   opts out of in-place mutation (``auto_evolve`` returns ``None``), so the *same*
   policy climbs/descends an **instance ladder** of three real, distinct,
   content-addressed worlds (``calc_sum`` → ``shapes_area`` → ``notes_app``). Every
   trajectory is tagged with the ``snapshot_id`` it ran against.

This is the anti-OpenEnv beat made concrete: the gym tracks the policy, live.

**Honest scope.** A sub-1B model on a laptop will not *master* SWE in this compute
budget; the run proves end-to-end *mechanics + a real reward signal* (real rollouts,
real tool calls, real grading, real GRPO steps, real curriculum decisions), not SOTA
capability. The curriculum carries policy *weights* across rungs (a shared model
object); Adam state resets per rung — a mechanics demo, not a tuned recipe.

Needs the ``trl`` extra (``uv sync --extra trl``) and downloads the model on first
run. Grading replays model-written code in the bare subprocess sandbox (the
trusted-code path) — fine for a model you control, not public adversarial traffic.

Run::

    uv run --extra trl python -m examples.trl_grpo_eval --smoke
    uv run --extra trl python -m examples.trl_grpo_eval
    uv run --extra trl python -m examples.trl_grpo_eval --start-instance notes_app
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from openrange_pack_sdk import Pack, Snapshot
from swe import SwePack
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from openrange.core.admit import AdmissionFailure, admit
from openrange.core.curriculum import Direction, auto_evolve
from openrange.core.episode import EpisodeReport
from openrange.trainers.trl import (
    OpenRangeEnv,
    build_grpo_dataset,
    env_trajectory,
    make_environment_factory,
    make_reward_func,
    reward_variance_policy,
)
from openrange.training import Trajectory, episode_reward, to_jsonl

# Easy → hard. ``swe.fix`` (one-line repairs) then ``swe.build`` (build-from-
# skeleton, dense partial credit). The variance policy climbs/descends this.
LADDER: tuple[str, ...] = ("calc_sum", "shapes_area", "notes_app")


def main() -> None:
    args = _parse_args()
    pack = SwePack()
    run_root = _resolve_root(args)
    device = _device(args)

    print("=== OpenRange × TRL GRPO — live, self-curricularizing SWE ===")
    print(f"model: {args.model}")
    print(f"device: {device}   run root: {run_root}")
    print(f"ladder: {' -> '.join(LADDER)}   rounds: {args.rounds}")

    # Load once and carry weights across rungs — the curriculum trains *one*
    # policy against a moving frontier, not a fresh model per world.
    model = AutoModelForCausalLM.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    idx = LADDER.index(args.start_instance)
    snapshot = _admit(pack, LADDER[idx])
    trajectories: list[Trajectory] = []

    for round_no in range(1, args.rounds + 1):
        print(
            f"\n--- round {round_no}/{args.rounds}: {LADDER[idx]} "
            f"[{snapshot.snapshot_id}] ---"
        )
        reports, round_trajs = _train_one_round(
            pack, snapshot, model, tokenizer, run_root, round_no, args
        )
        trajectories.extend(round_trajs)
        _print_round_rewards(reports)
        direction = reward_variance_policy(reports)
        evolved = auto_evolve(
            snapshot, *reports, pack=pack, policy=reward_variance_policy
        )
        snapshot, idx = _next_world(pack, snapshot, idx, direction, evolved)

    _emit(trajectories, run_root)


def _train_one_round(
    pack: Pack,
    snapshot: Snapshot,
    model: Any,
    tokenizer: Any,
    run_root: Path,
    round_no: int,
    args: argparse.Namespace,
) -> tuple[list[EpisodeReport], list[Trajectory]]:
    """Run one GRPO round over ``snapshot`` and return its reports + trajectories.

    A fresh dataset + ``environment_factory`` re-root the trainer on the round's
    world; ``model`` is shared so weights carry across rounds. The envs that
    ``GRPOTrainer`` built (``trainer.environments``) hold the last batch's graded
    episodes — that is the round's signal for the curriculum.
    """
    round_root = run_root / f"round-{round_no:02d}"
    # 1 distinct prompt is consumed per optimizer step; repeat so the dataset
    # comfortably covers ``max_steps`` without relying on epoch wrap-around.
    repeat = max(2, args.max_steps)
    dataset = Dataset.from_list(build_grpo_dataset(snapshot, repeat=repeat))
    factory = make_environment_factory(pack, [snapshot], round_root / "envs")
    config = _grpo_config(round_root / "trainer", args)
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[make_reward_func()],
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
        environment_factory=factory,
    )
    envs: list[OpenRangeEnv] = list(trainer.environments or ())
    try:
        trainer.train()
        reports = [env.report for env in envs if env.report is not None]
        trajectories: list[Trajectory] = []
        for env in envs:
            try:
                trajectories.append(env_trajectory(env))
            except RuntimeError:
                continue  # env never reset (e.g. dataset exhausted) → nothing
        return reports, trajectories
    finally:
        for env in envs:
            env.service.close()  # release each round's grading subprocesses


def _grpo_config(output_dir: Path, args: argparse.Namespace) -> Any:
    """A small, MPS-friendly GRPO config.

    ``per_device_train_batch_size == num_generations`` (with ``steps_per_generation
    = 1``, single process) makes one GRPO group per optimizer step and exactly
    ``num_generations`` rollout envs. ``use_vllm=False`` runs generation through
    transformers; ``beta=0`` skips the reference model (lighter on a laptop).
    """
    return GRPOConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.num_generations,
        num_generations=args.num_generations,
        steps_per_generation=1,
        gradient_accumulation_steps=1,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        beta=0.0,
        temperature=1.0,
        max_completion_length=args.max_completion_length,
        max_tool_calling_iterations=args.max_tool_calls,
        use_vllm=False,
        log_completions=False,
        logging_steps=1,
        report_to="none",
        save_strategy="no",
        bf16=False,
        fp16=False,
        use_cpu=args.cpu,
    )


def _next_world(
    pack: Pack,
    snapshot: Snapshot,
    idx: int,
    direction: Direction | None,
    evolved: Snapshot | None,
) -> tuple[Snapshot, int]:
    """Decide the next round's world from the curriculum signal.

    ``auto_evolve`` is the generic, pack-agnostic seam — a pack that wires the
    mutation hooks gets in-place evolution here for free. SWE opts out (returns
    ``None``), so the *same* ``reward_variance_policy`` direction climbs (harden) /
    descends (soften) / holds the **instance ladder** instead.
    """
    if evolved is not None:
        print("    auto_evolve: in-place mutation -> new world (generic seam)")
        return evolved, idx

    new_idx = idx
    if direction == "harden":
        new_idx = min(idx + 1, len(LADDER) - 1)
    elif direction == "soften":
        new_idx = max(idx - 1, 0)
    move = {"harden": "climb", "soften": "descend"}.get(direction or "", "hold")
    print(
        f"    curriculum: variance policy -> {direction or 'hold'} ({move}); "
        f"auto_evolve(SWE) -> None (opts out); ladder idx {idx} -> {new_idx}"
    )
    if new_idx == idx:
        return snapshot, idx
    return _admit(pack, LADDER[new_idx]), new_idx


def _print_round_rewards(reports: list[EpisodeReport]) -> None:
    if not reports:
        print("    rewards: (no graded rollouts this round)")
        return
    scalars = sorted(episode_reward(r).scalar for r in reports)
    mean = sum(scalars) / len(scalars)
    spread = scalars[-1] - scalars[0]
    resolved = sum(1 for r in reports if r.passed)
    pretty = ", ".join(f"{s:.2f}" for s in scalars)
    print(f"    rewards: [{pretty}]")
    print(
        f"    mean={mean:.2f}  spread={spread:.2f}  resolved={resolved}/{len(reports)}"
    )


def _emit(trajectories: list[Trajectory], out_dir: Path) -> None:
    if not trajectories:
        print("\nno trajectories produced")
        return
    path = out_dir / "trajectories.jsonl"
    path.write_text(to_jsonl(trajectories) + "\n", encoding="utf-8")
    worlds = sorted({t.snapshot_id for t in trajectories})
    resolved = sum(1 for t in trajectories if t.success)
    print("\ntraining seam: live GRPO rollouts -> (trajectory, reward)")
    print(f"  {len(trajectories)} trajectories across {len(worlds)} world(s)")
    print(f"  resolved {resolved}/{len(trajectories)} rollout(s)")
    print(f"  wrote JSONL (snapshot_id-tagged) -> {path}")


def _admit(pack: Pack, instance: str) -> Snapshot:
    result = admit(pack, manifest={"instance": instance}, max_repairs=0)
    if isinstance(result, AdmissionFailure):
        raise SystemExit(f"admission failed for {instance!r}: {result}")
    return result


def _device(args: argparse.Namespace) -> str:
    if args.cpu:
        return "cpu"
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _resolve_root(args: argparse.Namespace) -> Path:
    if args.run_root is not None:
        Path(args.run_root).mkdir(parents=True, exist_ok=True)
        return Path(args.run_root)
    args.runs_dir.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix="trl-grpo-", dir=args.runs_dir))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument(
        "--start-instance",
        choices=LADDER,
        default=LADDER[0],
        help="ladder rung to start on (default: calc_sum).",
    )
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=2)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--max-completion-length", type=int, default=256)
    parser.add_argument("--max-tool-calls", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--runs-dir", type=Path, default=Path("or-runs"))
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--cpu", action="store_true", help="force CPU (debug).")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="shrink everything for a fast mechanics check (1 round, tiny budget).",
    )
    args = parser.parse_args()
    if args.smoke:
        args.rounds = 1
        args.max_steps = 1
        args.num_generations = 2
        args.max_completion_length = 128
        args.max_tool_calls = 2
    return args


if __name__ == "__main__":  # pragma: no cover
    main()
