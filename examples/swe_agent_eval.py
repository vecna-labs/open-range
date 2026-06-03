"""Real-agent eval over OpenRange SWE worlds — a Codex/OpenAI LLM solves blind.

The sibling evals (``swe_eval`` / ``swe_build_eval``) feed the grader an *oracle*
— the answer key read from the world graph — to prove the *scorer* discriminates.
They measure the environment, not capability: nothing unknown is under test.

This one drives a *real* agent. The Codex CLI is rooted in the episode's
``solver_root`` with only the problem statement and the base working tree; the
held-out suite and the gold overlay never leave the graph, so a CLI agent on disk
*cannot* read them — it solves blind. The held-out suite then scores it, and the
same training seam (``openrange.training``) turns the episode into a
``(trajectory, reward)``. The number measures the model, not the grader.

Non-deterministic and online: needs a working ``codex`` CLI (OpenAI / ChatGPT
auth). Grading replays arbitrary model-written code; on macOS that is the bare
subprocess sandbox (the trusted-code path) — fine for a model you control, not for
public adversarial traffic.

Run::

    uv run python -m examples.swe_agent_eval
    uv run python -m examples.swe_agent_eval --instance notes_app
"""

from __future__ import annotations

import argparse
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from openrange_pack_sdk import LLMBackendError, LLMRequest, LLMResult, TaskSpec

from openrange.core.episode import AgentTurn, EpisodeReport
from openrange.llm import CodexBackend
from openrange.runtime import OpenRangeRun, RunConfig
from openrange.training import Trajectory, episode_trajectory, to_jsonl

# calc_sum is a one-line fix (swe.fix); notes_app is a build-from-skeleton
# (swe.build). One of each so the eval spans both task shapes.
_DEFAULT_INSTANCES = ("calc_sum", "notes_app")


def main() -> None:
    args = _parse_args()
    # Fail fast with a clear message if the codex binary isn't installed.
    CodexBackend(command=args.codex_command).preflight()
    run_root = _resolve_root(args)
    run = OpenRangeRun(RunConfig(run_root, dashboard=False))
    harness = CodexHarness(
        command=args.codex_command, model=args.model, timeout=args.agent_timeout
    )

    print("=== OpenRange SWE agent eval (real Codex LLM, solves blind) ===")
    print(f"run root: {run_root}")
    trajectories: list[Trajectory] = []
    for instance in args.instances:
        trajectories.append(_run_instance(run, harness, instance))

    _emit(trajectories, run_root)


def _run_instance(
    run: OpenRangeRun, harness: CodexHarness, instance: str
) -> Trajectory:
    snapshot = run.build(
        {
            "world": {"goal": f"solve the {instance} SWE task"},
            "pack": {"id": "swe"},
            "instance": instance,
        }
    )
    task = snapshot.tasks[0]
    print(f"\n--- {instance}: admitted {snapshot.snapshot_id}")
    print(f"    task {task.id} — {_first_line(task.instruction)}")

    svc = run.episode_service(snapshot)
    try:
        handle = svc.start_episode(snapshot, task.id)
        root = svc.solver_root(handle)
        before = _tree(root)
        turn = _solve(harness, task, root)
        after = _tree(root)
        svc.record_turn(handle, turn)
        report = svc.stop_episode(handle)
    finally:
        svc.close()

    _print_report(report, before, after)
    trajectory = episode_trajectory(report, [turn])
    _print_reward(trajectory)
    return trajectory


def _solve(harness: CodexHarness, task: TaskSpec, root: Path) -> AgentTurn:
    """Hand the whole episode to the Codex CLI; a backend failure is a failed
    episode (graded against whatever the agent left behind), not a crash."""
    try:
        result = harness.run(task.instruction, root)
        return AgentTurn(message=result.text)
    except LLMBackendError as exc:
        print(f"    agent backend failed: {exc}")
        return AgentTurn(message=f"backend error: {exc}")


def _print_report(
    report: EpisodeReport, before: Mapping[str, str], after: Mapping[str, str]
) -> None:
    result = report.episode_result
    status = "RESOLVED" if result.success else "UNRESOLVED"
    added = sorted(k for k in after if k not in before)
    modified = sorted(k for k in after if k in before and after[k] != before[k])
    print(f"    agent edits: +{added or '[]'}  ~{modified or '[]'}")
    print(f"    result: {status} — {result.reason}")
    passed = sum(1 for v in result.subgoals.values() if v)
    print(f"    subgoals: {passed}/{len(result.subgoals)} pass")
    for tid, ok in result.subgoals.items():
        print(f"      {'pass' if ok else 'FAIL'}  {tid}")


def _print_reward(traj: Trajectory) -> None:
    gate = "RESOLVED" if traj.success else "unresolved"
    print(f"    reward: scalar={traj.reward.scalar:.2f}  success={gate}")


def _emit(trajectories: list[Trajectory], out_dir: Path) -> None:
    if not trajectories:
        print("\nno episodes completed")
        return
    path = out_dir / "trajectories.jsonl"
    path.write_text(to_jsonl(trajectories) + "\n", encoding="utf-8")
    scalars = ", ".join(f"{t.reward.scalar:.2f}" for t in trajectories)
    resolved = sum(1 for t in trajectories if t.success)
    print("\ntraining seam: real agent episode -> (trajectory, reward)")
    print(f"  rewards across instances: [{scalars}]")
    print(f"  resolved {resolved}/{len(trajectories)} world(s)")
    print(f"  wrote {len(trajectories)} JSONL trajectory record(s) -> {path}")


@dataclass(frozen=True, slots=True)
class CodexHarness:
    """Spawns the Codex CLI with ``cwd`` set to the episode's agent root.

    ``workspace-write`` lets the agent edit files in its workspace and run the
    tests it writes; no network override — a SWE world has no server, and the
    held-out suite isn't on disk to be found.
    """

    command: str | Path = "codex"
    model: str | None = None
    timeout: float = 300.0

    def run(self, prompt: str, cwd: Path) -> LLMResult:
        return CodexBackend(
            command=self.command,
            model=self.model,
            cwd=cwd,
            sandbox="workspace-write",
            timeout=self.timeout,
        ).complete(LLMRequest(prompt))


def _tree(root: Path) -> dict[str, str]:
    """Snapshot the workspace as ``{relpath: contents}`` (text, lenient decode)
    so we can report what the agent added or changed. Bytecode caches are skipped
    — the agent runs code so they appear, but they aren't edits."""
    return {
        p.relative_to(root).as_posix(): p.read_text(encoding="utf-8", errors="replace")
        for p in sorted(root.rglob("*"))
        if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
    }


def _first_line(text: str) -> str:
    return text.strip().splitlines()[0] if text.strip() else ""


def _resolve_root(args: argparse.Namespace) -> Path:
    if args.run_root is not None:
        return Path(args.run_root)
    args.runs_dir.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix="swe-agent-eval-", dir=args.runs_dir))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--instance",
        dest="instances",
        action="append",
        metavar="NAME",
        help="SWE fixture to solve; repeatable (default: calc_sum, notes_app).",
    )
    parser.add_argument("--codex-command", type=Path, default=Path("codex"))
    parser.add_argument(
        "--model", default=None, help="Codex model; default lets the CLI choose."
    )
    parser.add_argument("--agent-timeout", type=float, default=300.0)
    parser.add_argument("--runs-dir", type=Path, default=Path("or-runs"))
    parser.add_argument("--run-root", type=Path)
    args = parser.parse_args()
    if not args.instances:
        args.instances = list(_DEFAULT_INSTANCES)
    return args


if __name__ == "__main__":  # pragma: no cover
    main()
