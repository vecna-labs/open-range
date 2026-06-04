"""The SWE pack's end-to-end example: a real agent solves a SWE world *blind*,
and the world's own held-out suite scores it.

``run_episode`` realizes the world, roots the agent in the episode's
``solver_root`` with only the problem statement and the base working tree, then
grades whatever it left behind against the held-out suite. That suite and the gold
overlay never leave the graph, so an agent on disk *cannot* read them — it solves
blind. The same training seam (``openrange.training``) turns the graded episode
into a ``(trajectory, reward)``: the number measures the model, not the grader.

The harness is the swap-in seam. ``CodexHarness`` drives the Codex CLI as the
reference; point the solver at your own endpoint to eval another model. Two
instances span both task shapes: ``calc_sum`` (``swe.fix``, a one-line repair)
and ``notes_app`` (``swe.build``, build-from-skeleton).

Non-deterministic and online: the reference harness needs a working ``codex`` CLI
(OpenAI / ChatGPT auth). Grading replays arbitrary model-written code; on macOS
that is the bare subprocess sandbox (the trusted-code path) — fine for a model you
control, not for public adversarial traffic.

Run::

    uv run python -m examples.swe_eval
    uv run python -m examples.swe_eval --instance notes_app
"""

from __future__ import annotations

import argparse
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from openrange_pack_sdk import LLMBackendError, LLMRequest, LLMResult, Snapshot

from openrange.core.episode import AgentTurn, EpisodeReport
from openrange.llm import CodexBackend
from openrange.runtime import EpisodeContext, OpenRangeRun, RunConfig, Solver
from openrange.training import EpisodeRun, Trajectory, to_jsonl

_DEFAULT_INSTANCES = ("calc_sum", "notes_app")


def main() -> None:
    args = _parse_args()
    CodexBackend(command=args.codex_command).preflight()
    run_root = _resolve_root(args)
    run = OpenRangeRun(RunConfig(run_root, dashboard=False))
    harness = CodexHarness(
        command=args.codex_command, model=args.model, timeout=args.agent_timeout
    )

    print("=== OpenRange SWE agent eval (real Codex LLM, solves blind) ===")
    print(f"run root: {run_root}")
    runs: list[EpisodeRun] = []
    for instance in args.instances:
        runs.append(_run_instance(run, harness, instance))

    _emit(runs, run_root)


def _run_instance(
    run: OpenRangeRun, harness: CodexHarness, instance: str
) -> EpisodeRun:
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

    ep = run.run_episode(snapshot, _codex_solver(harness))

    _print_report(ep.report, snapshot)
    _print_reward(ep.trajectory)
    return ep


def _codex_solver(harness: CodexHarness) -> Solver:
    """A backend failure is a failed episode — graded against whatever the agent
    left behind — so it is returned as a turn, not raised."""

    def solve(ctx: EpisodeContext) -> AgentTurn:
        try:
            result = harness.run(ctx.task.instruction, ctx.root)
            return AgentTurn(message=result.text)
        except LLMBackendError as exc:
            print(f"    agent backend failed: {exc}")
            return AgentTurn(message=f"backend error: {exc}")

    return solve


def _print_report(report: EpisodeReport, snapshot: Snapshot) -> None:
    result = report.episode_result
    status = "RESOLVED" if result.success else "UNRESOLVED"
    before = _base_files(snapshot)
    after = _workspace_files(report)
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


def _emit(runs: list[EpisodeRun], out_dir: Path) -> None:
    if not runs:
        print("\nno episodes completed")
        return
    trajectories = [ep.trajectory for ep in runs]
    path = out_dir / "trajectories.jsonl"
    path.write_text(to_jsonl(trajectories) + "\n", encoding="utf-8")
    scalars = ", ".join(f"{t.reward.scalar:.2f}" for t in trajectories)
    resolved = sum(1 for ep in runs if ep.success)
    print("\ntraining seam: real agent episode -> (trajectory, reward)")
    print(f"  rewards across instances: [{scalars}]")
    print(f"  resolved {resolved}/{len(runs)} world(s)")
    print(f"  wrote {len(trajectories)} JSONL trajectory record(s) -> {path}")


@dataclass(frozen=True, slots=True)
class CodexHarness:
    """``workspace-write`` lets the agent edit and run tests in its workspace; no
    network — a SWE world has no server, and the held-out suite isn't on disk."""

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


def _base_files(snapshot: Snapshot) -> dict[str, str]:
    repos = snapshot.graph.by_kind("repo")
    if not repos:
        return {}
    raw = repos[0].attrs.get("base_files", {})
    if not isinstance(raw, Mapping):
        return {}
    return {str(k): str(v) for k, v in raw.items()}


def _workspace_files(report: EpisodeReport) -> dict[str, str]:
    raw = report.final_state.get("workspace_files", {})
    if not isinstance(raw, Mapping):
        return {}
    return {str(k): str(v) for k, v in raw.items()}


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
