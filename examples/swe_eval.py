"""Oracle eval over an OpenRange SWE world — the real episode harness, no LLM.

Three episodes against one admitted world show the loop end to end:

- **oracle** applies the hidden gold fix and resolves the held-out suite;
- **iterative** drives the surfaced ``run_tests`` tool across two turns (write a
  reproduction → red → apply the fix → green), proving the multi-turn agent
  surface works and that the tool runs the agent's *own* tests, never the hidden
  grader;
- **no-op** leaves the bug in place and fails FAIL_TO_PASS.

Every episode is then shaped through the training seam (``openrange.training``)
into a ``(trajectory, reward)``, and the batch is written as JSONL — the
``EpisodeResult`` → (trajectory, reward) standard a trainer consumes, dense by
default (a solved episode scores 1.0, an unsolved one the fraction of its
held-out tests that pass).

Deterministic and offline — the gold fix is read from the world graph (the
oracle's answer key), never guessed, and the held-out tests never touch the
agent's workspace.

Run::

    uv run python -m examples.swe_eval
"""

from __future__ import annotations

import argparse
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path

from openrange_pack_sdk import Snapshot, TaskSpec

from openrange.core import PACKS, auto_evolve
from openrange.core.episode import AgentTurn, EpisodeReport, EpisodeService
from openrange.runtime import OpenRangeRun, RunConfig
from openrange.training import Trajectory, episode_trajectory, to_jsonl

MANIFEST: dict[str, object] = {
    "world": {"goal": "fix the bug so the held-out suite goes green"},
    "pack": {"id": "swe"},
    "instance": "calc_sum",
}

Solver = Callable[[Path], None]

# Per-fixture reproduction a scripted agent writes, then runs via the surfaced
# ``run_tests`` tool: red on the buggy base tree, green once the fix lands. Used
# only to demonstrate the multi-turn loop; the hidden grader is unaffected.
_REPROS: dict[str, str] = {
    "calc_sum": (
        "from calc.core import add\n\n\ndef test_repro():\n    assert add(2, 3) == 5\n"
    ),
    "shapes_area": (
        "from shapes.geometry import rectangle_area\n\n\n"
        "def test_repro():\n    assert rectangle_area(2, 3) == 6\n"
    ),
}


def main() -> None:
    args = _parse_args()
    pack = PACKS.resolve("swe")
    run_root_dir = _resolve_root(args)
    run = OpenRangeRun(RunConfig(run_root_dir, dashboard=False))

    manifest = {**MANIFEST, "instance": args.instance}
    snapshot = run.build(manifest)
    task = snapshot.tasks[0]
    print("=== OpenRange SWE eval (oracle + tool loop + training seam, no LLM) ===")
    print(f"build: admitted {snapshot.snapshot_id}")
    print(f"       task {task.id} — {_first_line(task.instruction)}")

    gold = _gold_files(snapshot)

    trajectories: list[Trajectory] = []
    svc = run.episode_service(snapshot)
    try:
        oracle, oracle_traj = _run_episode(
            svc, snapshot, task, _apply(gold), "oracle: apply gold fix"
        )
        trajectories.append(oracle_traj)
        repro = _REPROS.get(args.instance)
        if repro is not None:
            _, iter_traj = _run_iterative_episode(svc, snapshot, task, gold, repro)
            trajectories.append(iter_traj)
        _, noop_traj = _run_episode(
            svc, snapshot, task, _noop, "no-op: leave the bug in place"
        )
        trajectories.append(noop_traj)
    finally:
        svc.close()

    evolved = auto_evolve(snapshot, oracle, pack=pack)
    summary = (
        f"a harder world ({evolved.snapshot_id})"
        if evolved is not None
        else "None — SWE curriculum is milestone C in the design doc, not wired yet"
    )
    print(f"\ncurriculum: auto_evolve -> {summary}")
    _emit_trajectories(trajectories, run_root_dir)
    verdict = "RESOLVED" if oracle.passed else "FAILED"
    print(f"\neval complete: oracle {verdict} the world")


def _run_episode(
    svc: EpisodeService,
    snapshot: Snapshot,
    task: TaskSpec,
    solver: Solver,
    label: str,
) -> tuple[EpisodeReport, Trajectory]:
    handle = svc.start_episode(snapshot, task.id)
    root = svc.solver_root(handle)
    realized = ", ".join(
        sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())
    )
    solver(root)
    turn = AgentTurn(message=label)
    svc.record_turn(handle, turn)
    report = svc.stop_episode(handle)

    result = report.episode_result
    status = "RESOLVED" if result.success else "UNRESOLVED"
    print(f"\nepisode — {label}")
    print(f"  realized tree: {realized}  (held-out tests + gold stay in the graph)")
    print(f"  result: {status} — {result.reason}")
    for test_id, ok in result.subgoals.items():
        print(f"    {test_id}: {'pass' if ok else 'FAIL'}")
    trajectory = episode_trajectory(report, [turn])
    _print_reward(trajectory)
    return report, trajectory


def _run_iterative_episode(
    svc: EpisodeService,
    snapshot: Snapshot,
    task: TaskSpec,
    gold: Mapping[str, str],
    repro: str,
) -> tuple[EpisodeReport, Trajectory]:
    """Multi-turn loop over the surfaced ``run_tests`` tool: write a repro, run
    it (red), apply the fix, run it (green), then end the episode. The grader
    still runs the held-out suite at stop — the tool never exposes it."""
    handle = svc.start_episode(snapshot, task.id)
    root = svc.solver_root(handle)
    run_tests = svc.surface(handle)["run_tests"]
    print("\nepisode — iterative: run_tests tool loop (repro red -> fix -> green)")
    turns: list[AgentTurn] = []

    (root / "repro_test.py").write_text(repro, encoding="utf-8")
    red = run_tests(["repro_test.py"])
    turn_red = AgentTurn(
        message="run_tests(repro) before fix",
        tool_calls=({"tool": "run_tests", "args": {"node_ids": ["repro_test.py"]}},),
        tool_results=({"ok": red["ok"], "returncode": red["returncode"]},),
    )
    turns.append(turn_red)
    svc.record_turn(handle, turn_red)
    print(
        f"  turn 1 run_tests(repro): ok={red['ok']} rc={red['returncode']} "
        f"(sandbox isolation={red['isolation']})"
    )

    for rel, contents in gold.items():
        dest = root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(contents, encoding="utf-8")
    green = run_tests(["repro_test.py"])
    turn_green = AgentTurn(
        message="run_tests(repro) after fix",
        tool_calls=({"tool": "run_tests", "args": {"node_ids": ["repro_test.py"]}},),
        tool_results=({"ok": green["ok"], "returncode": green["returncode"]},),
    )
    turns.append(turn_green)
    svc.record_turn(handle, turn_green)
    print(f"  turn 2 run_tests(repro): ok={green['ok']} rc={green['returncode']}")

    _finish(root)
    report = svc.stop_episode(handle)
    result = report.episode_result
    status = "RESOLVED" if result.success else "UNRESOLVED"
    print(f"  result: {status} — {result.reason}  (held-out grader, not the tool)")
    trajectory = episode_trajectory(report, turns)
    _print_reward(trajectory)
    return report, trajectory


def _print_reward(traj: Trajectory) -> None:
    comps = traj.reward.components
    passed = sum(1 for v in comps.values() if v >= 1.0)
    print(
        f"  reward: scalar={traj.reward.scalar:.2f}  "
        f"subgoals={passed}/{len(comps)}  steps={len(traj.steps)}"
    )


def _emit_trajectories(trajectories: list[Trajectory], out_dir: Path) -> None:
    path = out_dir / "trajectories.jsonl"
    path.write_text(to_jsonl(trajectories) + "\n", encoding="utf-8")
    scalars = ", ".join(f"{t.reward.scalar:.2f}" for t in trajectories)
    print("\ntraining seam: episode -> (trajectory, reward)")
    print(f"  rewards across episodes: [{scalars}]")
    print(f"  wrote {len(trajectories)} JSONL trajectory record(s) -> {path}")


def _apply(gold: Mapping[str, str]) -> Solver:
    def solver(root: Path) -> None:
        for rel, contents in gold.items():
            dest = root / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(contents, encoding="utf-8")
        _finish(root)

    return solver


def _noop(root: Path) -> None:
    _finish(root)


def _finish(root: Path) -> None:
    (root / "result.json").write_text('{"done": true}\n', encoding="utf-8")


def _gold_files(snapshot: Snapshot) -> dict[str, str]:
    solution = snapshot.graph.by_kind("solution")[0]
    raw = solution.attrs.get("gold_files", {})
    if not isinstance(raw, Mapping):
        return {}
    return {str(k): str(v) for k, v in raw.items()}


def _first_line(text: str) -> str:
    return text.strip().splitlines()[0] if text.strip() else ""


def _resolve_root(args: argparse.Namespace) -> Path:
    if args.run_root is not None:
        return Path(args.run_root)
    args.runs_dir.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix="swe-eval-", dir=args.runs_dir))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--instance",
        default="calc_sum",
        help="fixture instance to build (e.g. calc_sum, shapes_area)",
    )
    parser.add_argument("--runs-dir", type=Path, default=Path("or-runs"))
    parser.add_argument("--run-root", type=Path)
    return parser.parse_args()


if __name__ == "__main__":  # pragma: no cover
    main()
