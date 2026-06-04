"""Build-and-grade eval over an OpenRange SWE *build* world — no LLM.

``swe.build`` is the long-horizon sibling of ``swe.fix``: the held-out suite
splits into unit tests that **shape** (dense partial credit) and integration
tests that **gate** success. Three oracle episodes against one admitted world
make the split visible:

- **gold** applies the full reference overlay; the pieces compose, every
  integration test passes, the episode succeeds (reward 1.0);
- **units-only** applies a half-built overlay whose pieces each pass their unit
  test but do not compose — ``add`` returns the rendered note instead of an id
  and never persists. Integration fails, so the episode does NOT succeed, yet it
  still earns the unit fraction as partial credit (reward ~0.6);
- **skeleton** leaves the ``NotImplementedError`` stubs in place and earns
  nothing (reward 0.0).

Each episode is shaped through the training seam (``openrange.training``) into a
``(trajectory, reward)`` and the batch is written as JSONL. The reward
discriminates composes-vs-doesn't even though two of the three episodes never
"succeed" — the dense signal a long-horizon trainer needs (an all-or-nothing
gate is zero almost everywhere, so group-relative advantage sees no variance to
learn from; see the SWE design doc and issue #243).

Deterministic and offline — the gold overlay is read from the world graph (the
oracle's answer key), never guessed, and the held-out tests never touch the
agent's workspace.

Run::

    uv run python -m examples.swe_build_eval
"""

from __future__ import annotations

import argparse
import tempfile
from collections.abc import Mapping
from pathlib import Path

from openrange_pack_sdk import Snapshot

from openrange.core.episode import AgentTurn
from openrange.runtime import EpisodeContext, OpenRangeRun, RunConfig, Solver
from openrange.training import EpisodeRun, Trajectory, to_jsonl

MANIFEST: dict[str, object] = {
    "world": {"goal": "build the notes service so the pieces compose"},
    "pack": {"id": "swe"},
    "instance": "notes_app",
}

# A half-built notes service: each piece passes its unit test (render works, the
# store works), but the pieces don't compose — add() returns the rendered note
# instead of a string id and never persists, and get() echoes its argument
# instead of reading the store. So every unit test passes and every integration
# test fails: the episode that makes the units-shape / integration-gate split
# visible (high partial credit, zero success).
_UNITS_ONLY_SERVICE = (
    "class NoteService:\n"
    "    def __init__(self, store):\n"
    "        self.store = store\n"
    "        self._next = 0\n\n"
    "    def render(self, title, body):\n"
    '        return f"{title}: {body}"\n\n'
    "    def add(self, title, body):\n"
    "        return self.render(title, body)\n\n"
    "    def get(self, note_id):\n"
    "        return note_id\n"
)


def main() -> None:
    args = _parse_args()
    run_root_dir = _resolve_root(args)
    run = OpenRangeRun(RunConfig(run_root_dir, dashboard=False))

    snapshot = run.build(MANIFEST)
    task = snapshot.tasks[0]
    print("=== OpenRange SWE build eval (units shape, integration gates, no LLM) ===")
    print(f"build: admitted {snapshot.snapshot_id}")
    print(f"       task {task.id} — {_first_line(task.instruction)}")

    gold = _gold_files(snapshot)
    tiers = _suite_tiers(snapshot)
    units_only = {**gold, "notes/service.py": _UNITS_ONLY_SERVICE}
    realized = ", ".join(sorted(_base_files(snapshot)))
    print(
        f"       suite: {len(tiers[0])} unit test(s) shape, "
        f"{len(tiers[1])} integration test(s) gate"
    )

    # Each episode is one ``run.run_episode`` call; the units-shape /
    # integration-gate split shows up in the graded result, not in a hand-rolled
    # harness loop.
    trajectories: list[Trajectory] = []
    gold_run = _run_and_print(
        run,
        snapshot,
        _overlay_solver(gold, "gold: full reference overlay"),
        realized,
        tiers,
    )
    trajectories.append(gold_run.trajectory)
    units_run = _run_and_print(
        run,
        snapshot,
        _overlay_solver(units_only, "units-only: pieces pass alone, don't compose"),
        realized,
        tiers,
    )
    trajectories.append(units_run.trajectory)
    skeleton_run = _run_and_print(
        run,
        snapshot,
        _noop_solver("skeleton: leave the stubs in place"),
        realized,
        tiers,
    )
    trajectories.append(skeleton_run.trajectory)

    _emit_trajectories(trajectories, run_root_dir)
    print("\neval complete: reward discriminated compose (1.00) vs partial vs none")


def _run_and_print(
    run: OpenRangeRun,
    snapshot: Snapshot,
    solver: Solver,
    realized: str,
    tiers: tuple[list[str], list[str]],
) -> EpisodeRun:
    ep = run.run_episode(snapshot, solver)
    result = ep.report.episode_result
    status = "RESOLVED" if result.success else "UNRESOLVED"
    print(f"\nepisode — {ep.report.agent_summary}")
    print(f"  realized tree: {realized}  (held-out tests stay in the graph)")
    print(f"  result: {status} — {result.reason}")
    _print_breakdown(result.subgoals, tiers)
    _print_reward(ep.trajectory)
    return ep


def _overlay_solver(overlay: Mapping[str, str], label: str) -> Solver:
    """Scripted oracle: write an overlay over the skeleton, then end the
    episode. The returned turn carries ``label`` as its message."""

    def solve(ctx: EpisodeContext) -> AgentTurn:
        _write_tree(ctx.root, overlay)
        _finish(ctx.root)
        return AgentTurn(message=label)

    return solve


def _noop_solver(label: str) -> Solver:
    """Scripted no-op: leave the skeleton stubs in place, then end the episode."""

    def solve(ctx: EpisodeContext) -> AgentTurn:
        _finish(ctx.root)
        return AgentTurn(message=label)

    return solve


def _print_breakdown(
    subgoals: Mapping[str, bool], tiers: tuple[list[str], list[str]]
) -> None:
    unit_ids, integ_ids = tiers
    up = sum(1 for t in unit_ids if subgoals.get(t))
    ip = sum(1 for t in integ_ids if subgoals.get(t))
    print(f"    units (shape):      {up}/{len(unit_ids)} pass")
    print(f"    integration (gate): {ip}/{len(integ_ids)} pass")


def _print_reward(traj: Trajectory) -> None:
    comps = traj.reward.components
    passed = sum(1 for v in comps.values() if v >= 1.0)
    gate = "RESOLVED" if traj.success else "unresolved"
    print(
        f"  reward: scalar={traj.reward.scalar:.2f}  subgoals={passed}/{len(comps)}  "
        f"success={gate}"
    )


def _emit_trajectories(trajectories: list[Trajectory], out_dir: Path) -> None:
    path = out_dir / "trajectories.jsonl"
    path.write_text(to_jsonl(trajectories) + "\n", encoding="utf-8")
    scalars = ", ".join(f"{t.reward.scalar:.2f}" for t in trajectories)
    print("\ntraining seam: episode -> (trajectory, reward)")
    print(f"  rewards across episodes: [{scalars}]")
    print(f"  wrote {len(trajectories)} JSONL trajectory record(s) -> {path}")


def _write_tree(root: Path, overlay: Mapping[str, str]) -> None:
    for rel, contents in overlay.items():
        dest = root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(contents, encoding="utf-8")


def _finish(root: Path) -> None:
    (root / "result.json").write_text('{"done": true}\n', encoding="utf-8")


def _gold_files(snapshot: Snapshot) -> dict[str, str]:
    solution = snapshot.graph.by_kind("solution")[0]
    raw = solution.attrs.get("gold_files", {})
    if not isinstance(raw, Mapping):
        return {}
    return {str(k): str(v) for k, v in raw.items()}


def _base_files(snapshot: Snapshot) -> dict[str, str]:
    repos = snapshot.graph.by_kind("repo")
    if not repos:
        return {}
    raw = repos[0].attrs.get("base_files", {})
    if not isinstance(raw, Mapping):
        return {}
    return {str(k): str(v) for k, v in raw.items()}


def _suite_tiers(snapshot: Snapshot) -> tuple[list[str], list[str]]:
    suite = snapshot.graph.by_kind("test_suite")[0]
    return (
        _str_list(suite.attrs.get("unit_tests")),
        _str_list(suite.attrs.get("integration_tests")),
    )


def _str_list(value: object) -> list[str]:
    return [str(v) for v in value] if isinstance(value, list) else []


def _first_line(text: str) -> str:
    return text.strip().splitlines()[0] if text.strip() else ""


def _resolve_root(args: argparse.Namespace) -> Path:
    if args.run_root is not None:
        return Path(args.run_root)
    args.runs_dir.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix="swe-build-eval-", dir=args.runs_dir))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=Path("or-runs"))
    parser.add_argument("--run-root", type=Path)
    return parser.parse_args()


if __name__ == "__main__":  # pragma: no cover
    main()
