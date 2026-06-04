"""Runtime convenience seam: ``OpenRangeRun.run_episode`` and ``EpisodeContext``.

``run_episode`` is the one call that replaces the
``episode_service → start_episode → record_turn → stop_episode → close`` loop a
harness would otherwise hand-roll. These tests drive it end to end through the
real webapp pack (no LLM): a scripted solver writes ``result.json`` and the
held-out grader scores it. A solving run and a no-op run prove the result
discriminates; a multi-turn solver proves each returned turn is recorded; and
the ``EpisodeContext`` accessors are exercised on their own.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from cyber_webapp.families.build.reference import api_list_reference
from openrange_pack_sdk import Snapshot, TaskSpec

from openrange.core.episode import AgentTurn, EpisodeError
from openrange.runtime import EpisodeContext, OpenRangeRun, RunConfig

MANIFEST = {
    "world": {"goal": "run_episode end to end"},
    "pack": {"id": "webapp"},
    "runtime": {"tick": {"mode": "off"}},
    "npc": [],
}


def _spec() -> TaskSpec:
    return TaskSpec(
        id="t",
        instruction="solve it",
        entrypoints=("e",),
        goal_nodes=(),
        feasibility_check="feasible",
        success_check="success",
    )


class TestEpisodeContext:
    def test_root_from_surface_string(self, tmp_path: Path) -> None:
        ctx = EpisodeContext(task=_spec(), surface={"solver_root": str(tmp_path)})
        assert ctx.root == tmp_path

    def test_root_accepts_path(self, tmp_path: Path) -> None:
        ctx = EpisodeContext(task=_spec(), surface={"solver_root": tmp_path})
        assert ctx.root == tmp_path

    def test_root_raises_when_absent(self) -> None:
        ctx = EpisodeContext(task=_spec(), surface={})
        with pytest.raises(EpisodeError, match="solver_root"):
            _ = ctx.root

    def test_base_url_from_surface(self) -> None:
        ctx = EpisodeContext(task=_spec(), surface={"base_url": "http://x"})
        assert ctx.base_url == "http://x"

    def test_base_url_raises_when_absent(self) -> None:
        ctx = EpisodeContext(task=_spec(), surface={})
        with pytest.raises(EpisodeError, match="base_url"):
            _ = ctx.base_url

    def test_exposes_task_and_surface(self) -> None:
        spec = _spec()
        ctx = EpisodeContext(task=spec, surface={"k": "v"})
        assert ctx.task is spec
        assert ctx.surface["k"] == "v"


@pytest.fixture(scope="module")
def snapshot(tmp_path_factory: pytest.TempPathFactory) -> Snapshot:
    root = tmp_path_factory.mktemp("runtime-build")
    return OpenRangeRun(RunConfig(root, dashboard=False)).build(MANIFEST)


def _build_task_id(snapshot: Snapshot) -> str:
    tasks = [t for t in snapshot.tasks if t.meta.get("family") == "webapp.build"]
    assert len(tasks) == 1, f"expected one webapp.build task, got {tasks}"
    return tasks[0].id


def _write_reference(ctx: EpisodeContext) -> None:
    (ctx.root / "result.json").write_text(
        json.dumps({"endpoint_impl": api_list_reference(1)}),
        encoding="utf-8",
    )


class TestRunEpisode:
    def test_solved_episode_scores_one(
        self, snapshot: Snapshot, tmp_path: Path
    ) -> None:
        run = OpenRangeRun(RunConfig(tmp_path, dashboard=False))

        def solve(ctx: EpisodeContext) -> AgentTurn:
            _write_reference(ctx)
            return AgentTurn(message="submitted reference handler")

        ep = run.run_episode(snapshot, solve, task_id=_build_task_id(snapshot))
        assert ep.success is True, ep.report.episode_result.reason
        assert ep.reward.scalar == 1.0
        assert ep.report.snapshot_id == snapshot.snapshot_id
        assert [s.message for s in ep.trajectory.steps] == [
            "submitted reference handler"
        ]

    def test_noop_solver_is_unsolved_and_zero_step(
        self, snapshot: Snapshot, tmp_path: Path
    ) -> None:
        run = OpenRangeRun(RunConfig(tmp_path, dashboard=False))

        def solve(ctx: EpisodeContext) -> None:
            return None

        ep = run.run_episode(snapshot, solve, task_id=_build_task_id(snapshot))
        assert ep.success is False
        assert ep.reward.scalar == 0.0
        assert ep.trajectory.steps == ()

    def test_multi_turn_solver_records_each_turn(
        self, snapshot: Snapshot, tmp_path: Path
    ) -> None:
        run = OpenRangeRun(RunConfig(tmp_path, dashboard=False))

        def solve(ctx: EpisodeContext) -> list[AgentTurn]:
            _write_reference(ctx)
            return [AgentTurn(message="inspecting"), AgentTurn(message="submitted")]

        ep = run.run_episode(snapshot, solve, task_id=_build_task_id(snapshot))
        assert ep.success is True, ep.report.episode_result.reason
        assert [s.message for s in ep.trajectory.steps] == ["inspecting", "submitted"]
        assert ep.report.agent_summary == "submitted"

    def test_default_task_id_runs_first_task(
        self, snapshot: Snapshot, tmp_path: Path
    ) -> None:
        run = OpenRangeRun(RunConfig(tmp_path, dashboard=False))

        def solve(ctx: EpisodeContext) -> None:
            return None

        ep = run.run_episode(snapshot, solve)
        assert ep.report.task_id == snapshot.tasks[0].id

    def test_solver_exception_propagates(
        self, snapshot: Snapshot, tmp_path: Path
    ) -> None:
        run = OpenRangeRun(RunConfig(tmp_path, dashboard=False))

        class SolverBoom(RuntimeError):
            pass

        def solve(ctx: EpisodeContext) -> AgentTurn:
            raise SolverBoom("solver failed")

        with pytest.raises(SolverBoom, match="solver failed"):
            run.run_episode(snapshot, solve, task_id=_build_task_id(snapshot))

    def test_base_url_exposed_on_realized_surface(
        self, snapshot: Snapshot, tmp_path: Path
    ) -> None:
        run = OpenRangeRun(RunConfig(tmp_path, dashboard=False))

        def solve(ctx: EpisodeContext) -> AgentTurn:
            return AgentTurn(message=ctx.base_url)

        ep = run.run_episode(snapshot, solve, task_id=_build_task_id(snapshot))
        assert ep.report.agent_summary.startswith("http://")
