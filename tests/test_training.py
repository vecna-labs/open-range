"""Training seam: ``EpisodeReport`` (+ recorded turns) → (Trajectory, Reward).

Covers the reward shaping (solved → 1.0, unsolved → subgoal fraction, the
no-subgoal edges), trajectory reconstruction from the harness's ``AgentTurn``
records, and the JSON / JSONL export a trainer consumes.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence

from openrange_pack_sdk import EpisodeReportLike, EpisodeResult, RewardAdapter

from openrange.core.episode import AgentTurn, EpisodeReport
from openrange.training import (
    EpisodeRun,
    episode_reward,
    episode_trajectory,
    to_jsonl,
)


def _report(
    *,
    success: bool,
    subgoals: Mapping[str, bool],
    reason: str = "",
    snapshot_id: str = "sha256:world",
    task_id: str = "swe.fix.calc",
) -> EpisodeReport:
    return EpisodeReport(
        snapshot_id=snapshot_id,
        task_id=task_id,
        episode_result=EpisodeResult(
            success=success, subgoals=dict(subgoals), reason=reason
        ),
    )


class TestReward:
    def test_solved_is_one(self) -> None:
        reward = episode_reward(
            _report(success=True, subgoals={"t1": True, "t2": True})
        )
        assert reward.scalar == 1.0
        assert reward.components == {"t1": 1.0, "t2": 1.0}

    def test_partial_credit_is_subgoal_fraction(self) -> None:
        reward = episode_reward(
            _report(success=False, subgoals={"t1": True, "t2": False})
        )
        assert reward.scalar == 0.5
        assert reward.components == {"t1": 1.0, "t2": 0.0}

    def test_unsolved_without_subgoals_is_zero(self) -> None:
        reward = episode_reward(_report(success=False, subgoals={}))
        assert reward.scalar == 0.0
        assert reward.components == {}

    def test_solved_without_subgoals_is_one(self) -> None:
        # success is the gate: a pack may green an episode with no subgoals.
        reward = episode_reward(_report(success=True, subgoals={}))
        assert reward.scalar == 1.0


class TestTrajectory:
    def test_rebuilds_steps_from_turns(self) -> None:
        report = _report(
            success=True,
            subgoals={"t1": True},
            reason="all pass",
            snapshot_id="sha256:w1",
            task_id="swe.fix.calc",
        )
        turns = [
            AgentTurn(
                message="red",
                tool_calls=({"tool": "run_tests", "args": {"node_ids": ["repro.py"]}},),
                tool_results=({"ok": False, "returncode": 1},),
            ),
            AgentTurn(
                message="green",
                tool_calls=({"tool": "run_tests"},),
                tool_results=({"ok": True, "returncode": 0},),
            ),
        ]
        traj = episode_trajectory(report, turns)
        assert traj.snapshot_id == "sha256:w1"
        assert traj.task_id == "swe.fix.calc"
        assert traj.success is True
        assert traj.reason == "all pass"
        assert traj.reward.scalar == 1.0
        assert [s.index for s in traj.steps] == [0, 1]
        assert traj.steps[0].message == "red"
        assert traj.steps[0].tool_results[0]["returncode"] == 1
        assert traj.steps[1].tool_results[0]["ok"] is True

    def test_without_turns_is_zero_step(self) -> None:
        traj = episode_trajectory(_report(success=False, subgoals={"t1": False}))
        assert traj.steps == ()
        assert traj.reward.scalar == 0.0
        assert traj.success is False

    def test_as_dict_is_json_serializable(self) -> None:
        report = _report(
            success=False, subgoals={"t1": True, "t2": False}, reason="1/2"
        )
        turns = [
            AgentTurn(
                message="m",
                tool_calls=({"tool": "x"},),
                tool_results=({"ok": True},),
                metadata={"k": "v"},
            )
        ]
        traj = episode_trajectory(report, turns)
        back = json.loads(json.dumps(traj.as_dict()))
        assert back["reward"]["scalar"] == 0.5
        assert back["reward"]["components"] == {"t1": 1.0, "t2": 0.0}
        assert back["steps"][0]["tool_calls"] == [{"tool": "x"}]
        assert back["steps"][0]["metadata"] == {"k": "v"}
        assert back["success"] is False


class TestEpisodeRun:
    def test_bundles_report_turns_and_shapes_trajectory(self) -> None:
        report = _report(success=True, subgoals={"t1": True}, reason="green")
        run = EpisodeRun(report=report, turns=(AgentTurn(message="done"),))
        assert run.success is True
        assert run.reward.scalar == 1.0
        assert [s.message for s in run.trajectory.steps] == ["done"]

    def test_defaults_to_zero_step_trajectory(self) -> None:
        run = EpisodeRun(report=_report(success=False, subgoals={"t1": False}))
        assert run.success is False
        assert run.reward.scalar == 0.0
        assert run.trajectory.steps == ()


class _ConstantAdapter(RewardAdapter):
    """Returns a fixed scalar regardless of the report."""

    def __init__(self, value: float) -> None:
        self._value = value

    def reward(self, report: EpisodeReportLike) -> float:
        del report
        return self._value


class _VectorAdapter(RewardAdapter):
    """Returns a fixed multi-objective vector."""

    def __init__(self, vector: Sequence[float]) -> None:
        self._vector = list(vector)

    def reward(self, report: EpisodeReportLike) -> Sequence[float]:
        del report
        return self._vector


class TestRewardAdapterParam:
    def test_scalar_adapter_sources_scalar_keeps_subgoal_components(self) -> None:
        report = _report(success=False, subgoals={"t1": True, "t2": False})
        reward = episode_reward(report, _ConstantAdapter(0.42))
        assert reward.scalar == 0.42
        # components stay the per-subgoal 1/0 vector, independent of the scalar.
        assert reward.components == {"t1": 1.0, "t2": 0.0}

    def test_none_adapter_preserves_default_behavior(self) -> None:
        report = _report(success=False, subgoals={"t1": True, "t2": False})
        assert episode_reward(report, None) == episode_reward(report)

    def test_vector_adapter_means_scalar_and_indexes_components(self) -> None:
        report = _report(success=True, subgoals={"t1": True})
        reward = episode_reward(report, _VectorAdapter([0.2, 0.8]))
        assert reward.scalar == 0.5
        assert reward.components == {"0": 0.2, "1": 0.8}

    def test_empty_vector_adapter_is_zero(self) -> None:
        report = _report(success=True, subgoals={"t1": True})
        reward = episode_reward(report, _VectorAdapter([]))
        assert reward.scalar == 0.0
        assert reward.components == {}

    def test_trajectory_threads_adapter_into_reward(self) -> None:
        report = _report(success=False, subgoals={"t1": False})
        traj = episode_trajectory(report, None, _ConstantAdapter(0.9))
        assert traj.reward.scalar == 0.9


def test_to_jsonl_one_line_per_trajectory() -> None:
    trajs = [
        episode_trajectory(_report(success=True, subgoals={"t": True}, task_id="a")),
        episode_trajectory(_report(success=False, subgoals={"t": False}, task_id="b")),
    ]
    rows = [json.loads(line) for line in to_jsonl(trajs).splitlines()]
    assert len(rows) == 2
    assert rows[0]["task_id"] == "a"
    assert rows[0]["reward"]["scalar"] == 1.0
    assert rows[1]["reward"]["scalar"] == 0.0
