"""Training seam: ``EpisodeReport`` (+ recorded turns) → (Trajectory, Reward).

Covers the reward shaping (solved → 1.0, unsolved → subgoal fraction, the
no-subgoal edges), trajectory reconstruction from the harness's ``AgentTurn``
records, and the JSON / JSONL export a trainer consumes.
"""

from __future__ import annotations

import json
from collections.abc import Mapping

from openrange_pack_sdk import EpisodeResult

from openrange.agent import AgentRollout
from openrange.core.episode import AgentTurn, EpisodeReport
from openrange.training import (
    EpisodeRun,
    Reward,
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
    baseline: Mapping[str, bool] | None = None,
) -> EpisodeReport:
    return EpisodeReport(
        snapshot_id=snapshot_id,
        task_id=task_id,
        episode_result=EpisodeResult(
            success=success,
            subgoals=dict(subgoals),
            reason=reason,
            baseline=dict(baseline or {}),
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


class TestRewardBaseline:
    """A subgoal the world hands over is not partial credit.

    The shape is ``swe.fix``: ``pass_to_pass`` is green before the agent edits
    anything, so uniform credit pays an agent that changes nothing.
    """

    def test_preconditions_are_not_credited(self) -> None:
        reward = episode_reward(
            _report(
                success=False,
                subgoals={"f2p": False, "p2p_a": True, "p2p_b": True},
                baseline={"p2p_a": True, "p2p_b": True},
            )
        )
        assert reward.scalar == 0.0
        assert reward.components == {"f2p": 0.0}

    def test_partial_credit_is_over_the_earnable_subgoals(self) -> None:
        reward = episode_reward(
            _report(
                success=False,
                subgoals={"f2p_a": True, "f2p_b": False, "p2p": True},
                baseline={"p2p": True},
            )
        )
        assert reward.scalar == 0.5
        assert reward.components == {"f2p_a": 1.0, "f2p_b": 0.0}

    def test_a_regressed_precondition_is_worth_nothing(self) -> None:
        reward = episode_reward(
            _report(
                success=False,
                subgoals={"f2p": True, "p2p": False},
                baseline={"p2p": True},
            )
        )
        assert reward.scalar == 0.0
        # The broken precondition is back in the vector: dropped, the loss is
        # invisible to every consumer that reads components rather than scalar.
        assert reward.components == {"f2p": 1.0, "p2p": 0.0}

    def test_nothing_earnable_is_zero(self) -> None:
        reward = episode_reward(
            _report(
                success=False,
                subgoals={"p2p": True},
                baseline={"p2p": True},
            )
        )
        assert reward.scalar == 0.0
        assert reward.components == {}

    def test_solved_still_scores_one_over_the_earnable_set(self) -> None:
        reward = episode_reward(
            _report(
                success=True,
                subgoals={"f2p": True, "p2p": True},
                baseline={"p2p": True},
            )
        )
        assert reward.scalar == 1.0
        assert reward.components == {"f2p": 1.0}


class TestRewardError:
    """An episode that graded nothing still shapes to 0.0.

    ``error`` does not change the scalar — a trainer reading a batch needs a
    number for every rollout. It marks the episode as a non-measurement so the
    consumers that *can* exclude it (the pool, a metrics pass) know to.
    """

    def test_an_errored_episode_still_scores_zero(self) -> None:
        report = _report(success=False, subgoals={})
        errored = EpisodeReport(
            snapshot_id=report.snapshot_id,
            task_id=report.task_id,
            episode_result=EpisodeResult(
                success=False, reason="grading failed", error="TimeoutExpired"
            ),
        )
        assert episode_reward(errored).scalar == 0.0
        assert episode_reward(errored).components == {}


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

    def test_a_rollout_exports_the_reward_it_was_graded_with(self) -> None:
        # The trajectory is what a trainer consumes, so it has to carry the
        # objective the caller actually ran, not the built-in one.
        report = _report(success=False, subgoals={"free": True, "earned": False})
        only_earned = Reward(scalar=0.0, components={"earned": 0.0})
        rollout = AgentRollout(
            snapshot_id=report.snapshot_id,
            task_id=report.task_id,
            steps=(),
            turns=(),
            report=report,
            reward=only_earned,
            terminal_reason="done",
        )
        assert episode_reward(report).scalar == 0.5  # what the default would say
        assert rollout.reward == rollout.trajectory.reward

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
