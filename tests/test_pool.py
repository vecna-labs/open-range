"""The pool ranks worlds by the reward the trainer optimizes, not always the default.

``_member_priority`` reads shaped ``Reward``s, so the ``reward_fn`` handed to
``WorldPool.update`` is the objective the ranking follows, and a subgoal the world
granted for free cannot inflate the regret term into headroom that is not there.
The threading through ``WorldPool.update`` / ``run_pool_curriculum`` is covered as
an integration test in ``test_curriculum`` (against a real admitted world).
"""

from __future__ import annotations

from openrange_pack_sdk import EpisodeResult

from openrange.core.episode import EpisodeReport
from openrange.pool import _member_priority
from openrange.training import Reward, episode_reward


def _report(
    success: bool,
    subgoals: dict[str, bool],
    baseline: dict[str, bool] | None = None,
) -> EpisodeReport:
    return EpisodeReport(
        snapshot_id="s",
        task_id="t",
        episode_result=EpisodeResult(
            success=success, subgoals=subgoals, baseline=baseline or {}
        ),
    )


def test_priority_follows_the_shaped_reward() -> None:
    reports = [_report(False, {"a": True, "b": False, "c": False})]
    shaped = _member_priority([episode_reward(r) for r in reports])
    flat = _member_priority([Reward(scalar=0.5) for _ in reports])
    assert shaped != flat


def test_regret_ignores_subgoals_the_world_granted() -> None:
    # Identical observed subgoals; the second world handed "a" over unearned, so
    # it offers one third less to learn and must not rank as if it offered more.
    earned = episode_reward(_report(False, {"a": True, "b": False, "c": False}))
    granted = episode_reward(
        _report(False, {"a": True, "b": False, "c": False}, {"a": True})
    )
    assert _member_priority([earned]) > _member_priority([granted])


def test_a_world_the_agent_almost_solved_outranks_a_solved_one() -> None:
    # The instructive world: everything earnable won, one precondition broken.
    # It must not sort with the solved world, or the pool evicts exactly the
    # world with the most left to learn.
    trap = episode_reward(_report(False, {"f2p": True, "p2p": False}, {"p2p": True}))
    solved = episode_reward(_report(True, {"f2p": True, "p2p": True}, {"p2p": True}))
    idle = episode_reward(_report(False, {"f2p": False, "p2p": True}, {"p2p": True}))
    assert _member_priority([solved]) < _member_priority([trap])
    assert _member_priority([trap]) < _member_priority([idle])


def test_priority_without_components_is_learnability_only() -> None:
    # No subgoals at all: regret has nothing to read, so priority is the
    # learnability term alone rather than an accidental zero.
    assert _member_priority([episode_reward(_report(True, {}))]) == 0.0
