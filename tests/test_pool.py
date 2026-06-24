"""The pool ranks worlds by the reward the trainer optimizes, not always the default.

``reward_fn`` threads into world priority (``_member_priority``); a custom reward
must change the ranking, or the pool evolves on a different objective than GRPO.
The threading through ``WorldPool.update`` / ``run_pool_curriculum`` is covered as
an integration test in ``test_curriculum`` (against a real admitted world).
"""

from __future__ import annotations

from openrange_pack_sdk import EpisodeResult

from openrange.core.episode import EpisodeReport
from openrange.pool import _member_priority
from openrange.training import Reward, episode_reward


def _report(success: bool, subgoals: dict[str, bool]) -> EpisodeReport:
    return EpisodeReport(
        snapshot_id="s",
        task_id="t",
        episode_result=EpisodeResult(success=success, subgoals=subgoals),
    )


def test_member_priority_defaults_to_episode_reward() -> None:
    reports = [_report(False, {"a": True, "b": False, "c": False})]
    assert _member_priority(reports) == _member_priority(reports, episode_reward)


def test_member_priority_uses_a_custom_reward_fn() -> None:
    reports = [_report(False, {"a": True, "b": False, "c": False})]
    custom = _member_priority(reports, lambda _r: Reward(scalar=0.5))
    assert custom != _member_priority(reports)
