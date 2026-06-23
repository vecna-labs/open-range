"""The pool ranks worlds by the SAME reward the trainer optimizes.

``reward_fn`` threads into ``EpisodeEnv`` so GRPO optimizes it; these prove it also
reaches the pool's world priority (``_member_priority``), so a caller with a custom
reward evolves the pool on the objective the policy is trained on, not the default
``episode_reward``. Pure: no pack, no trainer, no model.
"""

from __future__ import annotations

from types import SimpleNamespace

from openrange_pack_sdk import EpisodeResult

from openrange.core.episode import EpisodeReport
from openrange.pool import WorldPool, _Member, _member_priority, run_pool_curriculum
from openrange.training import Reward, episode_reward


def _report(success: bool, subgoals: dict[str, bool]) -> EpisodeReport:
    return EpisodeReport(
        snapshot_id="s",
        task_id="t",
        episode_result=EpisodeResult(success=success, subgoals=subgoals),
    )


def _half(_report: EpisodeReport) -> Reward:
    # A reward unlike episode_reward's subgoal fraction, so priority must shift.
    return Reward(scalar=0.5)


def test_member_priority_defaults_to_episode_reward() -> None:
    reports = [_report(False, {"a": True, "b": False, "c": False})]
    assert _member_priority(reports) == _member_priority(reports, episode_reward)


def test_member_priority_uses_a_custom_reward_fn() -> None:
    # default episode_reward scalar = 1/3 (one of three subgoals passed); a reward
    # that returns 0.5 lands learnability at its peak, so the priority must differ.
    reports = [_report(False, {"a": True, "b": False, "c": False})]
    assert _member_priority(reports, _half) != _member_priority(reports)


def _pool_with_one_member() -> tuple[WorldPool, _Member]:
    member = _Member(
        snapshot=SimpleNamespace(snapshot_id="s"),
        task_id="t",
        instruction="i",
        family="f",
        difficulty=0.0,
    )
    return WorldPool([member], difficulty_fn=lambda _s: 0.0, max_size=4), member


def test_update_threads_reward_fn_into_priority() -> None:
    pool, member = _pool_with_one_member()
    reports = {("s", "t"): [_report(False, {"a": True, "b": False, "c": False})]}
    pool.update(reports, pack=object(), evolve_top=0)
    default_priority = member.priority
    member.priority = 1.0
    pool.update(reports, pack=object(), evolve_top=0, reward_fn=_half)
    assert member.priority != default_priority


def test_run_pool_curriculum_threads_reward_fn() -> None:
    pool, member = _pool_with_one_member()
    reports = {("s", "t"): [_report(False, {"a": True, "b": False, "c": False})]}

    def run_round(_rows: object, _snaps: object) -> dict:
        return reports

    run_pool_curriculum(
        pool,
        run_round,
        rounds=1,
        pack=object(),
        groups=1,
        num_generations=1,
        evolve_top=0,
        reward_fn=_half,
    )
    custom_priority = member.priority
    member.priority = 1.0
    run_pool_curriculum(
        pool,
        run_round,
        rounds=1,
        pack=object(),
        groups=1,
        num_generations=1,
        evolve_top=0,
    )
    assert member.priority != custom_priority
