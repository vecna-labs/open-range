"""Cyber pentest reward ladder: subgoal chain → monotone partial credit.

Real ``EpisodeReport`` objects and the pack's own adapter. No mocks, no
patches, no test doubles.
"""

from __future__ import annotations

from collections.abc import Mapping

from cyber_webapp import CyberPentestRewardAdapter, WebappPack
from openrange_pack_sdk import EpisodeResult, RewardAdapter

from openrange.core.episode import EpisodeReport


def _report(*, success: bool, subgoals: Mapping[str, bool]) -> EpisodeReport:
    return EpisodeReport(
        snapshot_id="sha256:world",
        task_id="webapp.pentest.0",
        episode_result=EpisodeResult(success=success, subgoals=dict(subgoals)),
    )


def _chain(
    *, reached: bool, extracted: bool, matched: bool, success: bool
) -> EpisodeReport:
    return _report(
        success=success,
        subgoals={
            "reached_endpoint": reached,
            "extracted_anything": extracted,
            "matched_flag": matched,
        },
    )


class TestCyberPentestRewardAdapter:
    def test_solved_is_one(self) -> None:
        adapter = CyberPentestRewardAdapter()
        report = _chain(reached=True, extracted=True, matched=True, success=True)
        assert adapter.reward(report) == 1.0

    def test_clean_miss_is_zero(self) -> None:
        adapter = CyberPentestRewardAdapter()
        report = _chain(reached=False, extracted=False, matched=False, success=False)
        assert adapter.reward(report) == 0.0

    def test_reached_endpoint_only(self) -> None:
        adapter = CyberPentestRewardAdapter()
        report = _chain(reached=True, extracted=False, matched=False, success=False)
        assert adapter.reward(report) == 0.2

    def test_extracted_anything(self) -> None:
        adapter = CyberPentestRewardAdapter()
        report = _chain(reached=True, extracted=True, matched=False, success=False)
        assert adapter.reward(report) == 0.6

    def test_matched_flag_without_success_reaches_top_rung(self) -> None:
        # matched_flag is the deepest rung; a grader that flags it without the
        # submission gate still earns the full ladder value.
        adapter = CyberPentestRewardAdapter()
        report = _chain(reached=True, extracted=True, matched=True, success=False)
        assert adapter.reward(report) == 1.0

    def test_ladder_is_monotone_non_decreasing(self) -> None:
        adapter = CyberPentestRewardAdapter()
        rungs = [
            adapter.reward(
                _chain(reached=False, extracted=False, matched=False, success=False)
            ),
            adapter.reward(
                _chain(reached=True, extracted=False, matched=False, success=False)
            ),
            adapter.reward(
                _chain(reached=True, extracted=True, matched=False, success=False)
            ),
            adapter.reward(
                _chain(reached=True, extracted=True, matched=True, success=True)
            ),
        ]
        assert rungs == sorted(rungs)
        assert rungs == [0.0, 0.2, 0.6, 1.0]


class TestWebappPackWiring:
    def test_default_reward_adapter_is_the_cyber_ladder(self) -> None:
        adapter = WebappPack().default_reward_adapter()
        assert isinstance(adapter, CyberPentestRewardAdapter)
        assert isinstance(adapter, RewardAdapter)
