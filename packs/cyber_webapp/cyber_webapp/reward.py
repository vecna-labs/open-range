"""Reward shaping for ``webapp.pentest`` — the exploit-chain ladder."""

from __future__ import annotations

from openrange_pack_sdk import EpisodeReportLike, RewardAdapter


class CyberPentestRewardAdapter(RewardAdapter):
    """Monotone partial credit from the pentest subgoal chain.

    A solved episode scores ``1.0``. An unsolved one scores the deepest rung it
    reached along the natural exploit chain the grader emits — touching the
    endpoint (``reached_endpoint`` → ``0.2``), pulling anything back out of it
    (``extracted_anything`` → ``0.6``), then matching the flag
    (``matched_flag`` → ``1.0``). Closer chains score higher, so a trainer sees
    a gradient toward the exploit instead of a flat zero on every miss.
    """

    def reward(self, report: EpisodeReportLike) -> float:
        if report.passed:
            return 1.0
        subgoals = report.episode_result.subgoals
        rung = 0.0
        if subgoals.get("reached_endpoint"):
            rung = 0.2
        if subgoals.get("extracted_anything"):
            rung = 0.6
        if subgoals.get("matched_flag"):
            rung = 1.0
        return rung
