"""Per-domain ``RewardAdapter`` vs the generic default, no LLM or network.

Builds a handful of ``EpisodeReport``s by hand — a clean miss, endpoint reached,
data extracted, flag solved — and prints the reward each one earns under the
domain-agnostic ``SubgoalFractionRewardAdapter`` and the cyber pack's
``CyberPentestRewardAdapter`` (``WebappPack().default_reward_adapter()``). Then it
feeds one report through the training seam to show ``episode_trajectory`` sourcing
its scalar from a pack adapter.

Run::

    uv run python -m examples.reward_adapter_demo
"""

from __future__ import annotations

from collections.abc import Mapping

from cyber_webapp import WebappPack
from openrange_pack_sdk import EpisodeResult, SubgoalFractionRewardAdapter

from openrange.core.episode import EpisodeReport
from openrange.training import episode_trajectory


def _report(
    label: str, *, success: bool, subgoals: Mapping[str, bool]
) -> EpisodeReport:
    return EpisodeReport(
        snapshot_id="sha256:demo",
        task_id=f"webapp.pentest.{label}",
        episode_result=EpisodeResult(
            success=success, subgoals=dict(subgoals), reason=label
        ),
    )


def main() -> None:
    reports = [
        _report(
            "miss",
            success=False,
            subgoals={
                "reached_endpoint": False,
                "extracted_anything": False,
                "matched_flag": False,
            },
        ),
        _report(
            "reached",
            success=False,
            subgoals={
                "reached_endpoint": True,
                "extracted_anything": False,
                "matched_flag": False,
            },
        ),
        _report(
            "extracted",
            success=False,
            subgoals={
                "reached_endpoint": True,
                "extracted_anything": True,
                "matched_flag": False,
            },
        ),
        _report(
            "solved",
            success=True,
            subgoals={
                "reached_endpoint": True,
                "extracted_anything": True,
                "matched_flag": True,
            },
        ),
    ]

    generic = SubgoalFractionRewardAdapter()
    pack = WebappPack()
    cyber = pack.default_reward_adapter()

    print(f"{'case':<12}{'generic':>10}{'cyber':>10}")
    for report in reports:
        generic_reward = generic.reward(report)
        cyber_reward = cyber.reward(report)
        label = report.episode_result.reason
        print(f"{label:<12}{generic_reward:>10.2f}{cyber_reward:>10.2f}")

    extracted = reports[2]
    trajectory = episode_trajectory(extracted, adapter=cyber)
    print()
    print(
        "episode_trajectory('extracted', adapter=cyber).reward ->",
        trajectory.reward.as_dict(),
    )


if __name__ == "__main__":
    main()
