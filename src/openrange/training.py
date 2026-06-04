"""Training seam: an episode's structured outcome → (trajectory, reward).

OpenRange packs emit a *structured* ``EpisodeResult`` — a boolean ``success``
plus per-subgoal flags — and deliberately never a scalar reward (that is the
contract written on ``EpisodeResult`` itself). The shaping into something a
trainer consumes happens here, once, harness-side, for every pack: a generic,
trainer-agnostic default that turns any ``EpisodeReport`` (plus the turns the
harness recorded) into a replayable ``Trajectory`` carrying a scalar **and** a
per-subgoal **vector** ``Reward``.

This is the reference half of the training-integration standard
([#243](https://github.com/vecna-labs/open-range/issues/243) /
[#199](https://github.com/vecna-labs/open-range/issues/199)); a concrete trainer
(open-trajectory-gym, [#198](https://github.com/vecna-labs/open-range/issues/198))
is the first consumer. Three choices keep it trainer-agnostic:

- **Dense by default.** ``episode_reward`` gives a solved episode ``1.0`` and an
  unsolved one the *fraction* of its subgoals that passed — partial credit a
  trainer can learn from — with no per-domain knowledge. A pack that wants
  bespoke shaping writes its own function returning a ``Reward``.
- **The curriculum dimension rides along.** Each ``Trajectory`` is tagged with
  the ``snapshot_id`` it ran against, so trajectories logged across an
  ``evolve(...)`` curriculum stay attributable to the exact (hardened) world
  that produced them.
- **The export stays lean.** The raw ``final_state`` (for SWE, the whole edited
  workspace tree) is left *off* the trajectory on purpose; it remains on the
  ``EpisodeReport`` for callers that need it.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from openrange.core.episode import AgentTurn, EpisodeReport


@dataclass(frozen=True, slots=True)
class Reward:
    """A shaped reward. ``scalar`` (in ``[0, 1]``) is what a scalar-reward
    trainer reads; ``components`` is the per-subgoal vector for trainers that
    consume structured signals."""

    scalar: float
    components: Mapping[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"scalar": self.scalar, "components": dict(self.components)}


@dataclass(frozen=True, slots=True)
class TrajectoryStep:
    """One recorded agent turn, replayable by a trainer: the message, the tool
    calls it made, and what they returned."""

    index: int
    message: str | None
    tool_calls: tuple[Mapping[str, Any], ...] = ()
    tool_results: tuple[Mapping[str, Any], ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "message": self.message,
            "tool_calls": [dict(c) for c in self.tool_calls],
            "tool_results": [dict(r) for r in self.tool_results],
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class Trajectory:
    """One episode as a trainer consumes it: the ordered ``steps``, the terminal
    ``reward``, the binary ``success`` gate, and the ``snapshot_id`` / ``task_id``
    that anchor it to a specific (possibly evolved) world."""

    snapshot_id: str
    task_id: str
    steps: tuple[TrajectoryStep, ...]
    reward: Reward
    success: bool
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "task_id": self.task_id,
            "steps": [s.as_dict() for s in self.steps],
            "reward": self.reward.as_dict(),
            "success": self.success,
            "reason": self.reason,
        }


def episode_reward(report: EpisodeReport) -> Reward:
    """Shape an ``EpisodeReport`` into a dense, trainer-agnostic ``Reward``.

    ``components`` maps each subgoal to ``1.0`` / ``0.0``. ``scalar`` is ``1.0``
    for a solved episode; otherwise the fraction of subgoals that passed (partial
    credit), or ``0.0`` when the pack reports no subgoals.
    """
    subgoals = report.episode_result.subgoals
    components = {str(k): (1.0 if v else 0.0) for k, v in subgoals.items()}
    if report.passed:
        scalar = 1.0
    elif components:
        scalar = sum(components.values()) / len(components)
    else:
        scalar = 0.0
    return Reward(scalar=scalar, components=components)


def episode_trajectory(
    report: EpisodeReport,
    turns: Sequence[AgentTurn] | None = None,
) -> Trajectory:
    """Assemble a ``Trajectory`` from a report and the turns the harness recorded.

    ``EpisodeReport`` keeps only the last agent message (``agent_summary``), so
    the per-step record is rebuilt from ``turns`` — the same ``AgentTurn`` objects
    the harness passed to ``record_turn``. With no turns the trajectory still
    carries the terminal reward + outcome (a degenerate zero-step episode).
    """
    steps = tuple(
        TrajectoryStep(
            index=i,
            message=turn.message,
            tool_calls=tuple(turn.tool_calls),
            tool_results=tuple(turn.tool_results),
            metadata=dict(turn.metadata),
        )
        for i, turn in enumerate(turns or ())
    )
    return Trajectory(
        snapshot_id=report.snapshot_id,
        task_id=report.task_id,
        steps=steps,
        reward=episode_reward(report),
        success=report.passed,
        reason=report.episode_result.reason,
    )


@dataclass(frozen=True, slots=True)
class EpisodeRun:
    """One completed episode as a caller gets it back from
    ``OpenRangeRun.run_episode``: the terminal ``report`` and the ``turns`` the
    solver took. ``trajectory`` shapes the pair through the seam above;
    ``reward`` and ``success`` are the shortcuts most callers actually read.
    """

    report: EpisodeReport
    turns: tuple[AgentTurn, ...] = ()

    @property
    def trajectory(self) -> Trajectory:
        return episode_trajectory(self.report, self.turns)

    @property
    def reward(self) -> Reward:
        return self.trajectory.reward

    @property
    def success(self) -> bool:
        return self.report.passed


def to_jsonl(trajectories: Iterable[Trajectory]) -> str:
    """Serialize trajectories as JSON Lines — one ``Trajectory.as_dict()`` per
    line, the shape a trainer streams in."""
    return "\n".join(json.dumps(t.as_dict(), sort_keys=True) for t in trajectories)
