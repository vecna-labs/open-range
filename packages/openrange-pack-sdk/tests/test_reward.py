"""RewardAdapter seam: the ABC, the generic default, and the Pack hook.

Real concrete subclasses + real ``EpisodeResult`` objects. No mocks, no
patches, no test doubles.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from graphschema import Ontology, WorldGraph
from openrange_pack_sdk import (
    Backing,
    Builder,
    EpisodeReportLike,
    EpisodeResult,
    Manifest,
    Pack,
    PackPrior,
    RewardAdapter,
    RuntimeHandle,
    SubgoalFractionRewardAdapter,
    TaskFamily,
)


@dataclass(frozen=True)
class _Report:
    """A real ``EpisodeReportLike`` — the slice an adapter reads."""

    passed: bool
    episode_result: EpisodeResult
    final_state: Mapping[str, Any] = field(default_factory=dict)


def _report(*, success: bool, subgoals: Mapping[str, bool]) -> _Report:
    return _Report(
        passed=success,
        episode_result=EpisodeResult(success=success, subgoals=dict(subgoals)),
    )


class _NoopBuilder(Builder):
    def build(self, manifest: Manifest) -> Any:
        del manifest
        from openrange_pack_sdk import BuildResult

        return BuildResult(WorldGraph(ontology="test@0"), [])


class _MinimalPack(Pack):
    id = "minimal"
    version = "v0"

    def ontology(self) -> Ontology:
        return Ontology(id="test@0")

    def make_builder(self, prior: PackPrior | None) -> Builder:
        del prior
        return _NoopBuilder()

    def realize(self, graph: WorldGraph, backing: Backing) -> RuntimeHandle:
        raise NotImplementedError

    def task_families(self) -> list[TaskFamily]:
        return []


class _VectorAdapter(RewardAdapter):
    def reward(self, report: EpisodeReportLike) -> Sequence[float]:
        return [1.0 if hit else 0.0 for hit in report.episode_result.subgoals.values()]


class TestSubgoalFractionRewardAdapter:
    def test_solved_is_one(self) -> None:
        adapter = SubgoalFractionRewardAdapter()
        assert adapter.reward(_report(success=True, subgoals={"a": False})) == 1.0

    def test_partial_credit_is_subgoal_fraction(self) -> None:
        adapter = SubgoalFractionRewardAdapter()
        reward = adapter.reward(
            _report(success=False, subgoals={"a": True, "b": False, "c": False})
        )
        assert reward == 1 / 3

    def test_unsolved_without_subgoals_is_zero(self) -> None:
        adapter = SubgoalFractionRewardAdapter()
        assert adapter.reward(_report(success=False, subgoals={})) == 0.0

    def test_is_a_reward_adapter(self) -> None:
        assert isinstance(SubgoalFractionRewardAdapter(), RewardAdapter)


class TestPackDefaultRewardAdapter:
    def test_default_is_subgoal_fraction(self) -> None:
        adapter = _MinimalPack().default_reward_adapter()
        assert isinstance(adapter, SubgoalFractionRewardAdapter)

    def test_default_scores_like_the_generic_mapping(self) -> None:
        adapter = _MinimalPack().default_reward_adapter()
        assert (
            adapter.reward(_report(success=False, subgoals={"a": True, "b": False}))
            == 0.5
        )


class TestRewardAdapterVectorOutput:
    def test_subclass_may_return_a_vector(self) -> None:
        reward = _VectorAdapter().reward(
            _report(success=False, subgoals={"a": True, "b": False})
        )
        assert list(reward) == [1.0, 0.0]


class TestWidenedProtocol:
    def test_report_satisfies_runtime_checkable_protocol(self) -> None:
        report = _report(success=True, subgoals={"a": True})
        assert isinstance(report, EpisodeReportLike)
        assert report.episode_result.success is True
