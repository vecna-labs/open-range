"""The curriculum as an evolving pool of worlds-and-tasks.

See ``docs/design/evolving-pool-curriculum.md``. The pool is harness-side (core
ships no curriculum algorithm): it seeds wide, samples a round's prompts with a
mix floor so an easy tail always survives, and between rounds shifts slowly —
re-prioritising members by how much they can still teach, evolving the most
promising into admitted children while keeping their parents. Difficulty is
injected by the caller so the pool stays pack-agnostic.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from openrange_pack_sdk import Pack, Snapshot

from openrange.core.admit import AdmissionFailure, admit
from openrange.core.curriculum import (
    CurriculumPolicy,
    EvolutionGate,
    auto_evolve,
    direction_from_reports,
)
from openrange.core.episode import EpisodeReport
from openrange.training import episode_reward

DifficultyFn = Callable[[Snapshot], float]
PromptRow = dict[str, object]
RoundReports = Mapping[tuple[str, str], Sequence[EpisodeReport]]
RunRound = Callable[[list[PromptRow], list[Snapshot]], RoundReports]

_STALENESS_STEP = 0.1


@dataclass
class _Member:
    snapshot: Snapshot
    task_id: str
    instruction: str
    family: str
    difficulty: float
    priority: float = 1.0

    @property
    def key(self) -> tuple[str, str]:
        return (self.snapshot.snapshot_id, self.task_id)


def _member_priority(reports: Sequence[EpisodeReport]) -> float:
    scalars = [episode_reward(r).scalar for r in reports]
    mean = sum(scalars) / len(scalars)
    learnability = 1.0 - abs(2.0 * mean - 1.0)
    gaps = []
    for report in reports:
        subgoals = report.episode_result.subgoals
        if subgoals:
            achieved = sum(1 for hit in subgoals.values() if hit)
            gaps.append(1.0 - achieved / len(subgoals))
    # The regret term keeps a world stuck below its solvable ceiling (low reward
    # spread, so learnability alone would retire it) at the frontier.
    regret = sum(gaps) / len(gaps) if gaps else 0.0
    return learnability + regret


class WorldPool:
    def __init__(
        self,
        members: Sequence[_Member],
        *,
        difficulty_fn: DifficultyFn,
        max_size: int,
        mix_floor: float = 0.3,
    ) -> None:
        self._members: dict[tuple[str, str], _Member] = {m.key: m for m in members}
        self._difficulty_fn = difficulty_fn
        self._max_size = max_size
        self._mix_floor = mix_floor

    @classmethod
    def seed(
        cls,
        pack: Pack,
        manifests: Sequence[Mapping[str, object]],
        *,
        difficulty_fn: DifficultyFn,
        max_size: int,
        family: str | None = None,
        mix_floor: float = 0.3,
        max_repairs: int = 2,
    ) -> WorldPool:
        members: list[_Member] = []
        for manifest in manifests:
            result = admit(pack, dict(manifest), max_repairs=max_repairs)
            if isinstance(result, AdmissionFailure):
                continue
            cls._members_of(result, difficulty_fn(result), family, members)
        return cls(
            members, difficulty_fn=difficulty_fn, max_size=max_size, mix_floor=mix_floor
        )

    @staticmethod
    def _members_of(
        snapshot: Snapshot,
        difficulty: float,
        family: str | None,
        out: list[_Member],
    ) -> None:
        for task in snapshot.tasks:
            fam = str(task.meta.get("family", ""))
            if family is not None and fam != family:
                continue
            out.append(
                _Member(
                    snapshot=snapshot,
                    task_id=task.id,
                    instruction=task.instruction,
                    family=fam,
                    difficulty=difficulty,
                )
            )

    def __len__(self) -> int:
        return len(self._members)

    def keys(self) -> set[tuple[str, str]]:
        return set(self._members)

    def snapshots(self) -> list[Snapshot]:
        by_id: dict[str, Snapshot] = {}
        for member in self._members.values():
            by_id.setdefault(member.snapshot.snapshot_id, member.snapshot)
        return list(by_id.values())

    def round_rows(self, *, groups: int, num_generations: int) -> list[PromptRow]:
        rows: list[PromptRow] = []
        for member in self._select(groups):
            for _ in range(num_generations):
                rows.append(
                    {
                        "prompt": [{"role": "user", "content": member.instruction}],
                        "snapshot_id": member.snapshot.snapshot_id,
                        "task_id": member.task_id,
                    }
                )
        return rows

    def _easy_tier(self) -> set[tuple[str, str]]:
        ranked = sorted(self._members.values(), key=lambda m: (m.difficulty, m.key))
        return {m.key for m in ranked[: max(1, len(ranked) // 3)]}

    def _select(self, groups: int) -> list[_Member]:
        groups = min(groups, len(self._members))
        if groups == 0:
            return []
        easy = self._easy_tier()
        floor_n = min(round(self._mix_floor * groups), len(easy))
        by_priority = sorted(self._members.values(), key=lambda m: (-m.priority, m.key))
        chosen: list[_Member] = []
        taken: set[tuple[str, str]] = set()
        for member in by_priority:
            if len(chosen) >= floor_n:
                break
            if member.key in easy:
                chosen.append(member)
                taken.add(member.key)
        for member in by_priority:
            if len(chosen) >= groups:
                break
            if member.key not in taken:
                chosen.append(member)
                taken.add(member.key)
        return chosen

    def update(
        self,
        reports: RoundReports,
        *,
        pack: Pack,
        policy: CurriculumPolicy = direction_from_reports,
        gate: EvolutionGate | None = None,
        evolve_top: int = 1,
        max_repairs: int = 2,
    ) -> None:
        for member in self._members.values():
            ran = reports.get(member.key)
            if ran:
                member.priority = _member_priority(ran)
            else:
                member.priority += _STALENESS_STEP
        self._grow(reports, pack, policy, gate, evolve_top, max_repairs)
        self._bound()

    def _grow(
        self,
        reports: RoundReports,
        pack: Pack,
        policy: CurriculumPolicy,
        gate: EvolutionGate | None,
        evolve_top: int,
        max_repairs: int,
    ) -> None:
        ran = sorted(
            (m for m in self._members.values() if reports.get(m.key)),
            key=lambda m: (-m.priority, m.key),
        )
        for member in ran[:evolve_top]:
            child = auto_evolve(
                member.snapshot,
                *reports[member.key],
                pack=pack,
                policy=policy,
                gate=gate,
                max_repairs=max_repairs,
            )
            if child is None:
                continue
            difficulty = self._difficulty_fn(child)
            for task in child.tasks:
                if str(task.meta.get("family", "")) != member.family:
                    continue
                key = (child.snapshot_id, task.id)
                if key not in self._members:
                    self._members[key] = _Member(
                        snapshot=child,
                        task_id=task.id,
                        instruction=task.instruction,
                        family=member.family,
                        difficulty=difficulty,
                    )
                break

    def _bound(self) -> None:
        protected = self._easy_tier()
        evictable = sorted(
            (m for m in self._members.values() if m.key not in protected),
            key=lambda m: (m.priority, m.key),
        )
        while len(self._members) > self._max_size and evictable:
            del self._members[evictable.pop(0).key]


def run_pool_curriculum(
    pool: WorldPool,
    run_round: RunRound,
    *,
    rounds: int,
    pack: Pack,
    groups: int,
    num_generations: int,
    policy: CurriculumPolicy = direction_from_reports,
    gate: EvolutionGate | None = None,
    evolve_top: int = 1,
) -> WorldPool:
    for _ in range(rounds):
        rows = pool.round_rows(groups=groups, num_generations=num_generations)
        reports = run_round(rows, pool.snapshots())
        pool.update(reports, pack=pack, policy=policy, gate=gate, evolve_top=evolve_top)
    return pool
