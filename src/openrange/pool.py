"""The curriculum as an evolving pool of worlds-and-tasks.

See ``docs/design/evolving-pool-curriculum.md``. Trainer-agnostic: it depends
only on OpenRange core (admission, ``auto_evolve``) and the episode report, never
on a training backend, so any adapter (``openrange-trl``, …) drives it through
the caller-supplied ``run_round``. A mix floor keeps an easy tail in every round.
Difficulty is injected by the caller so the pool stays pack-agnostic.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass

from openrange_pack_sdk import Mutation, Pack, Snapshot

from openrange.core.admit import AdmissionFailure, admit
from openrange.core.curriculum import (
    CurriculumPolicy,
    EvolutionGate,
    SeedGate,
    auto_evolve,
    direction_from_reports,
)
from openrange.core.episode import EpisodeReport
from openrange.training import Reward, episode_reward

RewardFn = Callable[[EpisodeReport], Reward]

DifficultyFn = Callable[[Snapshot], float]
PromptRow = dict[str, object]
RoundReports = Mapping[tuple[str, str], Sequence[EpisodeReport]]
RunRound = Callable[[list[PromptRow], list[Snapshot]], RoundReports]
GateFactory = Callable[[Snapshot], EvolutionGate]

_STALENESS_STEP = 0.1
# Fresh children seat at this cap so a frontier world survives a round before eviction.
_MAX_PRIORITY = 2.0


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


def _gated_members(
    pack: Pack,
    manifests: Sequence[Mapping[str, object]],
    *,
    difficulty_fn: DifficultyFn,
    family: str | None,
    max_repairs: int,
    seed_gate: SeedGate | None,
) -> list[_Member]:
    members: list[_Member] = []
    for manifest in manifests:
        result = admit(pack, dict(manifest), max_repairs=max_repairs)
        if isinstance(result, AdmissionFailure):
            continue
        if seed_gate is not None and not seed_gate(result):
            continue
        _members_of(result, difficulty_fn(result), family, members)
    return members


def _rows_for(members: Iterable[_Member], num_generations: int) -> list[PromptRow]:
    rows: list[PromptRow] = []
    for member in members:
        for _ in range(num_generations):
            rows.append(
                {
                    "prompt": [{"role": "user", "content": member.instruction}],
                    "snapshot_id": member.snapshot.snapshot_id,
                    "task_id": member.task_id,
                }
            )
    return rows


def _snapshots_of(members: Iterable[_Member]) -> list[Snapshot]:
    by_id: dict[str, Snapshot] = {}
    for member in members:
        by_id.setdefault(member.snapshot.snapshot_id, member.snapshot)
    return list(by_id.values())


def _mean_pass_rate(report_groups: Iterable[Sequence[EpisodeReport]]) -> float:
    rates = [sum(1 for r in g if r.passed) / len(g) for g in report_groups if g]
    return sum(rates) / len(rates) if rates else 0.0


def _member_priority(
    reports: Sequence[EpisodeReport], reward_fn: RewardFn = episode_reward
) -> float:
    scalars = [reward_fn(r).scalar for r in reports]
    mean = sum(scalars) / len(scalars)
    learnability = 1.0 - abs(2.0 * mean - 1.0)
    gaps = []
    for report in reports:
        subgoals = report.episode_result.subgoals
        if subgoals:
            achieved = sum(1 for hit in subgoals.values() if hit)
            gaps.append(1.0 - achieved / len(subgoals))
    # Regret keeps a partly-solved world at the frontier that low reward-spread alone
    # would retire.
    regret = sum(gaps) / len(gaps) if gaps else 0.0
    return learnability + regret


def _compose_gate(
    gate: EvolutionGate | None,
    gate_factory: GateFactory | None,
    parent: Snapshot,
) -> EvolutionGate | None:
    built = gate_factory(parent) if gate_factory is not None else None
    gates = [g for g in (gate, built) if g is not None]
    if not gates:
        return None

    def combined(evolved: Snapshot, mutation: Mutation) -> bool:
        return all(g(evolved, mutation) for g in gates)

    return combined


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
        self._last_difficulty_gain: float | None = None

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
        seed_gate: SeedGate | None = None,
    ) -> WorldPool:
        members = _gated_members(
            pack,
            manifests,
            difficulty_fn=difficulty_fn,
            family=family,
            max_repairs=max_repairs,
            seed_gate=seed_gate,
        )
        return cls(
            members, difficulty_fn=difficulty_fn, max_size=max_size, mix_floor=mix_floor
        )

    def __len__(self) -> int:
        return len(self._members)

    def keys(self) -> set[tuple[str, str]]:
        return set(self._members)

    def snapshots(self) -> list[Snapshot]:
        return _snapshots_of(self._members.values())

    def round_rows(self, *, groups: int, num_generations: int) -> list[PromptRow]:
        return _rows_for(self._select(groups), num_generations)

    def _easy_tier(self) -> set[tuple[str, str]]:
        ranked = sorted(self._members.values(), key=lambda m: (m.difficulty, m.key))
        return {m.key for m in ranked[: max(1, len(ranked) // 3)]}

    def _select(self, groups: int) -> list[_Member]:
        groups = min(groups, len(self._members))
        if groups == 0:
            return []
        easy = self._easy_tier()
        floor_n = min(round(self._mix_floor * groups), len(easy), groups)
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
        gate_factory: GateFactory | None = None,
        evolve_top: int = 1,
        max_repairs: int = 2,
        reward_fn: RewardFn = episode_reward,
    ) -> bool:
        for member in self._members.values():
            ran = reports.get(member.key)
            if ran:
                member.priority = _member_priority(ran, reward_fn)
            else:
                member.priority = min(member.priority + _STALENESS_STEP, _MAX_PRIORITY)
        grown, capped, gain = self._grow(
            reports, pack, policy, gate, gate_factory, evolve_top, max_repairs
        )
        self._last_difficulty_gain = gain
        self._bound(grown)
        return capped

    def _grow(
        self,
        reports: RoundReports,
        pack: Pack,
        policy: CurriculumPolicy,
        gate: EvolutionGate | None,
        gate_factory: GateFactory | None,
        evolve_top: int,
        max_repairs: int,
    ) -> tuple[set[tuple[str, str]], bool, float | None]:
        grown: set[tuple[str, str]] = set()
        capped = False
        gains: list[float] = []
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
                gate=_compose_gate(gate, gate_factory, member.snapshot),
                max_repairs=max_repairs,
            )
            if child is None:
                # No admissible harder world passed the gate: the frontier is capped.
                capped = True
                continue
            difficulty = self._difficulty_fn(child)
            gains.append(difficulty - member.difficulty)
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
                        priority=_MAX_PRIORITY,
                    )
                    grown.add(key)
        return grown, capped, (max(gains) if gains else None)

    def _bound(self, protected_extra: set[tuple[str, str]]) -> None:
        protected = self._easy_tier() | protected_extra
        evictable = sorted(
            (m for m in self._members.values() if m.key not in protected),
            key=lambda m: (m.priority, m.key),
        )
        while len(self._members) > self._max_size and evictable:
            del self._members[evictable.pop(0).key]


@dataclass(frozen=True)
class RoundMetrics:
    train_solve_rate: float
    held_out_solve_rate: float | None = None
    frontier_capped: bool = False
    # Most any child advanced difficulty this round (signed; None if nothing evolved).
    # Unlike frontier_capped, near-zero here means children admit but only creep on
    # cosmetic decoys.
    difficulty_gain: float | None = None

    @property
    def generalization_gap(self) -> float | None:
        if self.held_out_solve_rate is None:
            return None
        return self.train_solve_rate - self.held_out_solve_rate


class EvalPool:
    """A held-out set of admitted worlds, measured each round but never sampled,
    evolved, or bounded, so train-vs-held-out solve-rate is the generalization
    signal (``docs/design/evolving-pool-curriculum.md`` §8).
    """

    def __init__(self, members: Sequence[_Member]) -> None:
        self._members = list(members)

    @classmethod
    def seed(
        cls,
        pack: Pack,
        manifests: Sequence[Mapping[str, object]],
        *,
        difficulty_fn: DifficultyFn,
        family: str | None = None,
        max_repairs: int = 2,
        seed_gate: SeedGate | None = None,
    ) -> EvalPool:
        members = _gated_members(
            pack,
            manifests,
            difficulty_fn=difficulty_fn,
            family=family,
            max_repairs=max_repairs,
            seed_gate=seed_gate,
        )
        return cls(members)

    def __len__(self) -> int:
        return len(self._members)

    def keys(self) -> set[tuple[str, str]]:
        return {m.key for m in self._members}

    def snapshots(self) -> list[Snapshot]:
        return _snapshots_of(self._members)

    def rows(self, *, num_generations: int) -> list[PromptRow]:
        return _rows_for(self._members, num_generations)

    def solve_rate(self, reports: RoundReports) -> float:
        return _mean_pass_rate(
            reports[m.key] for m in self._members if m.key in reports
        )


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
    gate_factory: GateFactory | None = None,
    evolve_top: int = 1,
    eval_pool: EvalPool | None = None,
    eval_round: RunRound | None = None,
    reward_fn: RewardFn = episode_reward,
) -> list[RoundMetrics]:
    """Run the curriculum, updating ``pool`` in place.

    When ``eval_pool`` is given it is measured each round but never trained on or
    evolved, so the train-vs-held-out gap is the generalization signal (§8). A
    scripted solver measures it with the same ``run_round``; a real trainer must
    pass an ``eval_round`` that rolls out and grades *without* a gradient step, or
    it would train on the held-out set and break the fence.

    ``reward_fn`` is the scalar the pool ranks worlds by (learnability/priority);
    pass the *same* one the trainer optimizes (the ``reward_fn`` given to
    ``openrange-trl``) so the pool evolves on the objective the policy is trained on.
    Defaults to :func:`episode_reward`. Evolution *direction* still tracks the pack's
    pass-rate (``check_success``), independent of the reward shaping.
    """
    measure = eval_round or run_round
    metrics: list[RoundMetrics] = []
    for _ in range(rounds):
        rows = pool.round_rows(groups=groups, num_generations=num_generations)
        reports = run_round(rows, pool.snapshots())
        held_out: float | None = None
        if eval_pool is not None and len(eval_pool):
            eval_reports = measure(
                eval_pool.rows(num_generations=num_generations), eval_pool.snapshots()
            )
            held_out = eval_pool.solve_rate(eval_reports)
        capped = pool.update(
            reports,
            pack=pack,
            policy=policy,
            gate=gate,
            gate_factory=gate_factory,
            evolve_top=evolve_top,
            reward_fn=reward_fn,
        )
        metrics.append(
            RoundMetrics(
                train_solve_rate=_mean_pass_rate(reports.values()),
                held_out_solve_rate=held_out,
                frontier_capped=capped,
                difficulty_gain=pool._last_difficulty_gain,
            )
        )
    return metrics
