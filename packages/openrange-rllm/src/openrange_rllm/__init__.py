"""rLLM AgentTrainer adapter for OpenRange — import-light.

OpenRange owns the world and the grade; rLLM owns the RL training loop. This thin
adapter maps one OpenRange agent rollout onto rLLM's ``Episode`` / ``Trajectory``
/ ``Step`` shapes and wraps the harness as an ``@rllm.rollout`` flow plus an
``@rllm.evaluator``. ``import openrange_rllm`` pulls none of rLLM — every rLLM
import is local to the function that needs it, so the module loads on a plain
machine and only the gated live training path imports the trainer.

The train-vs-eval split is just which endpoint the sampler points at. rLLM runs
the policy behind an OpenAI-compatible *gateway* that records each call's token
ids and logprobs; the rollout therefore leaves the per-token ``Step`` fields
empty and rLLM's trace enrichment fills them. So the adapter ships
``GatewaySampler`` — a :class:`~openrange.Sampler` that calls ``config.base_url``
through OpenRange's own OpenAI-compatible backend, and the gateway captures the
training signal.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import TYPE_CHECKING, Any

from openrange_pack_sdk import LLMRequest, Snapshot

from openrange import (
    AgentRollout,
    EpisodeService,
    SampleResult,
    arun_agent,
    episode_reward,
)
from openrange.llm import OpenAICompatibleBackend

if TYPE_CHECKING:
    from rllm.types import AgentConfig, Episode, Task

    from openrange import Reward, RunCapability, Sampler
    from openrange.core.episode import EpisodeReport

__all__ = [
    "GatewaySampler",
    "agent_rollout_to_episode",
    "build_rllm_dataset_rows",
    "make_evaluator",
    "make_rollout",
    "snapshot_resolver",
]


class GatewaySampler:
    """A :class:`~openrange.Sampler` that calls rLLM's gateway at
    ``config.base_url``. Token ids and logprobs are captured by the gateway, not
    here, so the returned :class:`~openrange.SampleResult` carries text only — the
    same loop serves eval by pointing ``base_url`` at any endpoint."""

    def __init__(self, config: AgentConfig) -> None:
        self._backend = OpenAICompatibleBackend(
            base_url=config.base_url, model=config.model
        )

    def complete(self, prompt: str, *, system: str | None = None) -> SampleResult:
        result = self._backend.complete(LLMRequest(prompt=prompt, system=system))
        return SampleResult(result.text)


def agent_rollout_to_episode(rollout: AgentRollout, *, task: Any = None) -> Episode:
    """Map one OpenRange rollout onto an rLLM ``Episode``.

    Each harness turn (one model call) becomes one ``Step`` carrying its action,
    observation and the model text, in call order so rLLM's gateway traces enrich
    them 1:1. The verifier's scalar lands on the trajectory and in ``artifacts``
    for the evaluator; the per-token fields stay empty for the gateway to fill."""
    from rllm.types import Episode, Step, Trajectory

    last = len(rollout.steps) - 1
    steps = [
        Step(
            action=step.command if step.command is not None else "finish",
            observation=step.output,
            model_response=step.sample.text,
            thought=step.sample.text,
            done=index == last,
            chat_completions=[
                {"role": "user", "content": step.prompt},
                {"role": "assistant", "content": step.sample.text},
            ],
        )
        for index, step in enumerate(rollout.steps)
    ]
    components = {
        name: float(value) for name, value in rollout.reward.components.items()
    }
    return Episode(
        task=task,
        is_correct=rollout.success,
        # Leave Episode.termination_reason unset — rLLM fills it with its own
        # TerminationReason enum (which lives behind its heavy engine import), and
        # its verl transform does ``reason.value``, so a raw string would crash
        # there. The value is preserved in ``artifacts`` below.
        trajectories=[
            Trajectory(name=rollout.task_id, steps=steps, reward=rollout.reward.scalar)
        ],
        artifacts={
            "reward": rollout.reward.scalar,
            "components": components,
            "snapshot_id": rollout.snapshot_id,
            "task_id": rollout.task_id,
            "terminal_reason": rollout.terminal_reason,
        },
        metrics=components,
    )


def build_rllm_dataset_rows(
    snapshots: Iterable[Snapshot], *, family: str
) -> list[dict[str, Any]]:
    """Turn admitted worlds into rLLM dataset rows for ``DatasetRegistry``.

    One row per task of ``family``. rLLM copies the whole row into
    ``Task.metadata`` (and ``Task.id`` is a fresh uuid on the verl backend), so
    each row carries ``snapshot_id`` / ``task_id`` for :func:`snapshot_resolver`
    to read back, and ``instruction`` becomes ``Task.instruction``."""
    rows: list[dict[str, Any]] = []
    for snapshot in snapshots:
        for task in snapshot.tasks:
            if task.meta.get("family") != family:
                continue
            rows.append(
                {
                    "id": f"{snapshot.snapshot_id}:{task.id}",
                    "instruction": task.instruction,
                    "snapshot_id": snapshot.snapshot_id,
                    "task_id": task.id,
                }
            )
    return rows


def snapshot_resolver(
    snapshots: Iterable[Snapshot],
) -> Callable[[Task], tuple[Snapshot, str]]:
    """A ``resolve`` for :func:`make_rollout`, keyed on the rows
    :func:`build_rllm_dataset_rows` emits — maps an rLLM ``Task`` back to its
    snapshot and task id through ``task.metadata``."""
    by_id = {snapshot.snapshot_id: snapshot for snapshot in snapshots}

    def resolve(task: Task) -> tuple[Snapshot, str]:
        metadata = task.metadata
        return by_id[metadata["snapshot_id"]], str(metadata["task_id"])

    return resolve


def make_rollout(
    service: EpisodeService,
    resolve: Callable[[Task], tuple[Snapshot, str | None]],
    *,
    bind_run: Callable[[Mapping[str, Any]], RunCapability],
    sampler_factory: Callable[[AgentConfig], Sampler] = GatewaySampler,
    reward_fn: Callable[[EpisodeReport], Reward] = episode_reward,
    **harness_kwargs: Any,
) -> Any:
    """Wrap the harness as an ``@rllm.rollout`` flow.

    ``resolve`` turns an rLLM ``Task`` into the OpenRange snapshot (and task id)
    to run — typically a lookup in a snapshot store keyed by ``task.metadata``.
    The flow runs one real episode on the shared ``service`` with a gateway-bound
    sampler and returns the graded ``Episode``."""
    from rllm.eval.rollout_decorator import rollout as rllm_rollout

    async def _flow(task: Task, config: AgentConfig) -> Episode:
        snapshot, task_id = resolve(task)
        sampler = sampler_factory(config)
        rollout = await arun_agent(
            service,
            snapshot,
            sampler,
            bind_run=bind_run,
            task_id=task_id,
            reward_fn=reward_fn,
            **harness_kwargs,
        )
        return agent_rollout_to_episode(rollout, task=task)

    return rllm_rollout(_flow, name="openrange")


def make_evaluator() -> Any:
    """Wrap the verifier's grade as an ``@rllm.evaluator``.

    The grade is intrinsic to the OpenRange episode — the consequence verifier
    ran at stop — so the rollout already stamped the scalar into ``artifacts``;
    this surfaces it as the ``EvalOutput`` rLLM writes back onto trajectories."""
    from rllm.eval.rollout_decorator import evaluator as rllm_evaluator
    from rllm.eval.types import EvalOutput, Signal

    def _evaluate(task: Any, episode: Episode) -> EvalOutput:
        reward = float(episode.artifacts.get("reward", 0.0))
        signals = [
            Signal(name=name, value=float(value))
            for name, value in episode.artifacts.get("components", {}).items()
        ]
        return EvalOutput(reward=reward, is_correct=episode.is_correct, signals=signals)

    return rllm_evaluator(_evaluate)
