"""One agent loop for both training and evaluation.

A gym owns worlds and the grade; this is the thin, framework-free agent loop that
turns a realized world into a graded trajectory. ``run_agent`` drives one
episode — brief the agent, sample an action, run it against the live world,
observe, repeat — then grades it at ``stop_episode``. Training and evaluation use
the *same* loop and differ only in the injected :class:`Sampler`: a training
sampler fills :attr:`SampleResult.logprobs` / :attr:`SampleResult.completion_token_ids`,
an evaluation sampler leaves them ``None``. The loop never grades the agent — the
consequence verifier runs independently at stop.

The agent acts through one primitive: a shell ``run`` capability bound to the
world's surface by the caller (``bind_run``). Core never binds docker itself, so a
container world brings a hardened sandbox and a local world brings whatever the
caller chooses; the agent composes ``curl`` / ``python`` itself and the harness
ships no per-verb tools.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from openrange_pack_sdk import Snapshot

from openrange.core.episode import AgentTurn, EpisodeReport, EpisodeService
from openrange.core.sandbox import CommandResult
from openrange.runtime import EpisodeContext
from openrange.training import (
    Reward,
    Trajectory,
    episode_reward,
    episode_trajectory,
)


class AgentError(RuntimeError):
    """The agent loop could not act — surfaced, never swallowed."""


@dataclass(frozen=True, slots=True)
class SampleResult:
    """One model completion. ``text`` is the only field the loop reads;
    ``completion_token_ids`` and ``logprobs`` are populated only by a training
    sampler (an evaluation sampler leaves them ``None``). That single difference
    is the whole train-vs-eval split — the loop is identical either way."""

    text: str
    completion_token_ids: tuple[int, ...] | None = None
    logprobs: tuple[float, ...] | None = None


@runtime_checkable
class Sampler(Protocol):
    """Turns a prompt into a completion. Eval and train differ only in whether
    the returned :class:`SampleResult` carries token-level signal."""

    def complete(self, prompt: str, *, system: str | None = None) -> SampleResult: ...


@runtime_checkable
class RunCapability(Protocol):
    """A bound shell the agent acts through: ``run`` one command, ``close`` when
    the episode ends. A hardened sandbox satisfies it."""

    def run(self, command: str, *, timeout: float) -> CommandResult: ...

    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class AgentAction:
    tool: str
    command: str = ""


@dataclass(frozen=True, slots=True)
class RolloutStep:
    """One turn the policy produced. ``sample`` is policy-generated — the only
    tokens a trainer learns from; ``prompt`` (the context shown) and ``output``
    (the tool result) are the spans a trainer masks. ``command`` is the parsed
    shell action, ``None`` when the agent finished."""

    prompt: str
    sample: SampleResult
    command: str | None
    output: str | None


@dataclass(frozen=True, slots=True)
class AgentRollout:
    """The trace one :func:`run_agent` produced, plus the graded report. ``steps``
    is the policy-vs-observation record a trainer consumes; ``turns`` is what was
    recorded on the episode (replayable via ``episode_trajectory``)."""

    snapshot_id: str
    task_id: str
    steps: tuple[RolloutStep, ...]
    turns: tuple[AgentTurn, ...]
    report: EpisodeReport
    reward: Reward
    terminal_reason: str

    @property
    def success(self) -> bool:
        return self.report.passed

    @property
    def trajectory(self) -> Trajectory:
        return episode_trajectory(self.report, self.turns)


def agent_briefing(ctx: EpisodeContext) -> str:
    """The task plus the live interface contract, for any harness's agent.

    A static instruction names a path, not the dynamic host:port a realized world
    binds; the briefing adds where the world actually is so the agent can reach
    it."""
    parts = [ctx.task.instruction]
    base_url = ctx.surface.get("base_url")
    solver_root = ctx.surface.get("solver_root")
    if isinstance(base_url, str):
        parts.append(
            f"The target web service is running at {base_url} — "
            "interact with it over HTTP."
        )
    elif solver_root is not None:
        parts.append(f"You are working in the directory {solver_root}.")
    return "\n\n".join(parts)


_ACTION_BLOCK = re.compile(r"```(run_shell|finish)\n(.*?)```", re.DOTALL)


def parse_action(text: str) -> AgentAction:
    """Parse one action from a model reply: a fenced ```run_shell``` / ```finish```
    block. A reply with no recognized block becomes a ``finish`` carrying the whole
    text, so a model that ignores the protocol still terminates rather than loops."""
    match = _ACTION_BLOCK.search(text)
    if match is None:
        return AgentAction(tool="finish", command=text.strip())
    return AgentAction(tool=match.group(1), command=match.group(2).strip())


def run_shell(
    surface: Mapping[str, Any], command: str, *, timeout: float = 120.0
) -> str:
    """Run one shell command through the surface's bound ``run`` capability and
    return its combined output."""
    run = surface.get("run")
    if not callable(run):
        raise AgentError(
            "surface has no 'run' capability; bind one via run_agent(bind_run=...)"
        )
    result: CommandResult = run(command, timeout=timeout)
    return result.output


_OBSERVATION = "Tool output:"


async def arun_agent(
    service: EpisodeService,
    snapshot: Snapshot,
    sampler: Sampler,
    *,
    bind_run: Callable[[Mapping[str, Any]], RunCapability],
    task_id: str | None = None,
    max_turns: int = 8,
    system_prompt: str | None = None,
    reward_fn: Callable[[EpisodeReport], Reward] = episode_reward,
) -> AgentRollout:
    """Drive one episode with ``sampler`` over the shell tool and grade it.

    The loop runs on the calling event loop while the blocking work — the
    sampler, the shell ``run``, and binding/closing the capability — is offloaded
    with :func:`asyncio.to_thread`, so :func:`arun_rollouts` overlaps many
    episodes' tool waits on one shared service. ``EpisodeService`` is touched only
    from the loop thread, so no locking is needed; concurrent use therefore
    targets rollout worlds with the auto-tick off. Train and eval differ only in
    ``sampler``; the service is left open so its warm pool survives across
    rollouts — the caller closes it.
    """
    handle = service.start_episode(snapshot, task_id)
    surface = service.surface(handle)
    capability = await asyncio.to_thread(bind_run, surface)
    steps: list[RolloutStep] = []
    turns: list[AgentTurn] = []
    terminal_reason = "max_turns"
    try:
        task = next(t for t in snapshot.tasks if t.id == handle.task_id)
        bound = {**surface, "run": capability.run}
        prompt = agent_briefing(EpisodeContext(task=task, surface=bound))
        for _ in range(max_turns):
            sample = await asyncio.to_thread(
                sampler.complete, prompt, system=system_prompt
            )
            action = parse_action(sample.text)
            if action.tool == "finish":
                turn = AgentTurn(
                    message=action.command or sample.text,
                    tool_calls=(
                        {"tool": "finish", "args": {"answer": action.command}},
                    ),
                )
                service.record_turn(handle, turn)
                turns.append(turn)
                steps.append(RolloutStep(prompt, sample, None, None))
                terminal_reason = "finished"
                break
            output = await asyncio.to_thread(run_shell, bound, action.command)
            turn = AgentTurn(
                message=sample.text,
                tool_calls=(
                    {"tool": "run_shell", "args": {"command": action.command}},
                ),
                tool_results=({"output": output},),
            )
            service.record_turn(handle, turn)
            turns.append(turn)
            steps.append(RolloutStep(prompt, sample, action.command, output))
            prompt = f"{prompt}\n\n{sample.text}\n\n{_OBSERVATION}\n{output}"
        report = service.stop_episode(handle)
        return AgentRollout(
            snapshot_id=report.snapshot_id,
            task_id=report.task_id,
            steps=tuple(steps),
            turns=tuple(turns),
            report=report,
            reward=reward_fn(report),
            terminal_reason=terminal_reason,
        )
    finally:
        await asyncio.to_thread(capability.close)


def run_agent(
    service: EpisodeService,
    snapshot: Snapshot,
    sampler: Sampler,
    *,
    bind_run: Callable[[Mapping[str, Any]], RunCapability],
    task_id: str | None = None,
    max_turns: int = 8,
    system_prompt: str | None = None,
    reward_fn: Callable[[EpisodeReport], Reward] = episode_reward,
) -> AgentRollout:
    """Synchronous one-episode driver — :func:`arun_agent` on a private loop."""
    return asyncio.run(
        arun_agent(
            service,
            snapshot,
            sampler,
            bind_run=bind_run,
            task_id=task_id,
            max_turns=max_turns,
            system_prompt=system_prompt,
            reward_fn=reward_fn,
        )
    )


async def arun_rollouts(
    service: EpisodeService,
    snapshot: Snapshot,
    sampler: Sampler,
    *,
    bind_run: Callable[[Mapping[str, Any]], RunCapability],
    task_ids: Sequence[str] | None = None,
    max_concurrency: int = 8,
    **kwargs: Any,
) -> list[AgentRollout]:
    """Run many episodes on one shared service, overlapping their tool waits up to
    ``max_concurrency`` at a time (all tasks when ``task_ids`` is ``None``; repeat
    an id to run several rollouts of one task). The shared ``sampler`` must be safe
    to call concurrently — a server-mode client is."""
    ids = list(task_ids) if task_ids is not None else [t.id for t in snapshot.tasks]
    limit = asyncio.Semaphore(max_concurrency)

    async def _one(task_id: str) -> AgentRollout:
        async with limit:
            return await arun_agent(
                service,
                snapshot,
                sampler,
                bind_run=bind_run,
                task_id=task_id,
                **kwargs,
            )

    return list(await asyncio.gather(*(_one(tid) for tid in ids)))


def run_rollouts(
    service: EpisodeService,
    snapshot: Snapshot,
    sampler: Sampler,
    *,
    bind_run: Callable[[Mapping[str, Any]], RunCapability],
    task_ids: Sequence[str] | None = None,
    **kwargs: Any,
) -> list[AgentRollout]:
    """Synchronous sequential rollouts — :func:`arun_rollouts` with no overlap,
    reusing the warm pool across episodes."""
    return asyncio.run(
        arun_rollouts(
            service,
            snapshot,
            sampler,
            bind_run=bind_run,
            task_ids=task_ids,
            max_concurrency=1,
            **kwargs,
        )
    )
