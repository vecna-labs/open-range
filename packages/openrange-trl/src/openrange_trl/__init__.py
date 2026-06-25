"""TRL GRPO adapter — torch-free.

The whole adapter lives here and imports only ``openrange`` + stdlib, so
``import openrange_trl`` works with no ``torch`` installed and every piece below
is deterministically unit-testable without a model. Only the gated
``tests/test_trl_live.py`` and the example notebooks (``examples/trl_grpo_*``)
import ``trl`` / ``torch`` and build a real ``GRPOTrainer``.

OpenRange owns the world + the grade; it owns nothing of the agent runtime. So
the policy's **tools are brought by the caller** (the user's harness), not
hard-coded here. A tool is a plain callable taking the live episode ``surface``
first, then the model's kwargs — this package ships the *mechanism* only, no
tools (``examples/tools.py`` has reference shims for the shell-less in-process
policy). The core public pieces map onto TRL's agentic GRPO
(the ``environment_factory`` path, ``transformers>=5.2``):

- ``EpisodeEnv`` — one rollout's environment over an ``EpisodeService`` episode.
  Constructed with the caller's ``tools``; it synthesizes one TRL-introspectable
  method per tool (so the trainer reflects them as the policy's tool surface) and
  binds each to the live ``surface`` at ``reset``. The first read of
  ``env.reward`` (via the reward func) lazily stops + grades the episode.
- ``build_grpo_dataset`` — a snapshot's tasks → GRPO prompt rows, each tagged
  with ``snapshot_id`` / ``task_id`` so trajectories stay attributable across an
  ``auto_evolve`` curriculum. The model sees the tools via TRL's tool schemas, so
  a row carries just the task instruction.
- ``make_environment_factory`` — the per-rollout factory TRL calls; the caller
  passes the ``tools`` the policy gets.
- ``make_reward_func`` — the TRL-shaped reward bridge; defers entirely to the
  pack's structured grade via ``episode_reward`` (no reward logic reinvented).
- ``reward_variance_policy`` — a ``CurriculumPolicy`` keyed on the signal GRPO
  actually consumes (reward *spread*): when a group's spread collapses there is
  no gradient, so evolve.
"""

from __future__ import annotations

import copy
import inspect
import subprocess
import types
import uuid
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from openrange_pack_sdk import Backing, EpisodeReportLike, Pack, Snapshot

from openrange.core.curriculum import Direction
from openrange.core.episode import (
    AgentTurn,
    EpisodeHandle,
    EpisodeReport,
    EpisodeService,
)
from openrange.core.sandbox import (
    SANDBOX_LABEL,
    AgentSandbox,
    SandboxError,
    track_resource,
    untrack_resource,
)
from openrange.pool import PromptRow, RoundReports, RunRound
from openrange.training import (
    Reward,
    Trajectory,
    episode_reward,
    episode_trajectory,
)

Tool = Callable[..., str]

# Tail tool output so a chatty surface can't flood the context window.
_OUTPUT_TAIL = 2000


def _tool_method(env: EpisodeEnv, fn: Tool) -> Any:
    """Build a TRL-introspectable bound method from a user tool fn.

    TRL reflects an env's public methods into tools (schema from the signature +
    docstring). The tool takes the live ``surface`` first; we hand TRL the same
    method with that parameter dropped and inject ``self._surface`` at call time.
    """
    params = list(inspect.signature(fn).parameters.values())[1:]
    ns: dict[str, Any] = {"_fn": fn}
    decl, forward = "", ""
    for p in params:
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD, p.POSITIONAL_ONLY):
            raise ValueError(
                f"tool {fn.__name__!r} parameter {p.name!r} must be "
                f"positional-or-keyword or keyword-only, not {p.kind.name}"
            )
        ns[f"_ann_{p.name}"] = p.annotation if p.annotation is not p.empty else str
        decl += f", {p.name}: _ann_{p.name}"
        if p.default is not p.empty:
            ns[f"_def_{p.name}"] = p.default
            decl += f" = _def_{p.name}"
        forward += f", {p.name}={p.name}"
    exec(  # noqa: S102 - source is built from the tool's own signature, not input
        f"def {fn.__name__}(self{decl}):\n    return self._invoke(_fn{forward})\n",
        ns,
    )
    method = ns[fn.__name__]
    method.__doc__ = fn.__doc__
    method.__annotations__["return"] = str
    return types.MethodType(method, env)


class EpisodeEnv:
    """One GRPO rollout over a single ``EpisodeService`` episode.

    Each caller-supplied tool becomes a public method TRL reflects as a tool,
    bound to the live ``surface`` at ``reset``; tool calls are fail-soft (a bad
    call costs reward, not the run). The first read of ``env.reward`` (via the
    reward func) lazily stops + grades the episode.

    With ``sandbox=True`` each episode gets its own throwaway :class:`AgentSandbox`
    (the agent's machine), and the live ``surface`` carries a ``run`` capability so a
    brought tool can run commands there — the trainer never runs an agent command.
    """

    def __init__(
        self,
        *,
        service: EpisodeService,
        snapshots: Mapping[str, Snapshot],
        tools: Sequence[Tool] = (),
        reward_fn: Callable[[EpisodeReport], Reward] = episode_reward,
        sandbox: bool = False,
    ) -> None:
        self.service = service
        self.snapshots = dict(snapshots)
        self.reward_fn = reward_fn
        self.reward: float = 0.0
        self.turns: list[AgentTurn] = []
        self.report: EpisodeReport | None = None
        self._handle: EpisodeHandle | None = None
        self._surface: Mapping[str, Any] | None = None
        self._finalized = False
        self._use_sandbox = sandbox
        self._sandbox: AgentSandbox | None = None
        self._network: str | None = None
        self._target_container: str | None = None
        self._tools: dict[str, Tool] = {}
        for fn in tools:
            if fn.__name__ in self._tools:
                raise ValueError(f"duplicate tool name: {fn.__name__!r}")
            # setattr below would silently shadow a same-named method/attribute.
            if hasattr(self, fn.__name__):
                raise ValueError(
                    f"tool name {fn.__name__!r} collides with an EpisodeEnv member"
                )
            self._tools[fn.__name__] = fn
            setattr(self, fn.__name__, _tool_method(self, fn))

    if TYPE_CHECKING:
        # Tools are attached dynamically, so the type checker can't see them;
        # declare any such access as a string-returning tool call.
        def __getattr__(self, name: str) -> Callable[..., str]: ...

    def reset(
        self,
        *,
        snapshot_id: str | None = None,
        task_id: str | None = None,
        **_: object,
    ) -> str:
        """Start a fresh episode; the returned observation (the live target URL or
        workspace listing the dataset can't know) is appended to the prompt.
        ``snapshot_id`` / ``task_id`` come from the dataset row.
        """
        self._teardown_sandbox()
        snapshot = self._resolve_snapshot(snapshot_id)
        handle = self.service.start_episode(snapshot, task_id)
        self._handle = handle
        surface = self.service.surface(handle)
        self._surface = self._start_sandbox(surface) if self._use_sandbox else surface
        self.reward = 0.0
        self.turns = []
        self.report = None
        self._finalized = False
        return self._initial_observation()

    def close(self) -> None:
        """Release the env's resources: tear down any sandbox and close the underlying
        ``EpisodeService`` (stopping live episodes and warm-pooled worlds). TRL never
        closes the envs it builds from a factory, so the factory's owner must."""
        self._teardown_sandbox()
        self.service.close()

    def _initial_observation(self) -> str:
        surface = self._surface or {}
        base_url = surface.get("base_url")
        if isinstance(base_url, str):
            return (
                f"A web service is running at {base_url}. Probe it with the "
                "available tools, then submit your answer."
            )
        solver_root = surface.get("solver_root")
        if solver_root is not None:
            names = sorted(p.name for p in Path(str(solver_root)).iterdir())
            return f"Workspace ready at {solver_root}. Files:\n" + "\n".join(names)
        return "Environment ready. Use the available tools."

    def _invoke(self, fn: Tool, **kwargs: Any) -> str:
        # str(): a non-str return must not crash the slice/_record below.
        out = self._safe(lambda: str(fn(self._require_surface(), **kwargs)))
        self._record(fn.__name__, kwargs, out)
        return out[-_OUTPUT_TAIL:]

    def _require_surface(self) -> Mapping[str, Any]:
        if self._surface is None:
            raise RuntimeError("tool called before reset()")
        return self._surface

    def _start_sandbox(self, surface: Mapping[str, Any]) -> Mapping[str, Any]:
        """Give the agent its own sandbox for this episode and hand it to the tools.

        The agent's tools run here, never in the trainer. An HTTP world (``base_url``)
        is a network target: the sandbox joins a private per-episode network the world
        is also on, so the agent reaches it by alias over the wire (not the host). A
        code world (``solver_root``) is mounted so the agent edits it as its own files.
        Either way the live ``run`` is injected into the surface, so a brought tool can
        call ``surface["run"](command)`` with the trainer unchanged.
        """
        base_url = surface.get("base_url")
        if isinstance(base_url, str):
            target = surface.get("target_container")
            if not isinstance(target, str):
                raise SandboxError(
                    "a sandboxed HTTP world needs a containerized target (CONTAINER "
                    "backing) so the sandbox can join its network and reach it by alias"
                )
            network = f"openrange-agent-net-{uuid.uuid4().hex[:12]}"
            # --internal: the network has no gateway, so the sandbox (running untrusted
            # agent code) can reach the target by alias yet CANNOT reach the host, the
            # internet, or other episodes' host-published ports. The label makes a
            # leaked network prunable; record + track the name before connect so a
            # failed connect still tears it down (here and via the atexit sweep).
            _run_docker(
                "network", "create", "--internal", "--label", SANDBOX_LABEL, network
            )
            self._network = network
            track_resource("network", network)
            _run_docker("network", "connect", "--alias", "target", network, target)
            self._target_container = target
            target_url = f"http://target:{surface.get('target_port', '8000')}"
            self._sandbox = AgentSandbox({"base_url": target_url}, network=network)
            self._sandbox.start()
            # The agent reaches the target by its in-network alias, not the host URL.
            return {**surface, "base_url": target_url, "run": self._sandbox.run}
        self._sandbox = AgentSandbox({"solver_root": surface.get("solver_root")})
        self._sandbox.start()
        return {**surface, "run": self._sandbox.run}

    def _teardown_sandbox(self) -> None:
        # Disposable: the sandbox dies with the episode so no state leaks to the next.
        if self._sandbox is not None:
            self._sandbox.close()
            self._sandbox = None
        if self._network is not None:
            # Detach the world (best-effort: stop_episode usually removed it already),
            # then drop the network so nothing dangles even on an un-finalized re-reset.
            if self._target_container is not None:
                _run_docker(
                    "network",
                    "disconnect",
                    "-f",
                    self._network,
                    self._target_container,
                    check=False,
                )
            _run_docker("network", "rm", self._network, check=False)
            untrack_resource("network", self._network)
            self._network = None
            self._target_container = None

    def _finalize(self) -> None:
        # Idempotent: the reward func may read env.reward more than once, and
        # stop_episode caches, so a double read is safe.
        if self._finalized or self._handle is None:
            self._finalized = True
            return
        self._finalized = True
        report = self.service.stop_episode(self._handle)
        self.report = report
        self.reward = self.reward_fn(report).scalar
        self._teardown_sandbox()

    def _resolve_snapshot(self, snapshot_id: str | None) -> Snapshot:
        if snapshot_id is not None:
            snapshot = self.snapshots.get(snapshot_id)
            if snapshot is None:
                raise KeyError(f"unknown snapshot_id {snapshot_id!r}")
            return snapshot
        if len(self.snapshots) == 1:
            return next(iter(self.snapshots.values()))
        raise ValueError(
            "reset() needs a snapshot_id when multiple snapshots are registered"
        )

    @staticmethod
    def _safe(fn: Callable[[], str]) -> str:
        try:
            return fn()
        except Exception as exc:  # fail-soft: a bad tool call costs reward only
            return f"error: {exc}"

    def _record(self, tool: str, args: Mapping[str, Any], result: str) -> None:
        turn = AgentTurn(
            tool_calls=({"tool": tool, "args": dict(args)},),
            tool_results=({"output": result},),
        )
        self.turns.append(turn)
        if self._handle is not None:
            self.service.record_turn(self._handle, turn)


def _run_docker(*args: str, check: bool = True) -> None:
    subprocess.run(["docker", *args], check=check, capture_output=True, timeout=60)


def build_grpo_dataset(snapshot: Snapshot, *, repeat: int = 1) -> list[dict[str, Any]]:
    """Turn a snapshot's tasks into GRPO prompt rows.

    One row per task (optionally ``repeat``-ed so a round has enough prompts):
    ``{"prompt": [{"role": "user", "content": task.instruction}], "snapshot_id",
    "task_id"}``. ``snapshot_id`` / ``task_id`` ride along as dataset columns —
    TRL forwards them to ``reset`` (which episode to start) and to the reward
    func, and they tag the exported trajectory to the exact (possibly evolved)
    world. The policy sees the available tools via TRL's tool schemas (the chat
    template), so the row carries only the task instruction. Torch-free; the live
    example wraps the rows in a ``datasets.Dataset``.
    """
    rows: list[dict[str, Any]] = []
    for _ in range(max(1, repeat)):
        for task in snapshot.tasks:
            rows.append(
                {
                    "prompt": [{"role": "user", "content": task.instruction}],
                    "snapshot_id": snapshot.snapshot_id,
                    "task_id": task.id,
                }
            )
    return rows


def make_reward_func(
    collector: dict[tuple[str, str], list[EpisodeReport]] | None = None,
) -> Callable[..., list[float]]:
    """Return a TRL-shaped ``reward_func(prompts, completions, ...)``.

    In the agentic path TRL passes the rollouts' ``environments``; this finalizes
    each (lazily stopping + grading the episode) and returns ``[env.reward, ...]``
    in order. All reward logic is the pack's structured grade shaped by
    ``episode_reward`` — the trainer only *reads* it. When ``collector`` is given,
    each graded report is also recorded under its ``(snapshot_id, task_id)``, so a
    curriculum reads one round's reports back from a multi-world batch (the trainer
    itself keeps only the last episode per env slot).
    """

    def reward_func(
        prompts: object = None,
        completions: object = None,
        completion_ids: object = None,
        *,
        environments: Sequence[EpisodeEnv] | None = None,
        **kwargs: object,
    ) -> list[float]:
        rewards: list[float] = []
        for env in environments or ():
            env._finalize()
            if collector is not None and env.report is not None:
                key = (env.report.snapshot_id, env.report.task_id)
                collector.setdefault(key, []).append(env.report)
            rewards.append(float(env.reward))
        return rewards

    return reward_func


def make_environment_factory(
    pack: Pack,
    snapshots: Sequence[Snapshot],
    run_root: str | Path,
    *,
    tools: Sequence[Tool],
    reward_fn: Callable[[EpisodeReport], Reward] = episode_reward,
    backing: Backing = Backing.PROCESS,
    sandbox: bool = False,
) -> Callable[[], EpisodeEnv]:
    """Build the zero-arg factory TRL calls once per rollout slot.

    The caller (the user's harness) supplies ``tools`` — the action surface the
    policy gets — bound to the world surface each ``reset``. Each factory call
    gets its own ``EpisodeService`` under a unique subdir, so the N envs in a
    GRPO generation batch are fully isolated. The factory closes over one round's
    ``snapshots`` (often a single, current world); the curriculum re-roots the
    next round by re-building the dataset + factory against the evolved snapshot.
    ``backing`` picks how each rollout realizes its world — PROCESS by default;
    CONTAINER (incl. the networked multi-service runtime) trains against the real
    containerized target. ``sandbox=True`` runs each episode's tools in their own
    throwaway :class:`AgentSandbox` (HTTP worlds need the CONTAINER backing so the
    sandbox can join the target's network).
    """
    snap_map = {s.snapshot_id: s for s in snapshots}
    base = Path(run_root)
    base.mkdir(parents=True, exist_ok=True)
    tool_list = tuple(tools)

    def factory() -> EpisodeEnv:
        service = EpisodeService(
            pack, base / f"env-{uuid.uuid4().hex[:8]}", backing=backing
        )
        return EpisodeEnv(
            service=service,
            snapshots=snap_map,
            tools=tool_list,
            reward_fn=reward_fn,
            sandbox=sandbox,
        )

    return factory


def env_trajectory(env: EpisodeEnv) -> Trajectory:
    """Export an env's last episode as a ``snapshot_id``-tagged ``Trajectory``.

    Finalizes the episode first if the reward was never read, so a caller can
    export trajectories without the reward func having run.
    """
    env._finalize()
    if env.report is None:
        raise RuntimeError("no completed episode to export; call reset() first")
    return episode_trajectory(env.report, env.turns)


def make_grpo_rounds(
    pack: Pack,
    *,
    model: Any,
    args: Any,
    tools: Sequence[Tool],
    run_root: str | Path,
    processing_class: Any = None,
    peft_config: Any = None,
    reward_fn: Callable[[EpisodeReport], Reward] = episode_reward,
    backing: Backing = Backing.PROCESS,
    sandbox: bool = False,
) -> tuple[RunRound, RunRound]:
    """A ``(train_round, eval_round)`` pair for
    :func:`openrange.pool.run_pool_curriculum`, sharing one live model.

    ``train_round`` drives one short ``trl.GRPOTrainer`` pass over the round's rows
    and carries the model forward (``peft_config`` wraps it once). ``eval_round``
    rolls the *same* mid-training model out over a held-out round and grades it under
    a frozen update (``learning_rate`` 0), so the held-out pool measures
    generalization without being trained on. Both return the graded reports keyed by
    ``(snapshot_id, task_id)`` and re-root the env factory onto the round's
    snapshots. ``args`` is a ``GRPOConfig`` whose ``max_steps`` bounds the round.
    ``trl`` is imported here so importing :mod:`openrange_trl` stays torch-free.
    """
    from datasets import Dataset
    from trl import GRPOTrainer

    base = Path(run_root)
    base.mkdir(parents=True, exist_ok=True)
    holder: dict[str, Any] = {"model": model, "peft": peft_config}
    frozen_args = copy.copy(args)
    frozen_args.learning_rate = 0.0

    def _round(
        rows: list[PromptRow],
        snapshots: list[Snapshot],
        run_args: Any,
        *,
        update: bool,
    ) -> RoundReports:
        collector: dict[tuple[str, str], list[EpisodeReport]] = {}
        factory = make_environment_factory(
            pack,
            snapshots,
            base / uuid.uuid4().hex[:8],
            tools=tools,
            reward_fn=reward_fn,
            backing=backing,
            sandbox=sandbox,
        )
        trainer = GRPOTrainer(
            model=holder["model"],
            reward_funcs=[make_reward_func(collector)],
            args=run_args,
            train_dataset=Dataset.from_list(rows),
            processing_class=processing_class,
            environment_factory=factory,
            peft_config=holder["peft"],
        )
        try:
            trainer.train()
            if update:
                holder["model"], holder["peft"] = trainer.model, None
        finally:
            # TRL builds one env per batch slot and never closes them; without this
            # each round's CONTAINER worlds (a per-service container stack) leak.
            for env in getattr(trainer, "environments", None) or []:
                env.close()
        return collector

    def train_round(rows: list[PromptRow], snapshots: list[Snapshot]) -> RoundReports:
        return _round(rows, snapshots, args, update=True)

    def eval_round(rows: list[PromptRow], snapshots: list[Snapshot]) -> RoundReports:
        return _round(rows, snapshots, frozen_args, update=False)

    return train_round, eval_round


def make_grpo_run_round(
    pack: Pack,
    *,
    model: Any,
    args: Any,
    tools: Sequence[Tool],
    run_root: str | Path,
    processing_class: Any = None,
    peft_config: Any = None,
    reward_fn: Callable[[EpisodeReport], Reward] = episode_reward,
    backing: Backing = Backing.PROCESS,
    sandbox: bool = False,
) -> RunRound:
    """The training half of :func:`make_grpo_rounds` — a ``run_round`` that trains
    one GRPO pass per round, for when no held-out eval is needed."""
    train_round, _ = make_grpo_rounds(
        pack,
        model=model,
        args=args,
        tools=tools,
        run_root=run_root,
        processing_class=processing_class,
        peft_config=peft_config,
        reward_fn=reward_fn,
        backing=backing,
        sandbox=sandbox,
    )
    return train_round


def reward_variance_policy(
    reports: Sequence[EpisodeReportLike],
    *,
    epsilon: float = 1e-9,
    harden_mean: float = 0.5,
    reward_fn: Callable[[EpisodeReport], Reward] = episode_reward,
) -> Direction | None:
    """Evolve only when GRPO's gradient has collapsed.

    GRPO learns from the *spread* of a group's rewards, so a round whose reward
    scalars are all (near-)equal yields no advantage signal. When the spread
    collapses this nudges the frontier toward the side that revives it — ``harden``
    if the group is mostly solving, ``soften`` if mostly failing. While the spread
    is alive it returns ``None`` (hold the world). It reads the dense scalar when a
    concrete ``EpisodeReport`` is present, else falls back to the binary ``passed``
    gate — a strict refinement of ``direction_from_reports`` keyed on what the
    trainer actually consumes.

    ``reward_fn`` must be the SAME one the trainer optimizes (the one passed to
    ``make_grpo_rounds``), or this keys on a different signal than GRPO's gradient.
    The pool calls a policy as ``policy(reports)``, so bind a custom reward with
    ``functools.partial(reward_variance_policy, reward_fn=my_reward_fn)``. Defaults
    to :func:`episode_reward`.
    """
    if not reports:
        return None
    scalars = [_report_scalar(r, reward_fn) for r in reports]
    mean = sum(scalars) / len(scalars)
    variance = sum((s - mean) ** 2 for s in scalars) / len(scalars)
    if variance > epsilon:
        return None
    return "harden" if mean >= harden_mean else "soften"


def _report_scalar(
    report: EpisodeReportLike,
    reward_fn: Callable[[EpisodeReport], Reward] = episode_reward,
) -> float:
    if isinstance(report, EpisodeReport):
        return reward_fn(report).scalar
    # CurriculumPolicy takes the EpisodeReportLike Protocol, but the trainer only
    # emits concrete EpisodeReport; this contract fallback needs a fake to hit.
    return 1.0 if report.passed else 0.0  # pragma: no cover


__all__ = [
    "EpisodeEnv",
    "Tool",
    "build_grpo_dataset",
    "env_trajectory",
    "make_environment_factory",
    "make_grpo_rounds",
    "make_grpo_run_round",
    "make_reward_func",
    "reward_variance_policy",
]
