"""One episode as three async stages over an ``EpisodeEnv``: init → run → eval.

Each rollout owns an isolated ``EpisodeService`` (the factory builds a fresh one per
call), so N rollouts run on threads without sharing world/episode state. The first
prompt is built here from the task instruction + ``reset()``'s observation — which
already carries the live, sandbox-correct interface URL — so the engine depends on no
example harness code.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from openrange.training import Trajectory
from openrange_trl import EpisodeEnv, env_trajectory
from openrange_trl.engine.async_utils import BlockingPool
from openrange_trl.engine.react import Policy, run_react
from openrange_trl.engine.schema import tool_schema


class AsyncRollout:
    """Drive one episode concurrently: realize + reset, run the ReAct loop, grade."""

    def __init__(
        self,
        factory: Callable[[], EpisodeEnv],
        *,
        policy: Policy,
        snapshot_id: str | None = None,
        task_id: str | None = None,
        max_iters: int = 8,
    ) -> None:
        self._factory = factory
        self._policy = policy
        self._snapshot_id = snapshot_id
        self._task_id = task_id
        self._max_iters = max_iters
        self._env: EpisodeEnv | None = None
        self._tool_schemas: list[dict[str, Any]] = []
        self._first_prompt = ""
        self._closed = False
        self.messages: list[dict[str, Any]] = []

    async def init(self, pool: BlockingPool) -> None:
        env = self._factory()
        self._env = env
        self._tool_schemas = [tool_schema(fn) for fn in env._tools.values()]
        obs = await pool.run(
            env.reset, snapshot_id=self._snapshot_id, task_id=self._task_id
        )
        instruction = self._instruction(env)
        self._first_prompt = f"{instruction}\n\n{obs}" if instruction else obs

    async def run(self, pool: BlockingPool) -> None:
        env = self._require_env()
        self.messages = await run_react(
            env,
            policy=self._policy,
            tool_schemas=self._tool_schemas,
            first_prompt=self._first_prompt,
            max_iters=self._max_iters,
            pool=pool,
        )

    async def eval(self, pool: BlockingPool) -> Trajectory:
        # env_trajectory finalizes the episode (grades + tears down the sandbox); the
        # world service is closed in aclose so cleanup also runs on the failure path.
        return await pool.run(env_trajectory, self._require_env())

    async def aclose(self, pool: BlockingPool) -> None:
        """Tear the episode's world + sandbox down, once. Safe on any exit — success,
        a mid-rollout failure, or cancellation — so a failed batch leaks nothing."""
        if self._env is None or self._closed:
            return
        self._closed = True
        env = self._env
        # _finalize is idempotent; it tears the sandbox down if eval() never ran.
        await pool.run(env._finalize)
        await pool.run(env.service.close)

    def _require_env(self) -> EpisodeEnv:
        if self._env is None:
            raise RuntimeError("rollout not initialized; call init() first")
        return self._env

    def _instruction(self, env: EpisodeEnv) -> str:
        snapshot_id = self._snapshot_id or next(iter(env.snapshots))
        snapshot = env.snapshots[snapshot_id]
        for task in snapshot.tasks:
            if self._task_id is None or task.id == self._task_id:
                return str(task.instruction)
        return ""
