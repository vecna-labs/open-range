"""A bounded thread pool for the blocking work an async rollout offloads.

Only the model round-trip is async; everything else a rollout does — ``EpisodeEnv``
reset, brought-tool calls, ``AgentSandbox.run`` (a blocking ``docker exec``), grading —
is synchronous, so it runs here instead of on the event loop. The pool is sized to the
dispatcher's concurrency: the default loop executor caps at ``min(32, cpu+4)`` (≈14 <
the 16-world target), and a single ``docker exec`` can hold its thread up to the
sandbox's 120s timeout, so a too-small pool would head-of-line-block other rollouts.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any, TypeVar

_T = TypeVar("_T")


class BlockingPool:
    """A thread pool plus an awaitable that runs a blocking call on it."""

    def __init__(self, workers: int) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max(1, workers))

    async def run(self, fn: Callable[..., _T], /, *args: Any, **kwargs: Any) -> _T:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, lambda: fn(*args, **kwargs))

    def close(self) -> None:
        self._executor.shutdown(wait=True)
