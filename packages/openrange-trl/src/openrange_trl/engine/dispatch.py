"""Run many rollouts concurrently, at most ``concurrency`` in flight.

While one rollout waits on the model, the others make progress; the cap bounds how
many worlds/sandboxes boot at once. ``pipeline`` (overlapping grade with model-wait via
a separate eval pool) lands with the world-reuse slice, where its payoff is real.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

from openrange.training import Trajectory
from openrange_trl.engine.async_utils import BlockingPool
from openrange_trl.engine.rollout import AsyncRollout


async def batch(
    rollouts: Sequence[AsyncRollout], *, concurrency: int
) -> list[Trajectory]:
    """Run each rollout's init → run → eval, with at most ``concurrency`` running.

    A rollout that fails (e.g. a model-server error) still tears its own world + sandbox
    down — ``aclose`` runs in a ``finally`` and ``return_exceptions`` lets every sibling
    finish its cleanup — so a failed batch leaks nothing; the first error is re-raised.
    """
    pool = BlockingPool(concurrency)
    semaphore = asyncio.Semaphore(concurrency)

    async def _one(rollout: AsyncRollout) -> Trajectory:
        async with semaphore:
            try:
                await rollout.init(pool)
                await rollout.run(pool)
                return await rollout.eval(pool)
            finally:
                await rollout.aclose(pool)

    try:
        results = await asyncio.gather(
            *(_one(r) for r in rollouts), return_exceptions=True
        )
    finally:
        pool.close()
    for result in results:
        if isinstance(result, BaseException):
            raise result
    return [r for r in results if isinstance(r, Trajectory)]
