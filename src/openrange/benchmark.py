"""Run a solver against an external benchmark with a gym-identical surface.

A benchmark (XBOW, CVE-Bench, Cybench, ...) is a fixed eval target, not a gym
world: it is never built, admitted, or evolved. ``run_benchmark`` boots the
target, hands the same ``Solver`` the same ``EpisodeContext`` a gym episode
would, and scores the submission with the benchmark's own scorer. Sharing the
solver surface is the point — a transfer gap then reflects capability, not a
surface mismatch between training and evaluation.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

from openrange_pack_sdk import EpisodeResult, TaskSpec

from openrange.runtime import EpisodeContext, Solver


class Benchmark(Protocol):
    """An external eval target. The adapter owns booting the real challenge
    and scoring it; OpenRange never admits, snapshots, or evolves it.

    Implementations exist per benchmark — a containerized XBOW challenge, a
    CVE-Bench app, a Cybench task — but all expose this one shape, so the
    runner stays benchmark-agnostic.
    """

    #: What the solver is asked to do, in the solver's own words.
    instruction: str

    def boot(self) -> Mapping[str, Any]:
        """Start the target and return its solver-facing surface, including at
        least a ``base_url``. The runner adds ``solver_root``. Must clean up
        after itself on failure; ``stop`` is still called either way."""
        ...

    def score(
        self,
        surface: Mapping[str, Any],
        submission: Mapping[str, Any],
    ) -> EpisodeResult:
        """Grade by the benchmark's own truth — a flag compared to the one it
        injected, an exploit effect checked against ``surface["base_url"]``,
        whatever the benchmark defines. Returns the same ``EpisodeResult`` a
        gym episode does, so gym and benchmark outcomes are handled alike."""
        ...

    def stop(self) -> None:
        """Tear the target down. Called once per run, including after a failed
        boot, so it must tolerate a partial start."""
        ...


def run_benchmark(
    benchmark: Benchmark,
    solver: Solver,
    *,
    root: str | Path,
) -> EpisodeResult:
    """Run ``solver`` once against ``benchmark`` and return the benchmark's score.

    The solver is handed the same ``EpisodeContext`` (``base_url``, ``root``,
    a ``result.json`` submission) it gets from a gym episode, so the only thing
    that differs between training and evaluation is what sits behind the URL.
    """
    workspace = Path(root)
    workspace.mkdir(parents=True, exist_ok=True)
    try:
        surface = {**benchmark.boot(), "solver_root": str(workspace)}
        task = _benchmark_task(benchmark.instruction)
        solver(EpisodeContext(task=task, surface=surface))
        return benchmark.score(surface, _read_submission(workspace))
    finally:
        benchmark.stop()


def _benchmark_task(instruction: str) -> TaskSpec:
    return TaskSpec(
        id="benchmark",
        instruction=instruction,
        entrypoints=(),
        goal_nodes=(),
        feasibility_check="",
        success_check="",
    )


def _read_submission(workspace: Path) -> Mapping[str, Any]:
    result = workspace / "result.json"
    if not result.exists():
        return {}
    parsed: Any = json.loads(result.read_text(encoding="utf-8"))
    # A solver that writes a non-object result.json submitted nothing usable.
    return parsed if isinstance(parsed, Mapping) else {}
