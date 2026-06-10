# Benchmarks

OpenRange is the **gym** — synthesized, admission-checked, evolvable worlds you
*train* on. A **benchmark** (XBOW, CVE-Bench, Cybench, …) is the opposite: a
fixed external target you *evaluate transfer against*. The two never mix.

| | gym world | benchmark |
| --- | --- | --- |
| built by | a pack builder, sampled fresh | shipped fixed, externally |
| goes through | build → admit → evolve | none of it |
| graded by | the pack's `TaskFamily.check_success` | the benchmark's own scorer |
| you may | train on it | only evaluate — never train, never evolve |

Putting a benchmark through admission would be a category error (you don't prove
a fixed eval target "solvable by construction"), and letting a builder near one
is contamination. So benchmarks run *outside* the build/admit/evolve pipeline,
through a thin runner.

## The one thing they share: the solver surface

The whole point of measuring sim-to-real transfer is that a gap reflects
*capability*, not a surface mismatch. So `run_benchmark` hands the solver the
**same `EpisodeContext`** a gym episode does — a `base_url`, an editable `root`,
and a `result.json` submission. One solver runs against both; the only thing
that changes is what sits behind the URL.

```python
from openrange import run_benchmark

result = run_benchmark(challenge, solver, root="runs/xbow-001")
# result is an EpisodeResult — the same type a gym episode returns
```

## Writing an adapter

A benchmark is anything satisfying the `Benchmark` protocol
(`src/openrange/benchmark.py`):

```python
class Benchmark(Protocol):
    instruction: str
    def boot(self) -> Mapping[str, Any]: ...   # start target, return {"base_url": ...}
    def score(self, surface, submission) -> EpisodeResult: ...
    def stop(self) -> None: ...
```

`boot` starts the real challenge and returns its surface; the runner adds
`solver_root`. `score` grades by the benchmark's own truth — a flag compared to
the one it injected, or an exploit effect checked against `surface["base_url"]`.
`stop` tears it down (called even after a failed boot).

The differences between benchmarks live entirely in the adapter: XBOW compares
an injected flag; CVE-Bench checks an exploit effect; a CTF compares a flag
string. The runner stays benchmark-agnostic.

## Status

- **Process-backed targets work today.** A benchmark whose `boot` starts a local
  HTTP service needs no extra infrastructure (see `tests/test_benchmark.py` for a
  real one).
- **Containerized targets are the next step.** XBOW, CVE-Bench, and PACEbench
  ship as Docker challenges, so their adapters' `boot` spins up a container —
  the same container runtime the high-fidelity gym backing needs
  ([#252](https://github.com/vecna-labs/open-range/issues/252)). Build that once
  and both ride on it.

## Integrity

Keep benchmarks out of any builder or training set; respect each benchmark's
canary string; prefer cleaned, hint-free forks where they exist (e.g.
`KeygraphHQ/xbow-validation-benchmarks`).
