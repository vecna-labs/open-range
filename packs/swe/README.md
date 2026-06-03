# OpenRange SWE pack

A software-engineering world: the agent edits a code repository until the repo's
**own held-out test suite** goes green. The pack is deliberately minimal because,
in SWE, *the world ships its own grader* — a repo's tests are the behavioral
contract, so there is nothing to hand-author.

The same world drives two task shapes (`swe/families/`): **`swe.fix`** repairs a
buggy repo until every held-out test passes — all-or-nothing, like SWE-bench;
**`swe.build`** builds a feature from a skeleton, where unit tests shape dense
partial credit and integration tests gate success (the long-horizon sibling).

## The shape (one instance = one SWE-bench-style task)

A realized world is isomorphic to a SWE-bench instance:

| SWE-bench            | OpenRange node / attr                          |
| -------------------- | ---------------------------------------------- |
| `repo@base_commit`   | `repo.base_files` (the working tree to edit)   |
| `test_patch`         | `test_suite.test_files` (held out)             |
| `FAIL_TO_PASS`       | `test_suite.fail_to_pass`                      |
| `PASS_TO_PASS`       | `test_suite.pass_to_pass`                      |
| gold patch           | `solution.gold_files` (HIDDEN answer key)      |
| `problem_statement`  | `repo.problem_statement`                       |

The file maps ride as JSON node attrs, so the graph content-addresses
byte-stably: the same instance always freezes to the same snapshot id.

## Admission proves the world is well-posed

This is the SWE generalization of the cyber pack's "reference handler passes +
mutation breaks the contract." `SweFix.check_feasibility` runs the repo's own
tests **twice**, before any agent ever sees the world:

1. **gold** (`base_files` + `gold_files`) must pass every FAIL_TO_PASS and
   PASS_TO_PASS test — the fix really resolves the task.
2. **base** (un-fixed) must *fail* every FAIL_TO_PASS while keeping PASS_TO_PASS
   green — the bug is real, the task is non-trivial, and the suite isn't broken
   independently of the fix.

A world that fails either check is rejected at admission. `SweBuild.check_feasibility`
is the same proof for a build: the gold overlay must green every tier (unit +
integration) and the bare skeleton must *fail* every integration test, so the gate
is real and the build is solvable before anyone is served the world.

## Grading an episode

`SweRuntime` materializes `base_files` into the agent's workspace on `reset()`
(the held-out tests and gold fix stay in the graph, never on disk). The agent
edits files and writes `result.json` to end the episode. `check_success` replays
the edited tree against the suite and reports success iff every FAIL_TO_PASS +
PASS_TO_PASS test passes — SWE-bench's "resolved" criterion.

A `swe.build` episode grades the same tree the same way but splits the verdict:
every test (unit + integration) is reported as a subgoal, while success gates on
the integration tier alone. So a half-built tree that passes its unit tests but
doesn't compose scores partial credit without resolving — the dense signal a
long-horizon episode needs.

## Three world-sources, one ontology + grader

The shipped builder is the **imported** source (a SWE-bench passthrough; the
fixture under `fixtures/` stands in for a fetched row). [The design
doc](DESIGN.md) describes two more producers behind the same ontology and
grader: **injected** (splice a
defect into a green repo) and **authored** (an LLM writes a feature + its tests +
the reference fix). All three freeze to the same world shape.

## Trust model — read before deploying

Grading runs the agent's patched code plus arbitrary repo tests: arbitrary code
execution. Every run is funneled through one chokepoint —
[`swe/sandbox.py`](swe/sandbox.py) — which applies the strongest isolation the
host supports and reports the backend that actually ran in
`SandboxResult.isolation`, so a caller never assumes a guarantee the host
couldn't give:

- **always** — wall-clock timeout (process-group kill on expiry) + best-effort
  CPU / file-size rlimits.
- **Linux + `bwrap`** — host bound read-only, workspace read-write, `/tmp` a
  tmpfs, network unshared for grading, PID/IPC/UTS isolated. Untrusted code on a
  disposable host.
- **macOS / no `bwrap`** — bare subprocess (timeout + rlimits only); *trusted*
  submissions only. The fallback is visible in `isolation`, never silent.

Seccomp syscall filtering and full container isolation are not yet enforced —
they are the prerequisite for *adversarial*, public-facing eval traffic. See
[DESIGN.md](DESIGN.md).
