# SWE pack — design

> The "written shape" for [#238](https://github.com/vecna-labs/open-range/issues/238):
> ontology, backing, agent surface, bug-injection + admission, and curriculum at
> enough specificity that the implementation issues (the tree at the end) don't
> re-litigate it. The [README](README.md) describes what *ships today*; this doc
> describes the *shape* and what's left to build.

## The bet, restated for SWE

SWE is the highest-signal agent benchmark domain right now, but the public benches
are *static* — a frozen scrape of GitHub issues that agents overfit to. OpenRange's
thesis is that agents trained against fresh, runnable, **admission-checked** worlds
generalize better. A SWE pack is where that thesis meets the domain everyone already
measures: a *generated, curriculum-evolvable* SWE gym that freezes to the same shape
as a SWE-bench instance but proves its own well-posedness before it is ever served.

The pack is deliberately **minimal**, because in SWE *the world ships its own
grader*: a repository's test suite is the behavioral contract. There is nothing to
hand-author for grading — you run the tests. The leverage is not in the grader; it
is in the **builder** (the "ancient" that produces worlds): it can import a real
repo, splice a defect into a green one, or have an LLM author a feature + its tests
+ a reference fix. All three freeze to one ontology and are graded by one runner.

## Resolving the open questions (#238)

1. **Generated, or seeded from real repos?** *Both* — three world-sources behind
   one ontology + grader (see [World-sources](#world-sources)). Ship **imported**
   (a real-repo passthrough) first because it most directly demonstrates
   "domain-agnostic core + state-of-the-art SWE gym"; **injected** (generated) and
   **authored** (LLM) follow without touching the ontology or grader.
2. **One language, or polyglot?** **Python first**, to reuse the existing
   subprocess pytest runner. The ontology carries a `language` attr and the *only*
   language-coupled seam is the test runner in `swe/grading.py`. A second language
   is a second runner adapter, not an ontology change.
3. **Own pack, or a task family on a shared "code" pack with `webapp.build`?**
   **Own `swe` pack.** It has a different ontology (repo / test_suite / solution vs.
   cyber's service graph), a different backing (filesystem test-runner vs. HTTP),
   and a different realizer. Folding them together would couple two unrelated world
   shapes. What *is* shared lives one level down, at the SDK: the grader **pattern**
   — run an untrusted submission against a held-out contract in a sandboxed
   subprocess — which both `webapp.build` and `swe.fix` consume.

## A realized world ≅ a SWE-bench instance

| SWE-bench           | OpenRange node / attr                        |
| ------------------- | -------------------------------------------- |
| `repo@base_commit`  | `repo.base_files` — `{path: contents}` tree  |
| `problem_statement` | `repo.problem_statement`                     |
| `test_patch`        | `test_suite.test_files` (held out)           |
| `FAIL_TO_PASS`      | `test_suite.fail_to_pass`                    |
| `PASS_TO_PASS`      | `test_suite.pass_to_pass`                    |
| gold patch          | `solution.gold_files` — HIDDEN overlay       |

File maps ride as JSON node attrs, so the graph content-addresses byte-stably: the
same instance always freezes to the same snapshot id.

## Ontology (`swe.repo@v1`)

Three node kinds, two edge kinds (`swe/ontology.py`):

- **`repo`** — `name`, `language` (default `python`), `problem_statement`,
  `base_files` (the buggy working tree the agent edits).
- **`test_suite`** — `test_files` (held out, never on disk), `fail_to_pass`,
  `pass_to_pass`, and the build tiers `unit_tests` / `integration_tests`. The
  repo's own tests *are* the contract.
- **`solution`** — `gold_files`, an overlay over `base_files` that resolves the
  task. Marked **HIDDEN**; it is the admission answer key and never reaches the
  agent.
- Edges: `repo --has_suite--> test_suite` and `repo --has_solution--> solution`,
  both 1:1.

Structural well-formedness the ontology can't express lives in three pack
invariants (`swe/invariants.py`): file maps are non-empty `{str: str}`; the suite
grades *something* (≥1 `fail_to_pass` for a fix, or ≥1 `integration_tests` for a
build); the F2P/P2P and unit/integration tiers are each disjoint; every test id
points at a file the suite ships; each repo links exactly one suite and one
solution. These are *cheap and structural* — the behavioral proof is admission's
job, below.

## Task families

One world shape, two task families that self-select on the suite (`swe/families/`):

- **`swe.fix`** — the SWE-bench-shaped repair. Claims any suite with a
  `fail_to_pass`; success is *all* F2P passing while P2P stays green — an
  all-or-nothing gate, like the bench.
- **`swe.build`** — the long-horizon sibling. Claims any suite with an
  `integration_tests` gate; the held-out suite splits into **unit tests that
  shape** (dense partial credit) and **integration tests that gate** success. A
  half-built tree whose pieces each pass their unit test but don't compose scores
  the unit fraction *without* resolving — the dense signal a long episode needs,
  where an all-or-nothing gate is zero almost everywhere.

The builder runs both; each shipped fixture is one shape or the other, so exactly
one task is emitted (a suite declaring both tiers would emit a task from each).
Both grade as a pure function of the snapshotted `workspace_files` and report
their full test vector as `subgoals`, which the training seam
(`openrange.training`) turns into the reward.

## Backing / realizer

No live process. The world is a code tree, so the realizer is an `OnDemandRuntime`
(`swe/realize.py`) — the SDK runtime built for exactly this ("fits SWE-style packs
where the world is a code workspace", `_runtime.py`). On `reset()` it materializes
`repo.base_files` into the agent's `solver_root`; the held-out `test_files` and the
gold overlay **stay in the graph, never on disk**, so they are hidden from the agent
by construction. The agent edits files and writes `result.json` to end the episode
(the `terminal()` sentinel). `collect_extras()` snapshots the edited tree as
`workspace_files` for the grader.

The realizer is offline-safe: the base tree comes from the graph, never a re-clone.
Only `Backing.PROCESS` is wired; other backings raise `NotImplementedError`.

## Agent surface

The interface is a **filesystem patch loop**, not an HTTP API:

- **read / edit files** under `solver_root` (the materialized base tree),
- **write `result.json`** to signal "done".

`check_success` (in each family under `swe/families/`) is a *pure function of
`final_state`*: it grades `final_state["workspace_files"]` (the snapshotted tree),
not live disk, so it is unit-testable with a synthetic tree — the same discipline
as trading and cyber.

The loop is **multi-turn**: the runtime exposes a `run_tests(node_ids=…)` tool via
`surface_extras` (`swe/realize.py`) so the agent runs tests against the *live*
workspace, reads the failures, edits, and repeats — in the same sandbox the grader
uses, mirroring the `webapp.build` stage-4 tool
([#237](https://github.com/vecna-labs/open-range/issues/237)). Crucially the tool
runs only whatever tests exist in the workspace (e.g. a reproduction the agent
wrote); it never injects the held-out grading suite, which stays in the graph and
is applied by `check_success` only at episode stop — so the agent gets a real local
test loop without ever seeing its scorer. A single-shot edit-then-grade episode
remains valid; the tool is optional.

## Bug-injection + admission (the well-posedness proof)

This is the SWE generalization of cyber's "reference handler passes + mutation
breaks the contract." Before any agent sees a world, `SweFix.check_feasibility` runs
the repo's own tests **twice**:

1. **gold** (`base_files` + `gold_files`) must pass *every* F2P ∪ P2P — the fix
   really resolves the task.
2. **base** (un-fixed) must **fail** every F2P while keeping P2P green — the bug is
   real, the task is non-trivial, and the suite isn't independently broken.

A world that fails either check is **rejected at admission**. Crucially, *the buggy
base is the admission mutation*: where cyber injects a vuln to prove the contract
bites, here the defect already lives in `base_files`, and admission proves it bites
the suite. This makes the three world-sources interchangeable — each just needs to
produce a `(base, suite, gold)` triple that survives the self-test.

`SweBuild.check_feasibility` runs the parallel proof for a build: the gold overlay
must green *every* tier (units ∪ integration), and the bare skeleton must **fail**
every integration test — proving the gate is real and the build is solvable before
any agent sees it.

## World-sources

One ontology, one grader, one admission self-test — three ways to produce the triple:

- **Imported** *(shipped in the spike)* — a SWE-bench-style passthrough. The
  instance recipe *is* the world; the builder loads it and lays it out
  (`swe/builder.py`, `swe/instances.py`). The `fixtures/calc_sum.json` micro-repo
  stands in for a fetched row.
- **Injected** *(planned)* — start from a green repo + suite and **AST-splice** a
  defect so base fails F2P, mirroring the cyber webapp vuln injector. Pure
  generation: no scraped issues. Reuses ontology + grader + admission unchanged.
- **Authored** *(planned)* — an LLM writes a feature, its tests, and the reference
  fix; admission's self-test is the quality gate that rejects ill-posed authored
  tasks. Depends on the injected machinery + an LLM backend
  ([#188](https://github.com/vecna-labs/open-range/issues/188)).

## Curriculum

`evolve(...)` hardens worlds via a pack `PackPrior` + `available_mutations`, the
same grow-via-prior seam the trading pack uses. SWE-shaped mutations:

- **harder defects** — multi-line, multi-file, or cross-module bugs vs. a one-token
  flip; defects that pass a naive read but fail the suite.
- **stricter suites** — add F2P cases, tighten edge-case coverage, promote a
  passing test to a regression guard (P2P).
- **less scaffolding** — thinner `problem_statement`, fewer hints.

Difficulty is measured against a reference solver (success-rate curves as worlds
harden), the demo target in
[#200](https://github.com/vecna-labs/open-range/issues/200). Injected is the source
where curriculum is most natural — the injector *is* the mutation operator.

## Trust model — read before deploying

Grading runs the agent's patched code **plus arbitrary repo tests**: that is
arbitrary code execution. Every run is funneled through one chokepoint,
`swe/sandbox.py`, which selects the strongest isolation the host supports and
records it in `SandboxResult.isolation` so the guarantee is observable, not
assumed:

- **Always** — a wall-clock timeout (process-group kill on expiry) and
  best-effort `RLIMIT_CPU` / `RLIMIT_FSIZE`.
- **Linux + `bwrap`** (probe-confirmed, not just `which`) — host bound
  read-only, workspace read-write, `/tmp` a tmpfs, `--unshare-net` for grading,
  and PID/IPC/UTS isolation. Untrusted code on a disposable host.
- **macOS / no namespaces** — a bare subprocess with the timeout + rlimits only,
  safe for *trusted* submissions. The `isolation` field makes the fallback
  visible rather than silent.

What remains for *adversarial*, public-facing traffic is **seccomp** syscall
filtering and full **container** isolation — the latter also being the path to
real per-instance dependency provisioning (below). That hardening is shared with
the cross-cutting sandboxing work in
[#202](https://github.com/vecna-labs/open-range/issues/202).

## What the spike already proved

The imported source is built and green (`packs/swe/`, `tests/test_swe_pack.py`):
the recipe is graph-native, the solution is HIDDEN, the graph content-hashes
deterministically, admission admits through all layers, the self-test rejects both
a no-bug world and a wrong-gold world, the realizer materializes only the base tree
(tests + gold stay off disk), and `check_success` resolves the gold tree while the
un-fixed base fails F2P / passes P2P. The shape below is therefore validated, not
hypothetical.

## Implementation issue tree

Sub-issues of [#238](https://github.com/vecna-labs/open-range/issues/238), grouped
by milestone. The spike (imported source + grader + admission + realizer) landed
first; subsequent work took it from a micro-repo proof toward a real, hardenable
gym — items marked **✓** are now in tree, **◐** partially, **○** still planned.

**A. Imported source → production**

- **✓ Real SWE-bench instance loader.** `swe/swebench.py` clones a real row's
  `repo@base_commit`, recovers the held-out tests + gold fix from its
  `test_patch`/`patch` diffs, and the row admits + grades end-to-end through the
  builder (`manifest["swebench"]`). *Remaining:* an offline row cache. *Scale
  ceiling:* the loader inlines the whole working tree into the graph; large
  monorepos are the lazy-clone-at-realize milestone
  ([#212](https://github.com/vecna-labs/open-range/issues/212)).
- **◐ Real-repo test execution.** Importability landed — the sandbox prepends
  `root` + `root/src` to `PYTHONPATH` so flat and `src/` layouts resolve, and the
  whole id set runs in one pytest invocation. *Remaining:* third-party
  dependency / editable installs, the per-instance **container-image** milestone
  (the model SWE-bench uses).

**B. Sandboxing (blocks public traffic)**

- **◐ Isolated test-runner backing.** `swe/sandbox.py` wraps every grading and
  tool run in Linux `bwrap` (read-only host / `--unshare-net` / pid-ipc-uts) plus
  rlimits and a wall-clock timeout, reporting the live backend in `isolation`;
  macOS degrades to a bare subprocess. *Remaining:* **seccomp** + full
  **container** for adversarial / public traffic, coordinated with
  [#202](https://github.com/vecna-labs/open-range/issues/202).

**C. Injected source (generated worlds)**

- **○ AST-splice defect injector.** A builder that turns a green repo + suite into
  an admitted buggy world, mirroring the cyber vuln injector. *Acceptance:* injected
  worlds pass the admission self-test; the defect is the admission mutation.
- **○ `available_mutations` + `evolve` curriculum.** Harder defects / stricter
  suites via the `PackPrior` seam. *Acceptance:* `evolve(...)` produces a strictly
  harder admitted world; success-rate curve demoed ([#200](https://github.com/vecna-labs/open-range/issues/200)).

**D. Authored source (LLM-generated worlds)**

- **○ LLM-authored task builder.** An LLM writes feature + tests + reference fix;
  admission's self-test is the quality gate. Depends on C's machinery + an LLM
  backend ([#188](https://github.com/vecna-labs/open-range/issues/188)).
  *Acceptance:* an authored world admits and is solvable by the reference fix.

**Cross-cutting**

- **✓ Agent test-runner tool (multi-turn surface).** `run_tests` is exposed via
  `surface_extras` (`swe/realize.py`) so the agent iterates instead of
  single-shotting; it runs the agent's own tests in the grading sandbox without
  exposing the held-out suite. Mirrors `webapp.build` stage 4
  ([#237](https://github.com/vecna-labs/open-range/issues/237)).
- **✓ Long-horizon `swe.build` family.** A second task family on the same world
  (`swe/families/build.py`): the held-out suite splits into unit tests that shape
  (partial credit) and integration tests that gate success, so a half-built tree
  whose pieces don't compose earns dense reward without resolving — the signal a
  long episode needs ([#243](https://github.com/vecna-labs/open-range/issues/243)).
