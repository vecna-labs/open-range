# Indictment experiment — results

**Question.** The standard recipe for generating training tasks is a *self-verifying*
loop: an LLM writes a world + a solver + a checker, and the world is admitted if the
solver passes the checker. Does that self-check ship worlds an *independent* verifier
rejects as trivially-solvable or unfaithful — i.e. is self-verification alone enough
for a measurement-grade gym?

## Method

LLM-generated vulnerable webapps (Claude sonnet/haiku), each a `handle(query, state)`
+ a flag planted in `state` (marked `__FLAG__`) + the generator's own solver and, in
the final run, the generator's own `check(response, flag)`. The harness
(`harness.py`) owns the flag value — injected fresh per run, so a generator cannot
hardcode it — and applies judge-free probes whose verdict owes nothing to the
generator's self-assessment:

- **self-admit**: the generator's own checker accepts the solver's response (final
  run) or, earlier, a *sound* flag-recovery check.
- **independent verdict** (always sound): `not_solvable` if the flag doesn't actually
  leak; `trivial` if it's reachable *without* the technique (its bare path, as a
  plain extra argument, or the app's own benign request redirected at it); `unfaithful`
  if a **computed control** the world supplies (`{{6*7}}`→42, `UNION SELECT 6*7`→42,
  `echo $((6*7))`) is *not* produced — proof the engine is faked — or a swap to a
  fresh flag isn't reflected.

Three runs: **A** 21 guided `command_injection` worlds; **B** 20 unguided
`command_injection` worlds; **C** 48 worlds across `command_injection`,
`sql_injection`, `ssti`, `path_traversal` (guided sonnet / unguided haiku), each with
the generator's **own checker** and a computed control.

## Result

| run | worlds | shipped (self-admit) | broken (self-caught) | confirmed admit-gap |
|---|---|---|---|---|
| A — guided cmdi | 21 | 21 | 0 | 1 (trivial) |
| B — unguided cmdi | 20 | 6 | 14 | 0 |
| C — 4 classes, real checkers | 48 | 43 | 5 | 1 (faked engine) |

Run C, by class (all guided cells were 6/6 faithful):

| class | unguided shipped | unguided faithful | gap |
|---|---|---|---|
| command_injection | 6 | 5 | **1** (partial shell) |
| sql_injection | 6 | 6 | 0 |
| ssti | 6 | 6 | 0 |
| path_traversal | 2 | 2 | 0 (4 broken, self-caught) |

**The admit-gap is small and consistent — ~2–4% of shipped worlds — across runs,
classes, guidance, and harness versions. Strengthening the experiment (4 classes,
real LLM checkers, bigger N) did not widen it.** The two confirmed cases:

- `log-file-viewer` (run A): a `tail` handler where `log=/etc/flag` reads the flag
  with no injection — arbitrary file read mislabelled command injection. Caught by the
  shortcut search.
- `command_injection__unguided_1` (run C): a toy shell that honors `;cat` but cannot
  compute its own control (`echo $((6*7))` → the literal string, not `42`) — a
  partial/faked engine. Caught by the control probe.

## Two predicted amplifiers that did *not* materialize

1. **Permissive LLM checkers.** The whole point of run C was to let generators write
   their own checker, expecting buggy ones to ship broken worlds. They didn't: the
   generated checkers were mostly sound (`flag in response`), so the 5 broken worlds
   were *self*-rejected, not shipped. The harness *can* catch a permissive checker
   (a `return True` checker on a non-leaking world is scored as a `not_solvable`
   admit-gap — tested), but real generators didn't write them.
2. **Weaker / unguided generation widening the gap.** It produces *more broken*
   worlds, not more contaminated ones — 14/20 unguided cmdi worlds (run B) and 4/6
   unguided path-traversal worlds (run C) don't even self-admit (faked tool output,
   the interface stub reused, exploits that echo the literal `__FLAG__`). The
   self-check correctly discards them.

## The harder finding: an independent verifier is itself hard to get right

This is the result that matters for the thesis. "Self-verification is insufficient"
is the easy half; "independent verification is *reliable*" is not. The harness had
failure modes in **both** directions, each found only by hand-auditing real worlds:

- **False negatives (under-counting).** v1 only used the generator's own benign/wrong
  queries and missed worlds where the flag is directly readable. It took three probe
  iterations — bare path, extra argument, benign-request-redirected — before it
  stopped passing `log-file-viewer`-style trivial worlds.
- **False positives (over-counting).** v2 judged faithfulness on the generator's
  self-labeled "wrong" query. Generators routinely mislabel a *valid* alternate
  technique as "wrong", so a real shell (`command_injection__unguided_2`) and real
  jinja2 (`ssti__unguided_0`) were flagged `unfaithful` — both actually faithful (their
  controls compute). Demoting wrong-vector to informational and judging faithfulness
  only on the computed control removed both, taking run C from a raw 3/43 to a
  defensible 1/43.

So the verifier needs the same scrutiny as the generator. The reliable signals are
**triviality** (reach the flag without the technique) and **faked-engine** (the world
can't satisfy its own computed control); the generator-supplied wrong-vector is not.

## Bottom line

For these classes and models, self-verification's blind spot is **small (~2–4%) and
consistent**, dominated by occasional trivial or partial-engine worlds in unguided
generation. That tail still matters for a measurement-grade gym — a `command_injection`
set that is even 2% arbitrary-file-read biases a per-class transfer number — and it is
invisible to the self-check. But the honest headline is not "the loop fails"; it is
**"the loop's gross failures (broken worlds) are self-caught, the residual impurity is
a small consistent tail, and the genuinely hard part is building an independent
verifier reliable enough to measure that tail."**

## Threats to validity

- The gap is a **lower bound** on triviality (a generic shortcut search can't be
  complete) and depends on each world supplying an honest computed control for
  faithfulness; `path_traversal` has no natural control, so its faithfulness leans on
  triviality probes alone.
- N = 89 worlds, one model family (Claude sonnet/haiku). A different generator family
  could differ.
- `PROCESS` backing only: real OS/exec-effect faithfulness (a real shell vs the toy
  one in `command_injection__unguided_1`) needs the container backing — until then,
  the control probe is the proxy.

## Reproduce

```
.venv/bin/python experiments/indictment/harness.py --dir experiments/indictment/runs
.venv/bin/python experiments/indictment/harness.py --dir experiments/indictment/runs_unguided
.venv/bin/python experiments/indictment/harness.py --dir experiments/indictment/runs_v2
```

Specs are the saved `*.json`. The harness is deterministic in verdict (the injected
flag is random but the verdict does not depend on its value).
