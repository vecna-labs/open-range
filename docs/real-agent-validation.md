# Real-agent validation — first live run of the loop

First end-to-end run of the OpenRange loop with a real LLM agent (not a
scripted stand-in), against `cyber_webapp` post-SDK-extraction. Harness:
`examples/codex_eval.py`, Codex CLI 0.128.0.

## Result: the thesis holds

A real model solved the `webapp.build` task unassisted — read the
instruction, wrote a handler into `result.json`, and the live
`EpisodeService → runtime.collect() → check_success` path graded it:

```python
def handle(query, state):
    import json
    items = []
    for record_id, fields in state["records"].items():
        item = dict(fields)
        item["id"] = record_id
        items.append(item)
    body = json.dumps({"items": items}).encode("utf-8")
    return 200, {"Content-Type": "application/json"}, body
```

`success: true`, reason "all contract cases pass", every subgoal green.
Real agent → real world → real grade. This is the first evidence the
loop works with a live model, not just in unit tests and the scripted
smoke (`tests/test_episode_loop_smoke.py`).

## Findings

Three real defects surfaced — none reachable by unit tests or the
scripted smoke, because each needs a real backend in the loop.

### 1. Stale hardcoded default model — FIXED

`CODEX_DEFAULT_MODEL = "gpt-5.3-codex-spark"` 400s on a ChatGPT account
("model is not supported when using Codex with a ChatGPT account").
`CodexBackend` always passed `--model`, overriding the model the user
already configured in `~/.codex/config.toml`.

Root fix: `CodexBackend.model` defaults to `None`; when `None`, `--model`
is omitted and the Codex CLI uses its own configured default. The
hardcoded constant is removed. Stop overriding the caller's model with a
value that goes stale.

### 2. A backend error aborts the whole multi-step run — FIXED

The first run raised `LLMBackendError` (the stale model), which
propagated uncaught through `_run_task` → `main` and crashed the eval
with a traceback. In a multi-step run, one transient backend failure
(timeout, rate limit) would discard all prior steps.

Root fix: `examples/codex_eval._run_task` catches `LLMBackendError`,
records it as a failed episode (the agent wrote no `result.json`, so
`check_success` grades a fail), prints the cause so it can't be mistaken
for an agent miss, and continues. Misconfiguration still surfaces
loudly; transient failures no longer nuke the run.

### 3. The curriculum silently stalls on build episodes — DEFERRED (Stage 2)

After the build episode passed, `auto_evolve` returned `None` and the
harness stopped. Mechanism, pinned by bisection (`auto_evolve(llm=None)`
succeeds; the real run with `llm` set returns `None`) plus reading
`enrich_mutations`:

- `webapp.build` has no `available_mutations` — solving build cannot
  harden build.
- The pass-rate policy still says "harden" (build was solved), routing
  to *pentest* mutations. The LLM correctly scores them ~0.0 (the agent
  never touched the pentest surface). `auto_evolve`'s strict
  `relevance > 0.0` filter drops the zero-scored mutation → no
  candidates → `None`.

The deeper root is **the build contract is graph-decoupled**: the
`ContractCase`s in `families/build/contracts.py` are static, not derived
from the world graph. So even a build `available_mutations` that mutates
the graph could not change build difficulty — the grader ignores the
graph's records and uses its own fixed inputs.

A rigorous fix is Stage-2-sized: either make the build contract
graph-derived, or wire more endpoint kinds (`auth`, `web`) so a
diversify mutation can switch a service's kind and change the contract
in play. A stub `available_mutations` that returns mutations which don't
actually change difficulty would be a band-aid, not a fix. Tracked as
the next focused effort.
