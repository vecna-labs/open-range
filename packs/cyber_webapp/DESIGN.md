# Cyber webapp pack — generation design

How this pack generates worlds, and why it generates them the way it does. The
[README](README.md) shows *what* one built world looks like; this explains the
*generator* behind it and the direction it's being taken: **staged, procedural,
constraint-propagating generation** that produces a wide range of exploit types
while staying solvable by construction.

Audience: anyone extending the builder, the vulnerability catalog, or the
ontology — and the sim-to-real study that depends on the gym being *broad*.

---

## 1. The bet, restated for generation

The gym's job is to be a **cheap, reproducible, solvable source of training
worlds** whose exploit skills transfer to real benchmarks. Three constraints
fall straight out of that and decide the whole design:

- **Reproducible.** `snapshot_id = graph.content_hash()`. Same builder + manifest
  + seed → the same world, byte for byte. A nondeterministic generator breaks the
  thing OpenRange is built on.
- **Cheap at scale.** The bet is worlds by the thousand. Per-world cost has to be
  near zero.
- **Solvable by construction.** Every task is admission-checked before an episode;
  a generator that mostly produces unsolvable worlds and leans on
  reject-and-repair is wasteful.

All three point the same way: **the correctness-critical core of generation is
procedural, not LLM-driven.** This is not anti-LLM; it's where the line falls.

---

## 2. Procedural owns correctness; the LLM owns variety

| | owns | why |
| --- | --- | --- |
| **Procedural** (the core) | the vuln mechanic, exploitability, feasibility, chaining, flag placement, base parameterization | must be deterministic, cheap, reproducible, solvable-by-construction |
| **LLM** (a later layer, *behind admission*) | open-ended structural diversity within a class, surface realism that pools can't cover | benefits from variety; a hallucination is **rejected by admission**, never trusted |

The line is sharp: **the LLM never generates the thing that must be correct.**
Admission is what makes any LLM use safe — generate, then verify the exploit
actually fires; a bad generation is dropped, not shipped. This is the
"self-verifying generation" the gym rests on.

This is well-trodden ground. Pre-LLM, **LAVA** (automated vulnerability addition)
and **NIST Juliet/SARD** (tens of thousands of procedurally generated CWE samples)
injected exploitable bugs *with known triggers* — self-verifying by construction.
That is already the OpenRange model.

Note even *realism* is procedural-first: realistic names and content come a long
way from curated pools sampled deterministically (`customer-portal`,
`alice@corp.example`), no model required ([#192](https://github.com/vecna-labs/open-range/issues/192)).
Reserve the LLM for diversity that pools and parameterized templates genuinely
can't reach — and accept that an LLM in the build path trades pure seed-determinism
for cache-keyed determinism (cache outputs by `(seed, prompt)`), which is a real
cost to pay only where it buys something.

---

## 3. Staged, constraint-propagating generation

The principle: **generate the world in ordered layers, each layer's output
*bounding* the next layer's choices.** Top-down, not flat. This is what keeps a
world coherent, makes feasibility hold incrementally instead of being discovered
after the fact, and keeps each step a small sampling problem.

The builder (`sampling.py::sample_graph`) **already does this in embryo.** It runs
network → services → hosts/endpoints → data store → flag → accounts → vulns, and
it already propagates one constraint: the flag's location fixes
`oracle_service_id`, which the vuln stage consumes (the oracle vuln must land on
the path to the flag).

What's missing is that **one stage hardcodes a single choice.** Flag placement is
always a DB record (`sample_graph`, the `record`/`data_store` block). That one
decision is why all three vuln classes are "leak-via-DB-response": the loot is
always a row, so only response-leak exploits can reach it. The narrowness is not
"few templates" — it is *one loot stage with one shape.*

The fix is to make loot placement a real layer that **picks a shape and emits it
as the constraint** the rest of the pipeline already knows how to consume:

```
loot-placement → picks loot shape ∈ {db-row, file, exec-reachable}   ← the constraint
        ↓ bounds
vuln-selection → picks an oracle vuln whose exploit reaches that shape
        ↓ bounds
realization    → renders the template + wires the exploit → flag path
```

Because the vuln is *chosen to match the loot*, the chain is reachable **by
construction** — no extra reject-and-repair. That is the deep win of staging:
solvability is assembled layer by layer. The same pattern generalizes upward to
enterprise scale ([#212](https://github.com/vecna-labs/open-range/issues/212)):
org → team → service → data → vuln, each layer bounding the next.

---

## 4. Exploit *shapes*, not CWE names

Organize the catalog by **exploit shape** — *how the flag is reached* — not by CWE
label. The shape is the unit of real work (realizer + feasibility); classes within
a shape are cheap templates on top.

| shape | how the flag is reached | classes | loot |
| --- | --- | --- | --- |
| **response-leak** *(have)* | exploit returns the flag in an HTTP response | `sql_injection`, `ssrf`, `broken_authz` | DB row |
| **file-read** *(new)* | exploit reads a file holding the flag | `path_traversal`, `lfi`, `xxe` | file |
| **code-exec** *(new)* | exploit runs code that reads the flag | `command_injection`, `ssti`, `deserialization` | exec-reachable file/env |

Shapes are also the **agent capability** the study measures: H2 ("which
capabilities survive simulation") is per-shape by nature. So shape-organization is
not tidiness — it is the study's axis.

---

## 5. Ontology decision: reuse `data_store`, no new kind

File-read and code-exec loot lives somewhere other than a DB row. The decision:
**reuse `data_store`, not a new node kind** — and the ontology already
accommodates it. `data_store.kind` is `{sql, kv, file, object}` and `engine` is
`{sqlite, postgres, mysql, redis, fs, s3}` (`ontology.py`), so filesystem loot is
just `kind=file, engine=fs`: **no ontology change, only realizer support.** Such a
store materializes its record as a real file under the owning service;
path-traversal reads it directly, command-injection `cat`s it. One shape, two
exploit classes. (The current sampler only ever emits `kind=kv, engine=redis` — the
filesystem values are unused, not unsupported.)

Feasibility generalizes from today's DB-path check to "a loot path of the matching
shape exists from the entrypoint" — the structural check stays per-*shape*, not
per-*class*.

---

## 6. The narrowness today, precisely

- **3 vuln classes, 1 shape.** `sql_injection`, `ssrf`, `broken_authz`
  (`vulnerabilities/__init__.py`) — all `family=code_web`, all targeting
  `endpoint`, all response-leak.
- **Structurally fixed templates.** The SQLi template is always
  `SELECT key, {col} FROM {table} WHERE key = '{input}'` — only names vary. An
  agent can learn *the template*, not "SQL injection." This is a transfer-study
  confound (a gap may reflect template-overfitting, not a real capability gap), so
  intra-class diversity matters for *validity*, not just polish — procedural
  parameterization first, the LLM layer later.

---

## 7. Goal — what this doc is here to make true

> **Generalize the loot → vuln → realization staging so the gym produces 3 exploit
> shapes (response-leak, file-read, code-exec) across ~8 vuln classes — every world
> solvable by construction because the vuln is chosen to match the loot — keeping
> every correctness-critical layer procedural, with the LLM diversity/realism layer
> left as a later admission-gated stage.**

### Work breakdown

1. **Loot-placement stage.** Lift the hardcoded DB-record block into a staged
   choice over loot shape (`db-row` / `file` / `exec-reachable`), prior-weighted,
   emitting the shape as the constraint.
2. **Filesystem loot.** Allow `data_store.engine = filesystem`; realizer
   materializes its record as a real file on the owning service.
3. **Shape-tagged catalog + selection.** Tag each `Vulnerability` with its shape;
   the vuln stage picks an oracle whose shape matches the placed loot.
4. **Two new shapes, end-to-end.** `path_traversal` (file-read) first — it stands
   up the whole file-store pipeline at lower risk — then `command_injection`
   (code-exec), which reuses the same in-memory file store for near-free.
5. **Feasibility per shape.** Generalize the pentest structural check to verify the
   matched loot→vuln path for each shape.
6. **Tests + proof.** A real pentest episode recovering the flag for each new
   shape; admission proves each world well-posed; determinism holds (same seed +
   shape → same snapshot).
7. **Fan-out.** `ssti`, `xxe`, `weak_credentials`, `idor` as cheap additions once
   their shape's pipeline exists.

### Status

Items 1–6 are **done** (`feat/cyber-staged-generation`): the staged loot→vuln
pipeline, the in-memory file store, shape-tagged catalog + shape-matched oracle
selection, and both new shapes — `path_traversal` (file-read) and
`command_injection` (code-exec) — each proven end to end by a real HTTP exploit
that recovers the flag (`tests/test_cyber_staged_generation.py`). Loot shape and
vuln-class mix are manifest-configurable (`loot_shapes` / `vuln_kinds`). The gym
now spans **3 exploit shapes across 5 classes**; item 7 (more classes per shape)
is the remaining fan-out. A `file` store is the in-memory PROCESS-backing
emulation; a container backing (#252) makes the file system and shell real.

This tracks [#190](https://github.com/vecna-labs/open-range/issues/190) (expand the
vuln catalog) and lays the staging groundwork for
[#212](https://github.com/vecna-labs/open-range/issues/212) (enterprise scale).

### Out of scope here

Client-side shapes (XSS, CSRF) need a victim NPC and wait. The LLM diversity layer
(§2) waits. Sandbox hardening of exec'd exploit code at training scale is
[#202](https://github.com/vecna-labs/open-range/issues/202).
