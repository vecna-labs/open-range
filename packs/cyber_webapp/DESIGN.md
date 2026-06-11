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

## 6. The narrowness this addressed (the starting point)

- **Was: 3 vuln classes, 1 shape.** `sql_injection`, `ssrf`, `broken_authz` —
  all `family=code_web`, all targeting `endpoint`, all response-leak. The staged
  pipeline took this to 9 classes across 3 shapes (see Status, §7).
- **Was: structurally fixed templates → now payload-context diversity.** The
  SQLi template used to be always `... WHERE key = '{input}'` — only names varied,
  so an agent learns *the template*, not "SQL injection." Because the agent only
  sees the HTTP surface (never server code), the fix is to vary the **injection
  context** the exploit must adapt to, sampled per build. Crucially the three
  contexts per class are **mutually exclusive**: the handler enforces each one's
  requirement, so a payload that solves one build *fails* the other two — an agent
  can't memorize one string and replay it. SQLi single/numeric/double quoting;
  cmdi separator/substitution/quoted (each strips the others' vectors); path
  traversal absolute-only/relative-`../`/`....//`-past-a-single-strip; SSTI
  attribute/comment/expr sink; XXE element/wrapped-root/scheme-prefix; SSRF
  scheme-block/host-allowlist/decimal-IP; IDOR direct/base64/prefixed; broken-authz
  single/dual-factor/encoded; weak-creds pair/combined/basic. A live 3×3 replay
  matrix per class confirms it: **all 9 classes are fully diagonal** (every
  off-diagonal cell rejects), so the single-payload replay floor is **~33%, down
  from ~67%** — an agent must learn all three techniques, not memorize one. (XXE's
  `element_content` vs `wrapped_root` was the last superset cell; it's closed by
  having `element_content` reflect only the root's *direct* text while
  `wrapped_root` nests the entity a level deeper — distinct positions, not a
  collapsed root-name swap.)

  *Wrong-context feedback.* A neutralized but attack-shaped attempt returns a
  response distinct from a benign miss (path traversal `403` vs `404`; cmdi
  `"input rejected"` vs the diagnostic echo; ssti `"template directive ignored"`
  vs a plain render) — so the agent learns it's hitting the right vuln class with
  the wrong technique. These reshape only the *non-leak* responses, so the
  replay matrix is unchanged.

  *Structural variety, honestly.* Path traversal samples base dirs of **varied
  depth (2–5)**, so the `../` count is build-specific structure the agent reads
  off the world. But the asymmetry is partly intrinsic: SQLi embeds world state
  (table + column) in the payload (≈108 distinct structures), while a file-read /
  cmd-exec payload embeds only the discovered path (a handful) — those classes
  carry their diversity in *three distinct techniques* per build, not in payload
  structure. *Threat to validity (documented):* for `sql_injection`, `idor`, and
  `weak_credentials` the three contexts are disjoint *serializations* of one skill
  (a quote/encoding swap), not three distinct competencies — they defeat replay
  but don't broaden the skill the way cmdi / path / ssti / xxe / ssrf /
  broken-authz do. Richer structural variety and natural-language realism remain
  the later LLM layer.

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

Items 1–7 are **done** (`feat/cyber-staged-generation`): the staged loot→vuln
pipeline, the in-memory file store, shape-tagged catalog + shape-matched oracle
selection, and the fan-out. The gym now spans **3 exploit shapes across 9
classes**, each proven end to end by a real HTTP exploit that recovers the flag
(`tests/test_cyber_staged_generation.py`):

| shape | classes |
| --- | --- |
| response-leak | `sql_injection`, `ssrf`, `broken_authz`, `idor`, `weak_credentials` |
| file-read | `path_traversal`, `xxe` |
| code-exec | `command_injection`, `ssti` |

Loot shape and vuln-class mix are manifest-configurable (`loot_shapes` /
`vuln_kinds`); decoy files are sampled into the content-addressed graph (not
hardcoded), so they vary by seed. Every class also samples a **payload-context
axis** per build (§6) so the correct exploit differs build-to-build, the flag
path is **discovered via a planted config** rather than guessed (and randomized
so brute force doesn't pay), and the 4 once-toy classes run **real engines**
(Jinja / `xml.sax` / `shlex`). This tracks
[#190](https://github.com/vecna-labs/open-range/issues/190) and lays the staging
groundwork for [#212](https://github.com/vecna-labs/open-range/issues/212).

### Default loot mix, and what stays an emulation

The default weights the response-leak shape (`db: 7`, `file: 3`) — the most common
real web-exploit class — while still producing file/exec worlds out of the box; a
study targets a shape or class by overriding `loot_shapes` / `vuln_kinds`. The
default is a starting point, not a claim about the "right" distribution.

At the `PROCESS` backing the loot store is an in-memory map (the flag never
lands on disk), but the exploits run against **real engines** wherever one fits
in-process: SQL injection hits a real sqlite engine, SSTI a real sandboxed Jinja
environment (`{{7*7}}` → 49, context dump leaks the store), XXE a real SAX parser
with external-entity resolution (a well-formed DOCTYPE/ENTITY/reference is
required), path traversal a real `posixpath` resolve, and command injection a
real `shlex` tokenizer honoring separators, `$()`/backtick substitution, quoting,
and arbitrary readers. So the *technique* — not a magic string — is what the
agent must produce, which is what makes it transfer. The one thing still emulated
is a real OS shell/filesystem with RCE escalation: a **container backing
([#252](https://github.com/vecna-labs/open-range/issues/252))** provides that.
Because no real shell executes at `PROCESS` (the interpreters only read the
store), there is nothing to sandbox yet;
exec-sandbox hardening ([#202](https://github.com/vecna-labs/open-range/issues/202))
lands with the container backing, before adversarial training traffic.

### Trainability: live-agent validation and the difficulty tier

Scripted-oracle tests (which exploit using the flag path read straight from the
graph) prove a world is *solvable by construction* — they do **not** prove a real
agent can solve it. Driving a real LLM agent through the actual episode harness
(`run.run_episode` with a `claude -p` solver) revealed that the validity-hardened
**standard** tier is too hard for a fresh agent: it solved only ~2 of 9 classes,
because the thin instruction left it unable to classify the vuln and the
discovery recon chain made file-loot a two-stage exploit it couldn't walk
(`command_injection` failed even with rich hints and a 20-minute budget). Validity
(replay-resistance, discovery-not-brute-force) and trainability trade off, and the
hardening pushed past one-step real-agent solvability.

So the gym carries a `difficulty` manifest knob:

| tier | instruction | use |
| --- | --- | --- |
| `standard` (default) | thin (endpoint only); blind recon + classification, mutually-exclusive contexts | the H2 transfer **measurement** target |
| `easy` / `guided` | names the vuln class, the flag's exact location, the sampled context, and a one-step payload recipe | **bootstrapping** — an agent learns to *execute* exploits before it has to *discover* them |

The agent still crafts and executes the real exploit at `easy`; only the recon and
classification are removed. A live-agent matrix (9 classes × 2 contexts, a real
agent through the real harness) solves **18/18 at `easy`** versus ~3/22 at
`standard` — the gym is real-agent-trainable via the `easy` tier and a
manifest-driven easy→standard curriculum. (Auto-curriculum via `auto_evolve` and a
richer default instruction are the natural next steps —
[#258](https://github.com/vecna-labs/open-range/issues/258).)

### Out of scope here

Client-side shapes (XSS, CSRF) need a victim NPC and wait. The LLM intra-class
diversity layer (§2) waits — params vary per build today, but the code *shape*
per class is fixed; richer structural variety is the documented next step.
