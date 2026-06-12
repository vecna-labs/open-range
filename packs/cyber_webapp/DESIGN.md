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

---

## 8. The verifier is the ceiling — verification, reward, and the path past plant-by-construction

§2 set the line: procedural owns correctness, the LLM owns variety behind
admission. This section answers the three questions that line raises once you take
it seriously — *what is the verifier, why does it set the agent's ceiling, and how
does generation move to the LLM without losing the measurement* — and records the
direction decided for the sim-to-real study.

### 8.1 Two ways to prove a world solvable

A generated world is training data only if it is provably solvable, and the proof
is always the same shape: exhibit a solution a checker accepts. Two places to put
that proof:

- **Plant-by-construction** (today, §3; the LAVA / Juliet lineage). Staging plants
  a known vuln so a known technique reaches a planted flag; `pentest.py::check_success`
  confirms `submitted == flag.value_ref`. Deterministic, cheap, reproducible — and
  **bounded by the catalog**: the agent can only learn the classes we plant.
- **Generate-then-verify** (the AgentWorld lineage). An LLM writes the world *and*
  a solver *and* a checker; admit if the solver passes the checker. General and
  realistic — but the proof is only as trustworthy as the LLM that wrote it, and
  **a generator and a verifier that are the same model share blind spots**: the toy
  `{{7*7}}` engines our own audit caught (§6, since fixed) *passed their own tests*.
  A self-checking LLM loop admits exactly those.

Neither is the answer alone. Plant-by-construction is measurement-grade but capped;
generate-then-verify scales but self-certifies. The synthesis is to **take the
LLM's generator and refuse its verifier-as-truth** — keep an independent verifier,
and never let the model own the flag or the checker.

### 8.2 The verification ladder

Order verifiers by trust, lowest ceiling to highest:

| rung | verifier | judge? | ceiling |
| --- | --- | --- | --- |
| 1 | **planted-flag match** (`check_success` today) | none | the catalog |
| 2 | **report ↔ graph structure** — agent's `{kind, endpoint, technique}` vs the graph's `vulnerability` node + `affects` edge | none | declared vulns |
| 3 | **invariant violation** — a `HIDDEN` value reaches output it shouldn't | none | the invariants you state |
| 4 | **execution effect** — a real boundary crossed in a sandbox | none | what you instrument |
| 5 | **LLM judge** | yes | the judge |

The design rule: **push verification down this ladder, reserve the judge for the
irreducible tail.** Rungs 1–4 are mechanical and judge-free; only genuinely
ambiguous findings (subtle logic flaws, disclosure of debatable sensitivity) need
rung 5. So the gym is *not* fundamentally capped at a judge — it is capped by how
much of "what counts as a violation" we can mechanize, and rungs 3–4 mechanize most
of security.

There is no `Claim` primitive in the graph (`_ir.py` has `Node`, `Edge`,
`Visibility`, `Role`). Rung 2's "ground truth" is already present as **edges**:
`holds` ("this record holds this secret in this field"), `affects` ("this vuln
affects this endpoint"), and `Visibility.HIDDEN` on the secret. A report-vs-graph
check matches against those — no new primitive, no judge.

### 8.3 The spine: one change unifies the ladder

The whole architecture lands on a single generalization of code that already
exists. `check_success` today asks *did the one planted flag appear in a response*:

```
expected = flag.attrs["value_ref"]; ok = submitted == expected
```

Generalize it to *did any `HIDDEN` value reach output it should not have* — and the
same function becomes rung 1 **and** rung 3. That one move:

- **keeps planted mode** (the planted flag is a `HIDDEN` value, so the check still
  fires);
- **unlocks emergent mode** (a leak the generator never planted still trips it — no
  planted flag required);
- **is the judge-free verifiable reward** a GRPO trainer needs (a programmatic
  check — the cyber analog of "is the math answer correct");
- **catches novel exploits**, because it watches the *consequence* (a hidden value
  escaped), not a *mechanism* (a specific CWE).

This is the spine of everything below: a ~10-line generalization of
`pentest.py::check_success`, validated against worlds we already trust. (Whether a
leak came via the *intended* technique or a shortcut is a separate question — the
mutual-exclusivity / no-shortcut probe of §6 is the validity gate; consequence
verification supplies the *reward*, the shortcut probe supplies the *label*.)

### 8.4 Instrument consequences, not mechanisms

Mechanisms are infinite and evolving; you cannot enumerate them, and enumerating
them *is* the catalog ceiling. **Consequences are few and stable** — an
unauthenticated read of `HIDDEN` data, a write across a boundary, code execution,
exfil of a planted canary. Instrument the consequence and a mechanism that reaches
it is confirmed regardless of how it got there — including one the generator never
intended. That is how the gym exceeds the model that builds it.

The honest qualifier: the oracle matches by substring, but it searches for the value
**and its cheap reversible encodings** — base64, hex, percent-encoding — by encoding
the *needle*, so a base64/hex/url-encoded exfil is caught, not only the literal form.
Still out (these would need decoding the body, not encoding the needle): gzip/binary
transforms, multibyte splits, bespoke schemes. The live per-response signal is raw and
un-de-duplicated; the offline verifier and the grader (which hold the graph) apply
containment de-duplication when multiple guarded values overlap.

How far this reaches is gated by backing (§"what stays an emulation"):

- **At `PROCESS` (now):** the only observable consequence is a value reaching an
  HTTP response — response-leak. `check_success`'s `flag_from_response` is already
  this in embryo; generalizing it to *any* `HIDDEN` value (8.3) is in reach today.
- **At `CONTAINER` ([#252](https://github.com/vecna-labs/open-range/issues/252) /
  [#202](https://github.com/vecna-labs/open-range/issues/202), Docker-blocked
  locally):** real OS effects — a file read outside web root, a process spawned —
  become observable. File-read and code-exec consequences light up only here. This
  is the gating dependency for the upper ladder, and it is the same container the
  benchmarks ride on.

### 8.5 Generation ≠ finding — why a mediocre builder is enough

Producing software with real flaws is an easier, *different* competence than
finding them (the generator/discriminator gap GANs and self-play exploit). A
mediocre LLM writing a webapp leaks genuine bugs it never intended; finding those
is real skill, uncorrelated with the builder's own finding ability. **The catch:**
this only holds for *emergent* bugs. The moment we plant a catalog class the bug is
not emergent and the ceiling is the catalog again. So the two modes coexist by
design:

| mode | proof | reproducible? | ceiling | role |
| --- | --- | --- | --- | --- |
| **planted** | construction + flag-match | fully (seed) | catalog | the controlled H2 **measurement** axis |
| **emergent** | consequence verification (8.3) | via build-time freeze | generated-software diversity | the ceiling-raising **research** axis |

The consequence verifier (8.3) is what *unifies* them: planted mode checks "the
planted value leaked," emergent mode checks "any hidden value leaked," same
function. Emergent mode is a real departure from §3's plant-by-construction and is
the new work; planted mode stays exactly as it is, because the study needs a
reproducible, known-ground-truth axis to measure transfer against. An LLM in the
build path trades pure seed-determinism for **generate-verify-freeze**: generate
once, verify by consequence, freeze to a content-addressed snapshot — the study
reads frozen worlds, so reproducibility holds.

### 8.6 Where the reward and the trainer live — the boundary

The gym builds, admits, and verifies worlds; it **never runs the agent or the RL
loop**. So:

- **Gym (this pack):** the verdict surface — `check_success` and its generalization
  (8.3), the report-vs-graph check (rung 2), the graph-wide invariant callables
  `Ontology.validate` already accepts. This is the verifiable reward *source*.
- **Trainer (`openrange_trl`, the consumer):** GRPO itself. GRPO removes the *value
  network*, not the reward — its judge-free property comes from the reward being
  *verifiable* (DeepSeek-R1-Zero: GRPO + rule reward, no critic, no reward model).
  The gym supplies that verifiable reward; the trainer computes group-relative
  advantage. `test_trl_cyber.py` already wires this: the world's held-out verdict,
  graded over HTTP, is the reward, and GRPO needs only that different actions earn
  different grades.

On reward *shape*: GRPO needs variance within a group, and a binary leak/no-leak
signal is sparse. The pentest verdict already returns **three rungs**
(`reached_endpoint → extracted_anything → matched_flag`, all graph-observable) —
that graded surface is the variance GRPO learns from, and it generalizes with 8.3.
(The exploit chain can densify further as potential-based shaping, but shaping
toward the *planted* chain biases against novel paths — use it for the
`easy`/bootstrapping tier, drop it when chasing emergent findings.)

Keeping GRPO in the trainer and the verdict in the gym is not pedantry — putting
rollout/eval in the gym is the category error this project has hit before.

### 8.7 So who sets the ceiling

Not the builder's finding ability (generation ≠ finding). Not the judge (mechanize
below it, rungs 1–4). The ceiling is **the diversity of software the generator can
emit × the expressiveness of the consequences and invariants we instrument** — both
higher and more honest limits than "a mediocre LLM" or "a judge's taste." The
co-evolution is productive because of the asymmetry: bugs are *easy to make*, *hard
to find*, *cheap to confirm once reached*, so generator and agent climb together
without either being a great vuln-hunter. The genuine frontier limit is a novel
*class* — a consequence type never instrumented; you cannot confirm a violation of
a property you never stated. That is real and far-off: consequence instrumentation
reaches novel *instances and chains* of known property-violations (most of real
pentesting); new categories stay human-seeded. A fine place for the wall.

### 8.8 First step — the experiment that earns the design

Before trusting any of this, prove the verifier is worth trusting, then use it to
indict the naive loop:

1. **Build + validate the consequence verifier on one class** (e.g.
   `command_injection`): generalize `check_success` to "any `HIDDEN` value leaked,"
   and confirm it against worlds we already trust — it must pass every existing
   faithful exploit and reject every trivial/neutralized negative in
   `test_cyber_staged_generation.py`. *The verifier is validated against known
   ground truth before it audits anything.*
2. **Run the indictment:** have an LLM generate world + solver + self-checker for
   that class, N times. Grade each by (a) its own self-check and (b) the validated
   verifier. **Measure the admit-gap** — worlds the self-check passes that the
   independent verifier rejects as trivially-solvable or unfaithful — and let the
   number be what it is (see §8.10).

Scope honestly: this runs at `PROCESS`, so it exercises the response-leak
consequence only; file-read / code-exec consequences wait on the container (8.4).
One class, one figure, the whole mechanism proven small.

### 8.10 The indictment runs — what they actually showed

A one-off validation experiment (run 2026-06-11; scaffolding not kept in the repo) ran
89 LLM-generated worlds (Claude sonnet/haiku) across `command_injection`,
`sql_injection`, `ssti`, `path_traversal`, guided and unguided, the final run carrying
each generator's **own checker** (the real loop) plus a computed faithfulness control.

| run | shipped | broken (self-caught) | admit-gap |
| --- | --- | --- | --- |
| A — guided cmdi (21) | 21 | 0 | 1 (trivial) |
| B — unguided cmdi (20) | 6 | 14 | 0 |
| C — 4 classes, real checkers (48) | 43 | 5 | 1 (faked engine) |

The honest result is **narrower than this section first predicted, and consistent:**
the admit-gap is **~2–4% of shipped worlds**, and did not widen with harder classes,
bigger N, or real LLM checkers. Findings:

1. The self-check is a **strong, necessary filter** — it catches the dominant failure,
   *unsolvable* worlds (14/20 unguided cmdi never self-admit: faked tool output, the
   stub reused, exploits that echo the literal marker).
2. Two predicted amplifiers **didn't materialise.** Generated checkers were mostly
   sound (`flag in response`), so broken worlds were self-rejected, not shipped; and
   weak generation produces *more broken* worlds, not *more contaminated* ones.
3. **The harder finding: an independent verifier is itself hard to get right.** The
   harness mis-fired in both directions, each found only by hand-auditing real worlds —
   false *negatives* (it took three probe iterations to stop passing `log-file-viewer`
   trivial worlds) and false *positives* (judging faithfulness on the generator's own
   "wrong" query flagged a real shell and real jinja2 as unfaithful; the fix is to judge
   only on the computed control). The reliable signals are **triviality** and
   **faked-engine**; the generator-supplied wrong-vector is not.

So the defensible claim is not "self-verification fails" but: **its gross failures
(broken worlds) are self-caught, the residual impurity is a small consistent tail
(~2–4%) invisible to the self-check, and the genuinely hard part is building an
independent verifier reliable enough to measure that tail.** That tail still matters —
a `command_injection` set even 2% arbitrary-file-read biases a per-class transfer
number — which is the independent verifier's job.

### 8.9 Status of this direction

| piece | state |
| --- | --- |
| planted-flag verifier (rung 1) | **done** — `check_success` |
| graded reward rungs for GRPO variance | **done** — pentest subgoals; `test_trl_cyber.py` |
| LLM behind admission (instruction, mutation enrichment) | **done** — `llm_generation.py`, strictly non-correctness-path |
| any-hidden-leak verifier (rung 1+3 spine, 8.3) | **done** — `consequence.py`, validated on all 9 classes |
| indictment: independent probes + run (8.8, 8.10) | **done** (one-off; scaffolding not kept) — gap ~2–4% |
| wire the verifier into live `check_success` (runtime leak-capture) | **done** — seed → app scan → `leaked_secret_ids` → `check_success` |
| LLM generates the *world* (emergent mode, 8.5) | **next** — not yet a gym mode |
| report ↔ graph check (rung 2) | designed, unbuilt |
| execution-effect consequences (rung 4) | **blocked** on container ([#252](https://github.com/vecna-labs/open-range/issues/252) / [#202](https://github.com/vecna-labs/open-range/issues/202)) |
| novel-class discovery | far-future, human-seeded (8.7) |

---

## 9. Emergent mode at scale: the realization ladder

§8 built the *verifier*. This is what it unlocks: stop templating worlds and let an
LLM **realize** them — keeping procedural as the architect and the verifier as the
gate, at rising fidelity.

The invariant at every rung: **procedural architects the graph** (topology, flag
placement, the solvability skeleton — the controllable, scalable, solvable-by-
construction part that is OpenRange's differentiator); **the LLM realizes each node**
into a real, varied service; **admission verifies** (the consequence oracle + the
shortcut/faithfulness probes of §8.10) that the realization is still solvable and not
*trivially* so; **the result freezes** to a content-addressed snapshot, so the study
stays reproducible even with an LLM in the build path.

Why the mix, not pure-LLM: an LLM asked for "a vulnerable world" gives *one* world,
low controllability, and — §8.10 measured this — mostly *broken* ones. The procedural
engine is the controllable variation source; the LLM is realism *per node, behind
admission*. The LLM never architects correctness.

The ladder (each rung an existing issue except M0):

| rung | the LLM realizes | runtime | issue |
| --- | --- | --- | --- |
| **M0** | a vuln *handler* — varied implementations within a class, dynamically admission-gated by run-the-exploit | `PROCESS` (today) | *new* |
| **M1** | a node as a real **container** image — real fs/shell ⇒ real RCE/file-read | `Backing.CONTAINER` | [#252](https://github.com/vecna-labs/open-range/issues/252) |
| **M2** | **multiple** networked services; graph edges become real links — SSRF→internal, pivot, credential reuse | containers + net | [#212](https://github.com/vecna-labs/open-range/issues/212), [#235](https://github.com/vecna-labs/open-range/issues/235) |
| **M3** | a **k8s** topology — pods/services/network-policies/RBAC; lateral movement + k8s-native classes (RBAC escalation, SA-token theft, netpol bypass, pod escape) | Kind | [#189](https://github.com/vecna-labs/open-range/issues/189) |

M0 is the realization *primitive* every rung is built from: the **dynamic admission
gate** — render the LLM's realization, run the intended exploit, confirm the flag
leaks via `consequence.detect_leak`, confirm a benign request does *not* — is what
makes letting an LLM write the world safe. (Today's admission is *structural* — a
graph-path check; an LLM realization needs *dynamic* admission, because the code
might be wrong.) Exec-effect faithfulness rides the container
([#202](https://github.com/vecna-labs/open-range/issues/202) sandbox). This is also
the sim-to-real fidelity ladder (`PROCESS` → `CONTAINER` → cluster) the H2 study
measures on.
