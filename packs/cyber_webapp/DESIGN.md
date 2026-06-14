# Cyber webapp pack — generation design

How this pack generates worlds, and why it generates them the way it does. The
[README](README.md) shows *what* one built world looks like; this explains the
*generator* behind it: **staged, procedural, constraint-propagating generation**
that produces a wide range of exploit types while staying solvable by construction.

Audience: anyone extending the builder, the vulnerability catalog, or the ontology —
and the sim-to-real study that depends on the gym being *broad*.

---

## 1. The bet, and the constraints it forces

The gym is a **cheap, reproducible, solvable source of training worlds** whose exploit
skills transfer to real benchmarks. Three constraints fall straight out of that and
decide the whole design:

- **Reproducible.** `snapshot_id = graph.content_hash()`. Same builder + manifest +
  seed → the same world, byte for byte. A nondeterministic generator breaks the thing
  OpenRange is built on.
- **Cheap at scale.** The bet is worlds by the thousand, so per-world cost is near zero.
- **Solvable by construction.** Every task is admission-checked before an episode; a
  generator that mostly produces unsolvable worlds and leans on reject-and-repair is
  wasteful.

All three point the same way: **the correctness-critical core of generation is
procedural, not LLM-driven.** Not anti-LLM — it's where the line falls.

---

## 2. Procedural owns correctness; the LLM owns variety

| | owns | why |
| --- | --- | --- |
| **Procedural** (the core) | the vuln mechanic, exploitability, feasibility, chaining, flag placement, base parameterization | must be deterministic, cheap, reproducible, solvable-by-construction |
| **LLM** (a layer behind admission) | open-ended structural diversity within a class, surface realism that pools can't cover | benefits from variety; a hallucination is **rejected by admission**, never trusted |

The line is sharp: **the LLM never generates the thing that must be correct.** Admission
makes any LLM use safe — generate, then verify the exploit actually fires; a bad
generation is dropped, not shipped. This is the "self-verifying generation" the gym
rests on. The lineage is pre-LLM: **LAVA** (automated vulnerability addition) and **NIST
Juliet/SARD** (procedurally generated CWE samples) injected exploitable bugs *with known
triggers* — self-verifying by construction, already the OpenRange model.

Even *realism* is procedural-first: realistic names and content come a long way from
curated pools sampled deterministically (`customer-portal`, `alice@corp.example`), no
model required ([#192](https://github.com/vecna-labs/open-range/issues/192)). The LLM is
reserved for diversity that pools and parameterized templates genuinely can't reach — and
an LLM in the build path trades pure seed-determinism for cache-keyed determinism (cache
by `(seed, prompt)`), a cost paid only where it buys something.

---

## 3. Staged, constraint-propagating generation

The world is generated in **ordered layers, each layer's output bounding the next
layer's choices** — top-down, not flat. This keeps a world coherent and makes
feasibility hold incrementally instead of being discovered after the fact, and keeps
each step a small sampling problem.

`sampling.py::sample_graph` runs network → services → hosts/endpoints → loot →
flag → accounts → vulns. The load-bearing propagation is loot → vuln:

```
loot-placement → picks loot shape ∈ {db-row, file, exec-reachable}   ← the constraint
        ↓ bounds
vuln-selection → picks an oracle vuln whose exploit reaches that shape
        ↓ bounds
realization    → renders the template + wires the exploit → flag path
```

Because the oracle vuln is *chosen to match the loot*, the chain is reachable **by
construction** — no reject-and-repair. That is the deep win of staging: solvability is
assembled layer by layer. The same pattern generalizes upward to enterprise scale
([#212](https://github.com/vecna-labs/open-range/issues/212)): org → team → service →
data → vuln, each layer bounding the next.

---

## 4. Exploit *shapes*, not CWE names

The catalog is organized by **exploit shape** — *how the flag is reached* — not by CWE
label. The shape is the unit of real work (realizer + feasibility); classes within a
shape are cheap templates on top. The shape is also the **agent capability** the study
measures (H2, "which capabilities survive simulation," is per-shape by nature), so
shape-organization is the study's axis, not tidiness.

| shape | how the flag is reached | classes |
| --- | --- | --- |
| **response-leak** | exploit returns the flag in an HTTP response | `sql_injection`, `ssrf`, `broken_authz`, `idor`, `weak_credentials` |
| **file-read** | exploit reads a file holding the flag | `path_traversal`, `xxe` |
| **code-exec** | exploit runs code that reads the flag | `command_injection`, `ssti` |

Each class is proven end to end by a real HTTP exploit that recovers the flag
(`tests/test_cyber_staged_generation.py`). Loot shape and class mix are
manifest-configurable (`loot_shapes` / `vuln_kinds`).

---

## 5. Ontology: file/exec loot reuses `data_store`

File-read and code-exec loot lives somewhere other than a DB row, and the ontology
already accommodates it: `data_store.kind` is `{sql, kv, file, object}` and `engine` is
`{sqlite, postgres, mysql, redis, fs, s3}` (`ontology.py`). So filesystem loot is just
`kind=file, engine=fs` — **no new node kind, only realizer support**. Such a store
materializes its record as a real file under the owning service; path-traversal reads it
directly, command-injection `cat`s it — one shape, two exploit classes. Feasibility is a
per-*shape* structural check ("a loot path of the matching shape exists from the
entrypoint"), not per-*class*.

---

## 6. Replay resistance: mutually-exclusive payload contexts

A structurally fixed template teaches *the template*, not the technique: if SQLi is
always `... WHERE key = '{input}'`, an agent memorizes one string and replays it. Since
the agent only ever sees the HTTP surface (never server code), each class instead samples
an **injection context** per build, and the three contexts of a class are **mutually
exclusive** — the handler enforces each one's requirement, so a payload that solves one
build *fails* the other two:

> SQLi single/numeric/double quoting; cmdi separator/substitution/quoted (each strips the
> others' vectors); path traversal absolute-only / relative-`../` / `....//`-past-a-single-strip;
> SSTI attribute/comment/expr sink; XXE element / wrapped-root / scheme-prefix; SSRF
> scheme-block / host-allowlist / decimal-IP; IDOR direct/base64/prefixed; broken-authz
> single/dual-factor/encoded; weak-creds pair/combined/basic.

A 3×3 live replay matrix per class is **fully diagonal** — every off-diagonal cell
rejects — so the single-payload replay floor is ~33%, down from ~67%: an agent must learn
all three techniques. (XXE's `element_content` vs `wrapped_root` is kept distinct by having
`element_content` reflect only the root's *direct* text while `wrapped_root` nests the
entity a level deeper.)

**Wrong-context feedback.** A neutralized but attack-shaped attempt returns a response
distinct from a benign miss (path traversal `403` vs `404`; cmdi `"input rejected"` vs the
diagnostic echo; ssti `"template directive ignored"` vs a plain render), so the agent
learns it has the right vuln class but the wrong technique. These reshape only the
*non-leak* responses, so the replay matrix is unchanged.

**The honest limit.** SQLi embeds world state (table + column) in the payload (≈108
distinct structures); a file-read / cmd-exec payload embeds only the discovered path. For
`sql_injection`, `idor`, and `weak_credentials` the three contexts are disjoint
*serializations* of one skill (a quote/encoding swap), not three distinct competencies —
they defeat replay but don't broaden the skill the way cmdi / path / ssti / xxe / ssrf /
broken-authz do. Richer structural variety is the LLM layer's job (§2).

---

## 7. What is real vs emulated, and the difficulty tiers

At the `PROCESS` backing the loot store is an in-memory map (the flag never lands on
disk), but the exploits run against **real engines** wherever one fits in-process: SQL
injection hits a real sqlite engine, SSTI a real sandboxed Jinja environment (`{{7*7}}` →
49, a context dump leaks the store), XXE a real SAX parser with external-entity resolution,
path traversal a real `posixpath` resolve, command injection a real `shlex` tokenizer
honoring separators, `$()`/backtick substitution, and quoting. So the *technique* — not a
magic string — is what the agent must produce, which is what transfers. The one thing
`PROCESS` emulates is a real OS shell/filesystem with RCE; the **`CONTAINER`** backing
(§9) makes those real. The default loot mix weights response-leak (`db: 7`, `file: 3`),
the most common real web-exploit class; it is a starting point, not a claim about the
"right" distribution.

Validity and trainability trade off: the replay-hardened, recon-required world is too
hard for a fresh agent to solve in one step, so the gym carries a `difficulty` knob.

| tier | instruction | use |
| --- | --- | --- |
| `standard` (default) | thin (endpoint only); blind recon + classification, mutually-exclusive contexts | the H2 transfer **measurement** target |
| `easy` / `guided` | names the vuln class, the flag's location, the sampled context, and a one-step payload recipe | **bootstrapping** — learn to *execute* exploits before having to *discover* them |

At `easy` the agent still crafts and executes the real exploit; only recon and
classification are removed. A live-agent matrix solves 18/18 at `easy` versus ~3/22 at
`standard`, so the gym is real-agent-trainable via an `easy → standard`
curriculum. Client-side shapes (XSS, CSRF) need a victim NPC and are out of scope here.

---

## 8. The verifier is the ceiling

§2 set the line: procedural owns correctness, the LLM owns variety behind admission. This
section answers what that line raises — *what the verifier is, why it sets the agent's
ceiling, and how generation moves to the LLM without losing the measurement.*

### 8.1 Two ways to prove a world solvable

A generated world is training data only if it is provably solvable, and the proof always
has the same shape: exhibit a solution a checker accepts. Two places to put that proof:

- **Plant-by-construction** (§3; the LAVA / Juliet lineage). Staging plants a known vuln
  so a known technique reaches a planted flag; `pentest.py::check_success` confirms
  `submitted == flag.value_ref`. Deterministic, cheap, reproducible — and **bounded by the
  catalog**: the agent can only learn the classes we plant.
- **Generate-then-verify** (the AgentWorld lineage). An LLM writes the world *and* a solver
  *and* a checker; admit if the solver passes the checker. General and realistic — but the
  proof is only as trustworthy as the LLM that wrote it, and **a generator and a verifier
  that are the same model share blind spots** (a self-checking loop admits exactly the toy
  engines it can't see are toys).

Neither alone is the answer: plant-by-construction is measurement-grade but capped;
generate-then-verify scales but self-certifies. The synthesis is to **take the LLM's
generator and refuse its verifier-as-truth** — keep an independent verifier, and never let
the model own the flag or the checker.

### 8.2 The verification ladder

Verifiers ordered by trust, lowest ceiling to highest:

| rung | verifier | judge? | ceiling |
| --- | --- | --- | --- |
| 1 | **planted-flag match** (`check_success`) | none | the catalog |
| 2 | **report ↔ graph structure** — agent's `{kind, endpoint, technique}` vs the graph's `vulnerability` node + `affects` edge | none | declared vulns |
| 3 | **invariant violation** — a `HIDDEN` value reaches output it shouldn't | none | the invariants you state |
| 4 | **execution effect** — a real boundary crossed in a sandbox | none | what you instrument |
| 5 | **LLM judge** | yes | the judge |

The rule: **push verification down this ladder, reserve the judge for the irreducible
tail.** Rungs 1–4 are mechanical and judge-free; only genuinely ambiguous findings (subtle
logic flaws, debatable disclosure) need rung 5. So the gym is not fundamentally capped at a
judge — it is capped by how much of "what counts as a violation" we mechanize, and rungs
3–4 mechanize most of security. Rung 2 needs no new primitive: its ground truth is already
in the graph as **edges** (`holds`, `affects`) and `Visibility.HIDDEN` on the secret.

### 8.3 The spine: one check unifies the ladder

The whole architecture lands on a single generalization of `check_success`. Instead of
*did the one planted flag appear in a response* (`submitted == flag.value_ref`), the
consequence verifier (`consequence.py`) asks *did any `HIDDEN` value reach output it
should not have* — so the same function serves rung 1 **and** rung 3. That one move:

- **keeps planted mode** (the planted flag is a `HIDDEN` value, so the check still fires);
- **unlocks emergent mode** (a leak the generator never planted still trips it);
- **is the judge-free verifiable reward** a GRPO trainer needs (a programmatic check — the
  cyber analog of "is the math answer correct");
- **catches novel exploits**, because it watches the *consequence* (a hidden value
  escaped), not a *mechanism* (a specific CWE).

Whether a leak came via the *intended* technique or a shortcut is a separate question: the
mutual-exclusivity / no-shortcut probe of §6 is the validity *gate* (the label),
consequence verification supplies the *reward*.

### 8.4 Instrument consequences, not mechanisms

Mechanisms are infinite and evolving — enumerating them *is* the catalog ceiling.
**Consequences are few and stable**: an unauthenticated read of `HIDDEN` data, a write
across a boundary, code execution, exfil of a canary. Instrument the consequence and a
mechanism that reaches it is confirmed regardless of how it got there — including one the
generator never intended. That is how the gym exceeds the model that builds it.

The oracle matches by substring, but searches for the value **and its cheap reversible
encodings** (base64, hex, percent-encoding) by encoding the *needle*, so an encoded exfil
is caught, not only the literal form. Out of reach (these need decoding the body, not
encoding the needle): gzip/binary transforms, multibyte splits, bespoke schemes. The live
per-response signal is raw; the offline verifier and grader (which hold the graph)
de-duplicate by containment when guarded values overlap. How far the verifier reaches is
gated by backing:

- **At `PROCESS`:** the only observable consequence is a value reaching an HTTP response —
  response-leak.
- **At `CONTAINER`:** real OS effects — a file read outside web root, a process spawned —
  become observable, so file-read and code-exec consequences light up — rung 4 is gated by
  this backing ([#252](https://github.com/vecna-labs/open-range/issues/252)); until it
  lands the verifier works at rungs 1–3.

### 8.5 Generation ≠ finding — why a mediocre builder is enough

Producing software with real flaws is an easier, *different* competence than finding them
(the generator/discriminator gap GANs and self-play exploit). A mediocre LLM writing a
webapp leaks genuine bugs it never intended; finding those is real skill, uncorrelated with
the builder's own finding ability. The catch: this holds only for *emergent* bugs — the
moment we plant a catalog class the bug is not emergent and the ceiling is the catalog
again. So the two modes coexist by design:

| mode | proof | reproducible? | ceiling | role |
| --- | --- | --- | --- | --- |
| **planted** | construction + flag-match | fully (seed) | catalog | the controlled H2 **measurement** axis |
| **emergent** | consequence verification (8.3) | via build-time freeze | generated-software diversity | the ceiling-raising **research** axis |

Emergent mode is a real departure from §3's plant-by-construction and is the new work;
planted mode stays exactly as it is, because the study needs a reproducible,
known-ground-truth axis to measure transfer against. The consequence verifier unifies them
— planted mode checks "the planted value leaked," emergent mode checks "any hidden value
leaked," same function. An LLM in the build path
trades pure seed-determinism for **generate-verify-freeze**: generate once, verify by
consequence, freeze to a content-addressed snapshot — the study reads frozen worlds, so
reproducibility holds.

### 8.6 Where the reward and the trainer live — the boundary

The gym builds, admits, and verifies worlds; it **never runs the agent or the RL loop.**

- **Gym (this pack):** the verdict surface — `check_success` and its consequence
  generalization (8.3), the report-vs-graph check (rung 2), the graph-wide invariant
  callables `Ontology.validate` accepts. This is the verifiable reward *source*.
- **Trainer (`openrange_trl`, the consumer):** GRPO itself. GRPO removes the *value
  network*, not the reward — its judge-free property comes from the reward being
  *verifiable* (DeepSeek-R1-Zero: GRPO + rule reward, no critic, no reward model).
  `test_trl_cyber.py` wires this: the world's held-out verdict, graded over HTTP, is the
  reward; the trainer computes group-relative advantage.

GRPO needs variance within a group, and a binary leak/no-leak signal is sparse. The
pentest verdict already returns **three graded rungs** (`reached_endpoint →
extracted_anything → matched_flag`, all graph-observable) — that surface is the variance
GRPO learns from, and it generalizes with the spine (§8.3). (Potential-based shaping toward the *planted* chain biases against novel
paths — use it for the `easy` tier, drop it when chasing emergent findings.) Putting
rollout/eval in the gym is the category error to avoid.

### 8.7 So who sets the ceiling

Not the builder's finding ability (generation ≠ finding). Not the judge (mechanize below
it, rungs 1–4). The ceiling is **the diversity of software the generator can emit × the
expressiveness of the consequences and invariants we instrument.** The co-evolution is
productive because of the asymmetry: bugs are *easy to make*, *hard to find*, *cheap to
confirm once reached*, so generator and agent climb together without either being a great
vuln-hunter. The genuine frontier limit is a novel *class* — a consequence type never
instrumented; you cannot confirm a violation of a property you never stated. That is real
and far-off: consequence instrumentation reaches novel *instances and chains* of known
property-violations (most of real pentesting); new categories stay human-seeded.

### 8.8 The admission gap

An independent consequence verifier, run against LLM-generated worlds (world + solver +
self-checker; 89 audited across four classes, guided and unguided), shows the durable shape
of generate-then-verify: the self-check is a strong,
necessary filter — it catches the dominant failure, *unsolvable* worlds — but leaves a
small, consistent tail (**~2–4% of shipped worlds**) that it passes and an independent
verifier rejects as trivial or unfaithful. The tail does not widen with harder classes or
real LLM checkers. The genuinely hard part is the independent verifier itself: it mis-fires
in both directions, and the reliable signals are **triviality** and **faked-engine**, not
the generator's own claimed wrong-vector. The tail still matters — a `command_injection` set
even 2% arbitrary-file-read biases a per-class transfer number, which is the verifier's job
to measure.

---

## 9. LLM-realized services on the procedural graph

§8 builds the verifier; this is what it unlocks — stop templating worlds and let an LLM
**realize** them, keeping procedural as the architect and the verifier as the gate, at
rising realism. The invariant at every stage:

- **procedural architects the graph** — topology, flag placement, the solvability skeleton:
  the controllable, scalable, solvable-by-construction part that is OpenRange's
  differentiator;
- **the LLM realizes each node** into a real, varied service;
- **admission verifies** (the consequence oracle + the shortcut/faithfulness probes of §8.8)
  that the realization is still solvable and not *trivially* so;
- **the result freezes** to a content-addressed snapshot, so the study stays reproducible
  even with an LLM in the build path.

An LLM asked for "a vulnerable world" gives *one* world, low controllability, and mostly
*broken* ones (§8.8). The procedural engine is the controllable variation source; the LLM is
realism *per node, behind admission*, and never architects correctness. Each stage adds
realism over the last, and is the sim-to-real progression (`PROCESS` → `CONTAINER` →
cluster) the study measures on:

| the LLM realizes | runtime | issue |
| --- | --- | --- |
| a vuln *handler* — varied implementations within a class, admission-gated by running the exploit | `PROCESS` | [#260](https://github.com/vecna-labs/open-range/issues/260) |
| a node as a real **container** — real fs/shell, so file-read / RCE actually execute | `Backing.CONTAINER` | [#252](https://github.com/vecna-labs/open-range/issues/252) ([#265](https://github.com/vecna-labs/open-range/issues/265)) |
| **multiple** networked services; graph edges become real links — SSRF→internal, pivot, credential reuse | containers + net | [#212](https://github.com/vecna-labs/open-range/issues/212), [#235](https://github.com/vecna-labs/open-range/issues/235) |
| a **k8s** topology — pods / services / network-policies / RBAC; lateral movement + k8s-native classes | Kind | [#189](https://github.com/vecna-labs/open-range/issues/189) |

The first stage (#260) is the realization *primitive* every later one builds on: the
**dynamic admission gate** — render the LLM's realization, run the intended exploit, confirm
the flag leaks via `consequence.detect_leak`, confirm a benign request does *not*. (Today's
structural admission is a graph-path check; an LLM realization needs *dynamic* admission,
because the code might be wrong.)

### The container backing

The `CONTAINER` backing runs the one generated app (not a bespoke app per class). It sets
`OPENRANGE_REALFS`, which flips the rendered app's surfaces from in-memory emulation to the
real container; `PROCESS` leaves it unset and stays byte-for-byte the emulation:

- **file_read** (path_traversal, xxe) becomes real with zero handler changes — the `files`
  surface is a real filesystem (`_RealFiles`, a real `open()` per path), so a traversal
  escape is real OS path resolution.
- **code_exec** command_injection runs a real `sh -c`, the §6 mutually-exclusive contexts
  preserved by the same naive per-context filter over a real shell.

`ContainerWebappRuntime` is the runtime episodes use, selected by `Backing.CONTAINER`: it
reuses the subprocess runtime (`docker run` is the supervised child), resolves the host port
with `docker port`, and reads the leak signal out of the running container. The world
container that runs attacker code is contained — capabilities dropped, no-new-privileges,
memory/cpu/pid caps (`hardening_run_args`, `CapEff` all-zero inside yet still exploitable).
The load-bearing check is **cross-backing parity**: the same snapshot + same exploit grades
*identically* on `PROCESS` and `CONTAINER` — only fidelity changes, not the task surface.
Remaining container hardening (read-only rootfs, egress policy, flag-out-of-image, unsandboxed
ssti) is [#265](https://github.com/vecna-labs/open-range/issues/265); sandboxing the `exec`'d
verifier source is the separate, host-side
[#202](https://github.com/vecna-labs/open-range/issues/202).

### Two environments, not one

A generated world is the *target* the agent attacks over its HTTP surface (`base_url`); the
agent never runs inside it. So the world image carries only what its OWN behavior needs:
when a vuln runs a real OS command server-side (command_injection shelling to `ping` /
`nslookup`), that tool is installed *because the server runs it*, and only in worlds with
that vuln (`required_apt_packages`; a file-read-only world installs nothing). A world is not
a toolbox — recon/exploit tooling lives in the attacking agent's own sandbox (`solver_root`),
which the harness brings, hitting the world only over the network.

---

## 10. Networked multi-service: real network position

The single-container backing (§9) mounts every service by path prefix on one server, so
"internal" services are just `/svc/<name>` paths and SSRF is emulated in-process. This stage
makes network position **real**: a **public** service (the agent's only entry, published)
holds the SSRF; an **internal** service (no published port, reachable only on the container
network) holds the flag. The flag is reachable **only** by pivoting — the agent exploits the
SSRF to make the public service fetch the internal service's URL.

- **Per-service realization** (`realize_services`) splits the world into one container per
  `service` node, each carrying only its own endpoints and the state of the data_stores it is
  `backed_by`. The flag stays in its owning internal service and never enters the public image.
- **The networked runtime** (`NetworkedContainerWebappRuntime`) runs those containers on a real
  docker network, each reachable by name, publishing only the public service. The leak signal
  aggregates across every service's request log. `WebappPack.realize` routes here when a world
  is networked-shaped (`_is_networked`).
- **Generation** re-homes the SSRF onto the public endpoint and adds the internal half — a
  `metadata_credential_leak` endpoint serving the flag, which the SSRF `enables`; feasibility
  and entrypoint selection follow that pivot.
- **The exploit is real.** Under `OPENRANGE_NETWORKED` the SSRF handler `urlopen`s the internal
  host across the network. Docker-gated tests recover the flag only through the pivot (a benign
  fetch and a direct hit on the internal path leak nothing) and confirm the same exploit grades
  identically on `PROCESS` (in-process read) and the networked `CONTAINER` (real fetch) — only
  fidelity changes.

---

## 11. Company worlds and synthesized credential chains

A real target is not a hand-shaped pair of services; it is a small company's estate, and the
interesting part of the attack is finding the way in. This stage grows the generator to a
believable medium-size company the agent recons and pivots through — opt-in and additive (the
direct-pivot worlds of §10 are untouched), fully realized (lazy realization #235 is deferred
until this ground is stable).

**The shape.** Procedural architects a coherent estate: a public `web` portal in the `dmz`,
and internal services on a separate segment (`api`, `auth`, one or more `db` — one bears the
flag) plus off-path decoys, so the agent has to tell signal from noise. Names, hosts, and
accounts are realistic (curated pools, §2); the internal services are unpublished real
containers, reached only by name. The agent recons an over-sharing config endpoint for the
internal hostnames, then drives the public SSRF to fetch one across the network. A benign
internal fetch leaks nothing.

**Synthesized credential chains.** The richest worlds, and the first *composed* rather than
hand-shaped. The SSRF gains an opt-in *proxy* mode — the agent drives the pivot to any
internal host by name (a real `urlopen` on `CONTAINER`; the same in-process `/svc/<host>`
dispatch on `PROCESS`, so parity holds). On it the engine **synthesizes** a credential-reuse
chain of *sampled depth* from one composable primitive: an entry host leaks a db credential,
each gated host validates the credential leaked one hop back and relays the next, the last
serves the flag — so one preset yields 1-, 2-, 3-hop chains, a distribution not a fixed shape.
The flag is reachable ONLY through the final gate: the db record's value is a decoy, the real
flag lives in the gated secret. That composable hop is the action a search-based sampler
([#193](https://github.com/vecna-labs/open-range/issues/193)) composes and scores — the
substrate for "synthesize, don't hand-shape."

Correctness is as in §2–§3: procedural owns topology, flag placement, and the
solvable-by-construction chain; feasibility proves reachability across services
(`_enable_closure`); cross-backing parity stays the load-bearing check. The next stages on
this ground are enterprise scale
([#212](https://github.com/vecna-labs/open-range/issues/212)) via graph-driven lazy
realization ([#235](https://github.com/vecna-labs/open-range/issues/235)), then k8s
([#189](https://github.com/vecna-labs/open-range/issues/189)).
