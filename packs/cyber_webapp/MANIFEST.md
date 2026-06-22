# `webapp` manifest — one dial, auto ↔ specific

The cross-pack basics (the manifest shape, the one key core reads) live in
[docs/manifest.md](../../docs/manifest.md). This is the `webapp` pack's own key
reference.

The `webapp` manifest is the gym's control surface, and it obeys **one rule**:

> A knob **absent** means *auto* — the seeded RNG samples it from the catalog,
> deterministically and LLM-free. A knob **present** is a *constraint*, **merged**
> onto the defaults (never a silent replace). Specificity is just how many knobs you
> supply.

So the same surface spans the spectrum: supply nothing → the gym decides; supply some
→ pin those, auto-fill the rest; supply all → an exactly-specified, byte-reproducible
world. The open end — letting an LLM generate *beyond* the catalog — is the same
surface's `generate` knob (see below), always behind the consequence verifier.

```jsonc
// fully-auto      — the gym decides; seeded + reproducible
{"pack": "webapp", "seed": 7}

// partial         — bias one vuln kind, auto-fill the rest (count, loot, placement)
{"pack": "webapp", "seed": 7, "topology": "flat", "vuln": {"weights": {"xxe": 5}}}

// fully-specific  — exactly these vulns on a flat world, no LLM, byte-reproducible
{"pack": "webapp", "seed": 7, "topology": "flat",
 "vuln": {"pin": [{"kind": "sql_injection"}, {"kind": "idor"}]}, "loot": {"db": 1, "file": 0}}
```

## Knobs

- `seed` (int) — the determinism axis. Same manifest + seed → byte-identical world.
- `topology` (`"flat"` | `"company"` | `"chain"`, default `"flat"`) — the world shape.
  `"flat"` is a single-service target; `"company"` is a segmented multi-service estate
  with a public SSRF foothold pivoting to an internal flag; `"chain"` extends that into a
  credential-reuse chain (loot a token, replay it hop-by-hop). `company`/`chain` **force**
  their networked shape, so `vuln`/`loot` are not tunable there (set them only on `flat`).
- `scale` (mapping) — count-range overrides, **merged** per key: maps
  `service_count` / `endpoints_per_service` / `vuln_count` to `{"min": int, "max": int}`.
- `vuln` (mapping, `flat` only) — exactly one of:
  - `{"weights": {kind: int}}` — **bias** the sampling pool toward these kinds (merged
    over the defaults; the rest stay available).
  - `{"pin": [{"kind": K}, ...]}` — place **exactly** these kinds, one each
    (`vuln_count` becomes the pin length). Kinds must be distinct and the list non-empty.
  Unknown kinds, or internal-only chain kinds (composed via `topology`), raise `PackError`.
- `loot` (mapping, `flat` only) — `{"db": int, "file": int}`, merged over the defaults,
  fixing where the flag lives (queryable rows vs an in-memory file).
- `chain` (mapping, `chain` only) — `{"depth": {"min": int, "max": int}}` pins the
  credential-chain length; absent, depth is sampled.
- `recon` (`"full"` | `"none"`, company/chain) — whether the world discloses its internal
  hostnames so the agent can recon the pivot targets, or withholds them.
- `instruction_tier` (string) — shapes the task **text** (e.g. how much the prompt
  spells out). Distinct from `world_difficulty` (#322), the solve-path-cost metric the
  builder records on the snapshot's lineage.
- `generate` (`false` | `"vuln"` | `"novel"` | `"service"` | `"world"`, default `false`) —
  **the open end of the dial.** `false` keeps the world purely procedural. `"vuln"` routes
  the frozen procedural snapshot through host-side *generate → verify → freeze* (`llm_realize.
  realize_generated`, the host injecting the LLM and the episode boot): an LLM realizes each
  vuln's handler and the **consequence verifier** keeps it only if the exploit leaks and a
  benign request does not. `"novel"` goes further — the LLM proposes a vulnerability **class
  the catalog does not have** (a new kind + handler + exploit recipe) for the skeleton, and
  the same gate admits it, re-seeding the flag and re-running to prove the exploit genuine;
  this is the only mode where "what kind" is left to the LLM rather than the catalog.
  `"service"` / `"world"` extend realization to whole services and worlds and are the next
  stages (#212). The knob is validated and recorded on the lineage at build time;
  realization is the separate host step, so `admit` alone returns the procedural world. See
  [DESIGN.md](DESIGN.md) §8 (the verifier is the ceiling) and §9 (the generation ladder).
  Worlds built this way are reproducible by **frozen-snapshot replay**, not by re-running
  the manifest (an LLM sits in the build path).

The seed is extracted by `ProceduralBuilder.build`; every other knob folds into the prior
in `WebappBuilder._effective_prior`. The retired keys `company`, `lateral_movement`,
`vuln_kinds`, `loot_shapes`, `recon_disclosure`, and `difficulty` now raise `PackError`
with a hint to their replacement.
