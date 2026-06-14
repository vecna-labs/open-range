# Enterprise-scale world generation (#212)

Design shape for [#212](https://github.com/vecna-labs/open-range/issues/212) (the
enterprise-scale umbrella) and [#261](https://github.com/vecna-labs/open-range/issues/261)
(the LLM-realization epic). It maps the space — ontology, sampler, realization tiers,
identity, networks, NPCs, surfaces — at enough specificity that the follow-up issues
filed under #212 don't re-litigate it. Grounded in the live pack at
`packs/cyber_webapp/cyber_webapp/` (the paths in #212's body predate the refactor and are
stale). The first realization tile of this shape — the LLM realizing a whole service's
benign surface — lands as code in the same PR (`llm_realize.realize_service_surface`).

## Enterprise-scale worlds — design shape (#212)

> Status: design / ADR section. Scope: map the space at enough specificity that follow-ups don't re-litigate it. Grounded in the live pack at `packs/cyber_webapp/cyber_webapp/` (the paths in #212's body are stale).

### 0. The thesis: grow the engine, don't rebuild it

The pieces enterprise scale needs are already load-bearing in the code, just sized small. The bet is to **scale the existing invariant**, not invent a second one:

- procedural **architects** the graph (`sample_graph`, `sampling.py:451`);
- the LLM **realizes** each node behind admission (`realize_world`, `llm_realize.py:431`);
- admission **verifies** by consequence (`classify_admission`, `realize_admit.py:33`);
- the result **freezes** to a content-addressed snapshot.

Three things change at 100–1000×: the sampler gains an **org-chart spine**, realization gains **tiers** (only the reachable slice runs), and the snapshot store gains **paging**. Everything else (ontology kinds, admission, feasibility, NPC ABC) extends additively.

The non-negotiable that must survive scale: **solvable-by-construction**. Today `WebappPentest.check_feasibility` proves one reachable `exposes→affects→enables*→backed_by→contains→holds` path to the flag (`families/pentest.py:64-112`). At enterprise scale we keep exactly one *planted oracle chain* and treat everything else as decoys/noise — so feasibility stays a bounded walk over the planted spine, independent of estate size.

---

### 1. Ontology — org hierarchy + AD-shaped identity

Today the ontology is flat: `host/service/endpoint/account/credential/secret/vulnerability/network/data_store/record`, with `account.role ∈ {user,admin,service}` and only `has_credential`/`can_access` identity edges (`ontology.py:88-100, 239-254`). Enterprise needs two new layers, added as new node/edge kinds (the realizer is a pure projection, so new kinds with no realizer mapping are *cold* by default — they cost nothing to run).

**Org hierarchy (new node kinds):** `org → division → department → team`, plus a `member_of` edge (account→team) and an `owns` edge (team→service / team→host). This is the spine the sampler walks; it also gives the UI its clustering key (§9) and gives ownership-based vulns somewhere to live (a team that owns a vault *and* a CI runner is a lateral-movement seed).

**AD-shaped identity (new node kinds, modeled on real AD, not CTF):**
- `user` (was `account.role=user`), `group`, `ou` (organizational unit), `gpo` (group policy object), `service_account`.
- Edges: `member_of` (user→group, group→group nesting), `in_ou` (user/computer→ou), `applies_to` (gpo→ou/group), `runs_as` (service→service_account), `delegates_to` (service_account→service_account — Kerberos-delegation-shaped lateral edges).
- A **joiner/mover/leaver** attribute axis on `user`: `lifecycle ∈ {joiner, active, mover, leaver}`. *Leaver-but-still-enabled* and *mover-with-stale-group* are the canonical identity vulns — they're graph facts (a `member_of` edge that should have been removed), so admission can verify them by reachability, no new oracle machinery.

Keep `account.role` as a back-compat alias that maps onto `{user, service_account, admin-group-member}` so today's worlds and tests (`families/pentest.py`, the 9 reference exploits) don't break.

**Service kinds beyond the current 7** (`ontology.py:42`): widen the enum to cover a real estate — `ci_cd, vault, crm, hris, helpdesk, vpn, ids, cloud_control_plane, fileshare(SMB), directory(AD/LDAP), object_store, code_host`. Most ship **cold** at first (graph-only, named, reachable in the topology) and get a *warm* synthetic responder or a *hot* container only when the agent's frontier reaches them (§3). This is how we get "dozens of service kinds" without writing dozens of realizers up front.

**Decision (Q3):** identity is a **shared sub-ontology**, not cyber-pack-private. AD-shaped identity is exactly the structure a future defense pack (#191), an HR/SaaS pack, or a phishing/supply-chain pack reuse. Put it in `graphschema` (or a small `openrange-identity` ontology module the cyber pack imports), so sibling packs compose the same node/edge kinds. The cyber pack contributes the *vuln* semantics; the identity *shape* is shared.

---

### 2. Sampler — hierarchical, org-chart-driven, solvable at scale

Replace the linear `sample_graph` (`sampling.py:451`) with a staged generator that samples **top-down from a budget**, not a flat count:

1. **Org skeleton.** From a manifest budget (`employees`, `divisions`, `services`), sample `org → divisions → departments → teams` with a realistic fan-out (power-law team sizes, not uniform). Curated name pools already exist for services (`sampling.py:167`) and people (`sampling.py:147`); extend them for divisions/departments.
2. **Estate per team.** Each team `owns` a few services/hosts drawn from the widened kind set, on a site/VLAN (§4). Service density follows the org chart — a 30-person eng division has CI/CD + code host + vault; a 5-person HR team has an HRIS + a fileshare. M&A debris and shadow IT = a *second* org subtree with weaker segmentation and one cross-trust edge.
3. **Identity population.** Materialize `user` nodes per team (with `member_of`/`in_ou`/`applies_to`), `service_account`s for services, and a small set of intentional identity flaws (leaver-still-enabled, over-broad group). Named characters are a thin overlay (§5).
4. **Plant ONE oracle chain.** This is the solvability spine. Pick an entry surface (a public service), then *compose hops* — the existing `_lateralize` credential-reuse primitive (`sampling.py:1231`) is already exactly a composable hop ("entry host leaks a credential, each gated host validates the credential one hop back, the last serves the flag"). At scale the chain is longer and crosses sites/identity (an SSRF → an internal service → a stolen service-account token → a vault), but it is *one* planted path. `check_feasibility` walks only this path, so feasibility cost is O(chain depth), not O(estate).
5. **Decoys/noise.** Everything else is reachable-but-dead: off-path services, benign endpoints, non-exploitable config. `dead_end_ratio` (`priors.py:53`) generalizes to "fraction of the estate that is noise."

**Solvable-by-construction at scale** = the planted chain is built before decoys, feasibility verifies only it, and the consequence verifier (`detect_leak`) grades on the single planted flag. Decoy volume can grow without bound without ever threatening solvability — which is precisely OpenRange's differentiator over "ask an LLM for a vulnerable world" (one world, mostly broken; DESIGN.md §8/§9).

**MCTS link (#193):** the composable hop in step 4 is the action the search sampler explores — `add_node / add_edge / inject_vuln / add_hop / reject`, rolled out by this procedural sampler, scored on chain depth × blast radius × novelty. The hierarchical sampler is the *rollout policy* MCTS sits on top of; ship the hierarchical baseline first (#193 is additive, per its own acceptance).

**Decision (Q2):** the org chart is **generated from a manifest budget**, with an optional **manifest override** for a specific org subtree. Pure-manifest authoring 1000 services by hand is the thing #212 says doesn't scale; pure-generated loses the ability to pin a scenario. Budget-in, override-where-you-care is the sweet spot, and it matches how `builder._effective_prior` already lets a manifest override count ranges and weights (`builder.py:43-94`).

---

### 3. Realization tiers — hot / warm / cold, driven by the frontier (#235, #272)

The whole estate is never realized at once. Tier each `service` node by its graph distance from the agent's current position/credentials — the `WorldGraph` already encodes reachability (`out_edges`/`in_edges`, `_enable_closure`), so the frontier is a graph query, not new state.

| tier | what runs | backing | when |
| --- | --- | --- | --- |
| **hot** | a real service: real fs/shell/network | `NetworkedContainerWebappRuntime` (today, `realize.py:333`) | the agent can reach it *now* (reachable set) |
| **warm** | synthetic in-process responses (banner, config, plausible 200s) — enough to recon, not to exploit | the existing `WebappRuntime` emulation path (`realize.py:46`), repurposed as a "warm responder" | the near-frontier (1–2 hops out): pre-warmed so promotion is instant |
| **cold** | nothing runs — graph-only node with a name/host/kind | none | out of reach |

**Promotion on reach:** when the agent's action extends the reachable set (pivots, steals a credential), the runtime promotes the newly-reachable cold/warm nodes to hot. This reuses the **`auto_evolve` re-admit seam** (`curriculum.py:65`) conceptually — a bounded graph patch + a local realize — rather than rebuilding the world.

**Admission-equivalence guarantee (the #235 hard requirement):** lazy realization must not change verifier outcomes vs a fully-realized world. The guarantee falls out of the design: the planted oracle chain is *always realized hot along its whole length the moment its entry is reached* (the chain is short and known by construction), and `classify_admission` only ever inspects the planted flag's leak. Decoys being warm/cold can't change a leak verdict because they hold no HIDDEN `value_ref`. Cross-backing parity (DESIGN.md §10: same snapshot grades identically on PROCESS and CONTAINER) extends to tier parity: warm-emulated and hot-container responses must grade identically on the *non-flag* path, which is exactly the parity the pack already tests.

**Where the LLM fits (Q5):** the LLM is a **seed-anchor enricher, never an architect and never a per-node scaler.** Concretely:
- **Anchors (LLM, ~10s of calls):** name the org, the divisions, the flagship products, the handful of *hot* services on the oracle chain, and the named characters (#192). One whole-graph-ish call per anchor cluster for coherence.
- **Fan-out (procedural, free):** every other name/host/account derives procedurally from the anchors via the curated pools (`sampling.py:147,167`). You never ask the LLM to name 5000 services — you ask it to name the 10 that matter and let procedural make the rest consistent.
- **Realization (LLM, behind admission, hot-only):** `realize_world` (`llm_realize.py:431`) already realizes a vuln handler per node behind the consequence gate. At scale it runs **only on the hot oracle-chain nodes** — the only nodes whose realism the agent can actually probe. Decoys never pay an LLM call.

**Decision (Q4):** tiered realization is a **core concern with a pack-supplied warm responder.** The frontier computation, tier bookkeeping, and promotion lifecycle are domain-agnostic (any pack with a reachability graph wants them) — they belong next to `Backing` and `Pack.realize(graph, backing)` (`_protocols.py:272`), the seam #235 names. The pack supplies *what a warm node says* (its synthetic responder) and *what a hot node is* (its container), but core owns *when* each tier is live. Add a `Backing.LAZY` (or a `realize(graph, backing, frontier=...)` arg) so the host drives tier transitions.

---

### 4. Networks — multi-site / VLAN / VPN / ZTNA / cloud VPC

Today: one bridge network per build, or a company preset with a dmz + one internal segment (`_add_networks`, `sampling.py:592`); `network.isolation ∈ {bridge,host,isolated}` (`ontology.py:165`). Extend:

- **`site` node** (HQ, branch, datacenter, cloud-region) and a `network.site` attribute; services belong to a site via their host.
- **VLAN/segment** as `network` nodes with a `segment_type ∈ {vlan, vpn, ztna, vpc, peering, transit}` attribute; `connected_to` already wires services to networks (`ontology.py:260`).
- **Reachability edges between networks**: `routes_to` (network→network) with attrs for the control (`firewall_allow`, `vpn_tunnel`, `ztna_policy`, `vpc_peering`). The agent's pivot crosses these — and a misconfigured `routes_to` (a flat VPN, an over-broad peering) is itself a planted oracle hop.
- **Realization mapping:** today networks are real docker networks (`NetworkedContainerWebappRuntime._create_network`, `realize.py:376`). Multi-site = multiple docker networks with explicit gateway containers for the allowed `routes_to` edges. The k8s backing (#189) is the *production* expression of the same graph (NetworkPolicies = `routes_to`); ship docker-network multi-site first, k8s later — exactly the §9 progression.

ZTNA/cloud-control-plane are where this stops being "the cyber webapp pack" and starts wanting sibling packs (§10).

---

### 5. NPCs — population models + named characters

Today NPCs are individual instances replicated by manifest `count` (`core/npc.py:79-112`), with two references: scripted `BrowsingUser` (`npcs/browsing_user.py:12`) and LLM-backed `CuriousEmployee` (AgentNPC, `npcs/curious_employee.py:19`). You can't hand-instantiate 1000.

Two layers:

- **Population model (new):** a `PopulationNPC` that, given the org chart, generates *aggregate background traffic* — request distributions over endpoints weighted by team/role, on a cadence — without one Python object per employee. It's one NPC driving N personas' worth of traffic statistically. This fills the request log realistically at scale (the log is what the leak oracle reads, `realize.py:451`) and is the believable noise an IDS-defense agent must filter. Keep it scripted (no LLM) so 1000 employees cost nothing.
- **Named characters (overlay):** a handful of `AgentNPC`s (the existing primitive, `_protocols.py:330`) for the people who matter to the scenario — the over-sharing admin, the phishable exec. These are the LLM-backed actors; the population model is the statistical backdrop. The named characters are also the anchors the LLM names in §3.

This builds directly on #74's "an NPC is an agent with tools" (now CLOSED/landed): population NPCs are scripted agents-with-tools; named characters are LLM agents-with-tools. No new NPC ABC needed.

---

### 6. Surfaces beyond HTTP

Today the only agent-facing surface is `http_get`/`http_get_json` over `base_url` (`realize.py:111`). Enterprise attack surface is broader. Model each surface as a **tool the runtime exposes in the interface** (the same seam NPCs and the verifier already consume, #74):

- **Email/phishing:** a mailbox surface (read/send) — the entry for a phish-to-foothold chain; pairs with named-character NPCs (§5) as victims. Likely a **sibling pack** (different ontology entry, victim-NPC heavy).
- **Endpoint:** a host shell/file surface — already half-real on `CONTAINER` (`OPENRANGE_REALFS`, DESIGN.md §9). Endpoint = "the agent has a foothold on a workstation," a natural promotion target in the tier model (§3).
- **SaaS/cloud control plane:** an API surface over a `cloud_control_plane` service (token-scoped) — IAM-misconfig and cloud-metadata pivots. The SSRF→metadata pivot already exists (`_networkize_ssrf`, `sampling.py:1079`); this generalizes it.
- **Supply chain:** a `code_host`/`ci_cd` surface where a poisoned dependency or a CI secret is the oracle hop. **Sibling pack** territory (build-system semantics).

**One-pack vs sibling-pack rule:** HTTP, endpoint, and cloud-API surfaces stay in the (renamed) `webapp→enterprise` pack because they share the realizer and the consequence oracle. Email/phishing and supply-chain get **sibling packs** that *import the shared identity + org sub-ontology* (§1) and compose worlds via shared graph kinds — same engine, different surface and victim model.

---

### 7. Admission-at-scale — "is the world interesting?" as graph search

Today admission verifies one oracle path dynamically (`classify_admission`, `realize_admit.py:33`) and structurally (`check_feasibility`, `families/pentest.py:64`). At scale, "valid" isn't enough — #193's point is that we want *interesting*. Make admission a **graph search over the planted spine plus the surrounding estate**:

- **Solvable (unchanged):** the planted oracle chain leaks the flag on the intended exploit, not on benign (already guaranteed by construction + `detect_leak`).
- **Not trivial (extend):** no *shorter* path to the flag than the planted chain. Search the reachable subgraph for any alternate `…→holds(flag)` path; reject if one is shorter than intended (the scale version of today's "benign request must not leak").
- **Interesting (new score, the #193 surface):** chain depth, blast radius (how much of the estate the chain touches), cross-domain hops (network/identity/surface boundaries crossed), novelty vs snapshot history. This is a score, not a gate, until #193 lands MCTS to *optimize* it.

Cost stays bounded: the search is over the *reachable* subgraph from the entry, which the tier model already computes — you never search 100k cold nodes.

---

### 8. Snapshot / storage — paged / columnar (#205)

Today: one monolithic `<snapshot_id>.json` per snapshot (`core/store.py:17-31`), and `content_hash()` re-serializes every node+edge sorted (`graphschema/_ir.py:256`). Both are O(n) and break past ~10⁴ entities.

- **Columnar / paged store:** nodes and edges in typed, paged tables (parquet/sqlite/jsonl-shards) keyed by id, so the runtime loads only the *reachable slice* (which the tier model already needs). The cold majority sits on disk untouched.
- **Incremental content hash:** a Merkle/rolling hash over sorted id-shards so a promotion patch re-hashes only changed pages, not the whole graph — preserving the content-addressed reproducibility the LLM-in-build-path story depends on (DESIGN.md §9).
- **Backward compatible:** keep the JSON store for small worlds; select the paged store by entity count. This is #205's "pick a snapshotting tech per backing class" applied to the *graph* store, complementing its runtime-state snapshotting.

---

### 9. UI — clustering / focus+context

The all-nodes dashboard can't draw 100k entities. Use the org hierarchy (§1) as the clustering key: render `org → division → department → team` collapsed by default; expand on focus; show the **planted oracle chain highlighted** and the agent's current frontier (hot/warm/cold tiers color-coded, §3). Search jumps to a node; focus+context keeps the chain visible while the rest stays clustered. This is downstream of the ontology/tier work — file it but don't block on it.

---

### 10. Answers to #212's six open questions (decisive)

1. **One pack or sibling pack?** **One pack.** Grow `webapp` (v2) into an `enterprise` pack in place — it reuses the realizer, the consequence oracle, the 9 reference exploits, and the networked container runtime. Spin **sibling packs only for genuinely different surfaces/victim models** (email/phishing, supply-chain), and have them import the shared identity+org sub-ontology.
2. **Org chart: manifest input or generated?** **Generated from a manifest budget**, with optional manifest override of a named subtree. Matches the existing budget-override pattern in `builder._effective_prior` (`builder.py:43`).
3. **Identity: cyber-pack ontology or shared sub-ontology?** **Shared sub-ontology** in `graphschema` (or `openrange-identity`). AD-shaped identity is reused by defense/HR/phishing packs; the cyber pack adds only the vuln semantics.
4. **Tiered realization: pack or core concern?** **Core**, with a pack-supplied warm responder and hot container. The frontier/tier lifecycle is domain-agnostic and belongs at the `Backing`/`Pack.realize` seam (#235).
5. **Where does the LLM live?** **Seed anchors → procedural fans out.** LLM names the ~10 anchors and realizes only the hot oracle-chain nodes behind admission (`realize_world`); procedural derives everything else. Never architect, never per-node-at-scale.
6. **Smallest world that's still "enterprise"?** **~200 services.** Below ~50 is the current "company" preset (`builder.py:70`); 200 forces the org spine, tiered realization, and paged snapshots to all earn their keep without requiring 1000-node infra to iterate. Target 200 as the v1 enterprise floor; design for 1000+ but don't gate v1 on it.

---

### 11. Build order (what unblocks what)

1. Identity + org sub-ontology (§1) — unblocks the sampler.
2. Hierarchical sampler with the planted oracle chain (§2) — unblocks everything downstream; keeps solvable-by-construction.
3. Tiered/lazy realization core seam (§3, #235) — unblocks running 200+ services.
4. Paged snapshots (§8, #205) — unblocks persisting them.
5. Population NPCs (§5), multi-site networks (§4), admission-at-scale score (§7) — additive realism.
6. MCTS sampler (#193), non-HTTP sibling packs (§6), UI (§9) — research / stretch.
---

## Proposed issue tree (follow-ups under #212 / #261)

Filed (in this order) as #275–#285, children of #212 / #261, in the build order of §11. The body sketch for each lives with the design section above.

1. **AD-shaped identity + org-hierarchy sub-ontology (shared)** — Add the org and identity layers #212 §1 calls for, as a SHARED sub-ontology (graphschema or a new openrange-identity module) so sibling packs reuse it.
2. **Hierarchical org-chart sampler with a planted oracle chain** — Replace the flat `sample_graph` (sampling.py:451) with a top-down budget-driven generator, per design §2.
3. **Tiered/lazy realization: hot/warm/cold driven by the reachability frontier (#235)** — Implement the core seam for #235 so a 200+ service estate runs without realizing the whole thing.
4. **LLM realizes a whole service (the realization tile), hot-only and admission-gated** — Extend `realize_world` (llm_realize.py:431) from realizing one vuln HANDLER to realizing a whole SERVICE node — its endpoints + behavior — on the hot oracle-chain nodes only.
5. **NPC population model + named-character overlay** — Add a population NPC so 1000 employees don't need 1000 Python objects, per design §5.
6. **Multi-site / VLAN / VPN / ZTNA / VPC networks** — Extend networks beyond one-bridge / dmz+internal, per design §4.
7. **Paged / columnar snapshot store + incremental content hash (#205)** — Make snapshots survive 100k+ entities, per design §8 and #205.
8. **Admission-at-scale: interestingness as graph search (link #193)** — Generalize admission from one-oracle-path to a bounded graph search over the reachable estate, per design §7.
9. **MCTS sampler over the hierarchical baseline (#193)** — Layer Monte Carlo Tree Search on top of the hierarchical sampler, per #193 and design §2/§7.
10. **Non-HTTP surfaces: endpoint + cloud-API (one-pack) and email/supply-chain (sibling packs)** — Add attack surfaces beyond HTTP, per design §6, modeling each as a tool in the runtime interface (the seam NPCs/verifier already use, #74).
11. **Enterprise dashboard: org-clustered focus+context UI** — Make the world viewable at 100k entities, per design §9.
