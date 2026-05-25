# openrange-webapp

The reference Pack for OpenRange: one **world-family** (`webapp`) with
two **TaskFamilies** that run against the same world graph.

| TaskFamily | id | What an agent does |
|---|---|---|
| `WebappBuild` | `webapp.build` | Implement a feature endpoint in an existing repo so it serves correctly. |
| `WebappPentest` | `webapp.pentest` | Discover the hidden flag by exploiting the running app. |

The two families entrypoint **different nodes** of the same world graph
— the build task entrypoints the repo, the pentest task entrypoints an
exposed endpoint — which is the load-bearing demo that "domain" lives
on the TaskFamily, not on the Pack. See `DESIGN.md` in the OpenRange
repo for the wider rationale.

This v1 ships:

- `webapp/ontology.py` — declarative `Ontology` value (`webapp@0.1.0`).
- `webapp/invariants.py` — graph-wide invariants (no orphan nodes,
  every secret held by a record).
- `webapp/families/build.py` — `WebappBuild` TaskFamily.
- `webapp/families/pentest.py` — `WebappPentest` TaskFamily.
- `webapp/builder.py` — `WebappBuilder` (hand-authored procedural
  sampler — no LLM dependency in v1; LLM enrichment lands in v2).
- `webapp/pack.py` — `WebappPack` wiring the above together.

What's NOT in v1 (deferred):

- Procedural sampling against the full PackPrior — v1 uses a simple
  hand-authored generator that produces a small, admittable world.
- LLM-driven instruction generation — TaskFamilies emit literal
  instructions in v1.
- Curriculum (`Mutation` / `available_mutations`) — TaskFamilies
  return `()` for `available_mutations` in v1.
- Realizer (Flask app generation, codegen) — `WebappPack.realize`
  returns a no-op `RuntimeHandle`. The episode loop layer that
  consumes this is itself being re-wired.

All of those are tracked as follow-ups. The v1 pack is the load-bearing
demonstration that the new shape *works* — two families on one world,
admitted into a content-addressed Snapshot.
