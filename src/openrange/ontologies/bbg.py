"""The BBG ontology — cognitive primitives for long-horizon agent memory.

The BBG ("Big Beautiful Graph") is a typed-property-graph shape used by any
runtime that maintains an agent's spatial memory across a task longer than
its context window. The ontology defined here is the *declarative schema*
for that shape — two node kinds, four edge kinds, the attribute slots and
their enums. It is plain data; no runtime, no perception, no NavPack.

The runtime that *uses* this ontology — the Wayfinder, with its three-tier
perception model, status-event log, NavPack proposer, and SDK adapters —
lives in `vecna/wayfinder`. That runtime vendors this module from OpenRange
so neither repo imports the other. The wire format (the `bbg@0.1.0` id and
the JSON shape declared in `CONTRACTS.md`) is the contract; this Python
module is its reference value.

Cognitive primitives, intentional minimum:

  thing       something encountered in the world. Traversable.
  thought     something the agent inferred. Anchored to the things it is about.
  traversed   the agent moved between things (with an outcome).
  part_of     spatial / logical containment.
  anchored_to a thought is about a thing.
  revises     a thought supersedes an earlier one.

Two state fields, shared by both node kinds:

  provenance  HOW the node entered the graph (immutable: trajectory /
              referenced / inferred).
  status      WHAT the agent currently believes about the node (mutable
              via a separate status-event log; never overwritten in place).

These primitives are deliberately scoped so the BBG carries *only* the
agent's epistemic state — what was seen, what was concluded — not the
world's ontic state. OpenRange's world graphs use a different ontology
(per-pack); the BBG and a world graph share the meta-model but never the
ontology.
"""

from __future__ import annotations

from openrange.world_ir import AttrSpec, AttrType, EdgeKind, NodeKind, Ontology

BBG_ONTOLOGY_ID = "bbg@0.1.0"


def wayfinder_ontology() -> Ontology:
    """Return the `bbg@0.1.0` ontology value.

    Returns a fresh `Ontology` each call so callers can mutate the result
    without affecting other consumers. The shape itself does not change
    between calls — that is the contract `bbg@0.1.0` encodes.

    Two node kinds (`thing`, `thought`) and four edge kinds (`traversed`,
    `part_of`, `anchored_to`, `revises`). Both node kinds share the same
    `provenance` (immutable) and `status` (mutable, logged) fields; the
    `status` enum differs by kind.
    """
    s = AttrSpec
    # provenance — HOW a node entered the BBG. Shared by both kinds, immutable.
    #   trajectory : the agent stepped here (a thing, by position)
    #   referenced : named by a thought / observe() without being visited
    #                (e.g. a hidden credential the agent only reasoned about)
    #   inferred   : the agent produced it by thinking (every thought)
    provenance = s(
        AttrType.ENUM,
        required=True,
        enum=["trajectory", "referenced", "inferred"],
        description="how the node entered the BBG; immutable",
    )
    return Ontology(
        id=BBG_ONTOLOGY_ID,
        node_kinds={
            "thing": NodeKind(
                "thing",
                attrs={
                    "label": s(
                        AttrType.STRING,
                        required=True,
                        description="human-readable name",
                    ),
                    "category": s(
                        AttrType.ENUM,
                        enum=["place", "object", "actor", "signal"],
                        description="cross-domain class; bridges to roles",
                    ),
                    "kind_hint": s(
                        AttrType.STRING, description="domain-type guess; distill input"
                    ),
                    "provenance": provenance,
                    # a thing's belief-state:
                    #   incidental : entered the BBG but the agent has not marked
                    #                it as mattering
                    #   salient    : the agent judged it matters — reached three
                    #                ways (a thought anchored it / repeated visits
                    #                / observe()). distill weights salient highest.
                    # one-way latch: incidental -> salient, never back.
                    "status": s(
                        AttrType.ENUM, required=True, enum=["incidental", "salient"]
                    ),
                    "explored": s(AttrType.BOOL, default=False),
                    "first_seen": s(
                        AttrType.INT, description="trajectory step first observed"
                    ),
                    "visits": s(AttrType.INT, default=0),
                },
                description="something encountered in the world; traversable",
            ),
            "thought": NodeKind(
                "thought",
                attrs={
                    "claim": s(
                        AttrType.STRING,
                        required=True,
                        description="the inferred statement",
                    ),
                    "provenance": provenance,  # always 'inferred' for a thought
                    # a thought's belief-state. An open thought is a live question;
                    # a refuted one is a place agents get fooled — both are prime
                    # task-mining signal, so neither is ever discarded.
                    "status": s(
                        AttrType.ENUM,
                        required=True,
                        enum=["open", "confirmed", "refuted"],
                    ),
                    "confidence": s(AttrType.FLOAT, default=0.5),
                    "formed_at": s(
                        AttrType.INT, description="trajectory step the thought arose"
                    ),
                },
                description="something the agent inferred; anchored to things",
            ),
        },
        edge_kinds={
            "traversed": EdgeKind(
                "traversed",
                endpoints=[("thing", "thing")],
                attrs={
                    "outcome": s(
                        AttrType.ENUM,
                        enum=["productive", "dead_end", "neutral"],
                        default="neutral",
                    ),
                    "count": s(AttrType.INT, default=1),
                    "cost": s(AttrType.FLOAT, description="effort / steps to cross"),
                },
                description="the agent moved from one thing to another",
            ),
            "part_of": EdgeKind(
                "part_of",
                endpoints=[("thing", "thing")],
                description="spatial / logical containment",
            ),
            "anchored_to": EdgeKind(
                "anchored_to",
                endpoints=[("thought", "thing")],
                description="what the thought is about",
            ),
            "revises": EdgeKind(
                "revises",
                endpoints=[("thought", "thought")],
                description="supersedes an earlier thought",
            ),
        },
    )
