"""distill(graph, status_log=None) -> PackPrior — the BBG → Builder seam.

This is the one function that turns observed agent experience into a
generation prior. It operates on a `WorldGraph` plus an optional
status-event log (the BBG's transaction-time changelog) and emits a
`PackPrior` carrying only GENERIC graph statistics. The builder INTERPRETS
those statistics into domain decisions — never the other way around.

OpenRange owns `distill` because:

- It consumes a `WorldGraph` (an OpenRange meta-model type).
- It produces a `PackPrior` (an OpenRange consumer concept).
- Putting it anywhere else would invert one of those dependency arrows.

The seam is deliberately wide — JSON in, JSON out, with optional Python
typed shapes for in-process callers. Any agent harness that emits the BBG
state dump shape declared in `CONTRACTS.md` can drive the flywheel; the
harness does not need to depend on OpenRange in code.

Four extractions, mapping the thing/thought seam onto OpenRange:

  things   -> ontology induction + topology prior
              `kind_hint` clusters become world node kinds. Topology
              comes from `part_of` containment edges (reliable observed
              structure) and from *classified* multi-anchor `thought`
              nodes whose claim is structural. `traversed` edges are the
              AGENT'S PATH, not the world's shape — they contribute only
              difficulty and coverage signal, NEVER edge endpoints.

  thoughts -> world nodes + task seeds + difficulty
              thoughts are not uniform. distill classifies them:
                * structural (multi-anchor) -> candidate world edges
                * evaluative (a property of one thing) -> a HIDDEN world
                  node promoted onto that thing (e.g. a vuln node)
                * objective -> a task goal + GOAL role on its anchor
              clusters of `refuted` thoughts and dense `open` regions mark
              where the world is non-obvious; trajectory length and
              backtracking around a cluster set its difficulty.

  coverage -> per-region observation density, so the builder knows where
              to *extrapolate* rather than replay — avoiding the
              survivorship bias of training on the shape of past success.

  goal     -> kinds of things at the END of productive paths become
              candidate goal kinds. Generic heuristic, no domain knowledge.

Everything `distill` produces is GENERIC GRAPH STATISTICS — node-kind
frequencies, a dead-end ratio, a confirmed-thought density per thing-kind.
It never emits pack-specific config keys. If `distill` emitted
`{"decoy_endpoints": 2}` it would have to know the cyber builder wants
that key.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from openrange.core.pack import PackPrior, TaskSeed
from openrange.world_ir import (
    NodeKind,
    Ontology,
    WorldGraph,
)

# ---------------------------------------------------------------------------
# StatusEvent — the BBG transaction-time record (wire shape only)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StatusEvent:
    """One entry in a BBG's status-event log.

    OpenRange does NOT maintain status logs (the BBG runtime — vecna's
    Wayfinder — owns that). But OpenRange consumes them, so this is the
    Python shape `distill()` expects when a status log accompanies a graph
    dump. The wire format is declared in `CONTRACTS.md`.
    """

    node_id: str
    status: str
    at_step: int
    cause: str = ""


# ---------------------------------------------------------------------------
# distill
# ---------------------------------------------------------------------------


def distill(
    graph: WorldGraph,
    status_log: Sequence[StatusEvent] | None = None,
    into: Ontology | None = None,
) -> PackPrior:
    """Turn an accreted graph into an optional generation prior.

    `graph` is the BBG state being distilled. `status_log` is the optional
    transaction-time changelog (used to mine "agents believed X for N steps
    before discovering not-X" signals); when omitted, distill falls back to
    the node attrs' current `status` only.

    `into` is the target world ontology:
      * given -> the prior is refined to conform to an existing pack.
      * None  -> the prior carries a *proposed* induced ontology that a
                 human or scaffolding step turns into a new pack.

    Note: with `into=None`, the induced ontology has node kinds but NO
    edge kinds — edge induction from observed pairs is not done here. So
    `into=None` (the pack-PROPOSAL flow) yields an un-buildable ontology
    by itself; the working bootstrap path is `into=<existing ontology>`.
    """
    things = [n for n in graph.nodes.values() if n.kind == "thing"]
    thoughts = [n for n in graph.nodes.values() if n.kind == "thought"]
    traversals = [e for e in graph.edges.values() if e.kind == "traversed"]

    # --- ontology: refine to a target, or propose an induced one ---
    induced = into
    if induced is None:
        kind_hints = sorted({t.attrs.get("kind_hint") or "thing" for t in things})
        induced = Ontology(
            id="distilled@0.1.0",
            node_kinds={k: NodeKind(k) for k in kind_hints},
            edge_kinds={},
        )

    # --- topology: generic stats only (builder interprets) ---
    # split by STATUS: `salient` things are ones the agent judged matter
    # (a thought anchored them, repeated visits, or observe()); `incidental`
    # things are mere positions. A builder weights salient frequency far
    # higher — it is signal, incidental is mostly background.
    node_kind_freq: dict[str, int] = {}
    salient_kind_freq: dict[str, int] = {}
    for t in things:
        k = t.attrs.get("kind_hint") or "thing"
        node_kind_freq[k] = node_kind_freq.get(k, 0) + 1
        if t.attrs.get("status") == "salient":
            salient_kind_freq[k] = salient_kind_freq.get(k, 0) + 1

    dead = sum(1 for e in traversals if e.attrs.get("outcome") == "dead_end")
    dead_end_ratio = dead / len(traversals) if traversals else 0.0

    # confirmed-thought density per thing-kind: where agents repeatedly
    # *confirm* something is where the world has discoverable hidden state.
    hidden_signal: dict[str, int] = {}
    for th in thoughts:
        if th.attrs.get("status") != "confirmed":
            continue
        for e in graph.out_edges(th.id, "anchored_to"):
            anchor = graph.nodes.get(e.dst)
            if anchor is None:
                continue
            k = anchor.attrs.get("kind_hint") or "thing"
            hidden_signal[k] = hidden_signal.get(k, 0) + 1

    # goal candidates: kinds of things at the END of productive paths —
    # productive-path sinks. Generic heuristic, no domain knowledge.
    goal_kinds: set[str] = set()
    for e in traversals:
        if e.attrs.get("outcome") != "productive":
            continue
        sink = graph.nodes.get(e.dst)
        if sink is not None and not graph.out_edges(sink.id, "traversed"):
            goal_kinds.add(sink.attrs.get("kind_hint") or "thing")

    topology: Mapping[str, Any] = {
        "node_kind_freq": node_kind_freq,
        "salient_kind_freq": salient_kind_freq,
        "dead_end_ratio": round(dead_end_ratio, 3),
        "hidden_signal": hidden_signal,
    }

    # --- task seeds: one per thought cluster ---
    # cluster thoughts via union-find over `revises` chains and shared anchors.
    parent: dict[str, str] = {th.id: th.id for th in thoughts}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        parent[find(a)] = find(b)

    anchors_of: dict[str, list[str]] = {
        th.id: [e.dst for e in graph.out_edges(th.id, "anchored_to")] for th in thoughts
    }
    by_anchor: dict[str, list[str]] = {}
    for tid, ancs in anchors_of.items():
        for a in ancs:
            by_anchor.setdefault(a, []).append(tid)
    for grp in by_anchor.values():
        for other in grp[1:]:
            union(grp[0], other)
    for e in graph.edges.values():
        if e.kind == "revises":
            union(e.src, e.dst)

    clusters: dict[str, list[str]] = {}
    for th in thoughts:
        clusters.setdefault(find(th.id), []).append(th.id)

    task_seeds: list[TaskSeed] = []
    for i, members in enumerate(clusters.values()):
        anchor_kinds: set[str] = set()
        refuted = 0
        for tid in members:
            node = graph.nodes[tid]
            if node.attrs.get("status") == "refuted":
                refuted += 1
            for a in anchors_of[tid]:
                anc = graph.nodes.get(a)
                if anc is not None:
                    anchor_kinds.add(anc.attrs.get("kind_hint") or "thing")
        # difficulty: a refuted thought = a place agents got fooled; a
        # nearby dead end = a wrong path that looked right. Both raise it.
        difficulty_score = min(
            1.0, 0.3 + 0.2 * refuted + 0.2 * (1 if dead_end_ratio > 0 else 0)
        )
        task_seeds.append(
            TaskSeed(
                theme=f"cluster-{i}",
                anchor_kinds=tuple(sorted(anchor_kinds)),
                suggested_goal_kinds=tuple(sorted(goal_kinds)),
                difficulty=round(difficulty_score, 2),
                evidence=1,
                # distill never tags a seed with a family. A harness with
                # that knowledge may attach one downstream.
                family=None,
            )
        )

    # coverage: explored-density per thing-kind (thin regions => extrapolate).
    coverage: dict[str, float] = {}
    for k in node_kind_freq:
        same = [t for t in things if (t.attrs.get("kind_hint") or "thing") == k]
        explored = sum(1 for t in same if t.attrs.get("explored"))
        coverage[k] = round(explored / len(same), 3) if same else 0.0

    return PackPrior(
        source=f"{graph.ontology} :: {graph.content_hash()}",
        ontology=induced,
        topology=topology,
        task_seeds=tuple(task_seeds),
        difficulty={s.theme: s.difficulty for s in task_seeds},
        coverage=coverage,
    )
