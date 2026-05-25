"""distill(graph, status_log=None) -> PackPrior — the BBG → Builder seam.

The one function that turns observed agent experience into a generation
prior. It operates on a BBG-shaped `WorldGraph` (any harness emitting
the wire format declared in `CONTRACTS.md` §6 can drive this) and emits
a `PackPrior` carrying only GENERIC graph statistics. A builder
INTERPRETS those statistics into domain decisions — never the other way
around.

OpenRange owns `distill` because:

- It consumes a `WorldGraph` (an OpenRange meta-model type).
- It produces a `PackPrior` (an OpenRange consumer concept).
- Putting it anywhere else would invert one of those dependency arrows.

What v1 extracts:

  topology     `node_kind_freq`     count of things per `kind_hint`
               `salient_kind_freq`  same restricted to status=salient
               `dead_end_ratio`     fraction of traversed edges marked
                                    `outcome=dead_end`
               `hidden_signal`      per-`kind_hint` count of anchors of
                                    confirmed thoughts; signals where a
                                    builder should place hidden state

  task_seeds   one per thought-cluster (union-find over shared anchors
               and `revises` chains). Each seed carries `anchor_kinds`
               (kinds of things the cluster anchored on),
               `suggested_goal_kinds` (kinds at the sinks of productive
               paths), and a `difficulty` raised by refuted thoughts
               and dead-end traversals in the source graph

  coverage     explored ratio per `kind_hint`; thin regions tell the
               builder where to extrapolate rather than replay

  induced      with `into=None`, a proposed `Ontology` carrying just the
   ontology    observed `kind_hint`s as node kinds (no edge induction in
               v1 — pure-bootstrap callers should pass `into=<existing
               pack ontology>` for the working refinement path)

What v1 does NOT yet do (tracked in ROADMAP.md):

- Multi-trajectory `distill` (merging evidence across many graphs).
- Edge induction in the `into=None` path.
- Status-log mining ("the agent believed X for N steps before
  discovering not-X"). The `status_log` parameter is accepted but
  currently unused — distill reads only each node's latest `status`
  attribute. The parameter is kept in the signature so the wire format
  stays stable; the contract widens, not narrows, when v2 starts
  consuming the log.

Everything `distill` produces is GENERIC GRAPH STATISTICS — node-kind
frequencies, a dead-end ratio, a confirmed-thought density per
thing-kind. It never emits a pack-specific config key.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from openrange.core.contracts import PackPrior, TaskSeed
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

    OpenRange does not maintain status logs — that is the harness's job.
    OpenRange consumes them via `distill()`, so this is the Python shape
    of the wire format declared in `CONTRACTS.md` §6.
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
    """Turn an accreted BBG-shaped graph into a `PackPrior`.

    `graph` is the BBG state being distilled — a `WorldGraph` conforming
    to the `bbg.wayfinder@0.1.0` ontology
    (`openrange.ontologies.bbg`).

    `status_log` is the optional transaction-time changelog. **Currently
    unused** in the v1 stats (distill reads only each node's latest
    `status` attr); the parameter is reserved for a future signal —
    "agents believed X for N steps before discovering not-X" — without
    breaking the wire format. See module docstring.

    `into` is the target world ontology:
      * given → the prior is refined to conform to an existing pack.
      * None  → the prior carries a *proposed* induced ontology built
                from observed `kind_hint`s. v1 induces node kinds only,
                no edge kinds, so `into=None` is not yet a buildable
                bootstrap path on its own; the working path is
                `into=<existing pack ontology>`.
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
    # (a thought anchored them, repeated visits, or observe()); incidental
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

    # goal candidates: kinds at the END of productive paths — productive-path
    # sinks. Generic heuristic, no domain knowledge.
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
                anchor_kinds=sorted(anchor_kinds),
                suggested_goal_kinds=sorted(goal_kinds),
                difficulty=round(difficulty_score, 2),
                evidence=1,
                # distill never tags a seed with a family — a harness with
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
        task_seeds=task_seeds,
        difficulty={s.theme: s.difficulty for s in task_seeds},
        coverage=coverage,
    )
