"""Tests for openrange.core.distill — the WorldGraph -> PackPrior seam.

The function is the only thing OpenRange owns that knows the BBG ontology
by name. These tests check:

- topology stats (kind_freq, salient_kind_freq, dead_end_ratio, hidden_signal)
- task seed clustering via revises chains and shared anchors
- goal-kind heuristic from productive-path sinks
- coverage extraction per kind
- the ontology induction path (into=None) vs the refinement path (into=...)
- source string carries `<ontology_id> :: <content_hash>`
- distill never tags a task seed with a family
- a status_log is accepted but does not change ontology-shape stats

Distill operates on any WorldGraph that conforms to the BBG ontology; the
fixtures here build one inline.
"""

from __future__ import annotations

from openrange.core.distill import StatusEvent, distill
from openrange.ontologies.bbg import BBG_ONTOLOGY_ID, wayfinder_ontology
from openrange.world_ir import Edge, Node, Ontology, WorldGraph

# ---------------------------------------------------------------------------
# fixtures — build small BBGs by hand
# ---------------------------------------------------------------------------


def _new_bbg() -> WorldGraph:
    return WorldGraph(ontology=BBG_ONTOLOGY_ID)


def _add_thing(
    g: WorldGraph,
    tid: str,
    kind_hint: str = "thing",
    status: str = "incidental",
    provenance: str = "trajectory",
    *,
    label: str | None = None,
    explored: bool = True,
    visits: int = 1,
    first_seen: int = 0,
) -> None:
    g.add_node(
        Node(
            tid,
            "thing",
            attrs={
                "label": label or tid,
                "kind_hint": kind_hint,
                "category": "place",
                "provenance": provenance,
                "status": status,
                "explored": explored,
                "visits": visits,
                "first_seen": first_seen,
            },
        )
    )


def _add_thought(
    g: WorldGraph,
    tid: str,
    anchors: list[str],
    status: str = "open",
    *,
    claim: str = "x",
    confidence: float = 0.5,
    formed_at: int = 0,
) -> None:
    g.add_node(
        Node(
            tid,
            "thought",
            attrs={
                "claim": claim,
                "provenance": "inferred",
                "status": status,
                "confidence": confidence,
                "formed_at": formed_at,
            },
        )
    )
    for a in anchors:
        g.add_edge(Edge(f"{tid}->{a}", "anchored_to", tid, a))


def _add_traversed(
    g: WorldGraph, src: str, dst: str, outcome: str = "neutral", count: int = 1
) -> None:
    g.add_edge(
        Edge(
            f"{src}->{dst}",
            "traversed",
            src,
            dst,
            attrs={"outcome": outcome, "count": count},
        )
    )


def _small_bbg() -> WorldGraph:
    """A BBG covering all the cases distill must handle.

    Two endpoints (one salient via anchored thought), one db (incidental),
    one credential reached only by reference (provenance=referenced, salient).
    A productive sink edge into the credential. A dead-end traversal between
    two endpoints. Two thoughts forming a refutation chain via `revises`.
    """
    g = _new_bbg()
    _add_thing(g, "thing.login", kind_hint="endpoint")
    _add_thing(g, "thing.dash", kind_hint="endpoint", status="salient")
    _add_thing(g, "thing.db", kind_hint="db")
    _add_thing(
        g,
        "thing.cred",
        kind_hint="cred",
        status="salient",
        provenance="referenced",
        explored=False,
    )

    _add_thought(
        g,
        "thought.0",
        anchors=["thing.dash"],
        status="refuted",
        claim="dash is the way in",
    )
    _add_thought(
        g,
        "thought.1",
        anchors=["thing.dash"],
        status="open",
        claim="dash is actually a dead end",
    )
    # thought.1 revises thought.0
    g.add_edge(Edge("rev.0->1", "revises", "thought.1", "thought.0"))
    _add_thought(
        g,
        "thought.2",
        anchors=["thing.cred"],
        status="confirmed",
        claim="the admin token lives at thing.cred",
    )

    _add_traversed(g, "thing.login", "thing.dash", outcome="dead_end")
    _add_traversed(g, "thing.login", "thing.db", outcome="productive")
    _add_traversed(g, "thing.db", "thing.cred", outcome="productive")
    return g


# ---------------------------------------------------------------------------
# topology stats
# ---------------------------------------------------------------------------


def test_topology_counts_node_kind_freq_by_kind_hint() -> None:
    p = distill(_small_bbg(), into=wayfinder_ontology())
    assert p.topology["node_kind_freq"] == {"endpoint": 2, "db": 1, "cred": 1}


def test_topology_salient_kind_freq_only_counts_salient_things() -> None:
    p = distill(_small_bbg(), into=wayfinder_ontology())
    # thing.dash (salient) + thing.cred (salient)
    assert p.topology["salient_kind_freq"] == {"endpoint": 1, "cred": 1}


def test_topology_dead_end_ratio() -> None:
    p = distill(_small_bbg(), into=wayfinder_ontology())
    # 1 dead_end out of 3 traversals
    assert p.topology["dead_end_ratio"] == round(1 / 3, 3)


def test_topology_hidden_signal_counts_confirmed_thought_anchors() -> None:
    p = distill(_small_bbg(), into=wayfinder_ontology())
    # thought.2 (confirmed) anchored to thing.cred (kind_hint=cred)
    assert p.topology["hidden_signal"] == {"cred": 1}


def test_topology_dead_end_ratio_zero_when_no_traversals() -> None:
    g = _new_bbg()
    _add_thing(g, "thing.x", kind_hint="x")
    p = distill(g, into=wayfinder_ontology())
    assert p.topology["dead_end_ratio"] == 0.0


# ---------------------------------------------------------------------------
# task seeds
# ---------------------------------------------------------------------------


def test_task_seed_clusters_revises_chain() -> None:
    """Two thoughts on the same anchor + a revises edge => one cluster."""
    p = distill(_small_bbg(), into=wayfinder_ontology())
    # thought.0 + thought.1 (refuted/open on thing.dash) cluster together;
    # thought.2 (confirmed on thing.cred) is its own cluster
    assert len(p.task_seeds) == 2


def test_task_seed_anchor_kinds_collected() -> None:
    p = distill(_small_bbg(), into=wayfinder_ontology())
    by_theme = {s.theme: s for s in p.task_seeds}
    # one cluster anchors on "endpoint" (thing.dash);
    # the other anchors on "cred" (thing.cred)
    all_kinds = {tuple(s.anchor_kinds) for s in p.task_seeds}
    assert ("endpoint",) in all_kinds
    assert ("cred",) in all_kinds
    # one seed per cluster
    assert len(by_theme) == 2


def test_task_seed_difficulty_rises_with_refuted_count_and_dead_ends() -> None:
    p = distill(_small_bbg(), into=wayfinder_ontology())
    # the refuted-thought cluster (containing thought.0, status=refuted)
    # should have higher difficulty than the all-confirmed cred cluster.
    seeds_by_kinds = {tuple(s.anchor_kinds): s.difficulty for s in p.task_seeds}
    assert seeds_by_kinds[("endpoint",)] > seeds_by_kinds[("cred",)]


def test_task_seed_family_is_never_set_by_distill() -> None:
    p = distill(_small_bbg(), into=wayfinder_ontology())
    # distill must never tag a seed with a family — that's a harness call
    for s in p.task_seeds:
        assert s.family is None


# ---------------------------------------------------------------------------
# goal kinds: productive-path sinks
# ---------------------------------------------------------------------------


def test_goal_kinds_collected_from_productive_path_sinks() -> None:
    p = distill(_small_bbg(), into=wayfinder_ontology())
    # thing.cred is a productive-path sink (no out-edges)
    sinks = {kind for s in p.task_seeds for kind in s.suggested_goal_kinds}
    assert "cred" in sinks
    # thing.db is also productive-reached but it HAS an out-edge -> not a sink
    assert "db" not in sinks
    # thing.dash is reached only via a dead-end -> never a goal candidate
    assert "endpoint" not in sinks


# ---------------------------------------------------------------------------
# coverage
# ---------------------------------------------------------------------------


def test_coverage_is_explored_density_per_kind() -> None:
    p = distill(_small_bbg(), into=wayfinder_ontology())
    # all endpoints and the db are explored=True; thing.cred is explored=False
    assert p.coverage["endpoint"] == 1.0
    assert p.coverage["db"] == 1.0
    assert p.coverage["cred"] == 0.0


# ---------------------------------------------------------------------------
# ontology induction vs refinement
# ---------------------------------------------------------------------------


def test_distill_into_none_induces_a_proposal_ontology() -> None:
    p = distill(_small_bbg(), into=None)
    assert p.ontology.id == "distilled@0.1.0"
    assert set(p.ontology.node_kinds) == {"endpoint", "db", "cred"}
    # no edge induction in the proposal path
    assert p.ontology.edge_kinds == {}


def test_distill_into_existing_ontology_keeps_it() -> None:
    target = Ontology(id="webapp@1")
    p = distill(_small_bbg(), into=target)
    assert p.ontology is target


# ---------------------------------------------------------------------------
# source string
# ---------------------------------------------------------------------------


def test_source_string_carries_ontology_and_content_hash() -> None:
    g = _small_bbg()
    p = distill(g, into=wayfinder_ontology())
    assert p.source.startswith(f"{BBG_ONTOLOGY_ID} :: sha256:")
    assert g.content_hash() in p.source


# ---------------------------------------------------------------------------
# status_log is accepted (currently doesn't change topology stats)
# ---------------------------------------------------------------------------


def test_status_log_argument_is_accepted_without_changing_stats() -> None:
    g = _small_bbg()
    log = [
        StatusEvent("thing.dash", "incidental", 0, "first-visit"),
        StatusEvent("thing.dash", "salient", 1, "anchor:thought.0"),
        StatusEvent("thought.0", "open", 0, "formed"),
        StatusEvent("thought.0", "refuted", 2, "revised-by:thought.1"),
    ]
    p_with_log = distill(g, status_log=log, into=wayfinder_ontology())
    p_without = distill(g, into=wayfinder_ontology())
    # topology is derived from node attrs (the latest state); the log is a
    # forward-compat hook for richer extraction. Today's stats should match.
    assert p_with_log.topology == p_without.topology
    assert p_with_log.coverage == p_without.coverage


# ---------------------------------------------------------------------------
# empty graph: distill must not crash
# ---------------------------------------------------------------------------


def test_distill_on_empty_graph_is_valid() -> None:
    g = _new_bbg()
    p = distill(g, into=wayfinder_ontology())
    assert p.topology["node_kind_freq"] == {}
    assert p.topology["dead_end_ratio"] == 0.0
    assert p.task_seeds == ()
    assert p.coverage == {}
