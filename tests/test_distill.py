"""Tests for `openrange.core.distill` — the WorldGraph -> PackPrior seam.

These tests exercise the new module against hand-built BBG-shaped
graphs. The function is the only thing OpenRange owns that recognises
the `bbg.wayfinder@0.1.0` ontology by structure.
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
    g: WorldGraph,
    src: str,
    dst: str,
    outcome: str = "neutral",
    count: int = 1,
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
    one credential reached only by reference (provenance=referenced,
    salient). A productive sink edge into the credential. A dead-end
    traversal between two endpoints. Two thoughts forming a refutation
    chain via `revises`.
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
    assert p.topology["salient_kind_freq"] == {"endpoint": 1, "cred": 1}


def test_topology_dead_end_ratio() -> None:
    p = distill(_small_bbg(), into=wayfinder_ontology())
    assert p.topology["dead_end_ratio"] == round(1 / 3, 3)


def test_topology_hidden_signal_counts_confirmed_thought_anchors() -> None:
    p = distill(_small_bbg(), into=wayfinder_ontology())
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
    p = distill(_small_bbg(), into=wayfinder_ontology())
    assert len(p.task_seeds) == 2


def test_task_seed_anchor_kinds_collected() -> None:
    p = distill(_small_bbg(), into=wayfinder_ontology())
    all_kinds = {tuple(s.anchor_kinds) for s in p.task_seeds}
    assert ("endpoint",) in all_kinds
    assert ("cred",) in all_kinds


def test_task_seed_difficulty_rises_with_refuted_count_and_dead_ends() -> None:
    p = distill(_small_bbg(), into=wayfinder_ontology())
    seeds_by_kinds = {tuple(s.anchor_kinds): s.difficulty for s in p.task_seeds}
    assert seeds_by_kinds[("endpoint",)] > seeds_by_kinds[("cred",)]


def test_task_seed_family_is_never_set_by_distill() -> None:
    p = distill(_small_bbg(), into=wayfinder_ontology())
    for s in p.task_seeds:
        assert s.family is None


def test_task_seed_is_mutable_so_harness_can_re_tag() -> None:
    """The design ref (crossdomain.py act 2) shows callers re-tagging
    seeds after distill returns. Confirm TaskSeed is mutable enough."""
    p = distill(_small_bbg(), into=wayfinder_ontology())
    p.task_seeds[0].family = "webapp.pentest"
    assert p.task_seeds[0].family == "webapp.pentest"


# ---------------------------------------------------------------------------
# goal kinds: productive-path sinks
# ---------------------------------------------------------------------------


def test_goal_kinds_collected_from_productive_path_sinks() -> None:
    p = distill(_small_bbg(), into=wayfinder_ontology())
    sinks = {kind for s in p.task_seeds for kind in s.suggested_goal_kinds}
    assert "cred" in sinks
    assert "db" not in sinks
    assert "endpoint" not in sinks


# ---------------------------------------------------------------------------
# coverage
# ---------------------------------------------------------------------------


def test_coverage_is_explored_density_per_kind() -> None:
    p = distill(_small_bbg(), into=wayfinder_ontology())
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
# status_log is accepted (currently unused per v1 contract)
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
    assert p.task_seeds == []
    assert p.coverage == {}
