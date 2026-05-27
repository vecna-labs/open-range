"""Tests for cyber_webapp's per-family mutation enumerator.

These tests build a real snapshot from ``WebappPack().make_builder(None)``
and exercise direction tagging, relevance scoring, family routing, and
patch realization end-to-end.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from cyber_webapp import WebappPack, WebappPentest
from cyber_webapp.mutation import available_mutations
from cyber_webapp.vulnerabilities import CATALOG as VULN_CATALOG
from graphschema import GraphPatch, WorldGraph, apply_patch
from openrange_pack_sdk import EpisodeReportLike, Mutation, Snapshot


def _build_snapshot(seed: int = 0) -> Snapshot:
    """Build a real snapshot via the pack's own builder + manifest path."""
    pack = WebappPack()
    result = pack.make_builder(None).build({"seed": seed})
    graph = result.graph
    return Snapshot(
        snapshot_id=graph.content_hash(),
        ontology_id=graph.ontology,
        graph=graph,
        tasks=tuple(result.tasks),
        lineage={},
    )


@dataclass(frozen=True)
class _ReportShim:
    """A minimal EpisodeReportLike that also carries `final_state`.

    The mutation enumerator's relevance heuristic narrows on a
    `final_state` attribute via a local Protocol (`_ReportWithFinalState`
    in `cyber_webapp.mutation`); the public Protocol the function
    signature names is `EpisodeReportLike` (just `passed`). This shim
    satisfies both — `passed` for the static type, `final_state` for the
    runtime narrowing path the relevance scorer takes.

    We don't pull in the full `EpisodeReport` dataclass because it
    lacks `passed` as a property (it exposes the same signal indirectly
    via `verifier_result["passed"]`).
    """

    passed: bool
    final_state: Mapping[str, Any] = field(default_factory=dict)


def _report_with_paths(
    snapshot_id: str,
    paths: Sequence[str],
    *,
    passed: bool = True,
) -> EpisodeReportLike:
    del snapshot_id  # not needed by available_mutations; kept for call-site clarity
    return _ReportShim(
        passed=passed,
        final_state=MappingProxyType({"requests_made": list(paths)}),
    )


def _vuln_paths(graph: WorldGraph, kind: str) -> list[str]:
    """HTTP paths of endpoints affected by vulnerabilities of ``kind``."""
    vuln_ids = {
        n.id
        for n in graph.nodes.values()
        if n.kind == "vulnerability" and n.attrs.get("kind") == kind
    }
    target_ids: set[str] = set()
    for edge in graph.edges.values():
        if edge.kind == "affects" and edge.src in vuln_ids:
            target_ids.add(edge.dst)
    return [
        str(n.attrs.get("path", ""))
        for n in graph.nodes.values()
        if n.id in target_ids and n.kind == "endpoint"
    ]


def test_available_mutations_returns_mutation_tuple() -> None:
    """The enumerator returns a non-empty tuple of `Mutation` values."""
    snap = _build_snapshot()
    options = available_mutations(snap.graph, "webapp.pentest", ())
    assert isinstance(options, tuple)
    assert options
    for opt in options:
        assert isinstance(opt, Mutation)
        assert isinstance(opt.patch, GraphPatch)
        assert opt.direction in {"harden", "soften", "diversify"}
        assert 0.0 <= opt.relevance <= 1.0


def test_available_mutations_tags_family_argument() -> None:
    """Every emitted Mutation carries the `family_id` it was called with."""
    snap = _build_snapshot()
    for family_id in ("webapp.pentest", "webapp.build", "custom.future_family"):
        options = available_mutations(snap.graph, family_id, ())
        assert options
        assert all(opt.family == family_id for opt in options)


def test_available_mutations_covers_every_catalog_kind() -> None:
    """Across the option list, every catalog kind shows up somewhere.

    A `harden` proposes an absent kind, `soften`/`diversify` operate on
    present kinds; together they exhaust the catalog.
    """
    snap = _build_snapshot()
    options = available_mutations(snap.graph, "webapp.pentest", ())
    kinds_seen: set[str] = set()
    for opt in options:
        # `harden` adds a new vuln node; `soften` removes vuln nodes;
        # `diversify` updates one vuln in place. The kind shows up in
        # the added node's attrs (harden), or in the note text for
        # remove/swap proposals.
        for node in opt.patch.nodes_added + opt.patch.nodes_updated:
            attr_kind = node.attrs.get("kind")
            if isinstance(attr_kind, str):
                kinds_seen.add(attr_kind)
        # For pure-removal patches the kind only appears in the note.
        for kind in VULN_CATALOG:
            if kind in opt.note:
                kinds_seen.add(kind)
    assert set(VULN_CATALOG).issubset(kinds_seen)


def test_directions_produce_different_patch_shapes() -> None:
    """harden / soften / diversify each produce a structurally distinct patch.

    harden:    adds a vuln node + an `affects` edge.
    soften:    removes vuln node(s) + dangling edge(s).
    diversify: updates one vuln node in place; no add, no remove.
    """
    snap = _build_snapshot()
    options = available_mutations(snap.graph, "webapp.pentest", ())
    by_direction: dict[str, list[Mutation]] = {}
    for opt in options:
        by_direction.setdefault(opt.direction, []).append(opt)
    # The seed=0 world carries multiple vuln kinds, so every direction
    # has at least one representative.
    assert {"harden", "soften", "diversify"}.issubset(by_direction.keys())

    for opt in by_direction["harden"]:
        assert opt.patch.nodes_added, "harden must add a vuln node"
        assert opt.patch.edges_added, "harden must add an affects edge"
        assert not opt.patch.nodes_removed
        assert not opt.patch.nodes_updated

    for opt in by_direction["soften"]:
        assert opt.patch.nodes_removed, "soften must remove vuln node(s)"
        assert not opt.patch.nodes_added
        assert not opt.patch.nodes_updated

    for opt in by_direction["diversify"]:
        assert opt.patch.nodes_updated, "diversify must update a vuln in place"
        assert not opt.patch.nodes_added
        assert not opt.patch.nodes_removed


def test_soften_relevance_is_floor_without_reports() -> None:
    """No reports → soften relevance equals the documented floor (0.05)."""
    snap = _build_snapshot()
    options = available_mutations(snap.graph, "webapp.pentest", ())
    soften_options = [o for o in options if o.direction == "soften"]
    assert soften_options
    for opt in soften_options:
        assert opt.relevance == 0.05


def test_relevance_climbs_when_agent_hits_vuln_endpoints() -> None:
    """Hits on a vuln-bearing endpoint push that kind's `soften`
    relevance well above the floor.
    """
    snap = _build_snapshot()
    kinds_in_world = sorted(
        {
            str(n.attrs.get("kind"))
            for n in snap.graph.nodes.values()
            if n.kind == "vulnerability"
        },
    )
    assert kinds_in_world
    target_kind = kinds_in_world[0]
    paths = _vuln_paths(snap.graph, target_kind)
    assert paths, f"no endpoint path found for {target_kind}"
    report = _report_with_paths(snap.snapshot_id, [paths[0]] * 10)

    options = available_mutations(snap.graph, "webapp.pentest", [report])
    soften_for_target = next(
        o for o in options if o.direction == "soften" and target_kind in o.note
    )
    assert soften_for_target.relevance > 0.5


def test_unrelated_paths_dont_inflate_relevance() -> None:
    """Paths that don't intersect a vuln-bearing endpoint don't drive
    that kind's soften relevance above the floor."""
    snap = _build_snapshot()
    kinds_in_world = sorted(
        {
            str(n.attrs.get("kind"))
            for n in snap.graph.nodes.values()
            if n.kind == "vulnerability"
        },
    )
    target_kind = kinds_in_world[0]
    report = _report_with_paths(snap.snapshot_id, ["/does/not/exist"] * 10)

    options = available_mutations(snap.graph, "webapp.pentest", [report])
    soften_for_target = next(
        o for o in options if o.direction == "soften" and target_kind in o.note
    )
    assert soften_for_target.relevance == 0.05


def test_available_mutations_is_deterministic() -> None:
    """Same `(graph, family_id, reports)` triple → identical option order
    and identical patches.
    """
    snap = _build_snapshot()
    first = available_mutations(snap.graph, "webapp.pentest", ())
    second = available_mutations(snap.graph, "webapp.pentest", ())
    assert len(first) == len(second)
    for a, b in zip(first, second, strict=True):
        assert a.direction == b.direction
        assert a.relevance == b.relevance
        assert a.note == b.note
        assert a.family == b.family
        # Patch equality is dataclass field-by-field.
        assert a.patch == b.patch


def test_pentest_family_tags_mutations_with_its_id() -> None:
    """`WebappPentest().available_mutations(snap, reports)` tags
    every Mutation with `family="webapp.pentest"`.
    """
    snap = _build_snapshot()
    options = WebappPentest().available_mutations(snap, [])
    assert options
    assert all(opt.family == "webapp.pentest" for opt in options)


def test_pentest_family_matches_procedural_floor_without_llm() -> None:
    """Without an LLM, the family delegates verbatim to the procedural
    enumerator; per-position patches and directions must match.
    """
    snap = _build_snapshot()
    family_options = WebappPentest().available_mutations(snap, [])
    proc_options = available_mutations(snap.graph, "webapp.pentest", [])
    assert len(family_options) == len(proc_options)
    for fam, proc in zip(family_options, proc_options, strict=True):
        assert fam.direction == proc.direction
        assert fam.relevance == proc.relevance
        assert fam.patch == proc.patch


def test_applying_harden_mutation_adds_a_vulnerability_node() -> None:
    """A `harden` patch increments the vulnerability count when applied."""
    snap = _build_snapshot()
    options = available_mutations(snap.graph, "webapp.pentest", ())
    harden = next(o for o in options if o.direction == "harden")

    before = sum(1 for n in snap.graph.nodes.values() if n.kind == "vulnerability")
    # Patches are designed to be applied to a copy — never mutate the
    # snapshot graph directly in a test.
    mutable = copy.deepcopy(snap.graph)
    apply_patch(mutable, harden.patch)
    after = sum(1 for n in mutable.nodes.values() if n.kind == "vulnerability")

    assert after == before + 1
    # The new vuln carries the same `kind` attr the patch announced.
    added_node = harden.patch.nodes_added[0]
    landed = mutable.nodes.get(added_node.id)
    assert landed is not None
    assert landed.kind == "vulnerability"
    assert landed.attrs.get("kind") == added_node.attrs.get("kind")


def test_applying_soften_mutation_removes_vulnerabilities() -> None:
    """A `soften` patch removes the vulns of its targeted kind from the graph."""
    snap = _build_snapshot()
    options = available_mutations(snap.graph, "webapp.pentest", ())
    soften = next(o for o in options if o.direction == "soften")

    removed_ids = set(soften.patch.nodes_removed)
    assert removed_ids, "soften patch must declare nodes to remove"

    mutable = copy.deepcopy(snap.graph)
    for nid in removed_ids:
        assert nid in mutable.nodes, "soften must target nodes that exist"
    apply_patch(mutable, soften.patch)

    for nid in removed_ids:
        assert nid not in mutable.nodes


def test_applying_diversify_mutation_changes_vuln_kind_in_place() -> None:
    """A `diversify` patch updates an existing vuln node's `kind` attr
    without changing the node-count.
    """
    snap = _build_snapshot()
    options = available_mutations(snap.graph, "webapp.pentest", ())
    diversify = next(o for o in options if o.direction == "diversify")

    updated_node = diversify.patch.nodes_updated[0]
    original = snap.graph.nodes.get(updated_node.id)
    assert original is not None, "diversify must target an existing vuln node"
    assert original.attrs.get("kind") != updated_node.attrs.get("kind"), (
        "diversify must change the kind"
    )

    before_count = len(snap.graph.nodes)
    mutable = copy.deepcopy(snap.graph)
    apply_patch(mutable, diversify.patch)

    assert len(mutable.nodes) == before_count
    landed = mutable.nodes[updated_node.id]
    assert landed.attrs.get("kind") == updated_node.attrs.get("kind")


def test_available_mutations_handles_empty_reports() -> None:
    """An empty reports sequence is the boot path — no exceptions."""
    snap = _build_snapshot()
    options = available_mutations(snap.graph, "webapp.pentest", ())
    assert options
    # All soften options should be at the floor (no signal).
    for opt in options:
        if opt.direction == "soften":
            assert opt.relevance == 0.05
