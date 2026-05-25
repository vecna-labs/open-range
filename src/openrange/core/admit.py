"""admit() — turn a manifest into a content-addressed `Snapshot`, or fail.

Admission is a LAYERED gate, on purpose:

    1. structural       : ids / edge-to-node shape (`_validate_structural`)
    2. ontology         : kinds, required attrs, enum/REF, endpoint pairs
                          (`_validate_conformance`)
    3. pack invariants  : Tier-3 callables the pack ships (`Pack.invariants()`)
    4. task bindings    : entrypoints/goal_nodes exist; entrypoints not HIDDEN
    5. task feasibility : each TaskFamily's `check_feasibility(graph, task)`

Layers 1+2 catch malformed worlds. Layer 3 catches structurally-valid but
semantically-broken worlds. Layer 4 catches mis-bound tasks. Layer 5
catches well-formed worlds that no one can actually solve.

Each layer catches a different bug; all of them are required.

The world graph is **timeless** — content-addressed, the graph IS its
content, no timestamps inside it. That is what keeps two identical
builds sharing one snapshot id. The build PROCESS still has a story
worth keeping (which pass ran, what a repair changed, why an attempt
was rejected); that story lives in `Snapshot.history` as a tuple of
`BuildEvent`s, BESIDE the graph, never inside it.

This is the deliberate asymmetry with the BBG: the BBG carries its own
transaction time inside, as a status-event log, because it has no other
identity to protect. A world graph DOES have an identity to protect
(content-addressing for reproducibility), so its history must live
alongside.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from openrange.core.pack import (
    FeasibilityVerdict,
    Pack,
    PackPrior,
    TaskFamily,
    TaskSpec,
)
from openrange.world_ir import (
    Edge,
    Issue,
    Node,
    Visibility,
    WorldGraph,
    validate,
)

# ---------------------------------------------------------------------------
# Frozen output
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BuildEvent:
    """One entry in a Snapshot's build history.

    Lives BESIDE the graph in `Snapshot.history`. Records what happened
    in the build process: which pass ran, what a repair changed, why an
    attempt was rejected. Build history is also where things that ARE
    NOT graph mutations get recorded (a failed feasibility check, a
    manifest-requested difficulty change) — which an in-graph log
    never could.

    `phase` is one of: "build", "validate", "feasibility", "repair",
    "freeze", "evolve".
    """

    seq: int
    phase: str
    detail: str
    refs: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "seq": self.seq,
            "phase": self.phase,
            "detail": self.detail,
        }
        if self.refs:
            d["refs"] = list(self.refs)
        return d


@dataclass(frozen=True)
class Snapshot:
    """An admitted, frozen world. CORE-owned, content-addressed, immutable.

    Every episode runs against one of these. `graph` is timeless;
    `history` is the build story (ordered BuildEvents); `lineage` holds
    the flat provenance facts (manifest, pack id+version, attempts).

    `snapshot_id == graph.content_hash()` — the only way to derive a
    snapshot id. Two identical builds (same builder, same manifest, same
    seed → same graph) share a snapshot id and are interchangeable for
    every downstream purpose.
    """

    snapshot_id: str
    ontology_id: str
    graph: WorldGraph
    tasks: tuple[TaskSpec, ...]
    lineage: Mapping[str, Any]
    history: tuple[BuildEvent, ...] = ()


@dataclass
class AdmissionFailure:
    """Returned when a candidate cannot be admitted within the repair budget.

    Carries the structured failure detail so a caller can inspect what
    went wrong without re-running validation.
    """

    issues: list[Issue]
    infeasible_tasks: list[str]
    attempts: int
    history: tuple[BuildEvent, ...] = ()


# ---------------------------------------------------------------------------
# Task-binding validation — CORE; generic; never branches on a kind
# ---------------------------------------------------------------------------


def validate_task_bindings(
    graph: WorldGraph,
    tasks: list[TaskSpec],
) -> list[Issue]:
    """Generic check that every task is soundly bound to its world.

    Checks the world-absolute facts CORE can reason about without
    knowing the domain:

      - every entrypoint and goal_node references a real node in the graph
      - an entrypoint is not HIDDEN — you cannot hand the agent a starting
        surface it is not supposed to be able to see
      - a goal MAY be HIDDEN — discovering it is often the point

    Whether a given node is the *right* entrypoint for a task is domain
    knowledge, owned by the TaskFamily that generated the task (and
    checked in its `check_feasibility`).
    """
    issues: list[Issue] = []
    for t in tasks:
        for nid in t.entrypoints:
            n = graph.nodes.get(nid)
            if n is None:
                issues.append(
                    Issue(
                        "error",
                        "task_dangling_entrypoint",
                        f"task {t.id!r}: entrypoint {nid!r} is not in the world graph",
                        t.id,
                    )
                )
            elif n.visibility is Visibility.HIDDEN:
                issues.append(
                    Issue(
                        "error",
                        "task_hidden_entrypoint",
                        f"task {t.id!r}: entrypoint {nid!r} is HIDDEN — cannot be "
                        f"a starting surface",
                        t.id,
                    )
                )
        for nid in t.goal_nodes:
            if nid not in graph.nodes:
                issues.append(
                    Issue(
                        "error",
                        "task_dangling_goal",
                        f"task {t.id!r}: goal {nid!r} is not in the world graph",
                        t.id,
                    )
                )
    return issues


# ---------------------------------------------------------------------------
# The admission loop
# ---------------------------------------------------------------------------


def admit(
    pack: Pack,
    manifest: Mapping[str, Any],
    prior: PackPrior | None = None,
    max_repairs: int = 2,
) -> Snapshot | AdmissionFailure:
    """Turn a manifest into a frozen Snapshot, or fail.

    The boundary between CORE and PACK runs THROUGH this function. Core
    holds the loop; the pack supplies each domain step; core checks the
    result. Every step below is tagged CORE or PACK.

    `prior` is the optional `PackPrior` seam — None means the pack falls
    back to its own hand-authored generation defaults. The builder has
    one code path; it never knows whether the prior was distilled or
    authored.
    """
    # PACK: the ontology VALUE.
    ontology = pack.ontology()

    # PACK: construct a builder. `prior` is the optional PackPrior seam.
    builder = pack.make_builder(prior)

    # PACK: family handles that admission will dispatch into.
    families = {f.id: f for f in pack.task_families()}

    # PACK: first candidate world + tasks.
    result = builder.build(manifest)

    # CORE: the build history accumulates BESIDE the (timeless) graph.
    history: list[BuildEvent] = [
        BuildEvent(
            0,
            "build",
            f"builder produced {len(result.graph.nodes)} nodes, "
            f"{len(result.tasks)} tasks",
            tuple(t.id for t in result.tasks),
        )
    ]

    errors: list[Issue] = []
    infeasible: list[str] = []

    for attempt in range(max_repairs + 1):
        # CORE: structural + conformance + pack invariants.
        issues = validate(result.graph, ontology, pack.invariants())
        # CORE: task-binding check.
        issues += validate_task_bindings(result.graph, result.tasks)
        errors = [i for i in issues if i.severity == "error"]
        history.append(
            BuildEvent(
                len(history),
                "validate",
                f"attempt {attempt + 1}: {len(errors)} error(s)",
                tuple(i.where for i in errors),
            )
        )

        # PACK: family feasibility per task; core dispatches by handle.
        infeasible = _run_feasibility(families, result.graph, result.tasks)
        history.append(
            BuildEvent(
                len(history),
                "feasibility",
                f"attempt {attempt + 1}: {len(infeasible)} infeasible task(s)",
                tuple(infeasible),
            )
        )

        if not errors and not infeasible:
            # CORE: freeze. Content-addressed id; immutable.
            history.append(
                BuildEvent(
                    len(history),
                    "freeze",
                    "world admitted and frozen",
                )
            )
            return Snapshot(
                snapshot_id=result.graph.content_hash(),
                ontology_id=ontology.id,
                graph=result.graph,
                tasks=tuple(result.tasks),
                lineage={
                    "manifest": dict(manifest),
                    "pack": pack.id,
                    "pack_version": pack.version,
                    "attempts": attempt + 1,
                    **dict(result.admission_meta),
                },
                history=tuple(history),
            )

        if attempt == max_repairs:
            break

        # PACK: builder repairs (or regenerates).
        result = builder.repair(result, errors, infeasible)
        history.append(
            BuildEvent(
                len(history),
                "repair",
                f"builder regenerated after attempt {attempt + 1}",
            )
        )

    return AdmissionFailure(
        issues=errors,
        infeasible_tasks=infeasible,
        attempts=max_repairs + 1,
        history=tuple(history),
    )


def _run_feasibility(
    families: Mapping[str, TaskFamily],
    graph: WorldGraph,
    tasks: list[TaskSpec],
) -> list[str]:
    """Dispatch each task to its TaskFamily's `check_feasibility`.

    Tasks naming a family the pack does not declare are treated as
    infeasible — the pack is the authority on what families it owns.
    """
    infeasible: list[str] = []
    for t in tasks:
        family = families.get(t.feasibility_check)
        if family is None:
            infeasible.append(t.id)
            continue
        verdict: FeasibilityVerdict = family.check_feasibility(graph, t)
        if not verdict.feasible:
            infeasible.append(t.id)
    return infeasible


# ---------------------------------------------------------------------------
# Helper: serialize a Snapshot for storage / read APIs.
# ---------------------------------------------------------------------------


def snapshot_to_dict(snap: Snapshot) -> dict[str, Any]:
    """JSON-ready projection of a Snapshot.

    Matches the wire shape declared in `CONTRACTS.md`. The natural
    consumer is a snapshot store or any read API that ships snapshots
    over the wire.
    """
    return {
        "snapshot_id": snap.snapshot_id,
        "ontology_id": snap.ontology_id,
        "graph": {
            "ontology": snap.graph.ontology,
            "nodes": [
                _node_dict(n)
                for n in sorted(snap.graph.nodes.values(), key=lambda n: n.id)
            ],
            "edges": [
                _edge_dict(e)
                for e in sorted(snap.graph.edges.values(), key=lambda e: e.id)
            ],
        },
        "tasks": [
            {
                "id": t.id,
                "instruction": t.instruction,
                "entrypoints": list(t.entrypoints),
                "goal_nodes": list(t.goal_nodes),
                "feasibility_check": t.feasibility_check,
                "success_check": t.success_check,
                **({"meta": dict(t.meta)} if t.meta else {}),
            }
            for t in snap.tasks
        ],
        "lineage": dict(snap.lineage),
        "history": [e.to_dict() for e in snap.history],
    }


def _node_dict(n: Node) -> dict[str, Any]:
    out: dict[str, Any] = {
        "id": n.id,
        "kind": n.kind,
        "attrs": dict(sorted(n.attrs.items())),
    }
    if n.roles:
        out["roles"] = sorted(r.value for r in n.roles)
    if n.visibility is not Visibility.PUBLIC:
        out["visibility"] = n.visibility.value
    return out


def _edge_dict(e: Edge) -> dict[str, Any]:
    out: dict[str, Any] = {"id": e.id, "kind": e.kind, "src": e.src, "dst": e.dst}
    if e.attrs:
        out["attrs"] = dict(sorted(e.attrs.items()))
    return out
