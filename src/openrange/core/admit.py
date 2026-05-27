"""Layered admission gate. See DESIGN.md §6."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from graphschema import (
    Edge,
    Issue,
    Node,
    Visibility,
    WorldGraph,
    validate,
)

from openrange.core.pack import (
    FeasibilityVerdict,
    Pack,
    PackPrior,
    TaskFamily,
    TaskSpec,
)


@dataclass(frozen=True)
class BuildEvent:
    """One entry in `Snapshot.history`.

    `phase` ∈ {"build", "validate", "feasibility", "repair", "freeze", "evolve"}.
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
    """An admitted, frozen world. `snapshot_id == graph.content_hash()`."""

    snapshot_id: str
    ontology_id: str
    graph: WorldGraph
    tasks: tuple[TaskSpec, ...]
    lineage: Mapping[str, Any]
    history: tuple[BuildEvent, ...] = ()


@dataclass
class AdmissionFailure:
    """Returned when a candidate cannot be admitted within the repair budget."""

    issues: list[Issue]
    infeasible_tasks: list[str]
    attempts: int
    history: tuple[BuildEvent, ...] = ()


def validate_task_bindings(
    graph: WorldGraph,
    tasks: list[TaskSpec],
) -> list[Issue]:
    """Check that every task entrypoint and goal references a real node;
    entrypoints must not be HIDDEN. Goals may be HIDDEN."""
    issues: list[Issue] = []
    for t in tasks:
        if not t.entrypoints:
            issues.append(
                Issue(
                    "error",
                    "task_no_entrypoint",
                    f"task {t.id!r}: must declare at least one entrypoint",
                    t.id,
                )
            )
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


def admit(
    pack: Pack,
    manifest: Mapping[str, Any],
    prior: PackPrior | None = None,
    max_repairs: int = 2,
) -> Snapshot | AdmissionFailure:
    """Turn a manifest into a frozen Snapshot, or fail."""
    ontology = pack.ontology()
    builder = pack.make_builder(prior)
    families = {f.id: f for f in pack.task_families()}
    result = builder.build(manifest)

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
        issues = validate(result.graph, ontology, pack.invariants())
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
    infeasible: list[str] = []
    for t in tasks:
        # An empty-entrypoint task is structurally rejected upstream
        # (``validate_task_bindings``); skipping feasibility here keeps
        # family ``check_feasibility`` implementations free to assume a
        # non-empty entrypoint tuple without crashing the admission loop.
        if not t.entrypoints:
            infeasible.append(t.id)
            continue
        family = families.get(t.feasibility_check)
        if family is None:
            infeasible.append(t.id)
            continue
        verdict: FeasibilityVerdict = family.check_feasibility(graph, t)
        if not verdict.feasible:
            infeasible.append(t.id)
    return infeasible


def snapshot_to_dict(snap: Snapshot) -> dict[str, Any]:
    """JSON-ready projection. See CONTRACTS.md §5 for the wire shape."""
    graph_block: dict[str, Any] = {
        "ontology": snap.graph.ontology,
        "nodes": [
            _node_dict(n) for n in sorted(snap.graph.nodes.values(), key=lambda n: n.id)
        ],
        "edges": [
            _edge_dict(e) for e in sorted(snap.graph.edges.values(), key=lambda e: e.id)
        ],
    }
    if snap.graph.meta:
        graph_block["meta"] = dict(sorted(snap.graph.meta.items()))
    return {
        "snapshot_id": snap.snapshot_id,
        "ontology_id": snap.ontology_id,
        "graph": graph_block,
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
