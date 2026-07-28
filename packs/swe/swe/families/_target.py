"""Shared repo/suite/solution resolution for the SWE task families.

Both ``swe.fix`` and ``swe.build`` work the same world shape — one repo, its
held-out ``test_suite``, and a HIDDEN ``solution`` — and need the same two
moves: pick that triple at generate time, and re-resolve it from a task's
``entrypoints``/``goal_nodes`` at feasibility/success time. That logic lives
here once so the families differ only in *how they grade* (fix flips
fail_to_pass; build shapes on unit_tests and gates on integration_tests), not in
how they find what to grade.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from graphschema import Node, WorldGraph
from openrange_pack_sdk import FeasibilityVerdict, TaskSpec


@dataclass(frozen=True)
class Target:
    """The repo under task, its grading suite, and the reference solution."""

    repo: Node
    suite: Node
    solution: Node


def pick_target(graph: WorldGraph) -> Target | None:
    """The repo + its suite + its solution, or ``None`` if the triple is
    incomplete (generate-time: skip rather than emit a broken task)."""
    repos = graph.by_kind("repo")
    if not repos:
        return None
    repo = repos[0]
    suite = first_dst(graph, repo.id, "has_suite", "test_suite")
    solution = first_dst(graph, repo.id, "has_solution", "solution")
    if suite is None or solution is None:
        return None
    return Target(repo, suite, solution)


def resolve_target(graph: WorldGraph, task: TaskSpec) -> Target | FeasibilityVerdict:
    """Re-resolve a task's triple from its entrypoint/goal, or an infeasible
    verdict naming what's missing (feasibility/success-time)."""
    if not task.entrypoints or not task.goal_nodes:
        return FeasibilityVerdict(False, "missing entrypoint or goal")
    repo = graph.nodes.get(task.entrypoints[0])
    if repo is None or repo.kind != "repo":
        return FeasibilityVerdict(False, "entrypoint is not a repo")
    suite = graph.nodes.get(task.goal_nodes[0])
    if suite is None or suite.kind != "test_suite":
        return FeasibilityVerdict(False, "goal is not a test_suite")
    if not any(e.dst == suite.id for e in graph.out_edges(repo.id, "has_suite")):
        return FeasibilityVerdict(False, "repo is not graded by the goal suite")
    solution = first_dst(graph, repo.id, "has_solution", "solution")
    if solution is None:
        return FeasibilityVerdict(False, "repo has no reference solution")
    return Target(repo, suite, solution)


def first_dst(
    graph: WorldGraph, src: str, edge_kind: str, dst_kind: str
) -> Node | None:
    for edge in graph.out_edges(src, edge_kind):
        node = graph.nodes.get(edge.dst)
        if node is not None and node.kind == dst_kind:
            return node
    return None


def str_map(value: object) -> dict[str, str]:
    if not isinstance(value, Mapping):
        return {}
    return {str(k): str(v) for k, v in value.items()}


def str_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(v) for v in value]


def grading_tree(target: Target, workspace: Mapping[str, object]) -> dict[str, str]:
    """The agent's edits to the files this world declared, and nothing else.

    The grader runs pytest over what it is given, and pytest loads whatever it
    finds: a ``conftest.py`` or ``pytest.ini`` the agent invented is collected
    like any other file. That is a route to the grade that does not go through
    the source — to a plugin hook that passes every test, or to a hang that
    makes the run report nothing at all. Only declared paths are graded, so a
    file the world never asked for cannot influence the result.
    """
    declared = str_map(target.repo.attrs.get("base_files"))
    return {
        str(path): str(content)
        for path, content in workspace.items()
        if str(path) in declared
    }
