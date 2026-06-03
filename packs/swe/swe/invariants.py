"""Pack invariants for the SWE world — admission layer 3 (structural only).

Cheap structural well-formedness the ontology can't express: the file maps are
actually ``{str: str}``, the suite grades something (a fail_to_pass for a fix or
an integration_tests gate for a build), the F2P/P2P and unit/integration tiers
are each disjoint, and every test id points at a file the suite ships. The
*behavioral* well-posedness (does the gold overlay green the suite? does the
base fail it?) needs to run pytest, so it lives in the family's
``check_feasibility``, not here — the same split as cyber (structural invariants
vs. ``grade_source`` in ``check_feasibility``).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TypeGuard

from graphschema import Issue, WorldGraph


def repo_has_base_files(graph: WorldGraph) -> list[Issue]:
    """Every repo ships a non-empty ``{str: str}`` base working tree."""
    issues: list[Issue] = []
    for repo in graph.by_kind("repo"):
        base = repo.attrs.get("base_files")
        if not _is_str_map(base) or not base:
            issues.append(
                Issue(
                    "error",
                    "repo_base_files_empty",
                    f"repo {repo.id!r} base_files is not a non-empty {{str: str}} map",
                    repo.id,
                )
            )
    return issues


def suite_well_formed(graph: WorldGraph) -> list[Issue]:
    """Each suite grades *something*, with disjoint F2P/P2P and unit/integration
    tiers and no test id naming a file it doesn't ship.

    A suite must declare at least one graded id: ``fail_to_pass`` (the fix task)
    or ``integration_tests`` (the build task's success gate). ``unit_tests`` and
    ``integration_tests`` are checked for danglers alongside F2P / P2P, and an id
    may not sit in both tiers (a test either shapes or gates, never both).
    """
    issues: list[Issue] = []
    for suite in graph.by_kind("test_suite"):
        test_files = suite.attrs.get("test_files")
        if not _is_str_map(test_files) or not test_files:
            issues.append(
                Issue(
                    "error",
                    "suite_test_files_empty",
                    f"test_suite {suite.id!r} test_files is not a non-empty "
                    "{str: str} map",
                    suite.id,
                )
            )
            continue
        f2p = _as_str_list(suite.attrs.get("fail_to_pass"))
        p2p = _as_str_list(suite.attrs.get("pass_to_pass"))
        units = _as_str_list(suite.attrs.get("unit_tests"))
        integration = _as_str_list(suite.attrs.get("integration_tests"))
        if not f2p and not integration:
            issues.append(
                Issue(
                    "error",
                    "suite_grades_nothing",
                    f"test_suite {suite.id!r} declares neither fail_to_pass (fix) "
                    "nor integration_tests (build) — it grades nothing",
                    suite.id,
                )
            )
            continue
        overlap = sorted(set(f2p) & set(p2p))
        if overlap:
            issues.append(
                Issue(
                    "error",
                    "suite_f2p_p2p_overlap",
                    f"test_suite {suite.id!r} lists {overlap} in both "
                    "fail_to_pass and pass_to_pass",
                    suite.id,
                )
            )
        tier_overlap = sorted(set(units) & set(integration))
        if tier_overlap:
            issues.append(
                Issue(
                    "error",
                    "suite_tier_overlap",
                    f"test_suite {suite.id!r} lists {tier_overlap} in both "
                    "unit_tests and integration_tests",
                    suite.id,
                )
            )
        paths = set(test_files)
        for tid in (*f2p, *p2p, *units, *integration):
            path = tid.split("::", 1)[0]
            if path not in paths:
                issues.append(
                    Issue(
                        "error",
                        "suite_test_id_dangling",
                        f"test_suite {suite.id!r} test id {tid!r} names file "
                        f"{path!r} not in test_files",
                        suite.id,
                    )
                )
    return issues


def solution_present(graph: WorldGraph) -> list[Issue]:
    """Each repo links exactly one suite and one solution; each solution ships a
    non-empty gold overlay."""
    issues: list[Issue] = []
    for repo in graph.by_kind("repo"):
        suites = len(graph.out_edges(repo.id, "has_suite"))
        solutions = len(graph.out_edges(repo.id, "has_solution"))
        if suites != 1:
            issues.append(
                Issue(
                    "error",
                    "repo_suite_cardinality",
                    f"repo {repo.id!r} has {suites} has_suite edges; need exactly 1",
                    repo.id,
                )
            )
        if solutions != 1:
            issues.append(
                Issue(
                    "error",
                    "repo_solution_cardinality",
                    f"repo {repo.id!r} has {solutions} has_solution edges; "
                    "need exactly 1",
                    repo.id,
                )
            )
    for solution in graph.by_kind("solution"):
        gold = solution.attrs.get("gold_files")
        if not _is_str_map(gold) or not gold:
            issues.append(
                Issue(
                    "error",
                    "solution_gold_files_empty",
                    f"solution {solution.id!r} gold_files is not a non-empty "
                    "{str: str} map",
                    solution.id,
                )
            )
    return issues


def _is_str_map(value: object) -> TypeGuard[Mapping[str, str]]:
    return isinstance(value, Mapping) and all(
        isinstance(k, str) and isinstance(v, str) for k, v in value.items()
    )


def _is_str_list(value: object) -> TypeGuard[list[str]]:
    return isinstance(value, list) and all(isinstance(v, str) for v in value)


def _as_str_list(value: object) -> list[str]:
    """Coerce a graph attr to ``list[str]`` (empty when absent/malformed)."""
    return list(value) if _is_str_list(value) else []
