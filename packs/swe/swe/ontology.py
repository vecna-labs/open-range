"""Ontology contract for the SWE pack — a code-repair world over a real repo.

The world's identity *is* the repo recipe: the working tree the agent edits
(``base_files``), the held-out test suite that grades it (``test_suite``), and
the reference fix that proves the task is solvable (``solution``). All three
ride graph-natively as JSON attrs, so the graph's content hash is byte-stable
across machines — the same instance always addresses to the same snapshot id.

This mirrors SWE-bench's instance shape (repo@base + test_patch + gold patch +
FAIL_TO_PASS / PASS_TO_PASS), but expressed as an OpenRange world so admission
can *prove* well-posedness before the world is ever served: the gold fix must
green the suite, and the un-fixed base must fail exactly the FAIL_TO_PASS tests.
"""

from __future__ import annotations

from graphschema import AttrSpec, AttrType, EdgeKind, NodeKind, Ontology

ONTOLOGY_ID = "swe.repo@v1"


def repo_ontology() -> Ontology:
    # fresh instance per call so callers can mutate without leaking
    s = AttrSpec
    return Ontology(
        id=ONTOLOGY_ID,
        node_kinds={
            "repo": NodeKind(
                "repo",
                attrs={
                    "name": s(AttrType.STRING, required=True),
                    "language": s(AttrType.STRING, default="python"),
                    "problem_statement": s(
                        AttrType.STRING,
                        required=True,
                        description="the bug report / feature ask the agent works",
                    ),
                    "base_files": s(
                        AttrType.JSON,
                        required=True,
                        description="{path: contents} working tree the agent edits; "
                        "holds the defect",
                    ),
                },
                description="a code workspace at its buggy base state",
            ),
            "test_suite": NodeKind(
                "test_suite",
                attrs={
                    "test_files": s(
                        AttrType.JSON,
                        required=True,
                        description="{path: contents} held-out tests; never "
                        "materialized into the agent's workspace",
                    ),
                    "fail_to_pass": s(
                        AttrType.JSON,
                        required=True,
                        description="test ids the fix must flip red -> green",
                    ),
                    "pass_to_pass": s(
                        AttrType.JSON,
                        default=[],
                        description="test ids that must stay green (regression guard)",
                    ),
                    "unit_tests": s(
                        AttrType.JSON,
                        default=[],
                        description="build-task shaping signal: per-piece tests that "
                        "give dense partial credit but do not gate success",
                    ),
                    "integration_tests": s(
                        AttrType.JSON,
                        default=[],
                        description="build-task success gate: end-to-end tests that "
                        "only pass when the pieces compose",
                    ),
                },
                description="the repo's own tests — the contract that grades a fix "
                "(or, for a build task, shapes via unit_tests and gates on "
                "integration_tests)",
            ),
            "solution": NodeKind(
                "solution",
                attrs={
                    "gold_files": s(
                        AttrType.JSON,
                        required=True,
                        description="{path: contents} overlay over base_files that "
                        "resolves the task; the admission answer key",
                    ),
                },
                description="the reference fix; HIDDEN — proves the task is solvable",
            ),
        },
        edge_kinds={
            "has_suite": EdgeKind(
                "has_suite",
                endpoints=[("repo", "test_suite")],
                src_max=1,
                dst_max=1,
                description="this suite grades this repo",
            ),
            "has_solution": EdgeKind(
                "has_solution",
                endpoints=[("repo", "solution")],
                src_max=1,
                dst_max=1,
                description="this fix resolves this repo's task",
            ),
        },
    )
