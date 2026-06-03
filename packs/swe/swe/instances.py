"""Load a SWE instance recipe (SWE-bench-shaped) into a world graph.

An instance is the *recipe*; the graph is the world. The recipe carries the
three file maps plus the FAIL_TO_PASS / PASS_TO_PASS test ids; ``to_graph`` lays
them out over the ``swe.repo@v1`` ontology. Fixtures ship under ``fixtures/`` so
the spike is fully offline — the imported (GitHub-passthrough) source will emit
the same shape from a fetched SWE-bench row.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from graphschema import Visibility, WorldGraph
from openrange_pack_sdk import add_edge, add_node

from swe.ontology import ONTOLOGY_ID

_FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"


@dataclass(frozen=True)
class SweInstance:
    """One SWE-bench-shaped task: a repo recipe plus its held-out grader."""

    instance_id: str
    name: str
    language: str
    problem_statement: str
    base_files: dict[str, str]
    gold_files: dict[str, str]
    test_files: dict[str, str]
    fail_to_pass: tuple[str, ...]
    pass_to_pass: tuple[str, ...]
    # Build-task dimension: unit_tests shape (dense partial credit), while
    # integration_tests gate success. Empty for a plain fix instance.
    unit_tests: tuple[str, ...] = ()
    integration_tests: tuple[str, ...] = ()


def load_instance(name: str) -> SweInstance:
    """Read ``fixtures/{name}.json`` into a :class:`SweInstance`."""
    path = _FIXTURE_DIR / f"{name}.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    return SweInstance(
        instance_id=str(raw["instance_id"]),
        name=str(raw["name"]),
        language=str(raw.get("language", "python")),
        problem_statement=str(raw["problem_statement"]),
        base_files={str(k): str(v) for k, v in raw["base_files"].items()},
        gold_files={str(k): str(v) for k, v in raw["gold_files"].items()},
        test_files={str(k): str(v) for k, v in raw["test_files"].items()},
        fail_to_pass=tuple(str(x) for x in raw["fail_to_pass"]),
        pass_to_pass=tuple(str(x) for x in raw.get("pass_to_pass", [])),
        unit_tests=tuple(str(x) for x in raw.get("unit_tests", [])),
        integration_tests=tuple(str(x) for x in raw.get("integration_tests", [])),
    )


def to_graph(instance: SweInstance) -> WorldGraph:
    """Lay an instance out over the ``swe.repo@v1`` ontology.

    File maps and test-id lists ride as JSON attrs so the graph
    content-addresses byte-stably. The solution is HIDDEN (the answer key); the
    suite is PUBLIC in the graph, but the realizer never materializes it into the
    agent's workspace, so it stays held-out behaviorally.
    """
    graph = WorldGraph(ontology=ONTOLOGY_ID)
    repo_id = f"repo:{instance.instance_id}"
    suite_id = f"suite:{instance.instance_id}"
    solution_id = f"solution:{instance.instance_id}"
    add_node(
        graph,
        kind="repo",
        id=repo_id,
        attrs={
            "name": instance.name,
            "language": instance.language,
            "problem_statement": instance.problem_statement,
            "base_files": dict(instance.base_files),
        },
    )
    add_node(
        graph,
        kind="test_suite",
        id=suite_id,
        attrs={
            "test_files": dict(instance.test_files),
            "fail_to_pass": list(instance.fail_to_pass),
            "pass_to_pass": list(instance.pass_to_pass),
            "unit_tests": list(instance.unit_tests),
            "integration_tests": list(instance.integration_tests),
        },
    )
    add_node(
        graph,
        kind="solution",
        id=solution_id,
        attrs={"gold_files": dict(instance.gold_files)},
        visibility=Visibility.HIDDEN,
    )
    add_edge(graph, kind="has_suite", src=repo_id, dst=suite_id)
    add_edge(graph, kind="has_solution", src=repo_id, dst=solution_id)
    return graph
