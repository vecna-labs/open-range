"""``swe.build`` — the agent builds a project from a skeleton until it composes.

The long-horizon sibling of ``swe.fix``. Where a fix flips one red test green,
a build implements several pieces that must work *together*: the suite splits
into two tiers on the same ``test_suite`` node —

- ``unit_tests`` **shape**: each piece tested in isolation earns dense partial
  credit, so a half-built project still lands a nonzero reward (the signal a
  trainer needs on a long episode — an all-or-nothing gate would be zero almost
  everywhere). They do *not* decide success.
- ``integration_tests`` **gate**: end-to-end tests that only pass once the
  pieces compose. Success is "every integration test passes" — you can green a
  thousand unit tests at 100% coverage and still fail integration.

Feasibility is the same self-test idea as fix: the gold overlay must green every
unit + integration test (the build is solvable and the suite isn't broken), and
the bare skeleton must fail every integration test (the gate is real — the
pieces don't compose for free).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from graphschema import WorldGraph
from openrange_pack_sdk import (
    EpisodeResult,
    FeasibilityVerdict,
    Manifest,
    PackPrior,
    TaskFamily,
    TaskSpec,
)

from swe.families._target import (
    Target,
    pick_target,
    resolve_target,
    str_list,
    str_map,
)
from swe.grading import run_tests


class SweBuild(TaskFamily):
    id = "swe.build"
    pack_id = "swe"

    def generate(
        self,
        graph: WorldGraph,
        manifest: Manifest,
        prior: PackPrior | None,
    ) -> list[TaskSpec]:
        del manifest, prior
        target = pick_target(graph)
        if target is None:
            return []
        # A build task is defined by its integration gate; a plain fix world
        # (no integration_tests) is swe.fix's to claim, not ours.
        if not str_list(target.suite.attrs.get("integration_tests")):
            return []
        return [
            self.make_task(
                instruction=_instruction(target),
                entrypoints=target.repo.id,
                goal_nodes=target.suite.id,
                index=str(target.repo.attrs.get("name", "repo")),
                meta={
                    "repo": str(target.repo.attrs.get("name")),
                    "language": str(target.repo.attrs.get("language")),
                    "unit_tests": str_list(target.suite.attrs.get("unit_tests")),
                    "integration_tests": str_list(
                        target.suite.attrs.get("integration_tests")
                    ),
                },
            ),
        ]

    def check_feasibility(
        self,
        graph: WorldGraph,
        task: TaskSpec,
    ) -> FeasibilityVerdict:
        target = resolve_target(graph, task)
        if isinstance(target, FeasibilityVerdict):
            return target
        units = str_list(target.suite.attrs.get("unit_tests"))
        integration = str_list(target.suite.attrs.get("integration_tests"))
        if not integration:
            return FeasibilityVerdict(
                False, "suite declares no integration_tests — nothing gates success"
            )
        test_files = str_map(target.suite.attrs.get("test_files"))
        base = str_map(target.repo.attrs.get("base_files"))
        gold = str_map(target.solution.attrs.get("gold_files"))
        all_ids = [*units, *integration]

        gold_report = run_tests({**base, **gold}, test_files, all_ids)
        if not gold_report.all_pass(all_ids):
            return FeasibilityVerdict(
                False,
                f"gold overlay does not green the suite ({gold_report.passed}/"
                f"{len(all_ids)} pass): {dict(gold_report.results)}",
            )

        base_report = run_tests(base, test_files, integration)
        if not base_report.all_fail(integration):
            return FeasibilityVerdict(
                False,
                "skeleton already passes an integration test — the gate is not "
                f"real or the project is pre-composed: {dict(base_report.results)}",
            )
        return FeasibilityVerdict(True)

    def check_success(
        self,
        graph: WorldGraph,
        task: TaskSpec,
        final_state: Mapping[str, Any],
    ) -> EpisodeResult:
        target = resolve_target(graph, task)
        if isinstance(target, FeasibilityVerdict):
            return EpisodeResult(
                success=False, reason=f"task target unresolvable: {target.reason}"
            )
        workspace = final_state.get("workspace_files")
        if not isinstance(workspace, Mapping) or not workspace:
            return EpisodeResult(
                success=False, reason="agent produced no workspace files"
            )
        tree = {str(k): str(v) for k, v in workspace.items()}
        units = str_list(target.suite.attrs.get("unit_tests"))
        integration = str_list(target.suite.attrs.get("integration_tests"))
        test_files = str_map(target.suite.attrs.get("test_files"))
        all_ids = [*units, *integration]
        report = run_tests(tree, test_files, all_ids)
        # Integration GATES success; units + integration both SHAPE the subgoal
        # vector the training seam turns into dense partial credit.
        composed = report.all_pass(integration)
        integ_pass = sum(1 for tid in integration if report.results.get(tid, False))
        return EpisodeResult(
            success=composed,
            subgoals={tid: report.results.get(tid, False) for tid in all_ids},
            reason=(
                f"all {len(integration)} integration tests pass — the pieces compose"
                if composed
                else f"{report.passed}/{len(all_ids)} pieces pass; integration gate "
                f"unmet ({integ_pass}/{len(integration)} integration green)"
            ),
        )


def _instruction(target: Target) -> str:
    name = target.repo.attrs.get("name")
    problem = target.repo.attrs.get("problem_statement")
    units = str_list(target.suite.attrs.get("unit_tests"))
    integration = str_list(target.suite.attrs.get("integration_tests"))
    unit_lines = "\n".join(f"  - {tid}" for tid in units)
    integ_lines = "\n".join(f"  - {tid}" for tid in integration)
    return f"""You are building the {name} project from a skeleton:

{problem}

Implement the files in your workspace so the pieces work — individually and
together. When you are done, write result.json (any JSON object, e.g.
{{"done": true}}) to end the episode.

Your tree is replayed against a held-out suite with two tiers. Unit tests check
each piece on its own and earn partial credit:

{unit_lines}

Integration tests check that the pieces compose end-to-end, and they gate
success — you only resolve the task when every one of them passes:

{integ_lines}

You never see the test files — build the code, not the tests."""
