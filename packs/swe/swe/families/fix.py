"""``swe.fix`` — the agent edits a buggy repo until its held-out suite is green.

Feasibility is the world's *self-test*, the SWE generalization of cyber's
"reference passes + mutation breaks": admission runs the repo's own tests twice
— the gold fix must make every FAIL_TO_PASS + PASS_TO_PASS test pass, and the
un-fixed base must fail every FAIL_TO_PASS while keeping PASS_TO_PASS green. A
world only admits if the bug is real, the fix resolves it, and the suite isn't
independently broken. Success replays the agent's edited tree against the same
suite — SWE-bench's "resolved" criterion.
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


class SweFix(TaskFamily):
    id = "swe.fix"
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
        # A fix task needs a red->green target; a build-only world (no
        # fail_to_pass) is swe.build's to claim, not ours.
        if not str_list(target.suite.attrs.get("fail_to_pass")):
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
                    "fail_to_pass": str_list(target.suite.attrs.get("fail_to_pass")),
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
        f2p = str_list(target.suite.attrs.get("fail_to_pass"))
        p2p = str_list(target.suite.attrs.get("pass_to_pass"))
        if not f2p:
            return FeasibilityVerdict(False, "suite declares no fail_to_pass tests")
        test_files = str_map(target.suite.attrs.get("test_files"))
        base = str_map(target.repo.attrs.get("base_files"))
        gold = str_map(target.solution.attrs.get("gold_files"))
        all_ids = [*f2p, *p2p]

        gold_report = run_tests({**base, **gold}, test_files, all_ids)
        if not gold_report.all_pass(all_ids):
            return FeasibilityVerdict(
                False,
                f"gold fix does not green the suite ({gold_report.passed}/"
                f"{len(all_ids)} pass): {dict(gold_report.results)}",
            )

        base_report = run_tests(base, test_files, all_ids)
        if not base_report.all_fail(f2p):
            return FeasibilityVerdict(
                False,
                "base state does not fail every fail_to_pass test — the bug "
                "is not real or the task is trivially passable: "
                f"{dict(base_report.results)}",
            )
        if p2p and not base_report.all_pass(p2p):
            return FeasibilityVerdict(
                False,
                "base state breaks a pass_to_pass test — the suite is "
                f"independently broken: {dict(base_report.results)}",
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
        f2p = str_list(target.suite.attrs.get("fail_to_pass"))
        p2p = str_list(target.suite.attrs.get("pass_to_pass"))
        test_files = str_map(target.suite.attrs.get("test_files"))
        all_ids = [*f2p, *p2p]
        report = run_tests(tree, test_files, all_ids)
        resolved = report.all_pass(all_ids)
        return EpisodeResult(
            success=resolved,
            subgoals={tid: report.results.get(tid, False) for tid in all_ids},
            reason=(
                "all held-out tests pass"
                if resolved
                else f"{report.passed}/{len(all_ids)} held-out tests pass"
            ),
        )


def _instruction(target: Target) -> str:
    name = target.repo.attrs.get("name")
    problem = target.repo.attrs.get("problem_statement")
    f2p = str_list(target.suite.attrs.get("fail_to_pass"))
    failing = "\n".join(f"  - {tid}" for tid in f2p)
    return f"""You are working in the {name} repository. A defect needs fixing:

{problem}

Edit the files in your workspace to resolve it. When you are done, write
result.json (any JSON object, e.g. {{"done": true}}) to end the episode.

Your edited tree is replayed against a held-out test suite; you succeed when
these tests go from failing to passing, without breaking the rest of the suite:

{failing}

You never see the test files — fix the code, not the tests."""
