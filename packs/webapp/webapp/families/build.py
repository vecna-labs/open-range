"""WebappBuild — the `webapp.build` TaskFamily.

The agent's job: open the repo, write a feature endpoint (typically
authentication-related), and ensure it serves correctly. Success comes
from the realizer's smoke test against the agent's modified code.

Entrypoint: a `repo` node. The agent gets a working tree.
Goal:       an `endpoint` node. The agent's edits must make this
            endpoint serve a 200 to a valid request.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from openrange import (
    EpisodeResult,
    FeasibilityVerdict,
    Manifest,
    PackPrior,
    TaskFamily,
    TaskSpec,
    WorldGraph,
)


class WebappBuild(TaskFamily):
    """Pose a build task against one webapp world.

    A v1 generation strategy: pick the (single) repo, pick one endpoint
    that has `auth_required=True` as the build target, write a literal
    instruction. Success-check reads `endpoint_serves_200` from the
    realizer's collected final-state.
    """

    id = "webapp.build"
    pack_id = "webapp"

    def generate(
        self, graph: WorldGraph, manifest: Manifest, prior: PackPrior | None
    ) -> list[TaskSpec]:
        del manifest, prior  # v1 is graph-driven; manifest/prior unused
        repos = graph.by_kind("repo")
        endpoints = graph.by_kind("endpoint")
        if not repos or not endpoints:
            return []
        # pick the auth-required endpoint if any; else the first one
        target = next(
            (e for e in endpoints if e.attrs.get("auth_required")),
            endpoints[0],
        )
        repo = repos[0]
        return [
            TaskSpec(
                id="webapp.build.0",
                instruction=(
                    f"Implement the {target.attrs['method']} "
                    f"{target.attrs['path']} endpoint in the {repo.attrs['name']} "
                    f"repo so it serves a 200 to a valid request."
                ),
                entrypoints=(repo.id,),
                goal_nodes=(target.id,),
                feasibility_check="webapp.build",
                success_check="webapp.build",
                meta={
                    "family": "webapp.build",
                    "difficulty": 0.4,
                    "target_path": target.attrs["path"],
                },
            )
        ]

    def check_feasibility(
        self, graph: WorldGraph, task: TaskSpec
    ) -> FeasibilityVerdict:
        """Verify the entrypoint repo exists and the goal endpoint is
        owned by a service in that repo (otherwise the agent cannot
        plausibly affect it from the repo's code)."""
        if not task.entrypoints or not task.goal_nodes:
            return FeasibilityVerdict(False, "missing entrypoint or goal")
        repo_id = task.entrypoints[0]
        if graph.nodes.get(repo_id, None) is None:
            return FeasibilityVerdict(False, "entrypoint repo not in graph")
        endpoint_id = task.goal_nodes[0]
        endpoint = graph.nodes.get(endpoint_id)
        if endpoint is None or endpoint.kind != "endpoint":
            return FeasibilityVerdict(False, "goal is not an endpoint")
        # walk endpoint <-exposes- service -owned_by-> repo
        for exposes in graph.in_edges(endpoint_id, "exposes"):
            service_id = exposes.src
            for owns in graph.out_edges(service_id, "owned_by"):
                if owns.dst == repo_id:
                    return FeasibilityVerdict(True)
        return FeasibilityVerdict(
            False,
            "no service owned by the repo exposes the goal endpoint",
        )

    def check_success(
        self, graph: WorldGraph, task: TaskSpec, final_state: Mapping[str, Any]
    ) -> EpisodeResult:
        """Read the realizer's collected final state.

        v1 looks for a single bool key `endpoint_serves_200`. A richer
        v2 might inspect HTTP status, body content, and unit-test
        results. v1 leaves room: returns subgoals reflecting whatever
        else the realizer collected.
        """
        del graph, task  # graph + task aren't needed for v1 scoring
        ok = bool(final_state.get("endpoint_serves_200"))
        return EpisodeResult(
            success=ok,
            subgoals={
                k: bool(v) for k, v in final_state.items() if isinstance(v, bool)
            },
            reason="endpoint serves 200"
            if ok
            else "endpoint did not serve 200 to the smoke test",
        )
