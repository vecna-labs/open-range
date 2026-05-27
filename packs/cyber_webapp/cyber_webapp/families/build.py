"""`webapp.build` TaskFamily — implement / repair a service endpoint."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from graphschema import Node, WorldGraph

from openrange.core.pack import (
    EpisodeReportLike,
    EpisodeResult,
    FeasibilityVerdict,
    LLMBackendLike,
    Manifest,
    Mutation,
    PackPrior,
    TaskFamily,
    TaskSpec,
)

if TYPE_CHECKING:
    from openrange.core.admit import Snapshot


class WebappBuild(TaskFamily):
    """Stub grader — `check_success` reads `endpoint_serves_200` which
    is set from a probe of `/` (always 200), so passes do not reflect
    agent work. See ROADMAP / cyber `webapp.build`."""

    id = "webapp.build"
    pack_id = "webapp"

    def generate(
        self,
        graph: WorldGraph,
        manifest: Manifest,
        prior: PackPrior | None,
    ) -> list[TaskSpec]:
        del manifest, prior
        target_service = self._pick_service(graph)
        if target_service is None:
            return []
        target_endpoint = self._pick_endpoint(graph, target_service.id)
        if target_endpoint is None:
            return []
        return [
            TaskSpec(
                id="webapp.build.0",
                instruction=(
                    f"Implement the {target_endpoint.attrs.get('method', 'GET')} "
                    f"{target_endpoint.attrs.get('path', '/')} endpoint in the "
                    f"{target_service.attrs.get('name', target_service.id)} service "
                    "so it serves a 200 to a valid request."
                ),
                entrypoints=(target_service.id,),
                goal_nodes=(target_endpoint.id,),
                feasibility_check="webapp.build",
                success_check="webapp.build",
                meta={
                    "family": "webapp.build",
                    "difficulty": 0.4,
                    "target_path": target_endpoint.attrs.get("path", "/"),
                },
            )
        ]

    def check_feasibility(
        self,
        graph: WorldGraph,
        task: TaskSpec,
    ) -> FeasibilityVerdict:
        if not task.entrypoints or not task.goal_nodes:
            return FeasibilityVerdict(False, "missing entrypoint or goal")
        service_id = task.entrypoints[0]
        service = graph.nodes.get(service_id)
        if service is None or service.kind != "service":
            return FeasibilityVerdict(False, "entrypoint is not a service")
        endpoint_id = task.goal_nodes[0]
        endpoint = graph.nodes.get(endpoint_id)
        if endpoint is None or endpoint.kind != "endpoint":
            return FeasibilityVerdict(False, "goal is not an endpoint")
        for e in graph.out_edges(service_id, "exposes"):
            if e.dst == endpoint_id:
                return FeasibilityVerdict(True)
        return FeasibilityVerdict(
            False,
            "service does not expose the goal endpoint",
        )

    def check_success(
        self,
        graph: WorldGraph,
        task: TaskSpec,
        final_state: Mapping[str, Any],
    ) -> EpisodeResult:
        del graph, task
        ok = bool(final_state.get("endpoint_serves_200"))
        return EpisodeResult(
            success=ok,
            subgoals={
                k: bool(v) for k, v in final_state.items() if isinstance(v, bool)
            },
            reason=(
                "endpoint serves 200"
                if ok
                else "endpoint did not serve 200 to the smoke test"
            ),
        )

    def available_mutations(
        self,
        snapshot: Snapshot,
        reports: Sequence[EpisodeReportLike],
        *,
        llm: LLMBackendLike | None = None,
    ) -> tuple[Mutation, ...]:
        del llm
        from cyber_webapp.mutation import available_mutations as _enumerate

        return _enumerate(snapshot.graph, self.id, reports)

    def _pick_service(self, graph: WorldGraph) -> Node | None:
        services = graph.by_kind("service")
        if not services:
            return None
        priority = {"auth": 0, "web": 1, "api": 2}
        services_sorted = sorted(
            services,
            key=lambda s: priority.get(s.attrs.get("kind", ""), 99),
        )
        return services_sorted[0]

    def _pick_endpoint(
        self,
        graph: WorldGraph,
        service_id: str,
    ) -> Node | None:
        exposed: list[Node] = []
        for e in graph.out_edges(service_id, "exposes"):
            ep = graph.nodes.get(e.dst)
            if ep is None:
                continue
            exposed.append(ep)
        if not exposed:
            return None
        exposed.sort(key=lambda ep: 0 if ep.attrs.get("auth_required") else 1)
        return exposed[0]
