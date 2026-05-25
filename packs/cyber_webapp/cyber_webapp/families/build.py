"""WebappBuild — the `webapp.build` TaskFamily.

The agent's job: open the source repo, implement / repair a service so
that a feature endpoint serves correctly. The realizer's smoke test
against the modified code yields the success signal.

  Entrypoint  : a `service` node — the agent's working tree is the
                code for that service.
  Goal node   : an `endpoint` node owned by that service. The agent's
                edits must make this endpoint serve a valid request.
  Feasibility : entrypoint service exists; goal endpoint is exposed by
                the entrypoint service (otherwise the agent can't
                affect it from the repo's code).
  Success     : realizer's `final_state["endpoint_serves_200"]` is True
                after the agent's edits.

The build family complements the pentest family: same world graph,
DIFFERENT entrypoint kind (service for build, endpoint for pentest).
That's the load-bearing demo that "domain" lives on the TaskFamily.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from openrange.core.contracts import (
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
from openrange.world_ir import Node, WorldGraph

if TYPE_CHECKING:
    from openrange.core.admit_loop import Snapshot


class WebappBuild(TaskFamily):
    """Generate / verify build tasks against a webapp world."""

    id = "webapp.build"
    pack_id = "webapp"

    def generate(
        self,
        graph: WorldGraph,
        manifest: Manifest,
        prior: PackPrior | None,
    ) -> list[TaskSpec]:
        del manifest, prior
        # Pick the auth service if present (a common starting point); else
        # the first service. Pick its first auth-required endpoint as the
        # build target; else its first endpoint.
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
        """Verify the entrypoint service exists and exposes the goal
        endpoint (otherwise the agent can't plausibly affect the
        endpoint from the service's code)."""
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
        # walk service -exposes-> endpoint
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
        """Read the realizer's collected smoke-test result.

        v1 reads a single bool key `endpoint_serves_200`. v2 may inspect
        HTTP status, body content, and unit-test outputs. v1 surfaces
        whatever the realizer collected as bool subgoals.
        """
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
        """Delegate to the pack's procedural mutation enumerator.

        The cyber pack ships a single `available_mutations(graph,
        family_id, reports)` in `cyber_webapp.mutation` that enumerates
        candidates for any family. We tag each Mutation with our own
        `family` so curriculum aggregation can route patches back to
        the right family.

        LLM enrichment is currently a no-op for this family — the
        build-side mutations are deterministic enough that LLM re-scoring
        adds little. The `llm` parameter is accepted for protocol
        conformance and reserved for v2 (LLM-driven novel mutation
        proposals beyond the procedural floor).
        """
        del llm
        from cyber_webapp.mutation import available_mutations as _enumerate

        return _enumerate(snapshot.graph, self.id, reports)

    # ----- helpers -------------------------------------------------------

    def _pick_service(self, graph: WorldGraph) -> Node | None:
        services = graph.by_kind("service")
        if not services:
            return None
        # Prefer auth services, then web, then anything else.
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
        # Prefer auth-required endpoints (more interesting build target).
        exposed.sort(key=lambda ep: 0 if ep.attrs.get("auth_required") else 1)
        return exposed[0]
