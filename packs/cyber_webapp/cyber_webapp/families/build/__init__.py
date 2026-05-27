"""``webapp.build`` TaskFamily — agent implements a service handler from spec.

The agent reads the task instruction (handler signature + behavioral spec +
sample state shape) and writes a Python source string for ``def handle(query,
state)`` into ``result.json`` under key ``endpoint_impl``. ``check_success``
runs the submitted source against a held-out behavioral contract in a
sandboxed subprocess and grades per-case.

At admission, ``check_feasibility`` also runs the contract against the kind's
reference impl (must pass) and against each registered mutation of the
reference (each must break at least one case), so an ill-posed task —
too-weak or contradictory contract — is rejected before an agent is asked
to solve it.

Only the ``api`` service kind is wired today. Adding a kind is a contract +
reference + mutations entry in ``_KIND_GENERATORS``. To use a different
generator set (custom contracts, test fixtures), construct
``WebappBuild(generators={...})``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from graphschema import Node, WorldGraph
from openrange_pack_sdk import (
    EpisodeResult,
    FeasibilityVerdict,
    Manifest,
    PackPrior,
    TaskFamily,
    TaskSpec,
)

from cyber_webapp.families.build.contracts import ContractCase, api_list_contract
from cyber_webapp.families.build.grading import grade_source
from cyber_webapp.families.build.mutations import api_wrong_field_name
from cyber_webapp.families.build.reference import api_list_reference

ContractFn = Callable[[], tuple[ContractCase, ...]]
ReferenceFn = Callable[[], str]
MutationFn = Callable[[str], str]
KindGenerators = Mapping[str, tuple[ReferenceFn, ContractFn, tuple[MutationFn, ...]]]


_KIND_GENERATORS: KindGenerators = {
    "api": (api_list_reference, api_list_contract, (api_wrong_field_name,)),
}


_INSTRUCTION_TEMPLATE = """Implement the {method} {path} handler for the {service} \
service.

Handler signature:

    def handle(
        query: dict[str, str],
        state: dict[str, Any],
    ) -> tuple[int, dict[str, str], bytes]

The handler must return a 3-tuple (status, headers, body). body must be bytes.

Behavioral spec:
- Respond with HTTP 200.
- Set Content-Type to application/json.
- Return a JSON object with a top-level field "items".
- "items" is a list; one entry per record in state["records"].
- Each entry includes the record's id (under "id") plus the record's fields.

The state shape your handler will be called with:
    state["records"]: dict[str, dict[str, Any]] mapping record id to a field dict.

Submit your implementation by writing to result.json in your workspace:
    {{"endpoint_impl": "def handle(query, state):\\n    ..."}}

The episode terminates when result.json appears. Your submission is graded
against a held-out behavioral test contract in a sandboxed subprocess.
"""


class WebappBuild(TaskFamily):
    """Agent implements a service handler from spec; grader runs a held-out
    behavioral contract against the submission."""

    id = "webapp.build"
    pack_id = "webapp"

    def __init__(self, *, generators: KindGenerators | None = None) -> None:
        self._generators: KindGenerators = (
            dict(_KIND_GENERATORS) if generators is None else dict(generators)
        )

    def generate(
        self,
        graph: WorldGraph,
        manifest: Manifest,
        prior: PackPrior | None,
    ) -> list[TaskSpec]:
        del manifest, prior
        target = self._pick_target(graph)
        if target is None:
            return []
        endpoint, service, kind = target
        instruction = _INSTRUCTION_TEMPLATE.format(
            method=str(endpoint.attrs.get("method", "GET")),
            path=str(endpoint.attrs.get("path", "/")),
            service=str(service.attrs.get("name", service.id)),
        )
        return [
            self.make_task(
                instruction=instruction,
                entrypoints=service.id,
                goal_nodes=endpoint.id,
                difficulty=0.4,
                meta={
                    "kind": kind,
                    "endpoint_path": str(endpoint.attrs.get("path", "/")),
                },
            ),
        ]

    def check_feasibility(
        self,
        graph: WorldGraph,
        task: TaskSpec,
    ) -> FeasibilityVerdict:
        target = self._resolve_target(graph, task)
        if isinstance(target, FeasibilityVerdict):
            return target
        kind = target[2]
        reference, contract, mutations = self._generators[kind]
        cases = contract()
        clean = grade_source(reference(), cases)
        if not clean.all_passed:
            return FeasibilityVerdict(
                False,
                f"reference impl for kind {kind!r} fails its own contract: "
                f"{clean.passed}/{clean.total} pass",
            )
        if not mutations:
            return FeasibilityVerdict(
                False,
                f"no admission mutations registered for kind {kind!r} — "
                "cannot validate contract distinguishes good from broken",
            )
        for index, mutation in enumerate(mutations):
            mutated = grade_source(mutation(reference()), cases)
            if mutated.all_passed:
                return FeasibilityVerdict(
                    False,
                    f"mutation {index} for kind {kind!r} did not break the "
                    "contract — task would be trivially passable",
                )
        return FeasibilityVerdict(True)

    def check_success(
        self,
        graph: WorldGraph,
        task: TaskSpec,
        final_state: Mapping[str, Any],
    ) -> EpisodeResult:
        target = self._resolve_target(graph, task)
        if isinstance(target, FeasibilityVerdict):
            return EpisodeResult(
                success=False,
                reason=f"task target unresolvable: {target.reason}",
            )
        kind = target[2]
        result = final_state.get("result")
        if not isinstance(result, Mapping):
            return EpisodeResult(
                success=False,
                reason="agent did not write result.json",
            )
        source = result.get("endpoint_impl")
        if not isinstance(source, str) or not source.strip():
            return EpisodeResult(
                success=False,
                reason="result.json missing non-empty 'endpoint_impl' string",
            )
        _, contract, _ = self._generators[kind]
        report = grade_source(source, contract())
        subgoals = {case.description: case.passed for case in report.cases}
        return EpisodeResult(
            success=report.all_passed,
            subgoals=subgoals,
            reason=(
                "all contract cases pass"
                if report.all_passed
                else f"{report.passed}/{report.total} contract cases pass"
            ),
        )

    def _pick_target(
        self,
        graph: WorldGraph,
    ) -> tuple[Node, Node, str] | None:
        for service in graph.by_kind("service"):
            kind = str(service.attrs.get("kind", ""))
            if kind not in self._generators:
                continue
            for edge in graph.out_edges(service.id, "exposes"):
                endpoint = graph.nodes.get(edge.dst)
                if endpoint is None or endpoint.kind != "endpoint":
                    continue
                return endpoint, service, kind
        return None

    def _resolve_target(
        self,
        graph: WorldGraph,
        task: TaskSpec,
    ) -> tuple[Node, Node, str] | FeasibilityVerdict:
        if not task.entrypoints or not task.goal_nodes:
            return FeasibilityVerdict(False, "missing entrypoint or goal")
        service = graph.nodes.get(task.entrypoints[0])
        if service is None or service.kind != "service":
            return FeasibilityVerdict(False, "entrypoint is not a service")
        endpoint = graph.nodes.get(task.goal_nodes[0])
        if endpoint is None or endpoint.kind != "endpoint":
            return FeasibilityVerdict(False, "goal is not an endpoint")
        if not any(
            edge.dst == endpoint.id for edge in graph.out_edges(service.id, "exposes")
        ):
            return FeasibilityVerdict(
                False,
                "service does not expose the goal endpoint",
            )
        kind = str(service.attrs.get("kind", ""))
        if kind not in self._generators:
            return FeasibilityVerdict(
                False,
                f"no build contract for service kind {kind!r}",
            )
        return endpoint, service, kind
