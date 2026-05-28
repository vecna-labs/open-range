"""``webapp.build`` — the agent writes a ``handle(query, state)`` into
``result.json``; a sandboxed grader scores it against a held-out contract.

Difficulty is a level on the target endpoint — for the ``api`` kind: L1 lists
records, L2 also returns their count, L3 also sorts them. ``available_mutations``
raises or lowers the level, which is how the curriculum hardens or softens the
task. Only ``api`` is wired; add kinds to ``_KIND_GENERATORS`` or inject your
own with ``WebappBuild(generators=...)``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from graphschema import GraphPatch, Node, WorldGraph
from openrange_pack_sdk import (
    EpisodeReportLike,
    EpisodeResult,
    FeasibilityVerdict,
    Manifest,
    Mutation,
    PackPrior,
    TaskFamily,
    TaskSpec,
)

from cyber_webapp.families.build.contracts import (
    API_MAX_LEVEL,
    ContractCase,
    api_list_contract,
)
from cyber_webapp.families.build.grading import grade_source
from cyber_webapp.families.build.mutations import api_wrong_field_name
from cyber_webapp.families.build.reference import api_list_reference

if TYPE_CHECKING:
    from openrange_pack_sdk import Snapshot


@dataclass(frozen=True)
class KindSpec:
    """Per-kind hooks: build the reference handler and contract for a level,
    plus bug-injectors that prove the contract rejects a broken handler."""

    reference: Callable[[int], str]
    contract: Callable[[int], tuple[ContractCase, ...]]
    admission_mutations: tuple[Callable[[str], str], ...]
    max_level: int


KindGenerators = Mapping[str, KindSpec]

_KIND_GENERATORS: KindGenerators = {
    "api": KindSpec(
        api_list_reference, api_list_contract, (api_wrong_field_name,), API_MAX_LEVEL
    ),
}


@dataclass(frozen=True)
class _Target:
    endpoint: Node
    service: Node
    kind: str
    spec: KindSpec
    level: int


def _endpoint_level(endpoint: Node, max_level: int) -> int:
    raw = endpoint.attrs.get("build_level", 1)
    level = raw if isinstance(raw, int) and not isinstance(raw, bool) else 1
    return max(1, min(level, max_level))


def _instruction(method: str, path: str, service: str, level: int) -> str:
    requirements = [
        "Respond with HTTP 200 and Content-Type application/json.",
        'Return a JSON object with an "items" list — one entry per record in '
        'state["records"], each carrying the record\'s id (as "id") and its fields.',
    ]
    if level >= 2:
        requirements.append('Include a top-level "count" of the number of items.')
    if level >= 3:
        requirements.append('Sort "items" by "id" ascending.')
    spec = "\n".join(f"- {line}" for line in requirements)
    return f"""Implement the {method} {path} handler for the {service} service.

    def handle(
        query: dict[str, str],
        state: dict[str, Any],
    ) -> tuple[int, dict[str, str], bytes]

state["records"] maps each record id to its field dict. Return (status,
headers, body) with body as bytes.

{spec}

Write your handler to result.json as
{{"endpoint_impl": "def handle(query, state): ..."}}. The episode ends when
result.json appears; it is graded against a held-out contract in a sandbox.
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
        instruction = _instruction(
            method=str(target.endpoint.attrs.get("method", "GET")),
            path=str(target.endpoint.attrs.get("path", "/")),
            service=str(target.service.attrs.get("name", target.service.id)),
            level=target.level,
        )
        return [
            self.make_task(
                instruction=instruction,
                entrypoints=target.service.id,
                goal_nodes=target.endpoint.id,
                difficulty=target.level / target.spec.max_level,
                meta={
                    "kind": target.kind,
                    "endpoint_path": str(target.endpoint.attrs.get("path", "/")),
                    "build_level": target.level,
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
        spec, level, kind = target.spec, target.level, target.kind
        cases = spec.contract(level)
        clean = grade_source(spec.reference(level), cases)
        if not clean.all_passed:
            return FeasibilityVerdict(
                False,
                f"reference impl for kind {kind!r} L{level} fails its own "
                f"contract: {clean.passed}/{clean.total} pass",
            )
        if not spec.admission_mutations:
            return FeasibilityVerdict(
                False,
                f"no admission mutations registered for kind {kind!r} — "
                "cannot validate contract distinguishes good from broken",
            )
        for index, mutation in enumerate(spec.admission_mutations):
            if grade_source(mutation(spec.reference(level)), cases).all_passed:
                return FeasibilityVerdict(
                    False,
                    f"mutation {index} for kind {kind!r} L{level} did not break "
                    "the contract — task would be trivially passable",
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
        report = grade_source(source, target.spec.contract(target.level))
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

    def available_mutations(
        self,
        snapshot: Snapshot,
        reports: Sequence[EpisodeReportLike],
        *,
        llm: object | None = None,
    ) -> tuple[Mutation, ...]:
        # Procedural, not LLM-scored: the pentest family's offense-flavored
        # enrichment has no signal for build and would zero these out. Harden
        # is the strong move when the agent passes; soften is a weak floor.
        del reports, llm
        target = self._pick_target(snapshot.graph)
        if target is None:
            return ()
        options: list[Mutation] = []
        if target.level < target.spec.max_level:
            options.append(
                self._level_mutation(target.endpoint, target.level + 1, "harden", 0.5)
            )
        if target.level > 1:
            options.append(
                self._level_mutation(target.endpoint, target.level - 1, "soften", 0.05)
            )
        return tuple(options)

    def _level_mutation(
        self,
        endpoint: Node,
        new_level: int,
        direction: str,
        relevance: float,
    ) -> Mutation:
        updated = replace(endpoint, attrs={**endpoint.attrs, "build_level": new_level})
        return self.make_mutation(
            direction=direction,
            relevance=relevance,
            patch=GraphPatch(nodes_updated=[updated]),
            note=f"build level {new_level} on {endpoint.id}",
        )

    def _pick_target(self, graph: WorldGraph) -> _Target | None:
        for service in graph.by_kind("service"):
            spec = self._generators.get(str(service.attrs.get("kind", "")))
            if spec is None:
                continue
            for edge in graph.out_edges(service.id, "exposes"):
                endpoint = graph.nodes.get(edge.dst)
                if endpoint is None or endpoint.kind != "endpoint":
                    continue
                kind = str(service.attrs.get("kind", ""))
                return _Target(
                    endpoint,
                    service,
                    kind,
                    spec,
                    _endpoint_level(endpoint, spec.max_level),
                )
        return None

    def _resolve_target(
        self,
        graph: WorldGraph,
        task: TaskSpec,
    ) -> _Target | FeasibilityVerdict:
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
        spec = self._generators.get(kind)
        if spec is None:
            return FeasibilityVerdict(
                False,
                f"no build contract for service kind {kind!r}",
            )
        return _Target(
            endpoint, service, kind, spec, _endpoint_level(endpoint, spec.max_level)
        )
