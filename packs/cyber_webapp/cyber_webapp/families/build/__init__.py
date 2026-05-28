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

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
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

ContractFn = Callable[[int], tuple[ContractCase, ...]]
ReferenceFn = Callable[[int], str]
MutationFn = Callable[[str], str]


@dataclass(frozen=True)
class KindSpec:
    """How to generate, grade, and harden the build task for one service kind.

    ``reference(level)`` / ``contract(level)`` produce the clean handler and
    the behavioral cases for a difficulty level in ``1..max_level``;
    ``admission_mutations`` are bug-injectors used only to prove the contract
    distinguishes a correct handler from a broken one.
    """

    reference: ReferenceFn
    contract: ContractFn
    admission_mutations: tuple[MutationFn, ...]
    max_level: int


KindGenerators = Mapping[str, KindSpec]


_KIND_GENERATORS: KindGenerators = {
    "api": KindSpec(
        reference=api_list_reference,
        contract=api_list_contract,
        admission_mutations=(api_wrong_field_name,),
        max_level=API_MAX_LEVEL,
    ),
}

# Curriculum relevance for build level mutations. Harden gets the mid-value
# (a passing agent is the signal to make the next level required); soften is
# a low floor so dropping a level is always an option when the agent stalls.
_HARDEN_RELEVANCE = 0.5
_SOFTEN_RELEVANCE = 0.05

_LEVEL_REQUIREMENTS = {
    2: '- Include a top-level "count" equal to the number of items.',
    3: '- Sort "items" by "id" in ascending order.',
}


@dataclass(frozen=True)
class _Target:
    """A resolved build target: the endpoint to implement, its service, the
    kind's spec, and the current difficulty level — resolved once so the
    generate / feasibility / success / mutation paths don't each re-derive it.
    """

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
    extra = "\n".join(_LEVEL_REQUIREMENTS[lvl] for lvl in range(2, level + 1))
    spec = (
        "- Respond with HTTP 200.\n"
        "- Set Content-Type to application/json.\n"
        '- Return a JSON object with a top-level field "items".\n'
        '- "items" is a list; one entry per record in state["records"].\n'
        "- Each entry includes the record's id (under \"id\") plus the record's "
        "fields."
    )
    if extra:
        spec = spec + "\n" + extra
    return f"""Implement the {method} {path} handler for the {service} service.

Handler signature:

    def handle(
        query: dict[str, str],
        state: dict[str, Any],
    ) -> tuple[int, dict[str, str], bytes]

The handler must return a 3-tuple (status, headers, body). body must be bytes.

Behavioral spec:
{spec}

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
        # Build difficulty is the contract level on the target endpoint;
        # harden raises it, soften lowers it. Procedural only — the
        # offense-flavored LLM the pentest family enriches with has no
        # signal for build and would zero these out.
        del reports, llm
        target = self._pick_target(snapshot.graph)
        if target is None:
            return ()
        options: list[Mutation] = []
        if target.level < target.spec.max_level:
            options.append(
                self._level_mutation(
                    target.endpoint, target.level + 1, "harden", _HARDEN_RELEVANCE
                )
            )
        if target.level > 1:
            options.append(
                self._level_mutation(
                    target.endpoint, target.level - 1, "soften", _SOFTEN_RELEVANCE
                )
            )
        return tuple(options)

    def _level_mutation(
        self,
        endpoint: Node,
        new_level: int,
        direction: str,
        relevance: float,
    ) -> Mutation:
        updated = Node(
            id=endpoint.id,
            kind=endpoint.kind,
            attrs={**dict(endpoint.attrs), "build_level": new_level},
            roles=set(endpoint.roles),
            visibility=endpoint.visibility,
            runtime=dict(endpoint.runtime),
            meta=dict(endpoint.meta),
        )
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
