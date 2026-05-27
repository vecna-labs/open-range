"""Tests for ``generate_task_instruction`` and ``enrich_mutations``."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence

import pytest
from cyber_webapp.llm_generation import (
    LLMGenerationError,
    enrich_mutations,
    generate_task_instruction,
)
from graphschema import (
    Edge,
    GraphPatch,
    Node,
    Visibility,
    WorldGraph,
)

from openrange.core.pack import EpisodeReportLike, Mutation
from openrange.llm import LLMBackend, LLMError, LLMRequest, LLMResult


class _FakeLLMBackend:
    """Scripted LLMBackend.

    Either ``response`` (a static dict the backend always returns as
    `parsed_json`) or ``responder`` (a callable receiving the parsed
    prompt and returning a dict) is supplied.

    Implements the `LLMBackend` Protocol — `complete(request)` returns
    an `LLMResult`. When ``raise_on_call`` is set, raises that exception
    instead so we can exercise the LLM-error fallback path in
    `enrich_mutations`.
    """

    def __init__(
        self,
        response: Mapping[str, object] | None = None,
        *,
        responder: (object | None) = None,
        raise_on_call: Exception | None = None,
    ) -> None:
        self._response = response
        self._responder = responder
        self._raise = raise_on_call
        self.calls: list[LLMRequest] = []

    def preflight(self) -> None:
        return None

    def complete(self, request: LLMRequest) -> LLMResult:
        self.calls.append(request)
        if self._raise is not None:
            raise self._raise
        if self._responder is not None:
            prompt = json.loads(request.prompt)
            assert callable(self._responder)
            data = self._responder(prompt)
        else:
            assert self._response is not None
            data = dict(self._response)
        return LLMResult(text=json.dumps(data), parsed_json=data)


def _trivial_graph() -> WorldGraph:
    """A minimal graph wired with a service, two endpoints, a vuln, a flag.

    `_summarize_graph` in llm_generation.py reads exactly these kinds
    (service, endpoint, vulnerability) and the `exposes` / `affects`
    edges between them. Everything else here is filler so the graph
    holds together.
    """
    g = WorldGraph(ontology="cyber.webapp@v1")
    g.add_node(
        Node(
            "svc_web",
            "service",
            attrs={
                "name": "web",
                "kind": "web",
                "language": "python",
                "exposure": "public",
            },
        )
    )
    g.add_node(
        Node(
            "ep_search",
            "endpoint",
            attrs={
                "path": "/search",
                "method": "GET",
                "auth_required": False,
                "behavior_ref": "web/search",
            },
        )
    )
    g.add_node(
        Node(
            "ep_login",
            "endpoint",
            attrs={
                "path": "/login",
                "method": "POST",
                "auth_required": False,
                "behavior_ref": "web/login",
            },
        )
    )
    g.add_node(
        Node(
            "vuln_sqli",
            "vulnerability",
            attrs={
                "kind": "sql_injection",
                "family": "code_web",
                "params": {"target_param": "q"},
            },
            visibility=Visibility.HIDDEN,
        )
    )
    g.add_node(
        Node(
            "secret_flag",
            "secret",
            attrs={
                "kind": "flag",
                "value_ref": "FLAG{x}",
                "description": "admin flag",
            },
            visibility=Visibility.HIDDEN,
        )
    )
    g.add_edge(Edge("e.svc-search", "exposes", "svc_web", "ep_search"))
    g.add_edge(Edge("e.svc-login", "exposes", "svc_web", "ep_login"))
    g.add_edge(Edge("e.vuln-search", "affects", "vuln_sqli", "ep_search"))
    return g


def test_generate_task_instruction_returns_llm_text() -> None:
    """The LLM-returned `instruction` flows back verbatim (whitespace-stripped)."""
    backend = _FakeLLMBackend(
        response={
            "instruction": (
                "Retrieve the admin flag from the running webapp and write "
                "it to result.json. Read OPENRANGE_TASK.json for base_url."
            ),
        },
    )
    instruction = generate_task_instruction(
        _trivial_graph(), "webapp.pentest::admin_flag", backend
    )
    assert instruction.startswith("Retrieve")
    assert "result.json" in instruction
    assert len(backend.calls) == 1


def test_generate_task_instruction_sees_graph_summary_in_prompt() -> None:
    """The prompt JSON carries the world summary the LLM is supposed to read."""
    captured: dict[str, object] = {}

    def _responder(prompt: Mapping[str, object]) -> Mapping[str, object]:
        captured.update(prompt)
        return {"instruction": "do the thing"}

    backend = _FakeLLMBackend(responder=_responder)
    _ = generate_task_instruction(_trivial_graph(), "webapp.pentest::t0", backend)
    assert captured["task_id"] == "webapp.pentest::t0"
    world = captured["world"]
    assert isinstance(world, dict)
    service_kinds = {s["kind"] for s in world["services"]}
    assert service_kinds == {"web"}
    endpoint_paths = {ep["path"] for ep in world["endpoints"]}
    assert endpoint_paths == {"/search", "/login"}
    vuln_kinds = {v["kind"] for v in world["vulnerabilities"]}
    assert vuln_kinds == {"sql_injection"}


def test_generate_task_instruction_rejects_empty_response() -> None:
    """An empty / whitespace-only instruction is not usable; raise."""
    backend = _FakeLLMBackend(response={"instruction": "   "})
    with pytest.raises(LLMGenerationError, match="usable instruction"):
        generate_task_instruction(_trivial_graph(), "t", backend)


def test_generate_task_instruction_propagates_llm_errors() -> None:
    """LLMError from the backend becomes LLMGenerationError up-stack."""
    backend = _FakeLLMBackend(raise_on_call=LLMError("backend exploded"))
    with pytest.raises(LLMGenerationError, match="LLM call failed"):
        generate_task_instruction(_trivial_graph(), "t", backend)


def test_generate_task_instruction_passes_json_schema_to_backend() -> None:
    """The backend should be called with a JSON-schema-constrained request
    so the model is forced to return the expected `instruction` field."""
    backend = _FakeLLMBackend(response={"instruction": "x"})
    _ = generate_task_instruction(_trivial_graph(), "t", backend)
    assert len(backend.calls) == 1
    req = backend.calls[0]
    assert req.json_schema is not None
    required = req.json_schema.get("required") or []
    assert isinstance(required, list)
    assert "instruction" in required


def _mutation(
    *,
    family: str = "webapp.pentest",
    direction: str = "harden",
    relevance: float = 0.5,
    note: str = "base note",
    nodes_added: tuple[Node, ...] = (),
    nodes_removed: tuple[str, ...] = (),
) -> Mutation:
    """Construct a Mutation with a fresh GraphPatch for the test."""
    patch = GraphPatch(
        nodes_added=list(nodes_added),
        nodes_removed=list(nodes_removed),
    )
    return Mutation(
        patch=patch,
        direction=direction,
        relevance=relevance,
        family=family,
        note=note,
    )


def test_enrich_mutations_passthrough_when_no_llm() -> None:
    """No LLM -> return the procedural list unchanged."""
    options = (_mutation(), _mutation(direction="soften", relevance=0.3))
    out = enrich_mutations(options, graph=_trivial_graph(), reports=[], llm=None)
    assert out is options


def test_enrich_mutations_passthrough_when_no_options() -> None:
    """Empty input -> empty output, no LLM call."""
    backend = _FakeLLMBackend(response={"mutations": []})
    out = enrich_mutations((), graph=_trivial_graph(), reports=[], llm=backend)
    assert out == ()
    assert backend.calls == []


def test_enrich_mutations_updates_relevance_and_note() -> None:
    """LLM-refined relevance and note replace the procedural values."""
    options = (
        _mutation(relevance=0.5, note="proc note 0"),
        _mutation(relevance=0.5, note="proc note 1", direction="soften"),
    )
    backend = _FakeLLMBackend(
        response={
            "mutations": [
                {"index": 0, "relevance": 0.95, "note": "LLM-scored: hot path"},
                {"index": 1, "relevance": 0.1, "note": "LLM-scored: cold path"},
            ],
        },
    )
    enriched = enrich_mutations(
        options,
        graph=_trivial_graph(),
        reports=[],
        llm=backend,
    )
    assert len(enriched) == 2
    assert enriched[0].relevance == pytest.approx(0.95)
    assert enriched[0].note == "LLM-scored: hot path"
    assert enriched[1].relevance == pytest.approx(0.1)
    assert enriched[1].note == "LLM-scored: cold path"
    # Direction, family, and patch are preserved (LLM cannot rewrite them).
    assert enriched[0].direction == "harden"
    assert enriched[1].direction == "soften"
    assert enriched[0].patch is options[0].patch
    assert enriched[1].patch is options[1].patch
    assert enriched[0].family == options[0].family


def test_enrich_mutations_clamps_relevance_to_unit_interval() -> None:
    """Out-of-range relevance values from the LLM are clamped to [0, 1]."""
    options = (_mutation(relevance=0.5), _mutation(relevance=0.5))
    backend = _FakeLLMBackend(
        response={
            "mutations": [
                {"index": 0, "relevance": 99.0, "note": "way too high"},
                {"index": 1, "relevance": -3.0, "note": "way too low"},
            ],
        },
    )
    enriched = enrich_mutations(
        options,
        graph=_trivial_graph(),
        reports=[],
        llm=backend,
    )
    assert enriched[0].relevance == pytest.approx(1.0)
    assert enriched[1].relevance == pytest.approx(0.0)


def test_enrich_mutations_preserves_base_on_missing_entry() -> None:
    """A mutation the LLM did not score keeps its procedural relevance/note."""
    options = (
        _mutation(relevance=0.42, note="kept-proc"),
        _mutation(relevance=0.5, note="replace-me"),
    )
    backend = _FakeLLMBackend(
        response={
            "mutations": [
                # only index 1 returned
                {"index": 1, "relevance": 0.7, "note": "LLM-updated"},
            ],
        },
    )
    enriched = enrich_mutations(
        options,
        graph=_trivial_graph(),
        reports=[],
        llm=backend,
    )
    assert enriched[0].relevance == pytest.approx(0.42)
    assert enriched[0].note == "kept-proc"
    assert enriched[1].relevance == pytest.approx(0.7)
    assert enriched[1].note == "LLM-updated"


def test_enrich_mutations_falls_back_on_llm_error() -> None:
    """LLM call failure -> return the procedural list verbatim."""
    options = (_mutation(relevance=0.5, note="kept"),)
    backend = _FakeLLMBackend(raise_on_call=LLMError("network down"))
    enriched = enrich_mutations(
        options,
        graph=_trivial_graph(),
        reports=[],
        llm=backend,
    )
    assert enriched == options


def test_enrich_mutations_falls_back_on_malformed_payload() -> None:
    """LLM returns the wrong shape -> return the procedural list verbatim."""
    options = (_mutation(relevance=0.5, note="kept"),)
    backend = _FakeLLMBackend(response={"mutations": "this is not a list"})
    enriched = enrich_mutations(
        options,
        graph=_trivial_graph(),
        reports=[],
        llm=backend,
    )
    assert enriched == options


def test_enrich_mutations_skips_non_string_notes() -> None:
    """If the LLM returns a non-string `note`, keep the procedural one."""
    options = (_mutation(relevance=0.5, note="proc-note"),)
    backend = _FakeLLMBackend(
        response={
            "mutations": [{"index": 0, "relevance": 0.9, "note": 12345}],
        },
    )
    enriched = enrich_mutations(
        options,
        graph=_trivial_graph(),
        reports=[],
        llm=backend,
    )
    # relevance still flows through; note is rejected and procedural note kept.
    assert enriched[0].relevance == pytest.approx(0.9)
    assert enriched[0].note == "proc-note"


def test_enrich_mutations_reads_request_log_from_reports() -> None:
    """The LLM prompt includes a `requests` section sourced from each
    report's `final_state["requests"]` rows."""
    captured: dict[str, object] = {}

    def _responder(prompt: Mapping[str, object]) -> Mapping[str, object]:
        captured.update(prompt)
        return {"mutations": []}

    class _Report:
        passed = False
        final_state: Mapping[str, object] = {
            "requests": [
                {
                    "method": "GET",
                    "path": "/search",
                    "status": 200,
                    "query": "q=' OR 1=1--",
                },
                {
                    "method": "POST",
                    "path": "/login",
                    "status": 401,
                },
            ],
        }

    reports: Sequence[EpisodeReportLike] = (_Report(),)
    backend = _FakeLLMBackend(responder=_responder)
    options = (_mutation(),)
    _ = enrich_mutations(
        options,
        graph=_trivial_graph(),
        reports=reports,
        llm=backend,
    )
    rows = captured.get("requests")
    assert isinstance(rows, list)
    assert any(
        row.get("path") == "/search" and row.get("query") == "q=' OR 1=1--"
        for row in rows
    )


# Quiet linters about unused symbols re-exported in case future tests grow.
_ = LLMBackend
