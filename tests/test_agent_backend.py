"""AgentBackend protocol + StrandsAgentBackend / CodexAgentBackend tests."""

from __future__ import annotations

from typing import Any

import pytest
from openrange_pack_sdk import (
    AgentBackendError,
    LLMRequest,
    LLMResult,
)

from openrange.agent_backend import (
    CodexAgentBackend,
    StrandsAgentBackend,
)

# ---------------------------------------------------------------------------
# StrandsAgentBackend
# ---------------------------------------------------------------------------


class _NoStrandsBackend(StrandsAgentBackend):
    # Override the import seams to raise — no monkey-patching needed.

    def _probe_strands(self) -> None:
        raise ImportError("No module named 'strands'")

    def _import_agent_class(self) -> type:
        raise ImportError("No module named 'strands'")


def test_strands_backend_preflight_raises_when_strands_missing() -> None:
    with pytest.raises(AgentBackendError, match="strands-agents"):
        _NoStrandsBackend().preflight()


def test_strands_backend_build_agent_raises_when_strands_missing() -> None:
    with pytest.raises(AgentBackendError, match="strands-agents"):
        _NoStrandsBackend().build_agent(system_prompt="x", tools=())


def test_strands_backend_preflight_passes_when_strands_installed() -> None:
    """When strands IS importable, preflight is a no-op."""
    pytest.importorskip("strands")
    StrandsAgentBackend().preflight()


def test_strands_backend_builds_agent_when_installed() -> None:
    """End-to-end: construct a real strands.Agent (no API call)."""
    pytest.importorskip("strands")
    backend = StrandsAgentBackend()

    def my_tool(x: str) -> str:
        """Echo the input.

        Args:
            x: Anything.
        """
        return x

    agent = backend.build_agent(system_prompt="be terse", tools=[my_tool])
    # The returned object is callable (strands.Agent.__call__).
    assert callable(agent)


# ---------------------------------------------------------------------------
# CodexAgentBackend
# ---------------------------------------------------------------------------


class _RecordingLLMBackend:
    def __init__(self) -> None:
        self.requests: list[LLMRequest] = []
        self.canned = LLMResult("ok")
        self.preflight_calls = 0

    def complete(self, request: LLMRequest) -> LLMResult:
        self.requests.append(request)
        return self.canned

    def preflight(self) -> None:
        # Default protocol no-op; tests that want a failure raise from
        # a one-off override (see ``test_codex_backend_preflight_*``).
        self.preflight_calls += 1


def test_codex_backend_rejects_tools() -> None:
    """CodexAgentBackend errors loudly if handed any tools."""
    backend = CodexAgentBackend(backend=_RecordingLLMBackend())

    def some_tool() -> None:
        return None

    with pytest.raises(AgentBackendError, match="does not support tool injection"):
        backend.build_agent(system_prompt="x", tools=[some_tool])


def test_codex_backend_drives_llm_for_tool_less_prompts() -> None:
    """Without tools, build_agent returns a callable that hits the LLM backend."""
    fake = _RecordingLLMBackend()
    backend = CodexAgentBackend(backend=fake)
    session = backend.build_agent(system_prompt="be terse", tools=())
    result = session("hello")
    assert isinstance(result, LLMResult)
    assert result.text == "ok"
    assert len(fake.requests) == 1
    assert fake.requests[0].prompt == "hello"
    assert fake.requests[0].system == "be terse"


def test_codex_backend_rejects_both_backend_and_model_args() -> None:
    with pytest.raises(AgentBackendError, match="not both"):
        CodexAgentBackend(backend=_RecordingLLMBackend(), model="some-model")


def test_codex_backend_preflight_delegates_to_custom_llm_backend() -> None:
    """A caller-supplied LLMBackend gets its own preflight called."""
    fake = _RecordingLLMBackend()
    backend = CodexAgentBackend(backend=fake)
    backend.preflight()
    assert fake.preflight_calls == 1


def test_codex_backend_preflight_surfaces_custom_llm_backend_failures() -> None:
    """A failing custom backend preflight raises AgentBackendError."""
    from openrange_pack_sdk import LLMBackendError

    class _BadBackend(_RecordingLLMBackend):
        def preflight(self) -> None:
            raise LLMBackendError("custom probe failed")

    backend = CodexAgentBackend(backend=_BadBackend())
    with pytest.raises(AgentBackendError, match="custom probe failed"):
        backend.preflight()


def test_codex_backend_preflight_errors_if_codex_cli_missing(tmp_path: Any) -> None:
    # A real-but-absent path makes ``shutil.which`` return None without
    # monkey-patching the resolver.
    from openrange.llm import CodexBackend

    nonexistent = tmp_path / "codex_does_not_exist"
    backend = CodexAgentBackend(backend=CodexBackend(command=nonexistent))
    with pytest.raises(AgentBackendError, match="codex CLI not found"):
        backend.preflight()
