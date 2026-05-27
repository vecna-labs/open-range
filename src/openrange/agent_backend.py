"""Agent backend protocol. `StrandsAgentBackend` + `CodexAgentBackend` ship."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Protocol

from openrange.core.errors import OpenRangeError
from openrange.llm import (
    CODEX_DEFAULT_MODEL,
    CodexBackend,
    LLMBackend,
    LLMBackendError,
    LLMRequest,
)

AgentSession = Callable[[str], Any]


class AgentBackendError(OpenRangeError):
    pass


class AgentBackend(Protocol):
    """Factory for agent sessions. `preflight` validates dependencies;
    `build_agent` returns a callable session. Backends without tool
    dispatch must raise on non-empty `tools`."""

    def preflight(self) -> None: ...

    def build_agent(
        self,
        *,
        system_prompt: str,
        tools: Sequence[Callable[..., Any]] = (),
    ) -> AgentSession: ...


class StrandsAgentBackend:
    """Wraps `strands.Agent`. Lazy-imports the optional SDK."""

    def __init__(self, *, model: str | None = None) -> None:
        self._model = model

    def preflight(self) -> None:
        try:
            import strands  # noqa: F401
        except ImportError as exc:
            raise AgentBackendError(
                "StrandsAgentBackend requires the optional 'strands-agents' "
                "package. Install with `pip install openrange[strands]`.",
            ) from exc

    def build_agent(
        self,
        *,
        system_prompt: str,
        tools: Sequence[Callable[..., Any]] = (),
    ) -> AgentSession:
        try:
            from strands import Agent
        except ImportError as exc:
            raise AgentBackendError(
                "StrandsAgentBackend requires the optional 'strands-agents' "
                "package. Install with `pip install openrange[strands]`.",
            ) from exc
        kwargs: dict[str, Any] = {
            "tools": list(tools),
            "system_prompt": system_prompt,
            "callback_handler": None,
        }
        if self._model is not None:
            kwargs["model"] = self._model
        agent: AgentSession = Agent(**kwargs)
        return agent


class CodexAgentBackend:
    """Wraps an `LLMBackend` (Codex CLI) for single-shot, tool-less agents.
    Raises on non-empty `tools`."""

    def __init__(
        self,
        *,
        backend: LLMBackend | None = None,
        model: str | None = None,
    ) -> None:
        if backend is not None and model is not None:
            raise AgentBackendError(
                "CodexAgentBackend: pass either 'backend' or 'model', not both",
            )
        self._backend: LLMBackend = (
            backend
            if backend is not None
            else CodexBackend(
                model=model if model is not None else CODEX_DEFAULT_MODEL,
            )
        )

    def preflight(self) -> None:
        # Delegate to the wrapped LLMBackend's own preflight — every
        # LLMBackend declares one (default no-op), so custom backends
        # can self-describe their checks instead of getting silently
        # skipped here.
        try:
            self._backend.preflight()
        except LLMBackendError as exc:
            raise AgentBackendError(
                f"CodexAgentBackend: backend preflight failed: {exc}",
            ) from exc

    def build_agent(
        self,
        *,
        system_prompt: str,
        tools: Sequence[Callable[..., Any]] = (),
    ) -> AgentSession:
        if tools:
            raise AgentBackendError(
                "CodexAgentBackend does not support tool injection. "
                "Use StrandsAgentBackend for NPCs that need tool dispatch.",
            )
        backend = self._backend

        def session(prompt: str) -> Any:
            return backend.complete(LLMRequest(prompt=prompt, system=system_prompt))

        return session
