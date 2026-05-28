"""Concrete agent backends. `StrandsAgentBackend` + `CodexAgentBackend` ship.

The ``AgentBackend`` Protocol and ``AgentBackendError`` live in
``openrange_pack_sdk``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from openrange_pack_sdk import (
    AgentBackendError,
    AgentSession,
    LLMBackend,
    LLMBackendError,
    LLMRequest,
)

from openrange.llm import CodexBackend


class StrandsAgentBackend:
    """Wraps `strands.Agent`. Lazy-imports the optional SDK.

    `_probe_strands` and `_import_agent_class` are extension points: tests
    subclass and override them to assert the missing-optional-dependency
    error path without monkey-patching ``builtins.__import__``.
    """

    def __init__(self, *, model: str | None = None) -> None:
        self._model = model

    def preflight(self) -> None:
        try:
            self._probe_strands()
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
            agent_cls = self._import_agent_class()
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
        agent: AgentSession = agent_cls(**kwargs)
        return agent

    def _probe_strands(self) -> None:
        import strands  # noqa: F401

    def _import_agent_class(self) -> Any:
        from strands import Agent

        return Agent


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
            backend if backend is not None else CodexBackend(model=model)
        )

    def preflight(self) -> None:
        # LLMBackend Protocol doesn't require preflight, but concrete impls
        # often have one. Call it only when present so minimal fakes pay
        # nothing.
        backend_preflight = getattr(self._backend, "preflight", None)
        if not callable(backend_preflight):
            return
        try:
            backend_preflight()
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
