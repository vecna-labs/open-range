"""The model-server seam: an async ``InferBackend`` + an OpenAI-compatible client.

The engine never generates in-process; it talks to a configurable OpenAI-style HTTP
endpoint (vLLM, TGI, a hosted API, …). The request/response *shaping* lives in
``protocol`` (pure, unit-covered); ``OpenAIBackend.generate`` is only the thin round
trip, so its I/O is exercised by the ``OPENRANGE_LIVE_MODEL_URL``-gated live test.
"""

from __future__ import annotations

import importlib
from typing import Any, Protocol

from openrange_trl.engine.protocol import (
    Action,
    build_chat_request,
    parse_chat_response,
)


class InferBackend(Protocol):
    """A policy that maps the running message log + tool schemas to the next action."""

    async def generate(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> Action: ...


class OpenAIBackend:
    """Generate via an OpenAI-style ``/v1/chat/completions`` endpoint.

    ``client`` is injectable (the live test passes a configured one); when omitted, a
    default async ``httpx`` client is created — ``httpx`` is imported lazily here so
    ``import openrange_trl.engine`` stays HTTP-free and needs only the ``engine`` extra
    when a backend is actually constructed.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        api_key: str | None = None,
        client: Any = None,
    ) -> None:
        self._model = model
        self._url = base_url.rstrip("/") + "/v1/chat/completions"
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        if client is None:  # pragma: no cover - real use only; tests inject a client
            httpx = importlib.import_module("httpx")
            client = httpx.AsyncClient(timeout=120.0)
        self._client = client

    async def generate(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> Action:
        # Live I/O — this whole body is covered only by the gated live-model test.
        request = build_chat_request(self._model, messages, tools)  # pragma: no cover
        response = await self._client.post(  # pragma: no cover
            self._url, json=request, headers=self._headers
        )
        response.raise_for_status()  # pragma: no cover
        payload: dict[str, Any] = response.json()  # pragma: no cover
        return parse_chat_response(payload)  # pragma: no cover

    async def aclose(self) -> None:  # pragma: no cover - live I/O cleanup
        await self._client.aclose()
