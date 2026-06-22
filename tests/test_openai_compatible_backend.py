from __future__ import annotations

import contextlib
import json
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, cast

import pytest
from openrange_pack_sdk import LLMBackendError, LLMRequest, LLMResult

import openrange as OR

ServerResponse = tuple[int, Mapping[str, object] | str, float]
RequestHandler = Callable[
    [Mapping[str, object], Mapping[str, str], str],
    ServerResponse,
]


@contextlib.contextmanager
def running_openai_server(
    handler: RequestHandler,
) -> Iterator[tuple[str, list[dict[str, object]]]]:
    requests: list[dict[str, object]] = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, _format: str, *_args: object) -> None:
            pass

        def do_POST(self) -> None:
            content_length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(content_length).decode("utf-8")
            payload = cast(dict[str, object], json.loads(body))
            headers = dict(self.headers.items())
            requests.append(
                {
                    "path": self.path,
                    "headers": headers,
                    "payload": payload,
                },
            )

            status, response, delay = handler(payload, headers, self.path)
            if delay:
                time.sleep(delay)
            response_bytes = (
                response.encode("utf-8")
                if isinstance(response, str)
                else json.dumps(response).encode("utf-8")
            )
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response_bytes)))
            self.end_headers()
            with contextlib.suppress(BrokenPipeError):
                self.wfile.write(response_bytes)

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host = cast(str, server.server_address[0])
        port = server.server_address[1]
        yield f"http://{host}:{port}", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_openai_compatible_backend_sends_plain_chat_completion() -> None:
    def handler(
        _payload: Mapping[str, object],
        _headers: Mapping[str, str],
        _path: str,
    ) -> ServerResponse:
        return 200, {"choices": [{"message": {"content": "plain answer"}}]}, 0

    with running_openai_server(handler) as (base_url, requests):
        result = OR.OpenAICompatibleBackend(
            base_url=f"{base_url}/v1",
            api_key="test-key",
            model="demo-model",
            extra_headers={"X-Test": "yes"},
        ).complete(LLMRequest("build a world", system="be precise"))

    assert result == LLMResult("plain answer")
    request = requests[0]
    payload = cast(dict[str, Any], request["payload"])
    assert request["path"] == "/v1/chat/completions"
    assert payload["model"] == "demo-model"
    assert payload["messages"] == [
        {"role": "system", "content": "be precise"},
        {"role": "user", "content": "build a world"},
    ]
    assert "response_format" not in payload
    headers = cast(dict[str, str], request["headers"])
    assert headers["Authorization"] == "Bearer test-key"
    assert headers["X-Test"] == "yes"


def test_openai_compatible_backend_requests_structured_json() -> None:
    schema = {
        "type": "object",
        "properties": {"ok": {"type": "boolean"}},
        "required": ["ok"],
    }

    def handler(
        payload: Mapping[str, object],
        _headers: Mapping[str, str],
        _path: str,
    ) -> ServerResponse:
        response_format = cast(dict[str, Any], payload["response_format"])
        assert response_format["type"] == "json_schema"
        assert response_format["json_schema"]["schema"] == schema
        assert response_format["json_schema"]["strict"] is False
        return 200, {"choices": [{"message": {"content": '{"ok": true}'}}]}, 0

    with running_openai_server(handler) as (base_url, _requests):
        result = OR.OpenAICompatibleBackend(
            base_url=f"{base_url}/v1",
            api_key="test-key",
            model="demo-model",
        ).complete(LLMRequest("return json", json_schema=schema))

    assert result.text == '{"ok": true}'
    assert result.parsed_json == {"ok": True}


def test_openai_compatible_backend_can_request_strict_json_schema() -> None:
    schema = {
        "type": "object",
        "properties": {"ok": {"type": "boolean"}},
        "required": ["ok"],
    }

    def handler(
        payload: Mapping[str, object],
        _headers: Mapping[str, str],
        _path: str,
    ) -> ServerResponse:
        response_format = cast(dict[str, Any], payload["response_format"])
        assert response_format["json_schema"]["strict"] is True
        return 200, {"choices": [{"message": {"content": '{"ok": true}'}}]}, 0

    with running_openai_server(handler) as (base_url, _requests):
        result = OR.OpenAICompatibleBackend(
            base_url=f"{base_url}/v1",
            api_key="test-key",
            model="demo-model",
            json_schema_strict=True,
        ).complete(LLMRequest("return json", json_schema=schema))

    assert result.parsed_json == {"ok": True}


def test_openai_compatible_backend_reads_api_key_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENRANGE_TEST_API_KEY", "env-key")

    def handler(
        _payload: Mapping[str, object],
        headers: Mapping[str, str],
        _path: str,
    ) -> ServerResponse:
        assert headers["Authorization"] == "Bearer env-key"
        return 200, {"choices": [{"message": {"content": "plain answer"}}]}, 0

    with running_openai_server(handler) as (base_url, _requests):
        result = OR.OpenAICompatibleBackend(
            base_url=f"{base_url}/v1",
            api_key_env="OPENRANGE_TEST_API_KEY",
            model="demo-model",
        ).complete(LLMRequest("hello"))

    assert result == LLMResult("plain answer")


def test_openai_compatible_backend_rejects_non_http_base_url() -> None:
    backend = OR.OpenAICompatibleBackend(
        base_url="file:///tmp/openai",
        api_key="test-key",
        model="demo-model",
    )

    with pytest.raises(LLMBackendError, match="base_url must use http or https"):
        backend.preflight()


def test_openai_compatible_backend_maps_http_errors() -> None:
    def handler(
        _payload: Mapping[str, object],
        _headers: Mapping[str, str],
        _path: str,
    ) -> ServerResponse:
        return 401, {"error": {"message": "bad api key"}}, 0

    with running_openai_server(handler) as (base_url, _requests):
        backend = OR.OpenAICompatibleBackend(
            base_url=f"{base_url}/v1",
            api_key="test-key",
            model="demo-model",
        )
        with pytest.raises(LLMBackendError, match="HTTP 401: bad api key"):
            backend.complete(LLMRequest("hello"))


def test_openai_compatible_backend_maps_timeouts() -> None:
    def handler(
        _payload: Mapping[str, object],
        _headers: Mapping[str, str],
        _path: str,
    ) -> ServerResponse:
        return 200, {"choices": [{"message": {"content": "late"}}]}, 0.2

    with running_openai_server(handler) as (base_url, _requests):
        backend = OR.OpenAICompatibleBackend(
            base_url=f"{base_url}/v1",
            api_key="test-key",
            model="demo-model",
            timeout=0.01,
        )
        with pytest.raises(LLMBackendError, match="timed out after 0.01 seconds"):
            backend.complete(LLMRequest("hello"))


def test_litellm_backend_completes_via_an_openai_compatible_server() -> None:
    pytest.importorskip("litellm")  # the optional ``litellm`` extra

    def handler(
        _payload: Mapping[str, object],
        _headers: Mapping[str, str],
        _path: str,
    ) -> ServerResponse:
        body = {
            "id": "cmpl-1",
            "object": "chat.completion",
            "model": "demo-model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "litellm answer"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        return 200, body, 0

    with running_openai_server(handler) as (base_url, requests):
        result = OR.LiteLLMBackend(
            model="openai/demo-model",
            api_base=f"{base_url}/v1",
            api_key="test-key",
        ).complete(LLMRequest("hello", system="be precise"))

    assert result.text == "litellm answer"
    payload = cast(dict[str, Any], requests[0]["payload"])
    assert payload["messages"] == [
        {"role": "system", "content": "be precise"},
        {"role": "user", "content": "hello"},
    ]


def test_litellm_backend_requires_a_model() -> None:
    with pytest.raises(LLMBackendError, match="requires a model"):
        OR.LiteLLMBackend(model="  ").preflight()


def test_openai_compatible_backend_passes_sampling_and_vendor_extras() -> None:
    def handler(
        _payload: Mapping[str, object],
        _headers: Mapping[str, str],
        _path: str,
    ) -> ServerResponse:
        return 200, {"choices": [{"message": {"content": "ok"}}]}, 0

    with running_openai_server(handler) as (base_url, requests):
        OR.OpenAICompatibleBackend(
            base_url=f"{base_url}/v1",
            model="demo-model",
            temperature=0.3,
            max_tokens=64,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        ).complete(LLMRequest("hi"))

    payload = cast(dict[str, Any], requests[0]["payload"])
    assert payload["temperature"] == 0.3
    assert payload["max_tokens"] == 64
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}


def test_openai_compatible_backend_omits_sampling_knobs_by_default() -> None:
    def handler(
        _payload: Mapping[str, object],
        _headers: Mapping[str, str],
        _path: str,
    ) -> ServerResponse:
        return 200, {"choices": [{"message": {"content": "ok"}}]}, 0

    with running_openai_server(handler) as (base_url, requests):
        OR.OpenAICompatibleBackend(
            base_url=f"{base_url}/v1", model="demo-model"
        ).complete(LLMRequest("hi"))

    payload = cast(dict[str, Any], requests[0]["payload"])
    assert "temperature" not in payload
    assert "max_tokens" not in payload
