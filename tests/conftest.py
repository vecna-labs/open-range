"""Shared test fixtures."""

from __future__ import annotations

import contextlib
import http.server
import json
import threading
from collections.abc import Callable, Iterator

import pytest

Respond = Callable[[str, str], tuple[int, str]]


@pytest.fixture
def chat_server() -> Callable[[Respond], contextlib.AbstractContextManager[str]]:
    """A real OpenAI-compatible HTTP endpoint for backend tests — not a test double.

    Returns a context-manager factory: ``with chat_server(respond) as base_url``, where
    ``respond(path, method)`` returns the ``(status, json body)`` the server replies
    with, so each test drives a concrete server reply. Yields the ``/v1`` base URL.
    """

    @contextlib.contextmanager
    def serve(respond: Respond) -> Iterator[str]:
        class _Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *_args: object) -> None:
                pass

            def _serve(self) -> None:
                status, body = respond(self.path, self.command)
                data = body.encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self) -> None:
                self._serve()

            def do_POST(self) -> None:
                self.rfile.read(int(self.headers.get("Content-Length", 0)))
                self._serve()

        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            port = server.server_address[1]
            yield f"http://127.0.0.1:{port}/v1"
        finally:
            server.shutdown()
            thread.join()
            server.server_close()

    return serve


@pytest.fixture
def chat_completion() -> Callable[[str], str]:
    """Build a minimal OpenAI ``/chat/completions`` response body for ``content``."""

    def build(content: str) -> str:
        return json.dumps({"choices": [{"message": {"content": content}}]})

    return build
