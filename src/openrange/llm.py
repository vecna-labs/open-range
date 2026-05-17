"""LLM backend contracts and Codex CLI implementation."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from openrange.core import OpenRangeError

CODEX_DEFAULT_MODEL = "gpt-5.3-codex-spark"
OPENAI_COMPATIBLE_DEFAULT_MODEL = "gpt-4o-mini"


class LLMError(OpenRangeError):
    """Base LLM error."""


class LLMRequestError(LLMError):
    """Raised when an LLM request cannot be serialized."""


class LLMBackendError(LLMError):
    """Raised when the backend process fails."""

    def __init__(self, message: str, *, returncode: int | None = None) -> None:
        super().__init__(message)
        self.returncode = returncode


@dataclass(frozen=True, slots=True)
class LLMRequest:
    prompt: str
    system: str | None = None
    json_schema: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        if self.json_schema is None:
            return
        try:
            json.dumps(self.json_schema)
        except TypeError as exc:
            raise LLMRequestError("json_schema must be JSON serializable") from exc

    def as_prompt(self) -> str:
        if self.system is None:
            return self.prompt
        return f"{self.system}\n\n{self.prompt}"


@dataclass(frozen=True, slots=True)
class LLMResult:
    text: str
    parsed_json: Mapping[str, object] | None = None


class LLMBackend(Protocol):
    def complete(self, request: LLMRequest) -> LLMResult: ...

    def preflight(self) -> None:
        """Cheap synchronous check that this backend is callable.

        Implementations verify imports, binaries, and configuration —
        no API calls. Default is a no-op so existing backends remain
        protocol-conformant; backends with checkable invariants
        (e.g. ``CodexBackend`` looking for the ``codex`` binary on
        PATH) override.
        """


@dataclass(frozen=True, slots=True)
class CodexBackend:
    command: str | Path = "codex"
    model: str = CODEX_DEFAULT_MODEL
    cwd: Path | None = None
    timeout: float = 120.0
    sandbox: str = "read-only"
    # Extra ``-c key=value`` args passed straight through to ``codex
    # exec``. The agent harness uses this to enable network egress when
    # running under ``workspace-write`` (``sandbox_workspace_write.
    # network_access=true``) without losing the read-restriction the
    # workspace sandbox provides.
    config_overrides: tuple[str, ...] = ()

    def preflight(self) -> None:
        """Verify the codex binary is reachable on PATH."""
        import shutil

        command = str(self.command)
        if shutil.which(command) is None:
            raise LLMBackendError(
                f"codex CLI not found on PATH ({command!r}). "
                "Install codex or override the 'command' field.",
            )

    def complete(self, request: LLMRequest) -> LLMResult:
        with tempfile.TemporaryDirectory() as tmp:
            schema_path = Path(tmp, "schema.json")
            output_path = Path(tmp, "output.json")
            command = [
                str(self.command),
                "exec",
                "--model",
                self.model,
                "--color",
                "never",
                "--ephemeral",
                "--sandbox",
                self.sandbox,
                "--skip-git-repo-check",
            ]
            for override in self.config_overrides:
                command.extend(("-c", override))
            if request.json_schema is not None:
                schema_path.write_text(
                    json.dumps(request.json_schema),
                    encoding="utf-8",
                )
                command.extend(
                    (
                        "--output-schema",
                        str(schema_path),
                        "--output-last-message",
                        str(output_path),
                    ),
                )
            completed = run_codex(
                command,
                input_text=request.as_prompt(),
                cwd=self.cwd,
                timeout=self.timeout,
            )
            if completed.returncode != 0:
                detail = completed.stderr.strip() or completed.stdout.strip()
                message = f"codex exit status {completed.returncode}: "
                raise LLMBackendError(
                    message + (detail or "no output"),
                    returncode=completed.returncode,
                )
            if request.json_schema is None:
                return LLMResult(completed.stdout.strip())
            try:
                raw = output_path.read_text(encoding="utf-8").strip()
            except OSError as exc:
                raise LLMBackendError(
                    "codex did not write --output-last-message",
                ) from exc
            return LLMResult(raw, parse_json_object(raw))


@dataclass(frozen=True, slots=True)
class OpenAICompatibleBackend:
    """HTTP backend for OpenAI Chat Completions-compatible providers.

    ``extra_headers`` are applied last, so callers can intentionally override
    the default ``Authorization`` bearer header for providers that use a
    different auth scheme.
    """

    base_url: str = "https://api.openai.com/v1"
    model: str = OPENAI_COMPATIBLE_DEFAULT_MODEL
    api_key: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    timeout: float = 120.0
    extra_headers: Mapping[str, str] | None = None
    json_schema_strict: bool = False

    def preflight(self) -> None:
        """Validate local configuration without making a network call."""
        if not self.base_url.strip():
            raise LLMBackendError("OpenAICompatibleBackend requires a base_url")
        scheme = urlparse(self.base_url).scheme
        if scheme not in {"http", "https"}:
            raise LLMBackendError(
                "OpenAICompatibleBackend base_url must use http or https",
            )
        if not self.model.strip():
            raise LLMBackendError("OpenAICompatibleBackend requires a model")
        if self.timeout <= 0:
            raise LLMBackendError("OpenAICompatibleBackend timeout must be positive")

    def complete(self, request: LLMRequest) -> LLMResult:
        self.preflight()
        http_request = Request(
            self._chat_completions_url(),
            data=json.dumps(self._payload(request)).encode("utf-8"),
            headers=self._headers(),
            method="POST",
        )
        try:
            with urlopen(http_request, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8")
        except HTTPError as exc:
            detail = _openai_compatible_error_detail(exc)
            raise LLMBackendError(
                f"OpenAI-compatible HTTP {exc.code}: {detail}",
                returncode=exc.code,
            ) from exc
        except TimeoutError as exc:
            raise _openai_compatible_timeout_error(self.timeout) from exc
        except URLError as exc:
            if isinstance(exc.reason, TimeoutError):
                raise _openai_compatible_timeout_error(self.timeout) from exc
            raise LLMBackendError(
                f"OpenAI-compatible request failed: {exc.reason}",
            ) from exc

        return _openai_compatible_result(raw, request)

    def _chat_completions_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/chat/completions"

    def _headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        api_key = (
            self.api_key if self.api_key is not None else os.getenv(self.api_key_env)
        )
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        if self.extra_headers is not None:
            headers.update(self.extra_headers)
        return headers

    def _payload(self, request: LLMRequest) -> dict[str, object]:
        messages: list[dict[str, str]] = []
        if request.system is not None:
            messages.append({"role": "system", "content": request.system})
        messages.append({"role": "user", "content": request.prompt})

        payload: dict[str, object] = {
            "model": self.model,
            "messages": messages,
        }
        if request.json_schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "openrange_response",
                    "strict": self.json_schema_strict,
                    "schema": request.json_schema,
                },
            }
        return payload


def run_codex(
    command: Sequence[str],
    *,
    input_text: str,
    cwd: Path | None,
    timeout: float,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            input=input_text,
            cwd=cwd,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise LLMBackendError(f"codex timed out after {timeout} seconds") from exc
    except OSError as exc:
        raise LLMBackendError(str(exc)) from exc


def _openai_compatible_error_detail(exc: HTTPError) -> str:
    body = exc.read().decode("utf-8", errors="replace").strip()
    if not body:
        return exc.reason
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        return body
    if isinstance(data, dict):
        error = data.get("error")
        if isinstance(error, dict):
            message = error.get("message")
            if isinstance(message, str) and message:
                return message
        message = data.get("message")
        if isinstance(message, str) and message:
            return message
    return body


def _openai_compatible_timeout_error(timeout: float) -> LLMBackendError:
    return LLMBackendError(
        f"OpenAI-compatible request timed out after {timeout} seconds",
    )


def _openai_compatible_result(raw: str, request: LLMRequest) -> LLMResult:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LLMBackendError(
            f"OpenAI-compatible response was not JSON: {exc}"
        ) from exc
    if not isinstance(data, dict):
        raise LLMBackendError("OpenAI-compatible response was not an object")

    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise LLMBackendError("OpenAI-compatible response had no choices")
    first = choices[0]
    if not isinstance(first, dict):
        raise LLMBackendError("OpenAI-compatible choice was not an object")
    message = first.get("message")
    if not isinstance(message, dict):
        raise LLMBackendError("OpenAI-compatible choice had no message")
    content = message.get("content")
    if not isinstance(content, str):
        raise LLMBackendError("OpenAI-compatible message content was not text")

    if request.json_schema is None:
        return LLMResult(content)
    return LLMResult(content, parse_json_object(content))


def parse_json_object(raw: str) -> Mapping[str, object]:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LLMBackendError(f"backend returned invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise LLMBackendError("backend returned JSON that is not an object")
    return cast(Mapping[str, object], data)
