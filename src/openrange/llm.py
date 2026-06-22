"""LLM backends.

The ``LLMBackend`` Protocol and ``LLMRequest`` / ``LLMResult`` value types live in
``openrange_pack_sdk``. This module ships the concrete backends: the CLI
``CodexBackend`` / ``ClaudeBackend``, the dependency-free HTTP
``OpenAICompatibleBackend``, and ``LiteLLMBackend`` for any provider LiteLLM
reaches — plus the impl-specific exceptions they raise.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from openrange_pack_sdk import LLMBackendError, LLMRequest, LLMResult

OPENAI_COMPATIBLE_DEFAULT_MODEL = "gpt-4o-mini"


@dataclass(frozen=True, slots=True)
class CodexBackend:
    command: str | Path = "codex"
    # None → don't pass --model; the codex CLI uses its own configured
    # default (~/.codex/config.toml). Hardcoding a model here overrides
    # that and breaks when the pinned model isn't available to the
    # caller's account.
    model: str | None = None
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
                "--color",
                "never",
                "--ephemeral",
                "--sandbox",
                self.sandbox,
                "--skip-git-repo-check",
            ]
            if self.model is not None:
                command += ["--model", self.model]
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


def parse_json_object(raw: str) -> Mapping[str, object]:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LLMBackendError(f"backend returned invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise LLMBackendError("backend returned JSON that is not an object")
    return cast(Mapping[str, object], data)


@dataclass(frozen=True, slots=True)
class ClaudeBackend:
    """An ``LLMBackend`` that drives the ``claude`` CLI in print mode (``-p``).

    Claude has no output-schema flag, so a structured request asks for a JSON object in
    the prompt and parses it out of the model's reply. Useful where codex is
    unavailable, or declines a task it flags as risky.
    """

    command: str | Path = "claude"
    model: str | None = None
    cwd: Path | None = None
    timeout: float = 180.0

    def preflight(self) -> None:
        """Verify the claude binary is reachable on PATH."""
        import shutil

        if shutil.which(str(self.command)) is None:
            raise LLMBackendError(
                f"claude CLI not found on PATH ({str(self.command)!r}). "
                "Install claude or override the 'command' field.",
            )

    def complete(self, request: LLMRequest) -> LLMResult:
        prompt = request.as_prompt()
        if request.json_schema is not None:
            prompt += (
                "\n\nReturn ONLY a JSON object matching this schema — no prose, no "
                "code fences:\n" + json.dumps(request.json_schema)
            )
        command = [str(self.command), "-p", prompt, "--output-format", "json"]
        if self.model is not None:
            command += ["--model", self.model]
        completed = _run_cli(
            command, cwd=self.cwd, timeout=self.timeout, label="claude"
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise LLMBackendError(
                f"claude exit status {completed.returncode}: {detail or 'no output'}",
                returncode=completed.returncode,
            )
        text = _claude_result_text(completed.stdout)
        if request.json_schema is None:
            return LLMResult(text)
        return LLMResult(text, parse_json_object(_first_json_object(text)))


def _run_cli(
    command: Sequence[str], *, cwd: Path | None, timeout: float, label: str
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            cwd=cwd,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise LLMBackendError(f"{label} timed out after {timeout} seconds") from exc
    except OSError as exc:
        raise LLMBackendError(str(exc)) from exc


def _claude_result_text(stdout: str) -> str:
    # `claude -p --output-format json` prints a result envelope whose `result` field is
    # the model's reply; fall back to raw stdout if it isn't that envelope.
    try:
        envelope = json.loads(stdout)
    except json.JSONDecodeError:
        return stdout.strip()
    if isinstance(envelope, dict) and isinstance(envelope.get("result"), str):
        return cast(str, envelope["result"])
    return stdout.strip()


def _first_json_object(text: str) -> str:
    # The reply may wrap JSON in ``` fences or add prose; pull out the object.
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fenced:
        return fenced.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        return text[start : end + 1]
    return text


@dataclass(frozen=True, slots=True)
class OpenAICompatibleBackend:
    """Dependency-free HTTP backend for OpenAI Chat Completions-compatible providers.

    Speaks the ``/chat/completions`` shape over stdlib ``urllib`` — OpenAI, vLLM,
    Ollama, llama.cpp, and the like. ``extra_headers`` are applied last, so a caller
    can override the default ``Authorization`` bearer for providers using a different
    auth scheme. The default ``model`` suits OpenAI's API; set it explicitly for local
    and third-party servers.
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


def _openai_compatible_error_detail(exc: HTTPError) -> str:
    body = exc.read().decode("utf-8", errors="replace").strip()
    if not body:
        return str(exc.reason)
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


@dataclass(frozen=True, slots=True)
class LiteLLMBackend:
    """An ``LLMBackend`` over LiteLLM, reaching any provider it supports.

    ``model`` is a LiteLLM model id (``"openai/gpt-4o-mini"``,
    ``"anthropic/claude-..."``, ``"hosted_vllm/<name>"``, ``"ollama/<name>"``, …);
    ``api_base`` points at a self-hosted or compatible endpoint. Needs the optional
    ``litellm`` extra (``uv sync --extra litellm``). ``extra_params`` passes provider
    arguments (e.g. ``temperature``) straight through to ``litellm.completion``.
    """

    model: str
    api_base: str | None = None
    api_key: str | None = None
    api_key_env: str | None = None
    timeout: float = 120.0
    extra_params: Mapping[str, object] | None = None
    json_schema_strict: bool = False

    def preflight(self) -> None:
        """Validate configuration and that the ``litellm`` extra is installed."""
        if not self.model.strip():
            raise LLMBackendError("LiteLLMBackend requires a model")
        if self.timeout <= 0:
            raise LLMBackendError("LiteLLMBackend timeout must be positive")
        _require_litellm()

    def complete(self, request: LLMRequest) -> LLMResult:
        litellm = _require_litellm()
        api_key = self.api_key
        if api_key is None and self.api_key_env is not None:
            api_key = os.getenv(self.api_key_env)
        params: dict[str, object] = {
            "model": self.model,
            "messages": self._messages(request),
            "timeout": self.timeout,
        }
        if self.api_base is not None:
            params["api_base"] = self.api_base
        if api_key is not None:
            params["api_key"] = api_key
        if request.json_schema is not None:
            params["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "openrange_response",
                    "strict": self.json_schema_strict,
                    "schema": request.json_schema,
                },
            }
        if self.extra_params is not None:
            params.update(self.extra_params)
        try:
            response = litellm.completion(**params)
        except Exception as exc:
            raise LLMBackendError(f"LiteLLM request failed: {exc}") from exc
        content = _litellm_content(response)
        if request.json_schema is None:
            return LLMResult(content)
        return LLMResult(content, parse_json_object(content))

    def _messages(self, request: LLMRequest) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        if request.system is not None:
            messages.append({"role": "system", "content": request.system})
        messages.append({"role": "user", "content": request.prompt})
        return messages


def _require_litellm() -> Any:
    try:
        import litellm
    except ImportError as exc:
        raise LLMBackendError(
            "LiteLLMBackend needs the optional 'litellm' extra "
            "(uv sync --extra litellm)",
        ) from exc
    return litellm


def _litellm_content(response: Any) -> str:
    try:
        content = response.choices[0].message.content
    except (AttributeError, IndexError, KeyError, TypeError) as exc:
        raise LLMBackendError("LiteLLM response had no message content") from exc
    if not isinstance(content, str):
        raise LLMBackendError("LiteLLM message content was not text")
    return content
