"""LLM backends.

The ``LLMBackend`` Protocol and ``LLMRequest`` / ``LLMResult`` value types live in
``openrange_pack_sdk``. This module ships the concrete backends — the CLI ones
(``CodexBackend``, ``ClaudeBackend``) and an HTTP one (``OpenAIBackend``, for any
OpenAI-compatible endpoint) — plus the impl-specific exceptions they raise.
"""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from openrange_pack_sdk import LLMBackendError, LLMRequest, LLMResult


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
class OpenAIBackend:
    """An ``LLMBackend`` for any OpenAI-compatible ``/chat/completions`` endpoint — a
    local ``llama-server`` or vLLM, or the OpenAI API itself.

    A structured request asks for a JSON object in the prompt and parses it out of the
    reply, so the backend works against servers without a structured-output flag (the
    same approach as ``ClaudeBackend``). ``api_key`` is sent as a bearer token when set
    and omitted otherwise, which local servers ignore. ``json_mode`` additionally asks
    the server for ``response_format=json_object`` on structured requests, which keeps a
    reasoning model's chain-of-thought out of ``content`` (servers expose it separately)
    so the reply parses cleanly; ``max_tokens`` bounds the reply, which a reasoning
    model needs to not run away. ``extra_body`` merges vendor extensions into the
    request — e.g. ``{"chat_template_kwargs": {"enable_thinking": False}}`` to turn a
    reasoning model's thinking off (some ignore the ``/no_think`` prompt token).
    """

    base_url: str = "http://localhost:8080/v1"
    model: str = "local"
    api_key: str | None = None
    temperature: float = 0.2
    max_tokens: int | None = None
    json_mode: bool = True
    extra_body: Mapping[str, object] | None = None
    timeout: float = 180.0

    def preflight(self) -> None:
        """Verify the endpoint is reachable (a ``GET /models`` round-trip)."""
        try:
            with urllib.request.urlopen(
                self._request("models", None), timeout=self.timeout
            ):
                return
        except OSError as exc:
            raise LLMBackendError(
                f"OpenAI-compatible endpoint not reachable at {self.base_url!r}: {exc}",
            ) from exc

    def complete(self, request: LLMRequest) -> LLMResult:
        prompt = request.prompt
        if request.json_schema is not None:
            prompt += (
                "\n\nReturn ONLY a JSON object matching this schema — no prose, no "
                "code fences:\n" + json.dumps(request.json_schema)
            )
        messages: list[dict[str, str]] = []
        if request.system is not None:
            messages.append({"role": "system", "content": request.system})
        messages.append({"role": "user", "content": prompt})
        payload: dict[str, object] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
        if request.json_schema is not None and self.json_mode:
            payload["response_format"] = {"type": "json_object"}
        if self.extra_body:
            payload.update(self.extra_body)
        text = _chat_content(self._post("chat/completions", payload))
        if request.json_schema is None:
            return LLMResult(text)
        return LLMResult(text, parse_json_object(_first_json_object(text)))

    def _request(
        self, path: str, payload: Mapping[str, object] | None
    ) -> urllib.request.Request:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return urllib.request.Request(
            f"{self.base_url.rstrip('/')}/{path}",
            data=None if payload is None else json.dumps(payload).encode("utf-8"),
            headers=headers,
        )

    def _post(self, path: str, payload: Mapping[str, object]) -> Mapping[str, object]:
        try:
            with urllib.request.urlopen(
                self._request(path, payload), timeout=self.timeout
            ) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace").strip()
            raise LLMBackendError(
                f"OpenAI endpoint HTTP {exc.code}: {detail or exc.reason}",
                returncode=exc.code,
            ) from exc
        except OSError as exc:
            raise LLMBackendError(f"OpenAI endpoint request failed: {exc}") from exc
        return parse_json_object(raw)


def _chat_content(envelope: Mapping[str, object]) -> str:
    choices = envelope.get("choices")
    if not isinstance(choices, list) or not choices:
        raise LLMBackendError("OpenAI endpoint returned no choices")
    choice = choices[0]
    message = choice.get("message") if isinstance(choice, Mapping) else None
    content = message.get("content") if isinstance(message, Mapping) else None
    if not isinstance(content, str):
        raise LLMBackendError("OpenAI endpoint choice has no string content")
    return content
