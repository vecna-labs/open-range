"""CLI LLM backends.

The ``LLMBackend`` Protocol and ``LLMRequest`` / ``LLMResult`` value types live in
``openrange_pack_sdk``. This module ships the concrete CLI backends — ``CodexBackend``
and ``ClaudeBackend`` — plus the impl-specific exceptions they raise.
"""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
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
