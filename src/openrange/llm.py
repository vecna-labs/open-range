"""LLM backend contracts and Codex CLI implementation."""

from __future__ import annotations

import json
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

from openrange.core import OpenRangeError

CODEX_DEFAULT_MODEL = "gpt-5.3-codex-spark"


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
        pass


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
