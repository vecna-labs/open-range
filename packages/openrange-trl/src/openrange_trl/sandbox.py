"""A throwaway machine the agent runs its own commands in — never the trainer.

The in-process TRL path hands the policy function-tools only because a token-emitting
model has no shell. The real model — the one SkyRL-Agent uses — is this: give the
agent its **own** hardened container with ``bash``/``curl``/``python``, let it run
**its own** commands against the world, and ship no tools from the harness. The
sandbox binds to whatever the world's ``surface`` exposes:

* an HTTP world (``base_url``) is a network target — the sandbox joins the world's
  docker network and the agent reaches it by alias (``curl http://target:8000/...``);
* a code world (``solver_root``) is a workspace — the sandbox bind-mounts it so the
  agent edits the tree as its own filesystem and the grader reads the edited tree.

Either way it is one :meth:`AgentSandbox.run` primitive (a real shell, like SkyRL's
``CmdRunAction``); only the binding differs. Untrusted agent code runs here under the
same hardening the worlds use — cap-drop ALL, no-new-privileges, memory/cpu/pid caps,
reachable only on the episode network — and never in the trainer process.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tempfile
import uuid
from collections.abc import Mapping
from pathlib import Path
from types import TracebackType
from typing import Any, NamedTuple

# Same base the worlds build on (container.py), plus a shell and the tools an agent
# reaches for; bash is already in the slim image, curl/git/certs are not.
_AGENT_DOCKERFILE = (
    "FROM python:3.13-slim\n"
    "RUN apt-get update \\\n"
    " && apt-get install -y --no-install-recommends bash curl ca-certificates git \\\n"
    " && rm -rf /var/lib/apt/lists/*\n"
)

# Mirrors the worlds' container hardening (cyber_webapp ``hardening_run_args``); kept
# here so the adapter doesn't depend on a pack. Contains attacker-controlled code: no
# capabilities, no privilege gain, bounded memory/cpu/pids.
_HARDENING = [
    "--cap-drop",
    "ALL",
    "--security-opt",
    "no-new-privileges",
    "--memory",
    "512m",
    "--cpus",
    "1.0",
    "--pids-limit",
    "256",
]


class SandboxError(RuntimeError):
    """The sandbox could not start, bind, or run — surfaced, never swallowed."""


class CommandResult(NamedTuple):
    """One command's outcome: its exit code and its combined stdout+stderr."""

    exit_code: int
    output: str


class AgentSandbox:
    """A hardened throwaway container the agent runs its own commands in.

    Bind it to a world ``surface``: an HTTP world (``base_url``) joins ``network`` and
    the target is reached by alias; a code world (``solver_root``) is bind-mounted at
    ``/workspace``. :meth:`run` executes one shell command; :meth:`close` removes the
    container (the network, if any, is the caller's). Usable as a context manager.
    """

    def __init__(
        self,
        surface: Mapping[str, Any],
        *,
        network: str | None = None,
        image: str | None = None,
    ) -> None:
        self._surface = surface
        self._network = network
        self._image = image
        self._cname: str | None = None

    def __enter__(self) -> AgentSandbox:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def start(self) -> None:
        """Build the agent image (once, reused) and run an idle hardened container
        bound to the world surface. The binding is validated before any image build,
        so a misconfigured surface fails fast without touching docker."""
        if self._cname is not None:
            raise SandboxError("sandbox already started")
        cname = f"openrange-agent-{uuid.uuid4().hex[:12]}"
        bindings = self._surface_bindings(cname)
        image = self._image or _build_agent_image()
        # Run as the invoking user, never root: it matches the owner of a bind-mounted
        # workspace (so the agent can edit it — cap-drop ALL strips root's DAC
        # override), and keeps the contained code off uid 0.
        user = f"{os.getuid()}:{os.getgid()}"
        subprocess.run(
            [
                "docker",
                "run",
                "-d",
                "--name",
                cname,
                *_HARDENING,
                "--user",
                user,
                *bindings,
                image,
                "sleep",
                "infinity",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        self._cname = cname

    def run(self, command: str, *, timeout: float = 120.0) -> CommandResult:
        """Run one shell command in the sandbox and return its exit code + output.
        The agent composes ``curl`` / ``python`` / file edits itself — this is the
        single generic primitive, not a fixed tool set."""
        done = subprocess.run(
            ["docker", "exec", self._require_started(), "bash", "-lc", command],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return CommandResult(done.returncode, done.stdout + done.stderr)

    def close(self) -> None:
        """Remove the sandbox container (its network, if any, is the caller's)."""
        if self._cname is None:
            return
        # Bounded so a wedged daemon can't hang teardown (this is the only docker call
        # on the code-world teardown path); best-effort, so no check.
        subprocess.run(
            ["docker", "rm", "-f", self._cname], capture_output=True, timeout=60
        )
        self._cname = None

    def _surface_bindings(self, cname: str) -> list[str]:
        base_url = self._surface.get("base_url")
        solver_root = self._surface.get("solver_root")
        if isinstance(base_url, str):
            if self._network is None:
                raise SandboxError(
                    "an HTTP world (base_url) needs a docker network to reach the "
                    "target by alias; pass network=..."
                )
            # Join the world's network; the agent reaches the target by its alias.
            return ["--network", self._network, "--network-alias", cname]
        if solver_root is not None:
            # Mount the workspace so the agent edits the code as its own filesystem;
            # the grader later reads the edited tree back on the host.
            root = Path(str(solver_root)).resolve()
            return ["-v", f"{root}:/workspace", "-w", "/workspace"]
        return []

    def _require_started(self) -> str:
        if self._cname is None:
            raise SandboxError("sandbox not started")
        return self._cname


def _build_agent_image() -> str:
    """Build the agent image once and reuse it: a content-derived tag means a second
    sandbox finds the image already present and skips the build."""
    digest = hashlib.sha256(_AGENT_DOCKERFILE.encode()).hexdigest()[:12]
    tag = f"openrange-agent:{digest}"
    present = subprocess.run(["docker", "image", "inspect", tag], capture_output=True)
    if present.returncode == 0:
        return tag
    context = Path(tempfile.mkdtemp(prefix="openrange-agent-"))
    try:
        (context / "Dockerfile").write_text(_AGENT_DOCKERFILE, encoding="utf-8")
        subprocess.run(
            ["docker", "build", "-q", "-t", tag, str(context)],
            check=True,
            capture_output=True,
            timeout=600,
        )
    finally:
        shutil.rmtree(context, ignore_errors=True)
    return tag
