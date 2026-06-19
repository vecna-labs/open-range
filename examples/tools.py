"""Reference tools for the TRL training path — examples, not part of OpenRange.

The realistic model is a sandbox: ``EpisodeEnv(sandbox=True)`` gives the agent its own
hardened container plus a ``run`` capability, and the agent composes ``curl`` /
``python`` itself — one shell primitive, no HTTP tools shipped from the harness, the
way a Kali/CTF agent works (proven end to end in ``tests/test_trl_sandbox_episode.py``).
``shell`` is that primitive; ``submit`` records the answer for the held-out grader.
``FILE_TOOLS`` are a shell-less convenience for editing a code world's workspace.

A tool is a plain callable taking the live episode ``surface`` first, then the model's
kwargs; ``EpisodeEnv`` reflects each as a TRL tool (schema from signature + docstring).
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any


def shell(surface: Mapping[str, Any], command: str) -> str:
    """Run a shell command on your own machine and return its combined output.

    Compose requests yourself, e.g. ``curl -s http://target:8000/items?id=1`` or
    ``curl -s -X POST --data 'user=admin&pw=x' http://target:8000/login``.

    Args:
        command: The shell command line to run.
    """
    run = surface.get("run")
    if not callable(run):
        return "error: no shell in this episode (start the env with sandbox=True)"
    return str(run(command).output)


def submit(surface: Mapping[str, Any], content: str) -> str:
    """Submit your final answer; the held-out grader reads ``result.json``.

    Args:
        content: A JSON object carrying the requested field, e.g.
            {"flag": "<the value you recovered>"}.
    """
    (Path(str(surface["solver_root"])) / "result.json").write_text(
        content, encoding="utf-8"
    )
    return f"submitted {len(content)} byte(s)"


class WorkspaceError(Exception):
    """A file actuator call that can't be honored — most importantly a path that
    escapes the workspace root, but also a missing file. The adapter's tool
    boundary catches it (fail-soft): a malformed call costs reward, never the run.
    """


class FileWorkspaceTools:
    """Sandboxed file IO rooted at one episode's ``solver_root``.

    Every path is resolved and asserted to stay under ``root`` — a
    ``write_file("../../etc/passwd")`` raises ``WorkspaceError``. Writing
    *inside* a throwaway temp ``solver_root`` is safe (grading already runs
    untrusted code sandboxed); escaping it is not.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()

    def _resolve(self, path: str) -> Path:
        candidate = (self.root / path).resolve()
        if candidate != self.root and self.root not in candidate.parents:
            raise WorkspaceError(f"path {path!r} escapes the workspace root")
        return candidate

    def read_file(self, path: str) -> str:
        target = self._resolve(path)
        if not target.is_file():
            raise WorkspaceError(f"no such file: {path!r}")
        return target.read_text(encoding="utf-8")

    def write_file(self, path: str, content: str) -> str:
        target = self._resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"wrote {len(content)} byte(s) to {path}"

    def list_dir(self, path: str = ".") -> str:
        target = self._resolve(path)
        if not target.exists():
            raise WorkspaceError(f"no such directory: {path!r}")
        if target.is_file():
            return path
        names = sorted(p.name + ("/" if p.is_dir() else "") for p in target.iterdir())
        return "\n".join(names) if names else "(empty)"

    def apply_patch(self, path: str, find: str, replace: str) -> str:
        target = self._resolve(path)
        if not target.is_file():
            raise WorkspaceError(f"no such file: {path!r}")
        original = target.read_text(encoding="utf-8")
        if find not in original:
            raise WorkspaceError(f"patch text not found in {path!r}")
        occurrences = original.count(find)
        target.write_text(original.replace(find, replace), encoding="utf-8")
        return f"patched {path} ({occurrences} occurrence(s))"


def read_file(surface: Mapping[str, Any], path: str) -> str:
    """Read a UTF-8 text file from the workspace and return its contents.

    Args:
        path: Path to the file, relative to the workspace root.
    """
    return FileWorkspaceTools(str(surface["solver_root"])).read_file(path)


def write_file(surface: Mapping[str, Any], path: str, content: str) -> str:
    """Create or overwrite a workspace file with the given contents.

    Args:
        path: Path to the file, relative to the workspace root.
        content: The full text to write into the file.
    """
    return FileWorkspaceTools(str(surface["solver_root"])).write_file(path, content)


def list_dir(surface: Mapping[str, Any], path: str = ".") -> str:
    """List the entries of a workspace directory.

    Args:
        path: Directory to list, relative to the workspace root (defaults to root).
    """
    return FileWorkspaceTools(str(surface["solver_root"])).list_dir(path)


def apply_patch(surface: Mapping[str, Any], path: str, find: str, replace: str) -> str:
    """Replace exact text in a workspace file (use this for small edits).

    Args:
        path: Path to the file to edit, relative to the workspace root.
        find: The exact text to search for in the file.
        replace: The text to substitute in place of every match.
    """
    return FileWorkspaceTools(str(surface["solver_root"])).apply_patch(
        path, find, replace
    )


def run_tests(surface: Mapping[str, Any], node_ids: str = "") -> str:
    """Run the workspace's own pytest suite and return a text summary.

    Runs only the tests visible in your workspace, never the held-out grader.

    Args:
        node_ids: Space-separated pytest targets; empty runs the whole suite.
    """
    fn = surface.get("run_tests")
    if not callable(fn):
        return "error: this world exposes no run_tests tool"
    res = fn(node_ids.split() or None)
    ok = bool(res.get("ok"))
    head = f"tests {'passed' if ok else 'failed'} (returncode={res.get('returncode')})"
    stdout = str(res.get("stdout") or "").strip()
    return f"{head}\n{stdout or '(no output)'}"


WEB_TOOLS = (shell, submit)
FILE_TOOLS = (read_file, write_file, list_dir, apply_patch, run_tests)
