"""SweRuntime — a thin OnDemandRuntime for the code-repair world.

No persistent process: the agent edits files under ``solver_root`` (the repo's
base working tree, materialized on each ``reset()``) and writes ``result.json``
to end the episode. The ``swe.fix`` grader replays the edited tree against the
held-out suite in a sandbox. The base tree comes from the graph, never a
re-clone, so realize is offline-safe; the held-out tests and gold fix live in
the graph too, but are never written to disk, so they stay hidden from the agent.

Multi-turn surface: ``surface_extras`` exposes a ``run_tests`` callable so an
agent can iterate — run tests against the *live* workspace, read the failures,
edit, repeat. It runs whatever tests exist under ``solver_root`` (e.g. a
reproduction the agent wrote), in the same sandbox the grader uses. It
deliberately does *not* inject the held-out grading suite: that stays hidden in
the graph and is applied only by ``swe.fix.check_success`` at episode stop, so
the agent gets a real local test loop without ever seeing its scorer.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from graphschema import WorldGraph
from openrange_pack_sdk import Backing, OnDemandRuntime, OpenRangeError, write_tree

from swe.sandbox import run_sandboxed

_RESULT_FILE = "result.json"
_SKIP_DIRS = frozenset({"__pycache__", ".pytest_cache"})
_RUN_TESTS_TIMEOUT = 60.0
_OUTPUT_TAIL = 4000  # cap tool output so a chatty suite can't flood agent context.


class SweRuntimeError(OpenRangeError):
    pass


class SweRuntime(OnDemandRuntime):
    def __init__(self, graph: WorldGraph, backing: Backing) -> None:
        if backing is not Backing.PROCESS:
            raise NotImplementedError(
                f"SweRuntime does not support backing={backing!r}; only "
                "Backing.PROCESS is wired (the repair world has no live process)"
            )
        super().__init__(graph)

    def prepare_env_files(self, graph: WorldGraph) -> Mapping[str, str]:
        # No persistent process → nothing to stage under pack_root.
        del graph
        return {}

    def reset(self) -> None:
        super().reset()
        assert self.solver_root is not None
        write_tree(self.solver_root, _base_files(self._graph))

    def collect_extras(self) -> Mapping[str, Any]:
        """Snapshot the agent's edited tree for the grader.

        ``check_success`` grades this map, so it stays a pure function of
        ``final_state`` (unit-testable with a synthetic tree, like the trading
        and cyber families).
        """
        if self.solver_root is None:
            return {"workspace_files": {}}
        return {"workspace_files": _read_tree(self.solver_root)}

    def surface_extras(self) -> Mapping[str, Any]:
        return {"run_tests": self._run_tests}

    def _run_tests(
        self,
        node_ids: Sequence[str] | None = None,
        *,
        timeout: float = _RUN_TESTS_TIMEOUT,
    ) -> dict[str, Any]:
        """Run pytest over the live workspace and report the outcome.

        ``node_ids`` are pytest targets relative to the workspace (a file, a
        ``file::test``, or empty to collect everything). Output is tailed so it
        can't flood the agent's context. Never raises for a failing/empty run —
        a red suite is the signal the agent is asking for.
        """
        if self.solver_root is None:
            raise SweRuntimeError("run_tests called before reset()")
        targets = [str(n) for n in (node_ids or [])]
        res = run_sandboxed(
            ["-m", "pytest", *targets, "-q", "-p", "no:cacheprovider", "--tb=short"],
            root=self.solver_root,
            timeout=min(float(timeout), _RUN_TESTS_TIMEOUT),
        )
        return {
            "ok": res.ok,
            "returncode": res.returncode,
            "timed_out": res.timed_out,
            "isolation": res.isolation,
            "stdout": _tail(res.stdout),
            "stderr": _tail(res.stderr),
        }


def _base_files(graph: WorldGraph) -> dict[str, str]:
    repos = graph.by_kind("repo")
    if not repos:
        return {}
    raw = repos[0].attrs.get("base_files")
    if not isinstance(raw, Mapping):
        return {}
    return {str(k): str(v) for k, v in raw.items()}


def _tail(text: str, limit: int = _OUTPUT_TAIL) -> str:
    if len(text) <= limit:
        return text
    return "…\n" + text[-limit:]


def _read_tree(root: Path) -> dict[str, str]:
    files: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if rel.name == _RESULT_FILE or _SKIP_DIRS.intersection(rel.parts):
            continue
        try:
            files[str(rel)] = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
    return files
