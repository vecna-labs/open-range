"""TRL GRPO adapter — torch-free.

The whole adapter lives here and imports only ``openrange`` + stdlib, so
``import openrange.trainers.trl`` works with no ``torch`` installed and every
piece below is deterministically unit-testable without a model. Only the live
``examples/trl_grpo_eval.py`` imports ``trl`` / ``torch`` and builds a real
``GRPOTrainer``. See ``DESIGN.md`` for why this seam is drawn here.

The five public pieces map onto TRL's agentic GRPO (the ``environment_factory``
path, ``transformers>=5.2``):

- ``OpenRangeEnv`` — one rollout's environment. ``reset(**row)`` starts a fresh
  ``EpisodeService`` episode and returns the live workspace listing (appended to
  the prompt); its public tool methods (``read_file`` / ``write_file`` /
  ``list_dir`` / ``apply_patch`` / ``run_tests``) are what TRL exposes to the
  policy as tools; the first read of ``env.reward`` (via the reward func) lazily
  stops + grades the episode through ``episode_reward``.
- ``FileWorkspaceTools`` — the sandboxed, path-traversal-guarded file IO the SWE
  *surface* never exposed (gap C): a tool-calling policy can now *change* the
  graded state, not just observe it. Harness-neutral, no TRL import.
- ``build_grpo_dataset`` — a snapshot's tasks → GRPO prompt rows, each tagged
  with ``snapshot_id`` / ``task_id`` so trajectories stay attributable across an
  ``auto_evolve`` curriculum.
- ``make_reward_func`` — the TRL-shaped reward bridge; defers entirely to the
  pack's structured grade via ``episode_reward`` (no reward logic reinvented).
- ``reward_variance_policy`` — a ``CurriculumPolicy`` keyed on the signal GRPO
  actually consumes (reward *spread*): when a group's spread collapses there is
  no gradient, so evolve.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from openrange_pack_sdk import EpisodeReportLike, Pack, Snapshot, TaskSpec

from openrange.core.curriculum import Direction
from openrange.core.episode import (
    AgentTurn,
    EpisodeHandle,
    EpisodeReport,
    EpisodeService,
)
from openrange.training import (
    Reward,
    Trajectory,
    episode_reward,
    episode_trajectory,
)

# Tail tool output so a chatty suite can't flood a tiny model's context window.
_OUTPUT_TAIL = 2000

_TOOL_GUIDE = (
    "You are solving a coding task in a sandboxed workspace. Use these tools:\n"
    "- list_dir(path): list a directory (default '.')\n"
    "- read_file(path): read a UTF-8 text file\n"
    "- write_file(path, content): create or overwrite a file\n"
    "- apply_patch(path, find, replace): replace exact text in a file\n"
    "- run_tests(node_ids): run pytest; node_ids is space-separated targets, "
    "empty runs all\n"
    "Edit files until the task is solved, then stop. A held-out test suite "
    "grades your final workspace."
)


class WorkspaceError(Exception):
    """A file actuator call that can't be honored — most importantly a path that
    escapes the workspace root, but also a missing file or a not-yet-reset env.

    Raised by ``FileWorkspaceTools`` and caught at the ``OpenRangeEnv`` tool
    boundary, which turns it into an error string (fail-soft): a malformed call
    from a weak model costs reward, never the run.
    """


class FileWorkspaceTools:
    """Sandboxed file IO rooted at one episode's ``solver_root``.

    Every path is resolved and asserted to stay under ``root`` — a
    ``write_file("../../etc/passwd")`` raises ``WorkspaceError``. Writing
    *inside* a throwaway temp ``solver_root`` is safe (grading already runs
    untrusted code sandboxed); escaping it is not. Harness-neutral: no TRL
    import, so a second trainer adapter can share it unchanged.
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


class OpenRangeEnv:
    """One GRPO rollout's environment, over a single ``EpisodeService``.

    Lifecycle (mirrors TRL's agentic loop): ``reset(**row)`` starts a fresh
    episode (each ``start_episode`` realizes its own ``solver_root``, so a group
    of N concurrent envs never collides); the public tool methods mutate that
    workspace; the first ``_finalize`` (driven by the reward func) stops the
    episode and grades the *final* tree into ``self.reward``. Tool methods are
    **fail-soft** — they return an error string rather than raising, so a weak
    model's bad call costs reward, not the run.

    Only ``reset`` + the five tool methods are public, so those are exactly the
    tools TRL exposes; everything else is underscore-prefixed (TRL skips it) or a
    plain data attribute (``reward`` / ``turns`` / ``report``).
    """

    def __init__(
        self,
        *,
        service: EpisodeService,
        snapshots: Mapping[str, Snapshot],
        reward_fn: Callable[[EpisodeReport], Reward] = episode_reward,
    ) -> None:
        self.service = service
        self.snapshots = dict(snapshots)
        self.reward_fn = reward_fn
        self.reward: float = 0.0
        self.turns: list[AgentTurn] = []
        self.report: EpisodeReport | None = None
        self._handle: EpisodeHandle | None = None
        self._surface: Mapping[str, Any] | None = None
        self._tools: FileWorkspaceTools | None = None
        self._finalized = False

    # -- lifecycle -----------------------------------------------------------

    def reset(
        self,
        *,
        snapshot_id: str | None = None,
        task_id: str | None = None,
        **_: object,
    ) -> str:
        """Start a fresh episode and return the initial workspace listing.

        ``snapshot_id`` / ``task_id`` come straight from the dataset row (the
        extra columns are absorbed by ``**_``). The returned text is appended to
        the prompt by TRL — it carries the *live* file listing the dataset can't
        know, since ``solver_root`` is materialized only at episode start.
        """
        snapshot = self._resolve_snapshot(snapshot_id)
        handle = self.service.start_episode(snapshot, task_id)
        self._handle = handle
        self._surface = self.service.surface(handle)
        self._tools = FileWorkspaceTools(self.service.solver_root(handle))
        self.reward = 0.0
        self.turns = []
        self.report = None
        self._finalized = False
        return f"Workspace ready. Files:\n{self._tools.list_dir('.')}"

    # -- tools (public → exposed to the policy) ------------------------------

    def read_file(self, path: str) -> str:
        """Read a UTF-8 text file from the workspace and return its contents.

        Args:
            path: Path to the file, relative to the workspace root.
        """
        out = self._safe(lambda: self._require_tools().read_file(path))
        self._record("read_file", {"path": path}, out)
        return out

    def write_file(self, path: str, content: str) -> str:
        """Create or overwrite a workspace file with the given contents.

        Args:
            path: Path to the file, relative to the workspace root.
            content: The full text to write into the file.
        """
        out = self._safe(lambda: self._require_tools().write_file(path, content))
        self._record("write_file", {"path": path, "content": content}, out)
        return out

    def list_dir(self, path: str = ".") -> str:
        """List the entries of a workspace directory.

        Args:
            path: Directory to list, relative to the workspace root (defaults to root).
        """
        out = self._safe(lambda: self._require_tools().list_dir(path))
        self._record("list_dir", {"path": path}, out)
        return out

    def apply_patch(self, path: str, find: str, replace: str) -> str:
        """Replace exact text in a workspace file (use this for small edits).

        Args:
            path: Path to the file to edit, relative to the workspace root.
            find: The exact text to search for in the file.
            replace: The text to substitute in place of every match.
        """
        out = self._safe(lambda: self._require_tools().apply_patch(path, find, replace))
        self._record("apply_patch", {"path": path, "find": find}, out)
        return out

    def run_tests(self, node_ids: str = "") -> str:
        """Run the workspace's own pytest suite and return a text summary.

        This runs only the tests visible in your workspace, never the held-out
        grading suite.

        Args:
            node_ids: Space-separated pytest targets; empty runs the whole suite.
        """
        out = self._safe(lambda: self._run_tests(node_ids))
        self._record("run_tests", {"node_ids": node_ids}, out)
        return out

    # -- internals (underscore → TRL skips these) ----------------------------

    def _run_tests(self, node_ids: str) -> str:
        surface = self._surface or {}
        fn = surface.get("run_tests")
        if not callable(fn):
            return "error: this world exposes no run_tests tool"
        targets = node_ids.split() or None
        res = fn(targets)
        if not isinstance(res, Mapping):
            return str(res)
        ok = bool(res.get("ok"))
        head = (
            f"tests {'passed' if ok else 'failed'} "
            f"(returncode={res.get('returncode')}, "
            f"isolation={res.get('isolation')})"
        )
        stdout = str(res.get("stdout") or "").strip()
        body = stdout[-_OUTPUT_TAIL:] if stdout else "(no output)"
        return f"{head}\n{body}"

    def _finalize(self) -> None:
        """Stop + grade the episode, caching the report and scalar reward.

        Idempotent: the reward func may read ``env.reward`` more than once, and
        ``stop_episode`` itself caches, so a double read is safe.
        """
        if self._finalized or self._handle is None:
            self._finalized = True
            return
        self._finalized = True
        report = self.service.stop_episode(self._handle)
        self.report = report
        self.reward = self.reward_fn(report).scalar

    def _resolve_snapshot(self, snapshot_id: str | None) -> Snapshot:
        if snapshot_id is not None:
            snapshot = self.snapshots.get(snapshot_id)
            if snapshot is None:
                raise KeyError(f"unknown snapshot_id {snapshot_id!r}")
            return snapshot
        if len(self.snapshots) == 1:
            return next(iter(self.snapshots.values()))
        raise ValueError(
            "reset() needs a snapshot_id when multiple snapshots are registered"
        )

    def _require_tools(self) -> FileWorkspaceTools:
        if self._tools is None:
            raise WorkspaceError("reset() has not been called")
        return self._tools

    @staticmethod
    def _safe(fn: Callable[[], str]) -> str:
        try:
            return fn()
        except Exception as exc:  # fail-soft: a bad tool call costs reward only
            return f"error: {exc}"

    def _record(self, tool: str, args: Mapping[str, Any], result: str) -> None:
        turn = AgentTurn(
            tool_calls=({"tool": tool, "args": dict(args)},),
            tool_results=({"output": result},),
        )
        self.turns.append(turn)
        if self._handle is not None:
            self.service.record_turn(self._handle, turn)


def build_grpo_dataset(
    snapshot: Snapshot,
    *,
    repeat: int = 1,
) -> list[dict[str, Any]]:
    """Turn a snapshot's tasks into GRPO prompt rows.

    One row per task (optionally ``repeat``-ed so a round has enough prompts):
    ``{"prompt": [{"role": "user", "content": ...}], "snapshot_id", "task_id"}``.
    ``snapshot_id`` / ``task_id`` ride along as dataset columns — TRL forwards
    them to ``reset`` (which episode to start) and to the reward func, and they
    tag the exported trajectory to the exact (possibly evolved) world. Torch-free
    by design; the live example wraps the rows in a ``datasets.Dataset``.
    """
    rows: list[dict[str, Any]] = []
    for _ in range(max(1, repeat)):
        for task in snapshot.tasks:
            rows.append(
                {
                    "prompt": [{"role": "user", "content": _task_prompt(task)}],
                    "snapshot_id": snapshot.snapshot_id,
                    "task_id": task.id,
                }
            )
    return rows


def _task_prompt(task: TaskSpec) -> str:
    return f"{task.instruction}\n\n{_TOOL_GUIDE}"


def make_reward_func() -> Callable[..., list[float]]:
    """Return a TRL-shaped ``reward_func(prompts, completions, ...)``.

    In the agentic path TRL passes the rollouts' ``environments``; this finalizes
    each (lazily stopping + grading the episode) and returns
    ``[env.reward, ...]`` in order. All reward logic is the pack's structured
    grade shaped by ``episode_reward`` — the trainer only *reads* it.
    """

    def reward_func(
        prompts: object = None,
        completions: object = None,
        completion_ids: object = None,
        *,
        environments: Sequence[OpenRangeEnv] | None = None,
        **kwargs: object,
    ) -> list[float]:
        rewards: list[float] = []
        for env in environments or ():
            env._finalize()
            rewards.append(float(env.reward))
        return rewards

    return reward_func


def make_environment_factory(
    pack: Pack,
    snapshots: Sequence[Snapshot],
    run_root: str | Path,
    *,
    reward_fn: Callable[[EpisodeReport], Reward] = episode_reward,
) -> Callable[[], OpenRangeEnv]:
    """Build the zero-arg factory TRL calls once per rollout slot.

    Each call gets its own ``EpisodeService`` under a unique subdir, so the N
    envs in a GRPO generation batch are fully isolated. The factory closes over
    one round's ``snapshots`` (often a single, current world); the curriculum
    re-roots the next round by re-building the dataset + factory against the
    evolved snapshot.
    """
    snap_map = {s.snapshot_id: s for s in snapshots}
    base = Path(run_root)
    base.mkdir(parents=True, exist_ok=True)

    def factory() -> OpenRangeEnv:
        env_root = base / f"env-{uuid.uuid4().hex[:8]}"
        service = EpisodeService(pack, env_root)
        return OpenRangeEnv(service=service, snapshots=snap_map, reward_fn=reward_fn)

    return factory


def env_trajectory(env: OpenRangeEnv) -> Trajectory:
    """Export an env's last episode as a ``snapshot_id``-tagged ``Trajectory``.

    Finalizes the episode first if the reward was never read, so a caller can
    export trajectories without the reward func having run.
    """
    env._finalize()
    if env.report is None:
        raise RuntimeError("no completed episode to export; call reset() first")
    return episode_trajectory(env.report, env.turns)


def reward_variance_policy(
    reports: Sequence[EpisodeReportLike],
    *,
    epsilon: float = 1e-9,
    harden_mean: float = 0.5,
) -> Direction | None:
    """Evolve only when GRPO's gradient has collapsed.

    GRPO learns from the *spread* of a group's rewards, so a round whose
    ``episode_reward`` scalars are all (near-)equal yields no advantage signal.
    When the spread collapses this nudges the frontier toward the side that
    revives it — ``harden`` if the group is mostly solving, ``soften`` if mostly
    failing. While the spread is alive it returns ``None`` (hold the world). It
    reads the dense scalar when a concrete ``EpisodeReport`` is present, else
    falls back to the binary ``passed`` gate — a strict refinement of
    ``direction_from_reports`` keyed on what the trainer actually consumes.
    """
    if not reports:
        return None
    scalars = [_report_scalar(r) for r in reports]
    mean = sum(scalars) / len(scalars)
    variance = sum((s - mean) ** 2 for s in scalars) / len(scalars)
    if variance > epsilon:
        return None
    return "harden" if mean >= harden_mean else "soften"


def _report_scalar(report: EpisodeReportLike) -> float:
    if isinstance(report, EpisodeReport):
        return episode_reward(report).scalar
    return 1.0 if report.passed else 0.0
