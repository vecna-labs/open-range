"""Deterministic tests for the torch-free TRL adapter (``openrange_trl``).

No torch, no trl, no LLM. Every seam is driven *at the seam itself* over **real**
SWE episodes (per ``.rules``, no mocks): the actuators mutate a real
``solver_root``, the reward bridge grades the real edited tree through
``episode_reward``, and the variance policy reads real ``EpisodeReport``s. This
proves the integration is correct — it does not measure a model (that is the
gated ``tests/test_trl_live.py``).

Some tests stop a real episode, which shells out to a sandboxed pytest to grade —
the same path the SWE pack's own tests take.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any

import pytest
from openrange_pack_sdk import Backing, EpisodeResult, Snapshot
from openrange_trl import (
    EpisodeEnv,
    build_grpo_dataset,
    env_trajectory,
    make_environment_factory,
    make_reward_func,
    reward_variance_policy,
)
from swe import SwePack
from swe.instances import load_instance

from openrange.core.admit import AdmissionFailure, admit
from openrange.core.curriculum import auto_evolve
from openrange.core.episode import EpisodeReport, EpisodeService

EnvMaker = Callable[[str], tuple[EpisodeEnv, Snapshot]]


def _in_root(surface: Mapping[str, Any], path: str) -> Path:
    root = Path(str(surface["solver_root"])).resolve()
    target = (root / path).resolve()
    if target != root and root not in target.parents:
        raise ValueError(f"path {path!r} escapes the workspace root")
    return target


def read_file(surface: Mapping[str, Any], path: str) -> str:
    """Read a workspace file.

    Args:
        path: Path to the file or directory, relative to the workspace root.
    """
    return _in_root(surface, path).read_text(encoding="utf-8")


def write_file(surface: Mapping[str, Any], path: str, content: str) -> str:
    """Write a file in the workspace.

    Args:
        path: Path to the file or directory, relative to the workspace root.
        content: The full text to write into the file.
    """
    target = _in_root(surface, path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return f"wrote {len(content)} byte(s) to {path}"


def list_dir(surface: Mapping[str, Any], path: str = ".") -> str:
    """List the entries of a workspace directory.

    Args:
        path: Path to the file or directory, relative to the workspace root.
    """
    target = _in_root(surface, path)
    if not target.exists():
        raise ValueError(f"no such directory: {path!r}")
    if target.is_file():
        return path
    names = sorted(p.name + ("/" if p.is_dir() else "") for p in target.iterdir())
    return "\n".join(names) if names else "(empty)"


def apply_patch(surface: Mapping[str, Any], path: str, find: str, replace: str) -> str:
    """Replace exact text in a workspace file.

    Args:
        path: Path to the file or directory, relative to the workspace root.
        find: The exact text to search for in the file.
        replace: The text to substitute for every match.
    """
    target = _in_root(surface, path)
    original = target.read_text(encoding="utf-8")
    if find not in original:
        raise ValueError(f"patch text not found in {path!r}")
    occurrences = original.count(find)
    target.write_text(original.replace(find, replace), encoding="utf-8")
    return f"patched {path} ({occurrences} occurrence(s))"


def run_tests(surface: Mapping[str, Any], node_ids: str = "") -> str:
    """Run the workspace's own pytest suite (never the held-out grader).

    Args:
        node_ids: Space-separated pytest targets; empty runs the whole suite.
    """
    fn = surface.get("run_tests")
    if not callable(fn):
        return "error: this world exposes no run_tests tool"
    res = fn(node_ids.split() or None)
    verb = "passed" if res.get("ok") else "failed"
    head = f"tests {verb} (returncode={res.get('returncode')})"
    return f"{head}\n{str(res.get('stdout') or '').strip() or '(no output)'}"


FILE_TOOLS = (read_file, write_file, list_dir, apply_patch, run_tests)


def _admit(instance: str) -> Snapshot:
    result = admit(SwePack(), manifest={"instance": instance}, max_repairs=0)
    assert not isinstance(result, AdmissionFailure), result
    return result


@pytest.fixture
def make_env(tmp_path: Path) -> Iterator[EnvMaker]:
    """Yield a factory for (env, snapshot) pairs over a real ``EpisodeService``;
    every service is closed on teardown so no grading subprocess leaks."""
    services: list[EpisodeService] = []

    def _make(instance: str) -> tuple[EpisodeEnv, Snapshot]:
        snapshot = _admit(instance)
        service = EpisodeService(SwePack(), tmp_path / f"svc{len(services)}")
        services.append(service)
        env = EpisodeEnv(
            service=service,
            snapshots={snapshot.snapshot_id: snapshot},
            tools=FILE_TOOLS,
        )
        return env, snapshot

    yield _make
    for service in services:
        service.close()


def _solve(env: EpisodeEnv, instance: str) -> None:
    for path, content in load_instance(instance).gold_files.items():
        env.write_file(path, content)


def test_factories_thread_backing_into_each_rollout_service(tmp_path: Path) -> None:
    # The factory builds each service before any realize, so the backing it threads
    # through is observable without booting docker — and the default must stay PROCESS.
    snapshot = _admit("calc_sum")
    default_factory = make_environment_factory(
        SwePack(), [snapshot], tmp_path / "proc", tools=FILE_TOOLS
    )
    container_factory = make_environment_factory(
        SwePack(),
        [snapshot],
        tmp_path / "cont",
        tools=FILE_TOOLS,
        backing=Backing.CONTAINER,
    )
    proc_env = default_factory()
    cont_env = container_factory()
    try:
        assert proc_env.service.backing is Backing.PROCESS
        assert cont_env.service.backing is Backing.CONTAINER
    finally:
        proc_env.service.close()
        cont_env.service.close()


def test_base_env_resets_with_no_tools(tmp_path: Path) -> None:
    # The env is usable with no tools at all: reset starts a real episode and
    # returns the live interface contract (here, the SWE workspace listing).
    snapshot = _admit("calc_sum")
    service = EpisodeService(SwePack(), tmp_path / "base")
    env = EpisodeEnv(service=service, snapshots={snapshot.snapshot_id: snapshot})
    try:
        assert env.reset().startswith("Workspace ready")
    finally:
        service.close()


def test_env_close_releases_the_service(tmp_path: Path) -> None:
    # TRL builds envs from a factory and never closes them, so env.close() is the
    # only hook to stop the live episode + warm worlds. Without it, CONTAINER rollouts
    # leak a container stack per env. close() is idempotent and safe before any reset.
    snapshot = _admit("calc_sum")
    service = EpisodeService(SwePack(), tmp_path / "close")
    env = EpisodeEnv(service=service, snapshots={snapshot.snapshot_id: snapshot})
    env.reset(snapshot_id=snapshot.snapshot_id)
    assert service._episodes  # a live episode is registered
    env.close()
    assert not service._episodes  # closed: nothing left running
    env.close()  # idempotent


class TestBuildDataset:
    def test_rows_carry_prompt_and_tags(self) -> None:
        snapshot = _admit("calc_sum")
        rows = build_grpo_dataset(snapshot)
        assert len(rows) == len(snapshot.tasks)
        ids = {t.id for t in snapshot.tasks}
        for row in rows:
            assert row["snapshot_id"] == snapshot.snapshot_id
            assert row["task_id"] in ids
            assert row["prompt"][0]["role"] == "user"
            assert isinstance(row["prompt"][0]["content"], str)

    def test_prompt_carries_the_task_instruction(self) -> None:
        snapshot = _admit("calc_sum")
        row = build_grpo_dataset(snapshot)[0]
        head = snapshot.tasks[0].instruction.strip().splitlines()[0]
        assert head in row["prompt"][0]["content"]

    def test_repeat_multiplies_rows(self) -> None:
        snapshot = _admit("calc_sum")
        base = build_grpo_dataset(snapshot)
        assert len(build_grpo_dataset(snapshot, repeat=3)) == 3 * len(base)


class TestEnvLifecycle:
    def test_reset_returns_live_workspace_listing(self, make_env: EnvMaker) -> None:
        env, snapshot = make_env("calc_sum")
        obs = env.reset(snapshot_id=snapshot.snapshot_id, task_id=snapshot.tasks[0].id)
        assert "calc" in obs  # the base tree ships a calc/ package

    def test_reset_picks_sole_snapshot_without_id(self, make_env: EnvMaker) -> None:
        env, _ = make_env("calc_sum")
        assert "calc" in env.reset()  # single registered snapshot → no id needed

    def test_gold_overlay_solves_and_rewards(self, make_env: EnvMaker) -> None:
        env, snapshot = make_env("calc_sum")
        env.reset(snapshot_id=snapshot.snapshot_id)
        _solve(env, "calc_sum")
        env._finalize()
        assert env.report is not None
        assert env.report.passed
        assert env.reward == 1.0

    def test_untouched_base_does_not_resolve(self, make_env: EnvMaker) -> None:
        env, snapshot = make_env("calc_sum")
        env.reset(snapshot_id=snapshot.snapshot_id)
        env._finalize()
        assert env.report is not None
        assert not env.report.passed
        # The bug (FAIL_TO_PASS) stays red, but the trivially-passing
        # PASS_TO_PASS test floors the dense reward at 0.5 — not zero, and
        # strictly below the gold's 1.0, so the group still has spread.
        assert env.reward == 0.5

    def test_build_partial_credit_is_dense(self, make_env: EnvMaker) -> None:
        # notes_app (swe.build): the bare skeleton fails every tier -> 0.0, but
        # the gold overlay greens all -> 1.0. Proves the dense seam end to end.
        env, snapshot = make_env("notes_app")
        env.reset(snapshot_id=snapshot.snapshot_id)
        env._finalize()
        assert env.reward == 0.0
        solved, snap2 = make_env("notes_app")
        solved.reset(snapshot_id=snap2.snapshot_id)
        _solve(solved, "notes_app")
        solved._finalize()
        assert solved.reward == 1.0

    def test_tool_calls_are_recorded(self, make_env: EnvMaker) -> None:
        env, snapshot = make_env("calc_sum")
        env.reset(snapshot_id=snapshot.snapshot_id)
        env.list_dir(".")
        env.read_file("calc/core.py")
        assert [t.tool_calls[0]["tool"] for t in env.turns] == [
            "list_dir",
            "read_file",
        ]

    def test_run_tests_tool_executes(self, make_env: EnvMaker) -> None:
        env, snapshot = make_env("calc_sum")
        env.reset(snapshot_id=snapshot.snapshot_id)
        out = env.run_tests("")
        assert "tests" in out  # "tests passed"/"tests failed" summary line
        assert env.turns[-1].tool_calls[0]["tool"] == "run_tests"

    def test_apply_patch_tool_edits_the_workspace(self, make_env: EnvMaker) -> None:
        env, snapshot = make_env("calc_sum")
        env.reset(snapshot_id=snapshot.snapshot_id)
        env.write_file("scratch.py", "return 0\n")
        out = env.apply_patch("scratch.py", "0", "1")
        assert "occurrence" in out
        assert env.read_file("scratch.py") == "return 1\n"  # edit landed on disk
        assert "apply_patch" in [t.tool_calls[0]["tool"] for t in env.turns]

    def test_bad_tool_call_fails_soft(self, make_env: EnvMaker) -> None:
        env, snapshot = make_env("calc_sum")
        env.reset(snapshot_id=snapshot.snapshot_id)
        out = env.write_file("../escape.py", "pwned")
        assert out.startswith("error")
        root = env.service.solver_root(env._handle)  # type: ignore[arg-type]
        assert not (root.parent / "escape.py").exists()

    def test_unknown_snapshot_id_raises(self, make_env: EnvMaker) -> None:
        env, _ = make_env("calc_sum")
        with pytest.raises(KeyError):
            env.reset(snapshot_id="sha256:does-not-exist")

    def test_reset_demands_id_when_multiple_registered(
        self, make_env: EnvMaker
    ) -> None:
        env, _ = make_env("calc_sum")
        other = _admit("notes_app")  # a second, distinct world on the ladder
        env.snapshots[other.snapshot_id] = other
        with pytest.raises(ValueError, match="snapshot_id"):
            env.reset()

    def test_tools_before_reset_fail_soft(self, make_env: EnvMaker) -> None:
        # No reset(): the file tools have no root and there's no surface. Every
        # call must degrade to an error string (never raise) and still be
        # recorded as a turn — a weak model that acts before reset loses reward,
        # not the run.
        env, _ = make_env("calc_sum")
        assert env.read_file("x").startswith("error")
        assert env.run_tests("").startswith("error")
        assert [t.tool_calls[0]["tool"] for t in env.turns] == [
            "read_file",
            "run_tests",
        ]


class TestRewardSpread:
    """The reward is a genuine [0, 1] discriminator over the *tool* path.

    GRPO learns from the spread of a group's rewards, so the integration only
    yields a gradient if different edits earn different grades. These drive the
    ``apply_patch`` tool to each distinct grade ``calc_sum`` admits — proving the
    spread is real and, just as importantly, mapping the trap a weak policy falls
    into: ``return a - b`` appears in *both* ``add`` and ``subtract``, so the
    naive replace-all fixes ``add`` but breaks ``subtract`` and nets right back to
    the 0.5 floor. Only the *targeted* edit reaches 1.0, so "learn to target the
    add block" is exactly the climb the gradient rewards. This is the
    deterministic floor under the live signal the notebook demonstrates at scale.
    """

    def _reward_after(
        self,
        env: EpisodeEnv,
        snapshot: Snapshot,
        edit: Callable[[EpisodeEnv], object],
    ) -> float:
        env.reset(snapshot_id=snapshot.snapshot_id)
        edit(env)
        env._finalize()
        return env.reward

    def test_targeted_fix_reaches_full_reward(self, make_env: EnvMaker) -> None:
        env, snapshot = make_env("calc_sum")
        reward = self._reward_after(
            env,
            snapshot,
            lambda e: e.apply_patch(
                "calc/core.py",
                "def add(a, b):\n    return a - b",
                "def add(a, b):\n    return a + b",
            ),
        )
        assert reward == 1.0
        assert env.report is not None and env.report.passed

    def test_naive_replace_all_nets_the_floor(self, make_env: EnvMaker) -> None:
        env, snapshot = make_env("calc_sum")
        reward = self._reward_after(
            env,
            snapshot,
            lambda e: e.apply_patch("calc/core.py", "return a - b", "return a + b"),
        )
        # FAIL_TO_PASS test_add now greens, but PASS_TO_PASS test_subtract reds:
        # one subgoal each way -> the same 0.5 an untouched workspace earns.
        assert reward == 0.5

    def test_breaking_pass_to_pass_sinks_below_the_floor(
        self, make_env: EnvMaker
    ) -> None:
        env, snapshot = make_env("calc_sum")
        reward = self._reward_after(
            env,
            snapshot,
            lambda e: e.apply_patch(
                "calc/core.py",
                "def subtract(a, b):\n    return a - b",
                "def subtract(a, b):\n    return a * b",
            ),
        )
        # Both tests red now -> below the floor, the bottom of the spread.
        assert reward == 0.0


class TestRewardFunc:
    def test_reward_func_grades_envs_in_order(self, make_env: EnvMaker) -> None:
        solved, snap = make_env("calc_sum")
        solved.reset(snapshot_id=snap.snapshot_id)
        _solve(solved, "calc_sum")
        unsolved, snap2 = make_env("calc_sum")
        unsolved.reset(snapshot_id=snap2.snapshot_id)
        reward_func = make_reward_func()
        # Solved earns the full 1.0; the untouched base floors at 0.5 (its
        # PASS_TO_PASS test passes for free) — distinct, and in env order.
        assert reward_func(environments=[solved, unsolved]) == [1.0, 0.5]

    def test_reward_func_is_idempotent(self, make_env: EnvMaker) -> None:
        env, snap = make_env("calc_sum")
        env.reset(snapshot_id=snap.snapshot_id)
        _solve(env, "calc_sum")
        reward_func = make_reward_func()
        first = reward_func(environments=[env])
        second = reward_func(environments=[env])  # double read is safe
        assert first == second == [1.0]

    def test_reward_func_empty_without_envs(self) -> None:
        assert make_reward_func()(environments=None) == []


class TestTrajectoryExport:
    def test_trajectory_tagged_and_stepped(self, make_env: EnvMaker) -> None:
        env, snapshot = make_env("calc_sum")
        env.reset(snapshot_id=snapshot.snapshot_id)
        env.list_dir(".")
        _solve(env, "calc_sum")
        traj = env_trajectory(env)
        assert traj.snapshot_id == snapshot.snapshot_id
        assert traj.task_id == snapshot.tasks[0].id
        assert traj.success
        assert traj.reward.scalar == 1.0
        assert len(traj.steps) == len(env.turns)
        assert traj.steps[0].tool_calls[0]["tool"] == "list_dir"

    def test_export_without_an_episode_raises(self, make_env: EnvMaker) -> None:
        env, _ = make_env("calc_sum")  # never reset → nothing to export
        with pytest.raises(RuntimeError, match="no completed episode"):
            env_trajectory(env)


class TestEnvFactory:
    def test_factory_isolates_concurrent_envs(self, tmp_path: Path) -> None:
        snapshot = _admit("calc_sum")
        factory = make_environment_factory(
            SwePack(), [snapshot], tmp_path, tools=FILE_TOOLS
        )
        a, b = factory(), factory()
        try:
            a.reset(snapshot_id=snapshot.snapshot_id)
            b.reset(snapshot_id=snapshot.snapshot_id)
            root_a = a.service.solver_root(a._handle)  # type: ignore[arg-type]
            root_b = b.service.solver_root(b._handle)  # type: ignore[arg-type]
            assert root_a != root_b
            a.write_file("calc/core.py", "TAINTED")
            assert b.read_file("calc/core.py") != "TAINTED"
        finally:
            a.service.close()
            b.service.close()


def _report(
    success: bool,
    subgoals: dict[str, bool],
    *,
    task_id: str = "t",
) -> EpisodeReport:
    return EpisodeReport(
        snapshot_id="sha256:test",
        task_id=task_id,
        episode_result=EpisodeResult(success=success, subgoals=subgoals),
    )


class TestVariancePolicy:
    def test_all_solved_collapse_hardens(self) -> None:
        reports = [_report(True, {"a": True}) for _ in range(4)]
        assert reward_variance_policy(reports) == "harden"

    def test_all_failed_collapse_softens(self) -> None:
        reports = [_report(False, {"a": False}) for _ in range(4)]
        assert reward_variance_policy(reports) == "soften"

    def test_mixed_outcomes_hold(self) -> None:
        reports = [_report(True, {"a": True}), _report(False, {"a": False})]
        assert reward_variance_policy(reports) is None

    def test_partial_credit_spread_holds(self) -> None:
        # Both fail, but dense partial credit differs -> spread alive -> hold.
        units_only = _report(False, {"u1": True, "u2": True, "i1": False})
        skeleton = _report(False, {"u1": False, "u2": False, "i1": False})
        assert reward_variance_policy([units_only, skeleton]) is None

    def test_uniform_partial_credit_collapses(self) -> None:
        # Identical partial credit (1/3 each): zero spread, low mean -> soften.
        flat = [_report(False, {"u1": True, "i1": False, "i2": False})] * 3
        assert reward_variance_policy(flat) == "soften"

    def test_empty_is_none(self) -> None:
        assert reward_variance_policy([]) is None

    def test_plugs_into_auto_evolve_noop_for_swe(self) -> None:
        # The policy is a CurriculumPolicy; auto_evolve accepts it. SWE opts out
        # of in-place mutation, so a zero-variance round still yields None — the
        # live curriculum rides the instance ladder instead.
        snapshot = _admit("calc_sum")
        tid = snapshot.tasks[0].id
        reports = [_report(True, {tid: True}, task_id=tid) for _ in range(3)]
        evolved = auto_evolve(
            snapshot, *reports, pack=SwePack(), policy=reward_variance_policy
        )
        assert evolved is None
