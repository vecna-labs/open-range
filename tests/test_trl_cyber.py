"""The cyber webapp pack, trained through the TRL adapter's generic ``EpisodeEnv``.

The live rungs run as a real rollout: each boots the pack on the CONTAINER backing
with ``sandbox=True``, and the agent acts the way a CTF agent does -- its own
``curl`` from its own sandbox, over the network, against the ``http://target:8000``
alias (no HTTP tool is shipped). Pins ``seed=0`` -- a deterministic world whose
pentest oracle is a SQL-injection delivered as a POST body -- so the reward surface
is reproducible.

The reward is the world's held-out verdict over the *HTTP* path: GRPO learns from
the spread of a group's rewards, so the integration only yields a gradient if
different actions earn different grades. ``webapp.pentest`` admits four:

    no request                        -> 0.0    (no subgoal)
    reach the endpoint                -> 0.333  (reached_endpoint)
    reach + submit a wrong value      -> 0.667  (reached + extracted_anything)
    exploit the vuln + submit flag    -> 1.0    (all three -> success)

These lock that surface in, and prove a real exploit-over-HTTP reaches ``1.0``. The
live rungs need docker (the sandbox joins the target's container network); the
dataset/plumbing/evolution checks stay torch-free and docker-free.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any

import pytest
from cyber_webapp import WebappPack
from cyber_webapp.reference_solver import Request, exploit_and_benign
from openrange_pack_sdk import Backing, EpisodeResult, Snapshot, TaskSpec
from openrange_trl import (
    EpisodeEnv,
    build_grpo_dataset,
    env_trajectory,
    make_environment_factory,
    make_reward_func,
    reward_variance_policy,
)

from openrange.core.admit import admit
from openrange.core.curriculum import auto_evolve
from openrange.core.episode import EpisodeReport, EpisodeService


def shell(surface: Mapping[str, Any], command: str) -> str:
    """Run a shell command on your machine and return its output.

    Args:
        command: The shell command line to run (e.g. a curl invocation).
    """
    return str(surface["run"](command).output)


def submit(surface: Mapping[str, Any], content: str) -> str:
    """Submit your final answer; the grader reads result.json.

    Args:
        content: A JSON object carrying the recovered field, e.g. {"flag": "..."}.
    """
    (Path(str(surface["solver_root"])) / "result.json").write_text(
        content, encoding="utf-8"
    )
    return f"submitted {len(content)} byte(s)"


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


def _docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        probe = subprocess.run(
            ["docker", "info"], capture_output=True, timeout=10, check=False
        )
    except Exception:  # noqa: BLE001 - a best-effort probe; any failure means "no"
        return False
    return probe.returncode == 0


gated = pytest.mark.skipif(
    not _docker_available(), reason="docker engine not reachable"
)


@pytest.fixture(scope="module")
def snapshot() -> Snapshot:
    result = admit(WebappPack(), manifest=_MANIFEST)
    assert isinstance(result, Snapshot), result
    return result


@pytest.fixture
def make_env(snapshot: Snapshot, tmp_path: Path) -> Iterator[EnvMaker]:
    # A real rollout env: the world realizes as a container and the agent's tools run
    # in their own sandbox, so each `shell` is the agent's own curl over the network.
    services: list[EpisodeService] = []

    def _make() -> EpisodeEnv:
        svc = EpisodeService(
            WebappPack(), tmp_path / f"svc{len(services)}", backing=Backing.CONTAINER
        )
        services.append(svc)
        return EpisodeEnv(
            service=svc,
            snapshots={snapshot.snapshot_id: snapshot},
            tools=[shell, submit],
            sandbox=True,
        )

    yield _make
    for svc in services:
        svc.close()


@pytest.fixture
def make_process_env(snapshot: Snapshot, tmp_path: Path) -> Iterator[EnvMaker]:
    # A docker-free env for the plumbing checks that never act over HTTP (pre-reset
    # fail-soft, report bookkeeping): PROCESS backing, no sandbox.
    services: list[EpisodeService] = []

    def _make() -> EpisodeEnv:
        svc = EpisodeService(WebappPack(), tmp_path / f"proc{len(services)}")
        services.append(svc)
        return EpisodeEnv(
            service=svc,
            snapshots={snapshot.snapshot_id: snapshot},
            tools=[shell, submit],
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


def _start(env: EpisodeEnv, snapshot: Snapshot, task: TaskSpec) -> str:
    return env.reset(snapshot_id=snapshot.snapshot_id, task_id=task.id)


def _grade(env: EpisodeEnv) -> float:
    env._finalize()
    return env.reward


def _curl(request: Request) -> str:
    # Frame the reference request as the agent's own curl against the in-network alias:
    # a body-shaped class is a POST with its body under content_type, else a GET.
    target = f"http://target:8000{request.path}"
    if request.method == "POST":
        return (
            f"curl -s -X POST -H 'Content-Type: {request.content_type}' "
            f"--data '{request.body or ''}' '{target}'"
        )
    return f"curl -s '{target}'"


def _reach(env: EpisodeEnv, snapshot: Snapshot, task: TaskSpec) -> str:
    return env.shell(f"curl -s 'http://target:8000{_entrypoint_url(snapshot, task)}'")


def _drive_exploit(env: EpisodeEnv, snapshot: Snapshot) -> str:
    exploit, _benign = exploit_and_benign(snapshot.graph, "sql_injection")
    return env.shell(_curl(exploit))


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
    @gated
    def test_reset_exposes_the_live_target(
        self, make_env: EnvMaker, snapshot: Snapshot
    ) -> None:
        task = _pentest_task(snapshot)
        env = make_env()
        obs = _start(env, snapshot, task)
        # The reset observation carries the in-network alias the dataset can't know.
        assert "http://target:" in obs

    @gated
    def test_tools_hit_the_server_and_record_turns(
        self, make_env: EnvMaker, snapshot: Snapshot
    ) -> None:
        task = _pentest_task(snapshot)
        env = make_env()
        _start(env, snapshot, task)
        out = env.shell(
            "curl -s -o /dev/null -w '%{http_code}' 'http://target:8000/'"
        )  # the app always serves "/"
        assert "200" in out
        assert env.submit('{"flag": "probe"}').startswith("submitted")
        assert len(env.turns) == 2  # shell + submit, both logged

    @gated
    def test_bad_get_is_fail_soft(self, make_env: EnvMaker, snapshot: Snapshot) -> None:
        task = _pentest_task(snapshot)
        env = make_env()
        _start(env, snapshot, task)
        # A 404 returns its status, not an exception; the rollout survives.
        out = env.shell(
            "curl -s -o /dev/null -w '%{http_code}' 'http://target:8000/no/such/route'"
        )
        assert "404" in out and "error" not in out

    def test_shell_before_reset_is_fail_soft(self, make_process_env: EnvMaker) -> None:
        # No episode started yet -> no surface -> fail-soft, not a crash.
        assert make_process_env().shell("curl http://target:8000/").startswith("error:")

    def test_submit_before_reset_is_fail_soft(self, make_process_env: EnvMaker) -> None:
        assert make_process_env().submit('{"flag": "x"}').startswith("error:")


class TestRewardRungs:
    """Each reward rung the pentest world admits, driven over real HTTP."""

    @gated
    def test_no_action_floors_at_zero(
        self, make_env: EnvMaker, snapshot: Snapshot
    ) -> None:
        task = _pentest_task(snapshot)
        env = make_env()
        _start(env, snapshot, task)
        assert _grade(env) == 0.0

    @gated
    def test_reaching_the_endpoint_earns_one_third(
        self, make_env: EnvMaker, snapshot: Snapshot
    ) -> None:
        task = _pentest_task(snapshot)
        env = make_env()
        _start(env, snapshot, task)
        # A plain GET to the POST-shaped endpoint is a 405, but it still *reaches* the
        # endpoint -- the exploit (the flag) needs the POST body, the rung below.
        _reach(env, snapshot, task)
        assert _grade(env) == pytest.approx(1 / 3)

    @gated
    def test_wrong_answer_earns_two_thirds(
        self, make_env: EnvMaker, snapshot: Snapshot
    ) -> None:
        task = _pentest_task(snapshot)
        env = make_env()
        _start(env, snapshot, task)
        _reach(env, snapshot, task)
        env.submit(json.dumps({"flag": "not-the-real-flag"}))
        assert _grade(env) == pytest.approx(2 / 3)

    @gated
    def test_exploit_over_http_reaches_full_reward(
        self, make_env: EnvMaker, snapshot: Snapshot
    ) -> None:
        task = _pentest_task(snapshot)
        truth = _truth_flag(snapshot, task)
        env = make_env()
        _start(env, snapshot, task)
        resp = _drive_exploit(env, snapshot)
        # The POST-body SQLi actually exfiltrated the hidden flag over HTTP.
        assert truth in resp
        env.submit(json.dumps({"flag": truth}))
        assert _grade(env) == 1.0
        assert env.report is not None and env.report.passed


class TestSeams:
    @gated
    def test_trajectory_is_snapshot_tagged(
        self, make_env: EnvMaker, snapshot: Snapshot
    ) -> None:
        task = _pentest_task(snapshot)
        env = make_env()
        _start(env, snapshot, task)
        _reach(env, snapshot, task)
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

    @gated
    def test_reward_func_grades_web_envs_in_order(
        self, make_env: EnvMaker, snapshot: Snapshot
    ) -> None:
        task = _pentest_task(snapshot)
        truth = _truth_flag(snapshot, task)

        solved = make_env()
        _start(solved, snapshot, task)
        _drive_exploit(solved, snapshot)
        solved.submit(json.dumps({"flag": truth}))

        floored = make_env()
        _start(floored, snapshot, task)

        rewards = make_reward_func()([], [], environments=[solved, floored])
        assert rewards == [1.0, 0.0]

    def test_reward_func_records_reports_by_world(
        self, make_process_env: EnvMaker, snapshot: Snapshot, tmp_path: Path
    ) -> None:
        # trainer.environments keeps only the last episode per slot, so a
        # multi-world batch needs the collector to read back every world's report.
        # No HTTP action here -- a floored episode per world is enough to key the
        # reports -- so this stays docker-free on the PROCESS backing.
        pack = WebappPack()
        task = _pentest_task(snapshot)
        env_a = make_process_env()
        _start(env_a, snapshot, task)
        _grade(env_a)

        other = admit(pack, manifest={**_MANIFEST, "seed": 1})
        assert isinstance(other, Snapshot)
        other_task = _pentest_task(other)
        svc = EpisodeService(pack, tmp_path / "other")
        env_b = EpisodeEnv(
            service=svc,
            snapshots={other.snapshot_id: other},
            tools=[shell, submit],
        )
        try:
            _start(env_b, other, other_task)
            _grade(env_b)
            collector: dict[tuple[str, str], list[EpisodeReport]] = {}
            make_reward_func(collector)([], [], environments=[env_a, env_b])
            assert set(collector) == {
                (snapshot.snapshot_id, task.id),
                (other.snapshot_id, other_task.id),
            }
            assert collector[(snapshot.snapshot_id, task.id)] == [env_a.report]
            assert collector[(other.snapshot_id, other_task.id)] == [env_b.report]
        finally:
            svc.close()

    @gated
    def test_factory_builds_isolated_envs(
        self, snapshot: Snapshot, tmp_path: Path
    ) -> None:
        factory = make_environment_factory(
            WebappPack(),
            [snapshot],
            tmp_path / "factory-envs",
            tools=[shell, submit],
            backing=Backing.CONTAINER,
            sandbox=True,
        )
        env = factory()
        assert isinstance(env, EpisodeEnv)
        task = _pentest_task(snapshot)
        try:
            obs = env.reset(snapshot_id=snapshot.snapshot_id, task_id=task.id)
            assert "http://target:" in obs
            code = env.shell(
                "curl -s -o /dev/null -w '%{http_code}' 'http://target:8000/'"
            )
            assert "200" in code
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

    @gated
    def test_curriculum_chains_distinct_worlds_across_rounds(
        self, snapshot: Snapshot, tmp_path: Path
    ) -> None:
        # Solving every round collapses the spread -> harden -> a fresh admitted
        # world; the lineage advances as a chain of distinct snapshots round over
        # round, not one fixed task. This is the loop the cyber notebook teaches.
        pack = WebappPack()
        snap = snapshot
        chain = [snap.snapshot_id]
        for i in range(3):
            task = _pentest_task(snap)
            reports = []
            for j in range(2):
                svc = EpisodeService(
                    pack, tmp_path / f"r{i}s{j}", backing=Backing.CONTAINER
                )
                env = EpisodeEnv(
                    service=svc,
                    snapshots={snap.snapshot_id: snap},
                    tools=[shell, submit],
                    sandbox=True,
                )
                env.reset(snapshot_id=snap.snapshot_id, task_id=task.id)
                _drive_exploit(env, snap)
                env.submit(json.dumps({"flag": _truth_flag(snap, task)}))
                env._finalize()
                assert env.report is not None and env.report.passed
                reports.append(env.report)
                svc.close()
            evolved = auto_evolve(
                snap, *reports, pack=pack, policy=reward_variance_policy
            )
            assert evolved is not None and evolved.snapshot_id not in chain
            snap = evolved
            chain.append(snap.snapshot_id)
        assert len(set(chain)) == 4  # root + three evolved, all distinct
