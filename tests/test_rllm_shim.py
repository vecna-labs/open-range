"""The rLLM adapter maps a real OpenRange rollout onto rLLM's training shapes.

Every seam is a real component, never a test double: a real deterministic
``Sampler``, a real host shell, a real PROCESS-backed webapp the agent reaches
over HTTP, the real verifier's grade, and rLLM's real pydantic ``Episode`` /
``Step`` / ``EvalOutput`` types (importable on CPU). The live ``AgentTrainer``
itself is not exercised here — that GPU path is the example — but the data the
trainer consumes is built and asserted end to end. Skipped where ``rllm`` is not
installed, so the default suite stays dependency-free.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from cyber_webapp import WebappPack
from openrange_pack_sdk import Snapshot
from openrange_rllm import (
    GatewaySampler,
    agent_rollout_to_episode,
    build_rllm_dataset_rows,
    make_evaluator,
    make_rollout,
    snapshot_resolver,
)

from openrange import EpisodeService, SampleResult, run_agent
from openrange.core.admit import admit
from openrange.core.sandbox import CommandResult

_SHELL = "```run_shell\necho probe\n```"
_FINISH = "```finish\ndone\n```"


class _Script:
    """A real deterministic ``Sampler``: one fenced reply per turn (last repeats)."""

    def __init__(self, *replies: str) -> None:
        self._replies = replies
        self._turn = 0

    def complete(self, prompt: str, *, system: str | None = None) -> SampleResult:
        reply = self._replies[min(self._turn, len(self._replies) - 1)]
        self._turn += 1
        return SampleResult(reply)


class _HostRun:
    """A real run capability backed by the host shell — the test's binding for a
    PROCESS world. It really runs commands."""

    def run(self, command: str, *, timeout: float = 120.0) -> CommandResult:
        done = subprocess.run(
            ["bash", "-lc", command],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return CommandResult(done.returncode, done.stdout + done.stderr)

    def close(self) -> None:
        return None


def _host_bind(_surface: object) -> _HostRun:
    return _HostRun()


def _snapshot() -> Snapshot:
    snap = admit(
        WebappPack(),
        manifest={
            "pack": {"id": "webapp"},
            "runtime": {"tick": {"mode": "off"}},
            "npc": [],
            "seed": 7,
            "loot": {"file": 1, "db": 0},
            "vuln": {"pin": [{"kind": "command_injection"}]},
        },
        max_repairs=3,
    )
    assert isinstance(snap, Snapshot), snap
    return snap


def _pentest(snap: Snapshot) -> str:
    return next(t for t in snap.tasks if t.meta.get("family") == "webapp.pentest").id


def test_agent_rollout_to_episode_maps_turns(tmp_path: Path) -> None:
    pytest.importorskip("rllm")
    from rllm.types import Episode

    snap = _snapshot()
    svc = EpisodeService(WebappPack(), tmp_path)
    try:
        rollout = run_agent(
            svc,
            snap,
            _Script(_SHELL, _FINISH),
            bind_run=_host_bind,
            task_id=_pentest(snap),
        )
    finally:
        svc.close()
    episode = agent_rollout_to_episode(rollout)
    assert isinstance(episode, Episode)
    assert len(episode.trajectories) == 1
    steps = episode.trajectories[0].steps
    assert len(steps) == len(rollout.steps) == 2
    assert steps[0].action == "echo probe"
    assert steps[0].observation == rollout.steps[0].output
    assert steps[0].done is False
    assert steps[0].logprobs == []
    assert steps[-1].action == "finish"
    assert steps[-1].done is True
    assert episode.is_correct == rollout.success
    assert episode.artifacts["reward"] == rollout.reward.scalar
    assert episode.trajectories[0].reward == rollout.reward.scalar
    assert set(episode.metrics) == {
        "reached_endpoint",
        "extracted_anything",
        "matched_flag",
    }
    assert episode.termination_reason is None
    assert episode.artifacts["terminal_reason"] == rollout.terminal_reason


def test_make_rollout_drives_a_real_episode(tmp_path: Path) -> None:
    pytest.importorskip("rllm")
    from rllm.types import AgentConfig, Episode, Task

    snap = _snapshot()
    pentest = _pentest(snap)
    svc = EpisodeService(WebappPack(), tmp_path)
    flow = make_rollout(
        svc,
        lambda _task: (snap, pentest),
        bind_run=_host_bind,
        sampler_factory=lambda _config: _Script(_SHELL, _FINISH),
    )
    try:
        episode = flow.run(
            Task(id=pentest, instruction="probe the target"),
            AgentConfig(base_url="http://unused", model="x", session_uid="t"),
        )
    finally:
        svc.close()
    assert isinstance(episode, Episode)
    assert episode.trajectories[0].steps
    assert episode.artifacts["task_id"] == pentest
    assert isinstance(episode.is_correct, bool)


def test_make_evaluator_surfaces_the_grade() -> None:
    pytest.importorskip("rllm")
    from rllm.types import Episode

    evaluate = make_evaluator()
    graded = evaluate(
        {},
        Episode(
            is_correct=True,
            artifacts={"reward": 0.75, "components": {"passed": 1.0, "leak": 0.5}},
        ),
    )
    assert graded.reward == 0.75
    assert graded.is_correct is True
    assert {signal.name for signal in graded.signals} == {"passed", "leak"}

    bare = evaluate({}, Episode(is_correct=False, artifacts={}))
    assert bare.reward == 0.0
    assert bare.is_correct is False
    assert bare.signals == []


def test_gateway_sampler_calls_the_endpoint(
    chat_server: Callable[..., Any], chat_completion: Callable[[str], str]
) -> None:
    pytest.importorskip("rllm")
    from rllm.types import AgentConfig

    def respond(_path: str, _method: str) -> tuple[int, str]:
        return 200, chat_completion("hi from gateway")

    with chat_server(respond) as base_url:
        config = AgentConfig(base_url=base_url, model="test-model", session_uid="t")
        sampler = GatewaySampler(config)
        result = sampler.complete("ping", system="be brief")
    assert isinstance(result, SampleResult)
    assert result.text == "hi from gateway"
    assert result.logprobs is None


def test_build_rllm_dataset_rows_carries_resolution_keys() -> None:
    snap = _snapshot()
    pentest = _pentest(snap)
    rows = build_rllm_dataset_rows([snap], family="webapp.pentest")
    assert rows
    assert {row["task_id"] for row in rows} == {pentest}
    assert all(row["snapshot_id"] == snap.snapshot_id for row in rows)
    assert all(row["instruction"] and row["id"] for row in rows)


def test_snapshot_resolver_maps_task_back() -> None:
    pytest.importorskip("rllm")
    from rllm.types import Task

    snap = _snapshot()
    row = build_rllm_dataset_rows([snap], family="webapp.pentest")[0]
    resolve = snapshot_resolver([snap])
    resolved, task_id = resolve(
        Task(id="opaque-uuid", instruction=row["instruction"], metadata=row)
    )
    assert resolved.snapshot_id == snap.snapshot_id
    assert task_id == row["task_id"]
