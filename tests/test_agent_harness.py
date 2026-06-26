"""The shared agent harness: one loop drives training and evaluation.

Every seam here is a real component, never a test double: the sampler is a real
deterministic policy (the :class:`~openrange.agent.Sampler` analog of the
reference solver), the run capability is a real host shell, and the world is a
real PROCESS-backed webapp the agent reaches over real HTTP. The pure tests pin
the loop branches; one integration test drives the real reference exploit and
asserts the independent consequence verifier observed the flag cross the wire.
"""

from __future__ import annotations

import asyncio
import re
import subprocess
from collections.abc import Callable
from pathlib import Path
from urllib.request import urlopen

import pytest
from cyber_webapp import WebappPack
from cyber_webapp.realize_admit import cmdi_exploit_and_benign
from openrange_pack_sdk import Snapshot, TaskSpec

from openrange.agent import (
    AgentError,
    SampleResult,
    agent_briefing,
    arun_rollouts,
    parse_action,
    run_agent,
    run_rollouts,
    run_shell,
)
from openrange.core.admit import admit
from openrange.core.episode import AgentTurn, EpisodeService
from openrange.core.sandbox import CommandResult
from openrange.runtime import EpisodeContext, OpenRangeRun, RunConfig

Reply = Callable[[str], str]


class _Policy:
    """A real ``Sampler``: a deterministic policy that answers with one function
    per turn (the last repeats). With ``logprobs`` it fills the token-level signal
    a training sampler carries; without, it is an eval sampler. It is a real
    protocol implementation, the sampler analog of the reference solver."""

    def __init__(self, *replies: Reply, logprobs: bool = False) -> None:
        self._replies = replies
        self._logprobs = logprobs
        self._turn = 0

    def complete(self, prompt: str, *, system: str | None = None) -> SampleResult:
        reply = self._replies[min(self._turn, len(self._replies) - 1)]
        self._turn += 1
        text = reply(prompt)
        if not self._logprobs:
            return SampleResult(text)
        count = len(text.split())
        return SampleResult(
            text,
            completion_token_ids=tuple(range(count)),
            logprobs=tuple(-0.1 for _ in range(count)),
        )


class _Fixed:
    """A real stateless Sampler — safe to call concurrently; one fixed reply."""

    def __init__(self, reply: str) -> None:
        self._reply = reply

    def complete(self, prompt: str, *, system: str | None = None) -> SampleResult:
        return SampleResult(self._reply)


class _HostRun:
    """A real run capability backed by the host shell — a sandbox without docker
    hardening, the test's binding for a PROCESS world. It really runs commands."""

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


def _finish(answer: str = "done") -> Reply:
    return lambda _prompt: f"```finish\n{answer}\n```"


def _shell(command: str) -> Reply:
    return lambda _prompt: f"```bash\n{command}\n```"


def _cmdi_snapshot() -> Snapshot:
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


def _pentest_task_id(snap: Snapshot) -> str:
    return next(t for t in snap.tasks if t.meta.get("family") == "webapp.pentest").id


def _exploit_reply(snap: Snapshot) -> Reply:
    exploit_req, _benign = cmdi_exploit_and_benign(snap.graph)

    def reply(prompt: str) -> str:
        match = re.search(r"running at (\S+)", prompt)
        assert match is not None, prompt
        target = f"{match.group(1)}{exploit_req.path}"
        if exploit_req.method == "POST":
            curl = (
                f"curl -s -X POST -H 'Content-Type: {exploit_req.content_type}' "
                f"--data '{exploit_req.body}' '{target}'"
            )
        else:
            curl = f"curl -s '{target}'"
        return f"```bash\n{curl}\n```"

    return reply


def test_real_exploit_is_run_and_the_leak_is_observed(tmp_path: Path) -> None:
    snap = _cmdi_snapshot()
    flag = str(snap.graph.nodes["secret_flag"].attrs["value_ref"])
    svc = EpisodeService(WebappPack(), tmp_path)
    try:
        rollout = run_agent(
            svc,
            snap,
            _Policy(_exploit_reply(snap), _finish("recovered")),
            bind_run=_host_bind,
            task_id=_pentest_task_id(snap),
        )
    finally:
        svc.close()
    assert rollout.terminal_reason == "finished"
    assert len(rollout.steps) == 2
    assert rollout.steps[0].command is not None
    assert flag in (rollout.steps[0].output or "")
    assert rollout.steps[-1].command is None
    leaked = rollout.report.final_state.get("leaked_secret_ids") or ()
    assert "secret_flag" in leaked
    assert isinstance(rollout.success, bool)


def test_max_turns_caps_the_loop_and_carries_train_signal(tmp_path: Path) -> None:
    snap = _cmdi_snapshot()
    svc = EpisodeService(WebappPack(), tmp_path)
    try:
        rollout = run_agent(
            svc,
            snap,
            _Policy(_shell("echo probe"), logprobs=True),
            bind_run=_host_bind,
            task_id=_pentest_task_id(snap),
            max_turns=2,
        )
    finally:
        svc.close()
    assert rollout.terminal_reason == "max_turns"
    assert len(rollout.steps) == 2
    assert all(step.command == "echo probe" for step in rollout.steps)
    assert all("probe" in (step.output or "") for step in rollout.steps)
    assert rollout.steps[0].sample.logprobs is not None
    assert rollout.steps[0].sample.completion_token_ids is not None
    assert rollout.trajectory.task_id == rollout.task_id


def test_finish_terminates_with_an_eval_sampler(tmp_path: Path) -> None:
    snap = _cmdi_snapshot()
    svc = EpisodeService(WebappPack(), tmp_path)
    try:
        rollout = run_agent(
            svc,
            snap,
            _Policy(_finish("nothing to report")),
            bind_run=_host_bind,
            task_id=_pentest_task_id(snap),
        )
    finally:
        svc.close()
    assert rollout.terminal_reason == "finished"
    assert len(rollout.steps) == 1
    assert rollout.steps[0].command is None
    assert rollout.steps[0].sample.logprobs is None
    assert rollout.success is False


def test_run_rollouts_runs_every_task_then_a_subset(tmp_path: Path) -> None:
    snap = _cmdi_snapshot()
    pentest = _pentest_task_id(snap)
    svc = EpisodeService(WebappPack(), tmp_path)
    try:
        every = run_rollouts(svc, snap, _Policy(_finish()), bind_run=_host_bind)
        subset = run_rollouts(
            svc, snap, _Policy(_finish()), bind_run=_host_bind, task_ids=[pentest]
        )
    finally:
        svc.close()
    assert len(every) == len(snap.tasks)
    assert [r.task_id for r in subset] == [pentest]


def test_arun_rollouts_overlaps_concurrent_episodes(tmp_path: Path) -> None:
    snap = _cmdi_snapshot()
    pentest = _pentest_task_id(snap)
    svc = EpisodeService(WebappPack(), tmp_path)
    try:
        rollouts = asyncio.run(
            arun_rollouts(
                svc,
                snap,
                _Fixed("```finish\ndone\n```"),
                bind_run=_host_bind,
                task_ids=[pentest, pentest, pentest],
                max_concurrency=3,
            )
        )
    finally:
        svc.close()
    assert len(rollouts) == 3
    assert all(r.task_id == pentest for r in rollouts)
    assert all(r.terminal_reason == "finished" for r in rollouts)


def test_parse_action_reads_blocks_and_falls_back_to_finish() -> None:
    shell = parse_action("intro\n```bash\ncurl -s http://x/\n```\noutro")
    assert shell.tool == "run_shell"
    assert shell.command == "curl -s http://x/"
    done = parse_action("```finish\nthe answer\n```")
    assert done.tool == "finish"
    assert done.command == "the answer"
    bare = parse_action("just prose, no block")
    assert bare.tool == "finish"
    assert bare.command == "just prose, no block"


def test_parse_action_accepts_standard_shell_fences_and_takes_the_last() -> None:
    # A shell action is the markdown code fence the model is trained to emit:
    # ```bash / ```sh / ```shell / ```console / ```zsh all read as one command.
    for lang in ("bash", "sh", "shell", "console", "zsh"):
        act = parse_action(f"thinking...\n```{lang}\nsubmit it\n```")
        assert act.tool == "run_shell", lang
        assert act.command == "submit it"
    # An illustrative block before the real action is not executed in its place —
    # the last recognized block is the one the model settled on.
    settled = parse_action(
        "maybe\n```bash\necho first\n```\non reflection:\n```bash\necho second\n```"
    )
    assert settled.tool == "run_shell"
    assert settled.command == "echo second"


def test_run_shell_requires_a_bound_run_capability() -> None:
    with pytest.raises(AgentError, match="run"):
        run_shell({}, "echo hi")


def _briefing_task() -> TaskSpec:
    return TaskSpec(
        id="t0",
        instruction="Recover the hidden admin flag.",
        entrypoints=("ep.backup",),
        goal_nodes=("secret.flag",),
        feasibility_check="webapp.pentest",
        success_check="webapp.pentest",
    )


def test_briefing_gives_an_http_world_its_url() -> None:
    ctx = EpisodeContext(
        task=_briefing_task(),
        surface={"base_url": "http://127.0.0.1:51991"},
    )
    briefing = agent_briefing(ctx)
    assert _briefing_task().instruction in briefing
    assert "http://127.0.0.1:51991" in briefing
    assert "over HTTP" in briefing


def test_briefing_gives_a_code_world_its_workspace() -> None:
    ctx = EpisodeContext(task=_briefing_task(), surface={"solver_root": "/tmp/ws"})
    briefing = agent_briefing(ctx)
    assert "/tmp/ws" in briefing
    assert "over HTTP" not in briefing


def test_briefing_is_just_the_instruction_for_an_opaque_world() -> None:
    ctx = EpisodeContext(task=_briefing_task(), surface={"mcp_endpoint": "stdio://x"})
    assert agent_briefing(ctx) == _briefing_task().instruction


def test_briefing_delivers_the_live_target_through_run_episode(tmp_path: Path) -> None:
    run = OpenRangeRun(RunConfig(tmp_path / "run", dashboard=False))
    snapshot = run.build(
        {
            "pack": {"id": "webapp"},
            "runtime": {"tick": {"mode": "off"}},
            "npc": [],
            "seed": 0,
        }
    )
    assert isinstance(snapshot, Snapshot), snapshot
    task = next(t for t in snapshot.tasks if t.meta.get("family") == "webapp.pentest")
    seen: dict[str, bool] = {}

    def solve(ctx: EpisodeContext) -> AgentTurn:
        brief = agent_briefing(ctx)
        seen["url_in_brief"] = ctx.base_url in brief
        with urlopen(ctx.base_url + "/", timeout=10) as resp:
            seen["reached"] = resp.status == 200
        return AgentTurn(message="probed")

    run.run_episode(snapshot, solve, task_id=task.id)
    assert seen["url_in_brief"]
    assert seen["reached"]
