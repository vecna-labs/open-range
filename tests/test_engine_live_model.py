"""The OpenAI-style backend, against a real server — gated, never mocked.

The construction test needs only ``httpx`` (no network). The generation and
full-episode tests need a real OpenAI-style endpoint: set ``OPENRANGE_LIVE_MODEL_URL``
(and optionally ``OPENRANGE_LIVE_MODEL``) — same gating as ``OPENRANGE_LIVE_TRL``.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
from cyber_webapp import WebappPack
from openrange_pack_sdk import Backing, Snapshot
from openrange_trl import make_environment_factory
from openrange_trl.engine import Action, AsyncRollout, OpenAIBackend, batch

from openrange.core.admit import admit
from openrange.training import Trajectory

_MODEL_URL = os.environ.get("OPENRANGE_LIVE_MODEL_URL")
_MODEL = os.environ.get("OPENRANGE_LIVE_MODEL", "default")
needs_model = pytest.mark.skipif(
    not _MODEL_URL,
    reason="set OPENRANGE_LIVE_MODEL_URL to a real OpenAI-style endpoint",
)


def test_openai_backend_builds_the_endpoint_and_headers() -> None:
    httpx = pytest.importorskip("httpx")
    client = httpx.AsyncClient()
    backend = OpenAIBackend("http://srv:8000/", "m", api_key="k", client=client)
    assert backend._url == "http://srv:8000/v1/chat/completions"
    assert backend._headers == {"Authorization": "Bearer k"}
    assert backend._model == "m"
    asyncio.run(client.aclose())


@needs_model
def test_openai_backend_generates_against_a_real_server() -> None:
    async def go() -> Action:
        backend = OpenAIBackend(str(_MODEL_URL), _MODEL)
        try:
            return await backend.generate(
                [{"role": "user", "content": "Reply with the single word: hi."}], []
            )
        finally:
            await backend.aclose()

    action = asyncio.run(go())
    assert action is not None  # a ToolCall or a Finish came back


def shell(surface: Mapping[str, Any], command: str) -> str:
    """Run a shell command on the agent's own sandbox machine.

    Args:
        command: the shell command to run.
    """
    return str(surface["run"](command).output)


def submit(surface: Mapping[str, Any], flag: str) -> str:
    """Submit the recovered flag for grading.

    Args:
        flag: the flag value to submit.
    """
    path = Path(str(surface["solver_root"])) / "result.json"
    path.write_text(json.dumps({"flag": flag}), encoding="utf-8")
    return "submitted"


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


@needs_model
@pytest.mark.skipif(not _docker_available(), reason="docker engine not reachable")
def test_a_real_model_drives_a_sandboxed_episode(tmp_path: Path) -> None:
    # The full slice end to end: a real model, in its own sandbox, drives a real cyber
    # episode via the engine. Asserts the MECHANICS (a graded trajectory comes back),
    # not that the model solves it — solving is a model-capability matter, not the seam.
    snap = admit(
        WebappPack(),
        manifest={
            "pack": {"id": "webapp"},
            "runtime": {"tick": {"mode": "off"}},
            "npc": [],
            "seed": 7,
            "loot_shapes": {"db": 1, "file": 0},
            "vuln_kinds": {"sql_injection": 1},
        },
    )
    assert isinstance(snap, Snapshot), snap
    task_id = next(t.id for t in snap.tasks if t.meta.get("family") == "webapp.pentest")
    factory = make_environment_factory(
        WebappPack(),
        [snap],
        tmp_path / "envs",
        tools=[shell, submit],
        backing=Backing.CONTAINER,
        sandbox=True,
    )

    async def go() -> list[Trajectory]:
        backend = OpenAIBackend(str(_MODEL_URL), _MODEL)
        rollout = AsyncRollout(
            factory,
            policy=backend.generate,
            snapshot_id=snap.snapshot_id,
            task_id=task_id,
            max_iters=6,
        )
        try:
            return await batch([rollout], concurrency=1)
        finally:
            await backend.aclose()

    [trajectory] = asyncio.run(go())
    assert 0.0 <= trajectory.reward.scalar <= 1.0
    assert trajectory.snapshot_id == snap.snapshot_id
