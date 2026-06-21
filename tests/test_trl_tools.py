"""The TRL adapter turns USER-supplied tools into the policy's tool surface.

Tools are brought by the caller (the user's harness), bound to the world surface,
and presented to TRL by method reflection. These prove the seam end to end with no
model: the synthesized methods carry the schema TRL reads, a *custom* tool the
adapter has never seen works against a live world (real BYO), and name collisions
are rejected. No mocks — a real cyber episode boots behind each tool call.
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import pytest
from cyber_webapp import WebappPack
from openrange_pack_sdk import Backing, Snapshot
from openrange_trl import EpisodeEnv, Tool

from openrange.core.admit import admit
from openrange.core.episode import EpisodeService


def shell(surface: Mapping[str, Any], command: str) -> str:
    """Run a shell command on your machine and return its output.

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
        content: A JSON object carrying the requested field.
    """
    (Path(str(surface["solver_root"])) / "result.json").write_text(
        content, encoding="utf-8"
    )
    return f"submitted {len(content)} byte(s)"


def run_tests(surface: Mapping[str, Any], node_ids: str = "") -> str:
    """Run the workspace's own pytest suite, never the held-out grader.

    Args:
        node_ids: Space-separated pytest targets; empty runs the whole suite.
    """
    fn = surface.get("run_tests")
    if not callable(fn):
        return "error: this world exposes no run_tests tool"
    res = fn(node_ids.split() or None)
    verdict = "passed" if res.get("ok") else "failed"
    head = f"tests {verdict} (returncode={res.get('returncode')})"
    return f"{head}\n{str(res.get('stdout') or '').strip() or '(no output)'}"


WEB_TOOLS = (shell, submit)

_MANIFEST = {
    "pack": {"id": "webapp"},
    "runtime": {"tick": {"mode": "off"}},
    "npc": [],
    "seed": 0,
    "loot": {"db": 1, "file": 0},
    "vuln": {"pin": [{"kind": "sql_injection"}]},
}


@pytest.fixture(scope="module")
def snapshot() -> Snapshot:
    snap = admit(WebappPack(), manifest=_MANIFEST)
    assert isinstance(snap, Snapshot), snap
    return snap


@pytest.fixture
def make_env(snapshot: Snapshot, tmp_path: Path) -> Iterator[Any]:
    services: list[EpisodeService] = []

    def _make(tools: list[Tool]) -> EpisodeEnv:
        svc = EpisodeService(WebappPack(), tmp_path / f"svc{len(services)}")
        services.append(svc)
        return EpisodeEnv(
            service=svc, snapshots={snapshot.snapshot_id: snapshot}, tools=tools
        )

    yield _make
    for svc in services:
        svc.close()


def _pentest_task(snapshot: Snapshot) -> Any:
    return next(t for t in snapshot.tasks if t.meta.get("family") == "webapp.pentest")


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


def test_user_tools_reflect_with_the_schema_trl_reads(make_env: Any) -> None:
    get_json_schema = pytest.importorskip("transformers.utils").get_json_schema

    env = make_env(list(WEB_TOOLS))
    fn = get_json_schema(env.shell)["function"]
    assert fn["name"] == "shell"
    props = fn["parameters"]["properties"]
    assert set(props) == {"command"}  # the surface-injection param is hidden from TRL
    assert props["command"]["type"] == "string"
    assert props["command"]["description"]  # carried through; get_json_schema needs it
    assert get_json_schema(env.submit)["function"]["name"] == "submit"


@gated
def test_a_custom_byo_tool_runs_against_a_live_world(
    snapshot: Snapshot, tmp_path: Path
) -> None:
    seen: list[str] = []

    def recon(surface: Mapping[str, Any], path: str) -> str:
        """Fetch a path on the target and note that it was visited.

        Args:
            path: the request path to fetch.
        """
        seen.append(path)
        return shell(surface, f"curl -s 'http://target:8000{path}'")

    # The adapter has never seen `recon`; the user brings it. It runs in the agent's
    # own sandbox against a real CONTAINER world over the network.
    svc = EpisodeService(WebappPack(), tmp_path / "svc", backing=Backing.CONTAINER)
    env = EpisodeEnv(
        service=svc,
        snapshots={snapshot.snapshot_id: snapshot},
        tools=[recon, submit],
        sandbox=True,
    )
    try:
        env.reset(snapshot_id=snapshot.snapshot_id, task_id=_pentest_task(snapshot).id)
        env.recon("/")
        assert seen == ["/"]  # the brought tool ran
        assert env.turns[-1].tool_calls[0]["tool"] == "recon"  # logged like any tool
    finally:
        svc.close()


def test_a_tool_with_a_default_arg_is_optional(
    make_env: Any, snapshot: Snapshot
) -> None:
    def note(surface: Mapping[str, Any], text: str = "ok") -> str:
        """Record a note.

        Args:
            text: the note text (optional).
        """
        return f"noted: {text}"

    env = make_env([note, submit])
    env.reset(snapshot_id=snapshot.snapshot_id, task_id=_pentest_task(snapshot).id)
    assert env.note() == "noted: ok"  # default preserved
    assert env.note("hi") == "noted: hi"


def test_initial_observation_falls_back_for_an_opaque_surface(make_env: Any) -> None:
    # A world that declares neither base_url nor solver_root still resets cleanly.
    env = make_env([])
    env._surface = {}
    assert env._initial_observation() == "Environment ready. Use the available tools."


def test_run_tests_tool_reports_when_world_has_no_runner() -> None:
    assert run_tests({}, "").startswith("error:")  # no run_tests in the surface


def test_duplicate_tool_names_are_rejected(make_env: Any) -> None:
    def probe(surface: Mapping[str, Any], path: str) -> str:
        """A tool brought twice under the same name.

        Args:
            path: x.
        """
        return ""

    with pytest.raises(ValueError, match="duplicate tool"):
        make_env([probe, probe])
