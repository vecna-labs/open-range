"""The TRL adapter turns USER-supplied tools into the policy's tool surface.

Tools are brought by the caller (the user's harness), bound to the world surface,
and presented to TRL by method reflection. These prove the seam end to end with no
model: the synthesized methods carry the schema TRL reads, a *custom* tool the
adapter has never seen works against a live world (real BYO), and name collisions
are rejected. No mocks — a real cyber episode boots behind each tool call.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import pytest
from cyber_webapp import WebappPack
from openrange_pack_sdk import Snapshot
from openrange_trl import EpisodeEnv, Tool

from examples.tools import WEB_TOOLS, http_get, submit
from openrange.core.admit import admit
from openrange.core.episode import EpisodeService

_MANIFEST = {
    "pack": {"id": "webapp"},
    "runtime": {"tick": {"mode": "off"}},
    "npc": [],
    "seed": 0,
    "loot_shapes": {"db": 1, "file": 0},
    "vuln_kinds": {"sql_injection": 1},
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


def test_user_tools_reflect_with_the_schema_trl_reads(make_env: Any) -> None:
    get_json_schema = pytest.importorskip("transformers.utils").get_json_schema

    env = make_env(list(WEB_TOOLS))
    fn = get_json_schema(env.http_get)["function"]
    assert fn["name"] == "http_get"
    assert "GET" in fn["description"]
    props = fn["parameters"]["properties"]
    assert set(props) == {"path"}  # the surface-injection param is hidden from TRL
    assert props["path"]["type"] == "string"
    assert props["path"]["description"]  # required by get_json_schema, carried through
    assert get_json_schema(env.submit)["function"]["name"] == "submit"


def test_a_custom_byo_tool_runs_against_a_live_world(
    make_env: Any, snapshot: Snapshot
) -> None:
    seen: list[str] = []

    def recon(surface: Mapping[str, Any], path: str) -> str:
        """Fetch a path on the target and note that it was visited.

        Args:
            path: the request path to fetch.
        """
        seen.append(path)
        return http_get(surface, path)

    # The adapter has never seen `recon`; the user brings it.
    env = make_env([recon, submit])
    env.reset(snapshot_id=snapshot.snapshot_id, task_id=_pentest_task(snapshot).id)
    out = env.recon("/")
    assert out.startswith("status=200")  # the user's tool hit the live server
    assert seen == ["/"]
    assert env.turns[-1].tool_calls[0]["tool"] == "recon"  # logged like any tool


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
    from examples.tools import run_tests

    assert run_tests({}, "").startswith("error:")  # no run_tests in the surface


def test_duplicate_tool_names_are_rejected(make_env: Any) -> None:
    def http_get(surface: Mapping[str, Any], path: str) -> str:
        """A second tool that collides on name.

        Args:
            path: x.
        """
        return ""

    with pytest.raises(ValueError, match="duplicate tool"):
        make_env([http_get, http_get])
