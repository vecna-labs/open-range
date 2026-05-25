from __future__ import annotations

import asyncio
import contextlib
import json
import textwrap
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Protocol, cast
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest
from cyber_webapp import WebappPack

import openrange as OR
from openrange.core.admit_loop import Snapshot, admit
from openrange.dashboard import (
    DashboardArtifactLog,
    DashboardEvent,
    DashboardHTTPServer,
    DashboardView,
    EventBridge,
    dashboard_event_from_mapping,
)
from openrange.dashboard import (
    read_dashboard_events as read_dashboard_artifact_events,
)
from openrange.llm import LLMBackendError, parse_json_object, run_codex

# Manifest shape under the new pack/admission contract: the pack id is
# the entry-point name registered in packs/cyber_webapp/pyproject.toml
# (``"webapp"``). A ``world`` block is optional — packs that don't
# pre-bake topology under that key (the webapp pack doesn't) leave it
# blank; the dashboard topology view falls back to the world graph for
# its services / edges / zones projection.
MANIFEST = {
    "world": {"goal": "find the admin flag", "title": "Ops Portal"},
    "pack": {"id": "webapp"},
    "seed": 0,
}


def _admit(manifest: dict[str, object] | None = None) -> Snapshot:
    """Admit the webapp pack against ``manifest``, asserting success.

    Centralizes the cast-or-fail pattern so individual tests don't
    branch on ``AdmissionFailure``; admission against the real
    ``WebappPack`` with ``seed=0`` is expected to succeed.
    """
    result = admit(WebappPack(), manifest if manifest is not None else MANIFEST)
    assert isinstance(result, Snapshot), result
    return result


class LineReader(Protocol):
    def readline(self) -> bytes: ...


def executable(tmp_path: Path, name: str, body: str) -> Path:
    path = tmp_path / name
    path.write_text(
        "#!/usr/bin/env python3\n" + textwrap.dedent(body),
        encoding="utf-8",
    )
    path.chmod(0o755)
    return path


@contextlib.contextmanager
def running_dashboard(view: DashboardView) -> Iterator[str]:
    server = DashboardHTTPServer(("127.0.0.1", 0), view)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host = cast(str, server.server_address[0])
        port = server.server_address[1]
        yield f"http://{host}:{port}"
    finally:
        view.bridge.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def read_http_json(url: str, *, method: str = "GET") -> dict[str, object]:
    request = Request(url, method=method)
    with urlopen(request, timeout=5) as response:
        return cast(dict[str, object], json.loads(response.read().decode()))


def read_sse_message(response: LineReader) -> dict[str, str]:
    fields: dict[str, str] = {}
    while True:
        line = response.readline().decode().rstrip("\r\n")
        if not line:
            return fields
        name, value = line.split(": ", 1)
        fields[name] = value


def wait_for_turn_count(
    view: DashboardView,
    task_id: str,
    count: int,
) -> list[dict[str, object]]:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        turns = view.turns(task_id)
        if len(turns) >= count:
            return turns
        time.sleep(0.05)
    return view.turns(task_id)


def read_dashboard_events(run_root: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in (run_root / "dashboard.events.jsonl")
        .read_text(
            encoding="utf-8",
        )
        .splitlines()
    ]


def wait_for_dashboard_action(
    run_root: Path,
    action: dict[str, object],
) -> dict[str, object]:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        for event in read_dashboard_events(run_root):
            data = cast(dict[str, object], event["data"])
            if data.get("action") == action:
                return event
        time.sleep(0.05)
    raise AssertionError(f"dashboard action was not persisted: {action}")


def test_llm_request_validation_and_json_parser() -> None:
    request = OR.LLMRequest("hello", system="system", json_schema={"type": "object"})
    assert request.as_prompt() == "system\n\nhello"
    assert OR.LLMRequest("hello").as_prompt() == "hello"
    assert parse_json_object('{"ok": true}') == {"ok": True}

    with pytest.raises(OR.LLMRequestError, match="JSON serializable"):
        OR.LLMRequest("bad", json_schema={"x": object()})
    with pytest.raises(LLMBackendError, match="invalid JSON"):
        parse_json_object("{")
    with pytest.raises(LLMBackendError, match="not an object"):
        parse_json_object("[]")


def test_codex_backend_runs_local_command_without_schema(tmp_path: Path) -> None:
    command = executable(
        tmp_path,
        "plain_backend.py",
        """
        import sys

        print(sys.stdin.read().strip().upper())
        """,
    )
    result = OR.CodexBackend(command=command, model="local", timeout=5).complete(
        OR.LLMRequest("hello", system="system"),
    )

    assert result == OR.LLMResult("SYSTEM\n\nHELLO")


def test_codex_backend_reads_schema_output_from_local_command(
    tmp_path: Path,
) -> None:
    command = executable(
        tmp_path,
        "json_backend.py",
        """
        import json
        import sys
        from pathlib import Path

        schema_path = Path(sys.argv[sys.argv.index("--output-schema") + 1])
        output_path = Path(sys.argv[sys.argv.index("--output-last-message") + 1])
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        output_path.write_text(
            json.dumps({"schema": schema["type"], "prompt": sys.stdin.read()}),
            encoding="utf-8",
        )
        print("ignored stdout")
        """,
    )
    result = OR.CodexBackend(command=command, model="local", timeout=5).complete(
        OR.LLMRequest("return json", json_schema={"type": "object"}),
    )

    assert result.parsed_json == {"schema": "object", "prompt": "return json"}
    assert json.loads(result.text)["schema"] == "object"


def test_codex_backend_reports_process_failures(tmp_path: Path) -> None:
    stderr_command = executable(
        tmp_path,
        "stderr_failure.py",
        """
        import sys

        print("boom", file=sys.stderr)
        raise SystemExit(7)
        """,
    )
    stdout_command = executable(
        tmp_path,
        "stdout_failure.py",
        """
        print("bad stdout")
        raise SystemExit(3)
        """,
    )
    silent_command = executable(
        tmp_path,
        "silent_failure.py",
        """
        raise SystemExit(4)
        """,
    )

    with pytest.raises(LLMBackendError, match="boom") as stderr_error:
        OR.CodexBackend(command=stderr_command, model="local", timeout=5).complete(
            OR.LLMRequest("hello"),
        )
    with pytest.raises(LLMBackendError, match="bad stdout"):
        OR.CodexBackend(command=stdout_command, model="local", timeout=5).complete(
            OR.LLMRequest("hello"),
        )
    with pytest.raises(LLMBackendError, match="no output"):
        OR.CodexBackend(command=silent_command, model="local", timeout=5).complete(
            OR.LLMRequest("hello"),
        )

    assert stderr_error.value.returncode == 7


def test_codex_backend_requires_schema_output_file(tmp_path: Path) -> None:
    command = executable(
        tmp_path,
        "missing_output.py",
        """
        import sys

        sys.stdin.read()
        """,
    )

    with pytest.raises(LLMBackendError, match="did not write"):
        OR.CodexBackend(command=command, model="local", timeout=5).complete(
            OR.LLMRequest("return json", json_schema={"type": "object"}),
        )


def test_run_codex_reports_os_errors_and_timeouts(tmp_path: Path) -> None:
    sleeper = executable(
        tmp_path,
        "sleeper.py",
        """
        import time

        time.sleep(5)
        """,
    )

    with pytest.raises(LLMBackendError, match="No such file|not found"):
        run_codex(
            [str(tmp_path / "missing-command")],
            input_text="hello",
            cwd=None,
            timeout=1,
        )
    with pytest.raises(LLMBackendError, match="timed out"):
        run_codex(
            [str(sleeper)],
            input_text="hello",
            cwd=None,
            timeout=0.01,
        )


def test_dashboard_http_server_can_start_without_snapshot() -> None:
    """Empty dashboard view exposes the new-shape topology / lineage.

    The pre-refactor topology had ``artifact_paths`` (per-artifact
    files admitted alongside the snapshot); the new admission shape
    doesn't materialize artifacts at admit time so the key is gone.
    The pre-refactor lineage had ``admission`` and ``nodes`` (a parent
    chain of LineageNode); the new shape replaces those with
    ``lineage`` (the flat provenance mapping ``admit()`` builds) plus
    ``history`` (the ordered ``BuildEvent`` tuple) and a forward-compat
    ``parent_snapshot_id`` hook.
    """
    view = DashboardView()

    empty_topology = {
        "snapshot_id": None,
        "world": {},
        "tasks": [],
        "services": [],
        "edges": [],
        "zones": [],
        "users": [],
        "green_personas": [],
    }
    empty_lineage = {
        "snapshot_id": None,
        "lineage": {},
        "history": [],
        "parent_snapshot_id": None,
    }
    assert view.topology() == empty_topology
    assert view.lineage() == empty_lineage
    assert view.briefing() == {
        "snapshot_id": None,
        "title": "",
        "goal": "",
        "entrypoints": [],
        "missions": [],
    }

    with running_dashboard(view) as base_url:
        briefing = read_http_json(base_url + "/api/briefing")
        actors = cast(list[dict[str, object]], read_http_json(base_url + "/api/actors"))
        topology = read_http_json(base_url + "/api/topology")
        state = read_http_json(base_url + "/api/state")
        lineage = read_http_json(base_url + "/api/lineage")
        inspection = read_http_json(base_url + "/api/inspect")
        reset = read_http_json(base_url + "/api/episode/reset", method="POST")

        assert topology == empty_topology
        assert briefing["snapshot_id"] is None
        assert actors == []
        assert state["snapshot_id"] is None
        assert state["status"] == "waiting_for_snapshot"
        assert state["latest_event"] is None
        assert lineage == empty_lineage
        assert inspection["topology"] == topology
        assert reset == {
            "status": "waiting_for_snapshot",
            "snapshot_id": None,
            "topology": topology,
        }


def test_dashboard_http_server_serves_static_assets_and_routes(
    tmp_path: Path,
) -> None:
    snapshot = _admit()
    view = DashboardView(snapshot)
    view.record_event(
        "agent_step",
        actor="red",
        target="webapp",
        data={"action": "browse"},
    )

    with running_dashboard(view) as base_url:
        with urlopen(base_url + "/", timeout=5) as response:
            html = response.read().decode()
        with urlopen(base_url + "/static/dashboard.css", timeout=5) as response:
            css = response.read().decode()
        with urlopen(base_url + "/static/dashboard.js", timeout=5) as response:
            dashboard_js = response.read().decode()

        briefing = read_http_json(base_url + "/api/briefing")
        topology = read_http_json(base_url + "/api/topology?ignored=1")
        lineage = read_http_json(base_url + "/api/lineage")
        state = read_http_json(base_url + "/api/state")
        narration = read_http_json(base_url + "/api/narrate")
        play = read_http_json(base_url + "/api/episode/play", method="POST")

        assert "OpenRange Dashboard" in html
        assert 'id="sim-canvas"' in html
        assert "/static/dashboard.css" in html
        assert "/static/dashboard.js" in html
        # Slim editorial chrome: brand bar + footer narrator + the
        # collapsible inspector rail with Build/World/Lineage/Activity
        # tabs. The actor panel is the rail's `actor` tab — toggled in
        # only when an actor is clicked.
        assert "OpenRange" in html
        assert 'id="topbar"' in html
        assert 'id="footbar-narrator"' in html
        assert 'id="rail"' in html
        assert 'data-tab="build"' in html
        assert 'data-tab="world"' in html
        assert 'data-tab="lineage"' in html
        assert 'data-tab="activity"' in html
        assert 'data-tab="actor"' in html
        assert 'id="build-banner"' in html
        assert 'id="toast-stack"' in html
        # Three.js scene + light theme tokens.
        assert "THREE.WebGLRenderer" in dashboard_js
        assert "--bg-0" in css
        assert ".dash-callout" in css
        assert ".rail-tab" in css
        assert briefing["snapshot_id"] == snapshot.snapshot_id
        assert topology["snapshot_id"] == snapshot.snapshot_id
        # The pre-refactor ``snapshot.admission`` attribute is replaced
        # by ``snapshot.lineage`` (the flat provenance Mapping) plus
        # ``snapshot.history`` (the ordered ``BuildEvent`` tuple); the
        # dashboard surfaces both under their own keys.
        assert lineage["lineage"] == dict(snapshot.lineage)
        assert lineage["history"] == [event.to_dict() for event in snapshot.history]
        assert cast(list[dict[str, object]], state["events"])[0]["data"] == {
            "action": "browse",
        }
        assert narration == {"narration": "red agent_step webapp"}
        assert play == {"status": "playing"}

        for request in (
            Request(base_url + "/missing"),
            Request(base_url + "/static/missing.css"),
            Request(base_url + "/static/../events.py"),
            Request(base_url + "/api/episode/missing", method="POST"),
        ):
            with pytest.raises(HTTPError) as error:
                urlopen(request, timeout=5).read()
            assert error.value.code == 404
            assert json.loads(error.value.read().decode()) == {"error": "not found"}


def test_dashboard_http_server_streams_events_and_narration(
    tmp_path: Path,
) -> None:
    snapshot = _admit()
    view = DashboardView(snapshot)
    first = view.record_event("agent_step", actor="red", target="webapp")

    with running_dashboard(view) as base_url:
        events = urlopen(base_url + "/api/events/stream", timeout=5)
        try:
            message = read_sse_message(events)
            second = view.record_event("env_turn", actor="agent", target="webapp")
            live_message = read_sse_message(events)
        finally:
            events.close()

        assert message["id"] == first.id
        assert message["event"] == "agent_step"
        assert json.loads(message["data"])["actor"] == "red"
        assert live_message["id"] == second.id
        assert json.loads(live_message["data"])["type"] == "env_turn"

        narration = urlopen(base_url + "/api/narrate/stream", timeout=5)
        try:
            narration_message = read_sse_message(narration)
        finally:
            narration.close()

        assert narration_message["id"] == first.id
        assert narration_message["event"] == "narration"
        assert json.loads(narration_message["data"]) == {
            "narration": "red agent_step webapp\nagent env_turn webapp",
        }


def test_dashboard_artifact_log_writes_builder_steps(tmp_path: Path) -> None:
    event_log = tmp_path / "dashboard.events.jsonl"
    state_path = tmp_path / "dashboard.json"
    log = DashboardArtifactLog(event_log, state_path, reset=True)

    first = log.record_builder_step(
        "build_started",
        {"pack_id": "cyber.webapp"},
    )
    with event_log.open("a", encoding="utf-8") as handle:
        handle.write("not-json\n")
        handle.write("[]\n")
    reopened = DashboardArtifactLog(event_log, state_path, reset=False)
    second = reopened.record_builder_step("builder_finished")
    malformed = dashboard_event_from_mapping(
        {
            "id": "bad",
            "type": "builder_step",
            "actor": "builder",
            "target": "snapshot",
            "time": "later",
            "data": [],
        },
    )
    events = read_dashboard_artifact_events(event_log)
    state = json.loads(state_path.read_text(encoding="utf-8"))

    assert read_dashboard_artifact_events(tmp_path / "missing.events.jsonl") == []
    assert first.id == "1:builder_step"
    assert second.id == "2:builder_step"
    assert malformed.time == 0.0
    assert malformed.data == {}
    assert [event.data["step"] for event in events] == [
        "build_started",
        "builder_finished",
    ]
    assert state["builder"]["steps"] == [
        {"pack_id": "cyber.webapp", "step": "build_started"},
        {"step": "builder_finished"},
    ]
    assert state["topology"] == {}

    live_event_log = tmp_path / "live-dashboard.events.jsonl"
    live_state = tmp_path / "live-dashboard.json"
    live_event_log.write_text(
        json.dumps(
            DashboardEvent("1:note", "note", "system", "dashboard", 0.0, {}).as_dict(),
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    view = DashboardView(
        event_log_path=live_event_log,
        state_path=live_state,
        reset_artifacts=False,
    )
    assert view.state()["event_count"] == 1


def test_dashboard_view_can_open_persisted_run_artifacts(
    tmp_path: Path,
) -> None:
    """Persisted dashboard.json round-trips into a stored-state DashboardView.

    Fixture uses the new wire shape: ``entrypoints`` is a list of
    node-id strings (the pre-refactor ``Entrypoint(kind, target)``
    object is gone), ``artifact_paths`` is no longer surfaced on the
    topology view (admission stopped materializing per-artifact files),
    and ``lineage`` carries a flat ``{lineage, history, parent_snapshot_id}``
    set instead of the pre-refactor ``admission`` / ``nodes`` chain.
    """
    event_log = tmp_path / "dashboard.events.jsonl"
    state_path = tmp_path / "dashboard.json"
    event_log.write_text(
        json.dumps(
            DashboardEvent(
                "1:env_turn",
                "env_turn",
                "agent",
                "webapp",
                0.0,
                {
                    "actor_kind": "agent",
                    "action": {"method": "GET"},
                    "target": "webapp",
                },
            ).as_dict(),
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    state_path.write_text(
        json.dumps(
            {
                "topology": {
                    "snapshot_id": "saved",
                    "world": {"title": "Saved Ops", "goal": "inspect"},
                    "tasks": [
                        {
                            "id": "task-1",
                            "instruction": "Inspect the saved run",
                            "entrypoints": ["webapp"],
                        },
                    ],
                    "services": [
                        {"id": "webapp", "kind": "http", "zone": "episode"},
                    ],
                    "edges": [],
                    "zones": ["episode"],
                    "users": [],
                    "green_personas": [],
                },
                "lineage": {
                    "snapshot_id": "saved",
                    "lineage": {},
                    "history": [],
                    "parent_snapshot_id": None,
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    view = DashboardView(
        event_log_path=event_log,
        state_path=state_path,
        reset_artifacts=False,
    )

    assert view.topology()["snapshot_id"] == "saved"
    assert view.briefing()["title"] == "Saved Ops"
    # Stored-state entrypoints can't resolve ``node_kind`` (no live
    # graph) so the dashboard records it as an empty string — the
    # ``stored_entrypoints`` helper handles the degradation.
    assert view.briefing()["entrypoints"] == [
        {"task_id": "task-1", "node_id": "webapp", "node_kind": ""},
    ]
    assert view.lineage()["snapshot_id"] == "saved"
    assert view.state()["snapshot_id"] == "saved"
    assert view.state()["status"] == "paused"
    assert view.state()["event_count"] == 1

    state_path.write_text("{", encoding="utf-8")
    assert DashboardView(state_path=state_path, reset_artifacts=False).briefing() == {
        "snapshot_id": None,
        "title": "",
        "goal": "",
        "entrypoints": [],
        "missions": [],
    }

    state_path.write_text(
        json.dumps(
            {
                "topology": {
                    "snapshot_id": "sparse",
                    "world": {"title": "Sparse"},
                    "tasks": [
                        "bad",
                        {"id": "task-2", "entrypoints": "bad"},
                        {"id": "task-3", "entrypoints": ["bad"]},
                    ],
                },
            },
        ),
        encoding="utf-8",
    )
    # Malformed stored tasks still surface as missions (with empty
    # instruction strings) but skip the non-string entrypoints — the
    # new ``stored_task_entrypoints`` helper requires str ids, so
    # ``["bad"]`` becomes a valid entrypoint while non-list / string
    # ``entrypoints: "bad"`` is dropped.
    sparse_briefing = DashboardView(
        state_path=state_path,
        reset_artifacts=False,
    ).briefing()
    assert sparse_briefing["snapshot_id"] == "sparse"
    assert sparse_briefing["title"] == "Sparse"
    assert sparse_briefing["goal"] == ""
    assert sparse_briefing["missions"] == [
        {"task_id": "task-2", "instruction": ""},
        {"task_id": "task-3", "instruction": ""},
    ]
    # task-3's ``["bad"]`` is a valid list-of-strings entrypoint, so
    # it surfaces; task-2's ``"bad"`` (not a list) does not.
    assert sparse_briefing["entrypoints"] == [
        {"task_id": "task-3", "node_id": "bad", "node_kind": ""},
    ]


def test_dashboard_records_actor_turns_from_env_actors(tmp_path: Path) -> None:
    snapshot = _admit()
    view = DashboardView(snapshot)
    agent_turn = OR.ActorTurn(
        task_id="find_admin_flag",
        actor_id="agent",
        actor_kind="agent",
        target="webapp",
        action={"method": "GET", "path": "/robots.txt"},
        observation={"status": 200},
        state={"result": {}},
        metadata={"entrypoint": "http"},
    )
    npc_turn = OR.ActorTurn(
        task_id="find_admin_flag",
        actor_id="mentor",
        actor_kind="npc",
        target="agent",
        action={"say": "inspect public hints"},
    )
    system_turn = OR.ActorTurn(
        task_id="audit",
        actor_id="clock",
        actor_kind="system",
        target="world",
        action={"tick": 1},
        state={"time": 1, "continuity": 0.65, "blue_reward": 0.1, "red_reward": 0.25},
    )

    note = view.record_event("note", actor="system", target="dashboard")
    first = view.record_turn(agent_turn)
    second = view.record_turn(npc_turn)
    third = view.record_turn(system_turn)
    state = view.state()
    events = cast(list[dict[str, object]], state["events"])

    assert agent_turn.as_dict() == {
        "task_id": "find_admin_flag",
        "actor_id": "agent",
        "actor_kind": "agent",
        "target": "webapp",
        "action": {"method": "GET", "path": "/robots.txt"},
        "observation": {"status": 200},
        "state": {"result": {}},
        "metadata": {"entrypoint": "http"},
    }
    assert npc_turn.as_dict()["observation"] is None
    assert npc_turn.as_dict()["metadata"] == {}
    assert note.as_dict()["data"] == {}
    assert first.as_dict()["data"] == agent_turn.as_dict()
    assert second.id == "3:env_turn"
    assert third.actor == "clock"
    assert [event["type"] for event in events] == [
        "note",
        "env_turn",
        "env_turn",
        "env_turn",
    ]
    assert view.turns() == [
        agent_turn.as_dict(),
        npc_turn.as_dict(),
        system_turn.as_dict(),
    ]
    assert view.turns("find_admin_flag") == [
        agent_turn.as_dict(),
        npc_turn.as_dict(),
    ]
    assert view.turns("missing") == []
    assert state["health"] == {
        "uptime": 65.0,
        "defense": 90.0,
        "integrity": 75.0,
    }
    actors = view.actors()
    assert [actor["actor_id"] for actor in actors] == [
        "agent",
        "clock",
        "mentor",
        "system",
    ]
    assert actors[0]["actor_kind"] == "agent"
    assert actors[0]["latest_action"] == {"method": "GET", "path": "/robots.txt"}
    assert actors[0]["targets"] == ["webapp"]
    assert actors[-1]["latest_event_type"] == "note"
    inspection = view.inspect()
    assert inspection["actors"] == actors
    assert inspection["turns"] == view.turns()
    assert inspection["state"] == view.state()


def test_openrange_run_can_disable_dashboard_artifacts(tmp_path: Path) -> None:
    """``dashboard=False`` keeps the run root free of dashboard artifacts.

    The pre-refactor assertion that ``agent_root.parent.parent ==
    run_root`` is no longer load-bearing: the new
    ``WebappRuntimeHandle`` realizer owns its own scratch dir (under
    the system temp tree) and surfaces ``agent_root`` from there, so
    the harness's run root only carries the dashboard / log artifacts.
    """
    run_root = tmp_path / "run"
    run = OR.OpenRangeRun(OR.RunConfig(run_root, dashboard=False))
    snapshot = run.build(MANIFEST)
    task = snapshot.tasks[0]
    svc = run.episode_service(snapshot)

    try:
        handle = svc.start_episode(snapshot, task.id)
        agent_root = svc.agent_root(handle)
    finally:
        svc.close()

    assert agent_root.exists()
    assert not (run_root / "dashboard.events.jsonl").exists()
    assert not (run_root / "dashboard.json").exists()


def test_run_config_starts_live_dashboard_internally(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    run = OR.OpenRangeRun(OR.RunConfig(run_root, dashboard_port=0))
    snapshot = run.build(MANIFEST)
    task = snapshot.tasks[0]
    svc = run.episode_service(snapshot)
    dashboard_handle = run.serve_dashboard(snapshot, port=0)

    try:
        svc.start_episode(snapshot, task.id)
        svc.start_episode(snapshot, task.id)
        state = read_http_json(dashboard_handle.url + "/api/state")
    finally:
        svc.close()
        dashboard_handle.close()

    assert state["snapshot_id"] == snapshot.snapshot_id
    # Two start_episode calls × 2 system turns each = 4 turns
    assert cast(int, state["turn_count"]) >= 2


def test_episode_each_start_gives_fresh_roots(tmp_path: Path) -> None:
    from openrange.dashboard import DashboardView

    snapshot = _admit()
    task = snapshot.tasks[0]
    run_root = tmp_path / "episode"
    run_root.mkdir()
    dashboard = DashboardView(
        snapshot,
        event_log_path=run_root / "dashboard.events.jsonl",
        state_path=run_root / "dashboard.json",
        reset_artifacts=True,
    )
    # ``EpisodeService`` now takes the Pack as the first positional arg
    # (resolved design Q1 — one service per Pack) so a service can
    # never realize a snapshot built by a different pack.
    svc = OR.EpisodeService(WebappPack(), run_root, dashboard=dashboard)
    first = svc.start_episode(snapshot, task.id)
    first_root = svc.agent_root(first)
    marker = first_root / "old.txt"
    marker.write_text("old", encoding="utf-8")
    try:
        second = svc.start_episode(snapshot, task.id)
        second_root = svc.agent_root(second)
    finally:
        svc.close()

    assert second_root.exists()
    assert first_root != second_root
    assert marker.exists()  # first episode's root still has its marker


def test_runtime_error_and_reader_paths(tmp_path: Path) -> None:
    """``EpisodeService.stop_episode`` raises on an unknown episode handle.

    The pre-refactor version of this test also exercised
    ``openrange.core.runtime_helpers`` (process spawning + stdout
    parsing + result/requests file readers) against the old-shape
    ``Entrypoint(kind, target, metadata)`` object and the old
    ``snapshot.world`` flat dict. Both seams are gone under the new
    pack shape: the entrypoint is now a node-id string on the task,
    the world is a graph not a flat dict, and the realizer (a
    ``RuntimeHandle`` returned by ``Pack.realize``) owns process /
    file mechanics directly. The helper-level error paths are
    covered by the pack's own ``WebappRuntimeHandle`` tests; this
    test only keeps the pack-agnostic surface.
    """
    snapshot = _admit()
    task = snapshot.tasks[0]
    svc = OR.EpisodeService(WebappPack(), tmp_path / "episode")

    bogus_handle = OR.EpisodeHandle("missing", snapshot.snapshot_id, task.id)
    with pytest.raises(OR.EpisodeError, match="unknown episode"):
        svc.stop_episode(bogus_handle)


def test_topology_surfaces_personas_from_manifest_when_pack_silent() -> None:
    """Manifest NPC entries with a ``name`` populate ``green_personas``.

    The cyber pack doesn't ship a ``green_personas`` list; without
    this fallback the dashboard scene can't seat persona NPCs at
    their desks before the first tick lands an event.
    """
    manifest = {
        **MANIFEST,
        "npc": [
            # Plain NPC (no ``name``): no persona row.
            {
                "type": "cyber.browsing_user",
                "config": {"cadence_ticks": 3, "paths": ["/"]},
            },
            # Persona NPC (with ``name``): one row per spawn slot.
            {
                "type": "cyber.office_persona",
                "config": {
                    "name": "Alice",
                    "role": "engineer",
                    "title": "Backend Engineer",
                    "tone": "dry, precise",
                    "colleagues": ["Bob"],
                    "home": "svc-web",
                },
            },
            {
                "type": "cyber.office_persona",
                "config": {
                    "name": "Bob",
                    "role": "it_admin",
                    "title": "Sec Eng",
                },
            },
        ],
    }
    snapshot = _admit(manifest)
    view = DashboardView(snapshot)
    topology = view.topology()
    personas = cast(list[dict[str, object]], topology["green_personas"])
    by_name = {p["display_name"]: p for p in personas}
    assert set(by_name) == {"Alice", "Bob"}
    assert by_name["Alice"]["role"] == "engineer"
    assert by_name["Alice"]["title"] == "Backend Engineer"
    assert by_name["Alice"]["tone"] == "dry, precise"
    assert by_name["Alice"]["colleagues"] == ["Bob"]
    assert by_name["Alice"]["home"] == "svc-web"
    assert by_name["Bob"]["role"] == "it_admin"
    assert by_name["Bob"]["colleagues"] == []
    assert by_name["Bob"]["home"] is None


def test_topology_persona_count_matches_manifest_count() -> None:
    """``count`` on a persona entry expands and disambiguates ids."""
    manifest = {
        **MANIFEST,
        "npc": [
            {
                "type": "cyber.office_persona",
                "count": 3,
                "config": {"name": "Triplet", "role": "ops"},
            },
        ],
    }
    snapshot = _admit(manifest)
    view = DashboardView(snapshot)
    personas = cast(
        list[dict[str, object]],
        view.topology()["green_personas"],
    )
    assert len(personas) == 3
    ids = sorted(str(p["id"]) for p in personas)
    assert ids == ["Triplet-1", "Triplet-2", "Triplet-3"]
    assert all(p["role"] == "ops" for p in personas)


def test_topology_skips_npc_entries_without_name() -> None:
    """Plain NPC entries (no ``name``) don't pollute ``green_personas``."""
    manifest = {
        **MANIFEST,
        "npc": [
            {
                "type": "cyber.admin_audit",
                "config": {"audit_path": "/openapi.json"},
            },
            {
                "type": "cyber.browsing_user",
                "config": {"paths": ["/"]},
            },
        ],
    }
    snapshot = _admit(manifest)
    view = DashboardView(snapshot)
    personas = view.topology()["green_personas"]
    assert personas == []


def test_event_bridge_replays_live_events_and_closes() -> None:
    with pytest.raises(ValueError, match="max_buffer"):
        EventBridge(max_buffer=0)

    bridge = EventBridge(max_buffer=2)
    bridge.push(DashboardEvent("1", "old", "red", "a", 0.0, {}))
    bridge.push(DashboardEvent("2", "backlog", "red", "b", 1.0, {}))
    bridge.push(DashboardEvent("3", "latest", "blue", "c", 2.0, {}))

    assert [event.id for event in bridge.snapshot_buffer()] == ["2", "3"]

    async def run() -> list[str]:
        received: list[str] = []
        stream = bridge.subscribe()
        received.append((await stream.__anext__()).id)
        received.append((await stream.__anext__()).id)
        bridge.push(DashboardEvent("4", "live", "green", "d", 3.0, {}))
        received.append((await stream.__anext__()).id)
        bridge.close()
        async for event in stream:
            received.append(event.id)
        return received

    assert asyncio.run(run()) == ["2", "3", "4"]

    sync_stream = bridge.subscribe_sync()
    assert [next(sync_stream).id, next(sync_stream).id] == ["3", "4"]
    bridge.close()
    with pytest.raises(StopIteration):
        next(sync_stream)
