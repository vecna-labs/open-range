"""End-to-end EpisodeService tests against the v1 cyber pack.

Every test in this module drives the OLD-shape `EpisodeService`
runtime — `start_episode`, `tick`, `stop_episode`, the verifier
result on `EpisodeReport`. That runtime layer has not yet been
migrated to the new pack/admission shape; it's slated for Round 3
(see `.claude/RUNTIME_MIGRATION_PLAN.md`, stage 3d, the
`core/episode.py` row marked STRUCTURAL ~120 LOC, HIGHEST RISK).

The new pack (`cyber_webapp.WebappPack`) registers as `"webapp"`
(not `"cyber.webapp"`) and no longer exposes a `verifier_result`
on its episode report — success is decided by
`TaskFamily.check_success(graph, task, final_state)`. Reviving
these tests requires a Round-3 `EpisodeService(pack, ...)` that
threads the Pack through and replaces the verifier-source seam
with the family.check_success call.

Until that lands, every test here is skipped at the module level.
Bodies are intentionally preserved verbatim so a Round-3 PR has a
known-good list of behaviors to keep alive (HTTP round-trip, NPC
tick scheduling, manifest NPC validation, dashboard recorder
plumbing, the model/backend mutual-exclusion check, the
LLM-opt-in `agent_backend` context injection).
"""

from __future__ import annotations

import json
import time
import urllib.request
from collections.abc import Callable  # noqa: TC003 — used in lazy-imported tests
from pathlib import Path
from typing import cast

import pytest

import openrange as OR
from openrange.core.builder import build
from openrange.core.episode import EpisodeService

pytestmark = pytest.mark.skip(
    reason=(
        "EpisodeService runtime migration is Round 3 — see "
        ".claude/RUNTIME_MIGRATION_PLAN.md (stage 3d). These tests "
        "exercise the old-shape EpisodeService/verifier_result flow; "
        "they will be revived once core/episode.py accepts a Pack "
        "and dispatches into TaskFamily.check_success."
    ),
)

V1_MANIFEST = {
    "pack": {"id": "cyber.webapp", "source": {"kind": "builtin"}},
    "mode": "simulation",
    "world": {},
    "runtime": {"tick": {"mode": "manual"}},
}


def _flag_value(snapshot: OR.Snapshot) -> str:
    assert snapshot.world_graph is not None
    for node in snapshot.world_graph.nodes:
        if node.type == "secret" and node.attrs.get("kind") == "flag":
            return str(node.attrs["value_ref"])
    raise AssertionError("snapshot has no flag secret")


def test_v1_episode_round_trip_with_scripted_solve(tmp_path: Path) -> None:
    """Build → start episode → agent hits HTTP → submits flag → verifier passes."""
    snapshot = build(V1_MANIFEST)
    flag = _flag_value(snapshot)

    service = EpisodeService(tmp_path / "runs")
    handle = service.start_episode(snapshot)
    try:
        # Scripted "agent": probe the service (so the request log records an
        # interaction; required by validate_public_interface_interaction)…
        base_url = service.base_url(handle)
        with urllib.request.urlopen(f"{base_url}/", timeout=2) as response:
            assert response.status == 200
        with urllib.request.urlopen(f"{base_url}/openapi.json", timeout=2) as response:
            assert response.status == 200
        # …then write the flag to the result file (in a real episode an
        # LLM agent would derive this via exploit; we shortcut for the
        # integration smoke test).
        agent_root = service.agent_root(handle)
        (agent_root / "result.json").write_text(
            json.dumps({"flag": flag}),
            encoding="utf-8",
        )
    finally:
        report = service.stop_episode(handle)

    assert report.verifier_result is not None
    assert report.verifier_result["passed"] is True, dict(report.verifier_result)
    result = cast("dict[str, object]", report.final_state["result"])
    assert result["flag"] == flag


def test_v1_episode_serves_openapi_discovery(tmp_path: Path) -> None:
    """The realized app exposes the discovery payload at /openapi.json."""
    snapshot = build(V1_MANIFEST)
    service = EpisodeService(tmp_path / "runs")
    handle = service.start_episode(snapshot)
    try:
        base_url = service.base_url(handle)
        with urllib.request.urlopen(f"{base_url}/openapi.json", timeout=2) as response:
            payload = json.loads(response.read().decode())
        assert "services" in payload
        assert any(svc["kind"] == "web" for svc in payload["services"])
    finally:
        service.stop_episode(handle)


def test_v1_episode_runs_npcs_during_ticks(tmp_path: Path) -> None:
    """Manifest NPCs make real HTTP requests on tick — verify via request log."""
    manifest = {
        **V1_MANIFEST,
        "npc": [
            {
                "type": "cyber.browsing_user",
                "count": 1,
                "config": {"cadence_ticks": 1, "paths": ["/openapi.json"]},
            },
        ],
    }
    snapshot = build(manifest)
    service = EpisodeService(tmp_path / "runs")
    handle = service.start_episode(snapshot)
    try:
        # Manual tick: each call advances NPCs and captures requests.
        service.tick(handle)
        service.tick(handle)
        service.tick(handle)
        # Allow the server's log to flush.
        time.sleep(0.05)
        # Stop with a fake result so the verifier can run.
        agent_root = service.agent_root(handle)
        (agent_root / "result.json").write_text(
            json.dumps({"flag": _flag_value(snapshot)}),
            encoding="utf-8",
        )
    finally:
        report = service.stop_episode(handle)
    requests = cast("list[dict[str, object]]", report.final_state["requests"])
    npc_paths = [str(row.get("path", "")) for row in requests]
    assert any("/openapi.json" in p for p in npc_paths), npc_paths


def test_v1_episode_unknown_npc_type_fails_cleanly(tmp_path: Path) -> None:
    """Bad manifest NPC type raises during start_episode, not silently."""
    from openrange.npc import NPCError

    manifest = {
        **V1_MANIFEST,
        "npc": [{"type": "does.not.exist"}],
    }
    snapshot = build(manifest)
    service = EpisodeService(tmp_path / "runs")
    with pytest.raises(NPCError, match="unknown NPC"):
        service.start_episode(snapshot)


def test_v1_episode_injects_agent_backend_into_npc_context(tmp_path: Path) -> None:
    """LLM-opt-in NPCs receive an ``agent_backend`` key; plain NPCs don't.

    Verifies the runtime honors ``requires_llm`` per-NPC and threads
    ``EpisodeService.npc_agent_backend`` through.
    """
    from collections.abc import Mapping, Sequence
    from typing import Any

    from openrange.agent_backend import AgentBackend
    from openrange.npc import NPC, NPCS, AgentNPC

    seen_contexts: dict[str, Mapping[str, Any]] = {}

    class _RecordingPlainNPC(NPC):
        def start(self, context: Mapping[str, Any]) -> None:
            seen_contexts["plain"] = dict(context)

        def step(self, interface: Mapping[str, Any]) -> None: ...

    class _SilentBackend:
        def preflight(self) -> None: ...

        def build_agent(self, **_kwargs: Any) -> Any:
            def session(_prompt: str) -> Any:
                return None

            return session

    class _RecordingAgentNPC(AgentNPC):
        def start(self, context: Mapping[str, Any]) -> None:
            super().start(context)
            seen_contexts["agent"] = dict(context)

        def _build_tools(
            self,
            interface: Mapping[str, Any],
        ) -> Sequence[Callable[..., Any]]:
            return ()

    NPCS.register("test.recording_plain", lambda config: _RecordingPlainNPC())
    NPCS.register(
        "test.recording_agent",
        lambda config: _RecordingAgentNPC(system_prompt="x", cadence_ticks=999),
    )
    try:
        manifest = {
            **V1_MANIFEST,
            "npc": [
                {"type": "test.recording_plain"},
                {"type": "test.recording_agent"},
            ],
        }
        snapshot = build(manifest)
        backend: AgentBackend = _SilentBackend()
        service = EpisodeService(tmp_path / "runs", npc_agent_backend=backend)
        handle = service.start_episode(snapshot)
        try:
            base_url = service.base_url(handle)
            with urllib.request.urlopen(f"{base_url}/", timeout=2) as response:
                assert response.status == 200
            (service.agent_root(handle) / "result.json").write_text(
                json.dumps({"flag": _flag_value(snapshot)}),
                encoding="utf-8",
            )
        finally:
            service.stop_episode(handle)
    finally:
        NPCS._factories.pop("test.recording_plain", None)
        NPCS._factories.pop("test.recording_agent", None)

    assert "agent_backend" not in seen_contexts["plain"]
    assert seen_contexts["agent"]["agent_backend"] is backend


def test_v1_episode_omits_backend_when_runtime_unset(tmp_path: Path) -> None:
    """LLM-opt-in NPC sees ``agent_backend = None`` when the service has no
    backend configured. The NPC should mark itself broken at start()."""
    from collections.abc import Mapping, Sequence
    from typing import Any

    from openrange.npc import NPCS, AgentNPC

    seen: dict[str, Mapping[str, Any]] = {}

    class _Recording(AgentNPC):
        def start(self, context: Mapping[str, Any]) -> None:
            super().start(context)
            seen["ctx"] = dict(context)

        def _build_tools(
            self,
            interface: Mapping[str, Any],
        ) -> Sequence[Callable[..., Any]]:
            return ()

    NPCS.register(
        "test.no_backend_runtime",
        lambda config: _Recording(system_prompt="x", cadence_ticks=999),
    )
    try:
        manifest = {**V1_MANIFEST, "npc": [{"type": "test.no_backend_runtime"}]}
        snapshot = build(manifest)
        service = EpisodeService(tmp_path / "runs")  # backend defaults to None
        handle = service.start_episode(snapshot)
        try:
            base_url = service.base_url(handle)
            with urllib.request.urlopen(f"{base_url}/", timeout=2) as response:
                assert response.status == 200
            (service.agent_root(handle) / "result.json").write_text(
                json.dumps({"flag": _flag_value(snapshot)}),
                encoding="utf-8",
            )
        finally:
            service.stop_episode(handle)
    finally:
        NPCS._factories.pop("test.no_backend_runtime", None)

    assert "agent_backend" in seen["ctx"]
    assert seen["ctx"]["agent_backend"] is None


def test_episode_service_rejects_both_backend_and_model(tmp_path: Path) -> None:
    """Both knobs together is a configuration error."""
    from openrange.agent_backend import StrandsAgentBackend
    from openrange.core.episode import EpisodeError

    with pytest.raises(EpisodeError, match="not both"):
        EpisodeService(
            tmp_path / "runs",
            npc_agent_backend=StrandsAgentBackend(),
            npc_llm_model="claude-sonnet-4-20250514",
        )


def test_episode_service_npc_llm_model_promotes_to_strands_backend(
    tmp_path: Path,
) -> None:
    """The model-id convenience auto-constructs a StrandsAgentBackend."""
    from openrange.agent_backend import StrandsAgentBackend

    service = EpisodeService(
        tmp_path / "runs",
        npc_llm_model="claude-sonnet-4-20250514",
    )
    assert isinstance(service.npc_agent_backend, StrandsAgentBackend)


def test_v1_episode_passes_record_action_to_npcs(tmp_path: Path) -> None:
    """Every NPC's start() context carries a callable ``record_action``.

    Plain NPCs (no LLM opt-in) get it just like AgentNPCs do — recording
    in-world actions to the dashboard is unrelated to LLM access. The
    callable must be invokable during step() and (when a dashboard is
    wired) must surface as an ``env_turn`` event tagged with the
    NPC's actor_id.
    """
    from collections.abc import Mapping
    from typing import Any

    from openrange.dashboard import DashboardView
    from openrange.npc import NPC, NPCS

    seen: dict[str, Any] = {}

    class _RecordingNPC(NPC):
        @property
        def actor_id(self) -> str:
            return "TestActor"

        def start(self, context: Mapping[str, Any]) -> None:
            seen["record_action"] = context.get("record_action")

        def step(self, interface: Mapping[str, Any]) -> None:
            recorder = seen.get("record_action")
            if recorder is not None:
                recorder({"speak": "hello"})

    NPCS.register("test.recording_speech", lambda config: _RecordingNPC())
    try:
        manifest = {**V1_MANIFEST, "npc": [{"type": "test.recording_speech"}]}
        snapshot = build(manifest)
        view = DashboardView(snapshot)
        service = EpisodeService(tmp_path / "runs", dashboard=view)
        handle = service.start_episode(snapshot)
        try:
            base_url = service.base_url(handle)
            with urllib.request.urlopen(f"{base_url}/", timeout=2) as response:
                assert response.status == 200
            service.tick(handle)  # NPC.step fires; emits speech event
            (service.agent_root(handle) / "result.json").write_text(
                json.dumps({"flag": _flag_value(snapshot)}),
                encoding="utf-8",
            )
        finally:
            service.stop_episode(handle)
    finally:
        NPCS._factories.pop("test.recording_speech", None)

    # The recorder was passed and was callable.
    assert callable(seen["record_action"])

    # And the speech event made it into the dashboard event stream.
    def _is_hello(event: Any) -> bool:
        if event.actor != "TestActor" or event.type != "env_turn":
            return False
        if not isinstance(event.data, Mapping):
            return False
        action = event.data.get("action")
        if not isinstance(action, Mapping):
            return False
        return action.get("speak") == "hello"

    speech_events = [
        event for event in view.bridge.snapshot_buffer() if _is_hello(event)
    ]
    assert speech_events, "expected a speech event from the recording NPC"
