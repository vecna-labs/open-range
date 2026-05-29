"""Tests for the cyber webapp pack's realizer.

Three concerns:

1. ``cyber_webapp.codegen._realize_graph`` walks a sampled WorldGraph
   and produces a ``{path: source}`` mapping carrying ``app.py`` and
   ``seed.json``.
2. ``WebappPack().realize(graph, Backing.PROCESS)`` returns a
   ``WebappRuntime`` satisfying the ``RuntimeHandle`` Protocol.
3. The handle's ``reset()`` materializes those files to disk, starts a
   subprocess, exposes the agent surface, and ``stop()`` cleans up.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Mapping
from typing import cast

import pytest
from cyber_webapp import WebappPack, WebappRuntime
from cyber_webapp.codegen import _realize_graph
from cyber_webapp.codegen.entrypoint import (
    APP_FILE_NAME,
    REQUEST_LOG_NAME,
    RESULT_FILE_NAME,
    SEED_FILE_NAME,
)
from graphschema import WorldGraph
from openrange_pack_sdk import Backing, RuntimeHandle


def _sample_graph(seed: int = 0) -> WorldGraph:
    """A graph drawn the way admission draws it — same path the runtime takes."""
    build_result = WebappPack().make_builder(None).build({"seed": seed})
    return build_result.graph


def test_realize_graph_emits_app_and_seed_files() -> None:
    """The codegen returns a plain mapping containing both required files."""
    files = _realize_graph(_sample_graph())
    assert APP_FILE_NAME in files
    assert SEED_FILE_NAME in files
    assert isinstance(files[APP_FILE_NAME], str)
    assert isinstance(files[SEED_FILE_NAME], str)


def test_realize_graph_app_py_compiles_across_seeds() -> None:
    """Every sampled seed produces a syntactically valid app.py."""
    for seed in range(6):
        files = _realize_graph(_sample_graph(seed))
        # `compile` raises SyntaxError if the rendered template is malformed.
        compile(files[APP_FILE_NAME], f"<seed-{seed}>", "exec")


def test_realize_graph_seed_json_carries_expected_keys() -> None:
    """seed.json is valid JSON and carries accounts/secrets/records/schema."""
    files = _realize_graph(_sample_graph())
    payload = json.loads(files[SEED_FILE_NAME])
    assert isinstance(payload, dict)
    for key in ("accounts", "secrets", "records", "schema"):
        assert key in payload, f"seed.json missing top-level key {key!r}"
    # The schema is what the SQLi handler reads against — must name a table
    # and the column we'll be SELECTing on.
    schema = payload["schema"]
    assert isinstance(schema, dict)
    assert "table" in schema
    assert "key_column" in schema
    assert "value_column" in schema


def test_realize_graph_seed_json_holds_flag_value() -> None:
    """The flag value sampled into the graph round-trips into seed.secrets."""
    graph = _sample_graph()
    flag_node = next(
        n
        for n in graph.nodes.values()
        if n.kind == "secret" and n.attrs.get("kind") == "flag"
    )
    expected_flag = str(flag_node.attrs["value_ref"])

    files = _realize_graph(graph)
    payload = json.loads(files[SEED_FILE_NAME])
    secrets = payload["secrets"]
    # The seeder mirrors the flag under every leak_field broken_authz might
    # pick, plus the canonical "flag" key.
    assert secrets.get("flag") == expected_flag


def test_realize_graph_is_deterministic_in_graph() -> None:
    """Same graph → byte-identical files. The codegen is a pure function."""
    graph = _sample_graph()
    first = _realize_graph(graph)
    second = _realize_graph(graph)
    assert first == second


def test_pack_realize_returns_webapp_runtime_handle() -> None:
    """The pack's realize() returns a concrete WebappRuntime."""
    graph = _sample_graph()
    handle = WebappPack().realize(graph, Backing.PROCESS)
    assert isinstance(handle, WebappRuntime)


def test_pack_realize_satisfies_runtime_handle_protocol() -> None:
    """The handle is duck-typed against the RuntimeHandle Protocol."""
    graph = _sample_graph()
    handle = WebappPack().realize(graph, Backing.PROCESS)
    # runtime_checkable Protocol — isinstance covers method presence.
    assert isinstance(handle, RuntimeHandle)


def test_pack_realize_rejects_non_process_backings() -> None:
    """Only Backing.PROCESS is wired today; the others must raise."""
    graph = _sample_graph()
    pack = WebappPack()
    for backing in (Backing.CONTAINER, Backing.SIMULATOR, Backing.HYBRID):
        with pytest.raises(NotImplementedError):
            pack.realize(graph, backing)


def test_handle_reset_materializes_files_and_starts_process() -> None:
    """reset() writes the rendered files to disk and exposes a base_url.

    After reset() the per-episode workspace has the rendered ``app.py``
    on disk under the pack root, the request log file pre-touched, and
    the HTTP subprocess listening on a port the surface reports.
    """
    handle = WebappPack().realize(_sample_graph(), Backing.PROCESS)
    handle.reset()
    try:
        surface = handle.surface()
        base_url = cast(str, surface["base_url"])
        assert base_url.startswith("http://127.0.0.1:")
        # The solver_root the surface reports must be a real directory.
        solver_root = cast(str, surface["solver_root"])
        from pathlib import Path  # noqa: PLC0415

        assert Path(solver_root).is_dir()
        # The pack root sits next to the agent root with app.py + seed.
        # We don't assert the seed.json still exists on disk (the
        # generated app unlinks it after loading), but app.py must.
        env_root = Path(solver_root).parent
        assert (env_root / "pack" / APP_FILE_NAME).exists()
        # The request log is pre-touched so poll_events() before any HTTP
        # traffic returns () instead of erroring. SubprocessRuntimeHandle
        # writes all prepared files under pack/.
        assert (env_root / "pack" / REQUEST_LOG_NAME).exists()
    finally:
        handle.stop()


def test_handle_serves_root_route_after_reset() -> None:
    """The generated app actually listens and responds to GET /."""
    handle = WebappPack().realize(_sample_graph(), Backing.PROCESS)
    handle.reset()
    try:
        surface = handle.surface()
        base_url = cast(str, surface["base_url"])
        with urllib.request.urlopen(base_url + "/", timeout=2) as response:
            assert response.status == 200
    finally:
        handle.stop()


def test_handle_returns_404_for_unknown_route() -> None:
    """A path no endpoint claims still 404s — the dispatcher is honest."""
    handle = WebappPack().realize(_sample_graph(), Backing.PROCESS)
    handle.reset()
    try:
        surface = handle.surface()
        base_url = cast(str, surface["base_url"])
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(
                base_url + "/__no_such_route__",
                timeout=2,
            )
        assert exc_info.value.code == 404
    finally:
        handle.stop()


def test_handle_collect_reports_smoke_test_signal() -> None:
    """collect() exposes endpoint_serves_200 — the build-family success key."""
    handle = WebappPack().realize(_sample_graph(), Backing.PROCESS)
    handle.reset()
    try:
        collected = handle.collect()
        assert isinstance(collected, Mapping)
        # The build family reads this. A freshly-reset generated app
        # serves / with 200, so the probe should agree.
        assert collected["endpoint_serves_200"] is True
        # The pentest family reads these two. With no agent interaction
        # the flag is None and the request list is empty.
        assert collected["flag_from_response"] is None
        assert collected["requests_made"] == []
    finally:
        handle.stop()


def test_handle_collect_returns_empty_before_reset() -> None:
    """Before reset(), collect() returns an empty mapping rather than crashing."""
    handle = WebappPack().realize(_sample_graph(), Backing.PROCESS)
    # No reset(); no process; no solver_root.
    collected = handle.collect()
    assert dict(collected) == {}


def test_handle_terminal_flips_when_result_file_appears() -> None:
    """terminal() flips True once the agent writes result.json."""
    from pathlib import Path  # noqa: PLC0415

    handle = WebappPack().realize(_sample_graph(), Backing.PROCESS)
    handle.reset()
    try:
        done, reason = handle.terminal()
        assert done is False
        assert reason is None
        # Simulate the agent finishing: drop a result.json next to where
        # the handle expects it.
        surface = handle.surface()
        solver_root = Path(cast(str, surface["solver_root"]))
        (solver_root / RESULT_FILE_NAME).write_text(
            json.dumps({"flag": "x"}),
            encoding="utf-8",
        )
        done, reason = handle.terminal()
        assert done is True
        assert reason == "solver wrote result"
    finally:
        handle.stop()


def test_handle_stop_is_idempotent() -> None:
    """stop() can be called twice without error — the second is a no-op."""
    handle = WebappPack().realize(_sample_graph(), Backing.PROCESS)
    handle.reset()
    handle.stop()
    # Should not raise.
    handle.stop()


def test_handle_stop_cleans_up_tempdirs() -> None:
    """stop() removes env_root and every checkpoint snapshot dir."""
    from pathlib import Path

    raw = WebappPack().realize(_sample_graph(), Backing.PROCESS)
    handle = cast(WebappRuntime, raw)
    handle.reset()
    env_root = handle._env_root
    assert env_root is not None and env_root.exists()
    state = handle.checkpoint()
    checkpoint_dir = Path(
        cast(str, cast(Mapping[str, object], state)["solver_root_snapshot"]),
    )
    assert checkpoint_dir.exists()
    handle.stop()
    assert not env_root.exists()
    assert not checkpoint_dir.exists()
    assert handle._env_root is None
    assert handle._solver_root is None
    assert handle._pack_root is None


def test_build_discovery_reads_title_from_graph_meta() -> None:
    """build_discovery returns the sampler-stashed title from `graph.meta`.

    Falls back to "Internal Services" only when `discovery_title` is
    missing (e.g. minimal hand-built graphs).
    """
    from cyber_webapp.codegen.discovery import build_discovery

    graph = _sample_graph(seed=42)
    expected = graph.meta.get("discovery_title")
    assert isinstance(expected, str) and expected
    payload = build_discovery(graph)
    assert payload["title"] == expected
    assert payload["title"] != "Internal Services"


def test_build_discovery_falls_back_when_meta_missing() -> None:
    """A graph without `discovery_title` in meta gets the fallback title."""
    from cyber_webapp.codegen.discovery import build_discovery

    bare = WorldGraph(ontology="cyber.webapp@v2")
    payload = build_discovery(bare)
    assert payload["title"] == "Internal Services"
