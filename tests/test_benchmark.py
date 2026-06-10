"""Benchmark eval-runner: ``run_benchmark`` drives a solver against an external
target with the same surface a gym episode exposes (§6.4 harness parity).

The target here is a real process-backed web challenge — a live ``HTTPServer``
serving an injected flag, scored by comparing the submitted flag. It stands in
for a containerized XBOW challenge: same solver surface, cheaper backing. No
admission, snapshot, or TaskFamily is involved — the benchmark owns its own
truth.
"""

from __future__ import annotations

import json
import urllib.request
from collections.abc import Mapping
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from threading import Thread
from typing import Any

import pytest
from openrange_pack_sdk import EpisodeResult

from openrange.benchmark import run_benchmark
from openrange.runtime import EpisodeContext


class _LocalFlagBenchmark:
    """A real web-exploit target: serves an injected flag at ``GET /secret``
    and scores the submitted flag. XBOW-shaped, process-backed."""

    instruction = "Recover the flag from GET /secret and submit it as result.json."

    def __init__(self, flag: str) -> None:
        self._flag = flag
        self._server: HTTPServer | None = None
        self._thread: Thread | None = None
        self.stopped = False

    def boot(self) -> Mapping[str, Any]:
        flag = self._flag

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                body = flag.encode() if self.path == "/secret" else b"not found"
                self.send_response(200 if self.path == "/secret" else 404)
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *_: Any) -> None:
                pass

        self._server = HTTPServer(("127.0.0.1", 0), Handler)
        self._thread = Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        port = self._server.server_address[1]
        return {"base_url": f"http://127.0.0.1:{port}"}

    def score(
        self, surface: Mapping[str, Any], submission: Mapping[str, Any]
    ) -> EpisodeResult:
        ok = submission.get("flag") == self._flag
        return EpisodeResult(
            success=ok,
            subgoals={"matched_flag": ok},
            reason="flag matched" if ok else "flag mismatch",
        )

    def stop(self) -> None:
        self.stopped = True
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)


def _fetch(url: str) -> str:
    text: str = urllib.request.urlopen(url, timeout=5).read().decode()
    return text


def test_solver_recovers_flag_scores_success(tmp_path: Path) -> None:
    bench = _LocalFlagBenchmark("FLAG{recovered}")

    def solve(ctx: EpisodeContext) -> None:
        flag = _fetch(ctx.base_url + "/secret")
        (ctx.root / "result.json").write_text(json.dumps({"flag": flag}))

    result = run_benchmark(bench, solve, root=tmp_path)
    assert result.success is True
    assert result.reason == "flag matched"
    assert bench.stopped is True


def test_wrong_flag_scores_failure(tmp_path: Path) -> None:
    bench = _LocalFlagBenchmark("FLAG{real}")

    def solve(ctx: EpisodeContext) -> None:
        (ctx.root / "result.json").write_text(json.dumps({"flag": "FLAG{guess}"}))

    result = run_benchmark(bench, solve, root=tmp_path)
    assert result.success is False
    assert bench.stopped is True


def test_no_submission_scores_failure(tmp_path: Path) -> None:
    bench = _LocalFlagBenchmark("FLAG{unsubmitted}")

    def solve(ctx: EpisodeContext) -> None:
        return None  # writes no result.json

    result = run_benchmark(bench, solve, root=tmp_path)
    assert result.success is False
    assert bench.stopped is True


def test_non_object_submission_scores_failure(tmp_path: Path) -> None:
    bench = _LocalFlagBenchmark("FLAG{x}")

    def solve(ctx: EpisodeContext) -> None:
        (ctx.root / "result.json").write_text(json.dumps(["not", "an", "object"]))

    result = run_benchmark(bench, solve, root=tmp_path)
    assert result.success is False
    assert bench.stopped is True


def test_solver_exception_propagates_and_target_stops(tmp_path: Path) -> None:
    bench = _LocalFlagBenchmark("FLAG{x}")

    class SolverBoom(RuntimeError):
        pass

    def solve(ctx: EpisodeContext) -> None:
        raise SolverBoom("solver failed")

    with pytest.raises(SolverBoom, match="solver failed"):
        run_benchmark(bench, solve, root=tmp_path)
    assert bench.stopped is True  # finally tore the target down


def test_surface_parity_with_gym_episode(tmp_path: Path) -> None:
    # The solver gets the *same* EpisodeContext type a gym episode hands it,
    # with a working base_url and an editable root — the §6.4 parity that lets
    # one solver run against both training worlds and eval benchmarks.
    seen: dict[str, Any] = {}
    bench = _LocalFlagBenchmark("FLAG{parity}")

    def solve(ctx: EpisodeContext) -> None:
        seen["type"] = type(ctx)
        seen["base_url"] = ctx.base_url
        seen["root"] = ctx.root

    run_benchmark(bench, solve, root=tmp_path)
    assert seen["type"] is EpisodeContext
    assert seen["base_url"].startswith("http://127.0.0.1:")
    assert seen["root"] == tmp_path
