"""End-to-end smoke for the trading pack: the real episode loop closes.

admit (fixture-backed, offline) → start_episode → the realizer hands the agent
``bars.json`` → a scripted strategy is written to ``result.json`` →
stop_episode (collect → the trade.pnl grader replays it look-ahead-safe in a
sandbox) → EpisodeReport → auto_evolve. No LLM, no network.
"""

from __future__ import annotations

import json
import urllib.error
from collections.abc import Mapping
from pathlib import Path

import pytest
from openrange_pack_sdk import Snapshot
from trading import TradingPack
from trading import data as trading_data

from openrange.core import auto_evolve
from openrange.core.admit import admit
from openrange.core.episode import EpisodeService


class _PassingReport:
    """Stand-in EpisodeReportLike that forces a 'harden' direction."""

    passed: bool = True
    final_state: Mapping[str, object] = {}


@pytest.fixture
def offline(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*args: object, **kwargs: object) -> object:
        raise urllib.error.URLError("offline test")

    monkeypatch.setattr(trading_data, "fetch_daily", _boom)


def _admit(tmp_path: Path) -> Snapshot:
    snap = admit(
        TradingPack(),
        manifest={"seed": 7, "cache_dir": str(tmp_path / "cache")},
        max_repairs=0,
    )
    assert isinstance(snap, Snapshot), snap
    return snap


def test_trading_episode_grades_strategy_and_evolves(
    offline: None, tmp_path: Path
) -> None:
    snap = _admit(tmp_path)
    task = snap.tasks[0]
    svc = EpisodeService(TradingPack(), tmp_path / "runs")
    try:
        handle = svc.start_episode(snap, task.id)
        agent_root = svc.agent_root(handle)
        # the realizer handed the agent the price window to study
        bars = json.loads((agent_root / "bars.json").read_text(encoding="utf-8"))
        assert len(bars) >= 2
        assert "close" in bars[0]
        # scripted "agent": a buy-and-hold strategy
        (agent_root / "result.json").write_text(
            json.dumps({"strategy": "def decide(history):\n    return 1.0\n"}),
            encoding="utf-8",
        )
        report = svc.stop_episode(handle)
    finally:
        svc.close()
    # the grader replayed the strategy end to end (pass/fail depends on the
    # window, but the backtest ran and produced metrics)
    assert "pnl=" in report.episode_result.reason

    # a passing signal hardens the world — the patch path raises the return
    # target and re-admits a new snapshot.
    evolved = auto_evolve(snap, _PassingReport(), pack=TradingPack())
    assert evolved is not None
    assert evolved.snapshot_id != snap.snapshot_id
    assert any(event.phase == "evolve" for event in evolved.history)
