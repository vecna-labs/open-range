"""End-to-end smoke for the trading pack: the real episode loop closes.

admit (cache-seeded, offline) → start_episode → the realizer hands the agent
``bars.json`` → a scripted strategy is written to ``result.json`` →
stop_episode (collect → the trade.pnl grader replays it look-ahead-safe in a
sandbox) → EpisodeReport → auto_evolve. No LLM, no network.

The window is deterministic in the seed, so a real synthetic bars file is
written at the exact cache path the real ``load_window`` reads — no patching.
"""

from __future__ import annotations

import json
import random
from datetime import timedelta
from pathlib import Path

from openrange_pack_sdk import EpisodeResult, Snapshot
from trading import TradingPack
from trading import data as trading_data

from openrange.core import auto_evolve
from openrange.core.admit import admit
from openrange.core.episode import EpisodeReport, EpisodeService

_SEED = 7
_WINDOW_DAYS = 180
_PRODUCT = "BTC-USD"


def _seed_cache(tmp_path: Path) -> None:
    rng = random.Random(_SEED)
    start, end = trading_data.select_window(rng.getrandbits(32), _WINDOW_DAYS)
    closes = [100.0]
    for i in range(1, 30):
        closes.append(closes[-1] * (1.01 if i % 2 else 0.995))
    bars = []
    prev = closes[0]
    for i, close in enumerate(closes):
        bars.append(
            {
                "day": (start + timedelta(days=i)).isoformat(),
                "open": f"{prev:.2f}",
                "high": f"{max(prev, close) * 1.002:.2f}",
                "low": f"{min(prev, close) * 0.998:.2f}",
                "close": f"{close:.2f}",
                "volume": "1000",
            }
        )
        prev = close
    cache = tmp_path / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"{_PRODUCT}_{start.isoformat()}_{end.isoformat()}.json"
    path.write_text(json.dumps(bars), encoding="utf-8")


def _admit(tmp_path: Path) -> Snapshot:
    _seed_cache(tmp_path)
    snap = admit(
        TradingPack(),
        manifest={"seed": _SEED, "cache_dir": str(tmp_path / "cache")},
        max_repairs=0,
    )
    assert isinstance(snap, Snapshot), snap
    return snap


def test_trading_episode_grades_strategy_and_evolves(tmp_path: Path) -> None:
    snap = _admit(tmp_path)
    task = snap.tasks[0]
    svc = EpisodeService(TradingPack(), tmp_path / "runs")
    try:
        handle = svc.start_episode(snap, task.id)
        solver_root = svc.solver_root(handle)
        # the realizer handed the agent the price window to study
        bars = json.loads((solver_root / "bars.json").read_text(encoding="utf-8"))
        assert len(bars) >= 2
        assert "close" in bars[0]
        # scripted "agent": a buy-and-hold strategy
        (solver_root / "result.json").write_text(
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
    passing = EpisodeReport(
        snapshot_id=snap.snapshot_id,
        task_id=task.id,
        episode_result=EpisodeResult(success=True),
    )
    evolved = auto_evolve(snap, passing, pack=TradingPack())
    assert evolved is not None
    assert evolved.snapshot_id != snap.snapshot_id
    assert any(event.phase == "evolve" for event in evolved.history)
