"""Trading pack: the builder embeds the price series graph-natively (one bar
node per day, not a blob), invariants catch broken series, the world admits
through all five layers, and regime selection respects the prior's intensity.

Tests are hermetic without patching: each build's window is deterministic in
its seed, so a real synthetic bars file is written at the exact cache path the
real ``load_window`` will read. No network is ever touched.
"""

from __future__ import annotations

import json
import random
from datetime import date, timedelta
from pathlib import Path

import pytest
from openrange_pack_sdk import BuildResult
from trading import TradingPack
from trading import data as trading_data
from trading.families import TradePnl
from trading.families.backtest import perfect_foresight_return, run_backtest
from trading.invariants import account_can_trade, bars_contiguous, ohlc_sane
from trading.sampling import _select_window

_DEFAULT_SEED = 7
_DEFAULT_WINDOW_DAYS = 180
_DEFAULT_PRODUCT = "BTC-USD"
_SEEDED_BARS = 30


def _oscillating_bars(start: date, closes: list[float]) -> list[dict[str, str]]:
    bars: list[dict[str, str]] = []
    prev = closes[0]
    for i, close in enumerate(closes):
        opn = prev
        high = max(opn, close) * 1.002
        low = min(opn, close) * 0.998
        bars.append(
            {
                "day": (start + timedelta(days=i)).isoformat(),
                "open": f"{opn:.2f}",
                "high": f"{high:.2f}",
                "low": f"{low:.2f}",
                "close": f"{close:.2f}",
                "volume": "1000",
            }
        )
        prev = close
    return bars


def _uptrend_closes(count: int) -> list[float]:
    closes = [100.0]
    for i in range(1, count):
        closes.append(closes[-1] * (1.01 if i % 2 else 0.995))
    return closes


def _seed_cache(
    cache_dir: Path, product: str, start: date, end: date, bars: list[dict[str, str]]
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{product}_{start.isoformat()}_{end.isoformat()}.json"
    path.write_text(json.dumps(bars), encoding="utf-8")


def _candidate_windows(
    seed: int, window_days: int, candidates: int
) -> list[tuple[date, date]]:
    rng = random.Random(seed)
    return [
        trading_data.select_window(rng.getrandbits(32), window_days)
        for _ in range(candidates)
    ]


def _manifest(tmp_path: Path, **overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "seed": _DEFAULT_SEED,
        "cache_dir": str(tmp_path / "cache"),
    }
    base.update(overrides)
    return base


def _seed_default_window(tmp_path: Path) -> None:
    (start, end) = _candidate_windows(_DEFAULT_SEED, _DEFAULT_WINDOW_DAYS, 1)[0]
    _seed_cache(
        tmp_path / "cache",
        _DEFAULT_PRODUCT,
        start,
        end,
        _oscillating_bars(start, _uptrend_closes(_SEEDED_BARS)),
    )


def _build(tmp_path: Path, **overrides: object) -> BuildResult:
    _seed_default_window(tmp_path)
    builder = TradingPack().make_builder(None)
    return builder.build(_manifest(tmp_path, **overrides))


class TestBuilderGraph:
    def test_bars_are_nodes_not_a_blob(self, tmp_path: Path) -> None:
        graph = _build(tmp_path).graph
        assert len(graph.by_kind("instrument")) == 1
        assert len(graph.by_kind("bar")) == _SEEDED_BARS
        assert len(graph.by_kind("account")) == 1
        assert len(graph.by_kind("risk_limit")) == 1
        has_bar = [e for e in graph.edges.values() if e.kind == "has_bar"]
        assert len(has_bar) == _SEEDED_BARS
        assert len([e for e in graph.edges.values() if e.kind == "trades"]) == 1
        assert len([e for e in graph.edges.values() if e.kind == "limits"]) == 1
        # each bar is a typed, inspectable node — not an opaque blob attr
        bar = graph.by_kind("bar")[0]
        assert {"seq", "day", "open", "high", "low", "close", "volume"} <= set(
            bar.attrs
        )

    def test_provenance_recorded_in_meta(self, tmp_path: Path) -> None:
        result = _build(tmp_path)
        assert result.graph.meta["data_source"] == "cache"
        assert result.admission_meta["data_source"] == "cache"

    def test_content_hash_is_deterministic(self, tmp_path: Path) -> None:
        assert (
            _build(tmp_path).graph.content_hash()
            == _build(tmp_path).graph.content_hash()
        )

    def test_task_wired_to_account_and_instrument(self, tmp_path: Path) -> None:
        result = _build(tmp_path)
        assert len(result.tasks) == 1
        task = result.tasks[0]
        assert task.entrypoints[0].startswith("account:")
        assert task.goal_nodes[0].startswith("instrument:")
        assert task.feasibility_check == "trade.pnl"


class TestInvariants:
    def test_clean_world_passes_all(self, tmp_path: Path) -> None:
        graph = _build(tmp_path).graph
        assert ohlc_sane(graph) == []
        assert bars_contiguous(graph) == []
        assert account_can_trade(graph) == []

    def test_ohlc_sane_flags_low_above_high(self, tmp_path: Path) -> None:
        graph = _build(tmp_path).graph
        graph.by_kind("bar")[0].attrs["low"] = "9999999"
        assert any(i.code == "bar_low_gt_high" for i in ohlc_sane(graph))

    def test_ohlc_sane_flags_unparseable(self, tmp_path: Path) -> None:
        graph = _build(tmp_path).graph
        graph.by_kind("bar")[0].attrs["close"] = "not-a-number"
        assert any(i.code == "bar_unparseable" for i in ohlc_sane(graph))

    def test_bars_contiguous_flags_gap(self, tmp_path: Path) -> None:
        graph = _build(tmp_path).graph
        graph.by_kind("bar")[1].attrs["seq"] = 99
        assert any(i.code == "bar_seq_noncontiguous" for i in bars_contiguous(graph))


class TestAdmission:
    def test_world_admits_through_all_layers(self, tmp_path: Path) -> None:
        from openrange.core.admit import AdmissionFailure, admit

        _seed_default_window(tmp_path)
        result = admit(TradingPack(), manifest=_manifest(tmp_path), max_repairs=0)
        assert not isinstance(result, AdmissionFailure)
        assert result.snapshot_id == result.graph.content_hash()


class TestRegimeSelection:
    def test_intensity_picks_volatility_quantile(self, tmp_path: Path) -> None:
        cache = tmp_path / "cache"
        windows = _candidate_windows(0, 4, 3)
        swings = {
            0: [100, 100, 100, 100],
            1: [100, 101, 100, 101],
            2: [100, 150, 100, 150],
        }
        for i, (start, end) in enumerate(windows):
            _seed_cache(
                cache,
                "X",
                start,
                end,
                _oscillating_bars(start, [float(c) for c in swings[i]]),
            )
        kw = {"cache_dir": cache, "fixture": tmp_path / "absent.json"}
        # calmest at intensity 0, most volatile at 1, the middle at 0.5
        calm = _select_window(random.Random(0), "X", 4, 3, 0.0, **kw)
        wild = _select_window(random.Random(0), "X", 4, 3, 1.0, **kw)
        mid = _select_window(random.Random(0), "X", 4, 3, 0.5, **kw)

        def _swing(window: trading_data.Window) -> float:
            return max(float(b.close) for b in window.bars)

        assert _swing(calm) == 100.0
        assert _swing(mid) == 101.0
        assert _swing(wild) == 150.0


class TestDataValidation:
    def test_select_window_rejects_out_of_range_length(self) -> None:
        with pytest.raises(ValueError, match="length_days"):
            trading_data.select_window(7, 0)
        with pytest.raises(ValueError, match="length_days"):
            trading_data.select_window(7, 10_000)

    def test_validate_rejects_empty_series(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            trading_data._validate([], date(2020, 1, 1), date(2020, 1, 10))

    def test_validate_rejects_unordered_days(self) -> None:
        bars = [
            trading_data.Bar("2020-01-02", "1", "1", "1", "1", "1"),
            trading_data.Bar("2020-01-01", "1", "1", "1", "1", "1"),
        ]
        with pytest.raises(ValueError, match="strictly increasing"):
            trading_data._validate(bars, date(2020, 1, 1), date(2020, 1, 10))

    def test_validate_rejects_days_outside_window(self) -> None:
        bars = [trading_data.Bar("2020-02-01", "1", "1", "1", "1", "1")]
        with pytest.raises(ValueError, match="outside window"):
            trading_data._validate(bars, date(2020, 1, 1), date(2020, 1, 10))


def _bars_from_closes(closes: list[int]) -> list[dict[str, object]]:
    return [
        {
            "seq": i,
            "day": f"2020-01-{i + 1:02d}",
            "open": str(c),
            "high": str(c),
            "low": str(c),
            "close": str(c),
            "volume": "1",
        }
        for i, c in enumerate(closes)
    ]


class TestBacktestGrader:
    def test_buy_and_hold_profits_on_rising_series(self) -> None:
        report = run_backtest(
            "def decide(history):\n    return 1.0\n",
            _bars_from_closes([100, 101, 102, 103, 104]),
            initial_cash=1000.0,
            cost_rate=0.0,
            return_target=0.02,
            max_drawdown_limit=0.5,
        )
        assert report.ok and report.passed
        assert report.pnl > 0.0
        assert report.max_drawdown == 0.0

    def test_all_cash_never_meets_positive_target(self) -> None:
        report = run_backtest(
            "def decide(history):\n    return 0.0\n",
            _bars_from_closes([100, 110, 121]),
            initial_cash=1000.0,
            cost_rate=0.0,
            return_target=0.05,
            max_drawdown_limit=0.5,
        )
        assert report.ok
        assert report.pnl == 0.0
        assert not report.return_met and not report.passed

    def test_replay_feeds_growing_history_no_lookahead(self) -> None:
        # Invest only once two bars are visible: captures the second +10% leg
        # but misses the first — proving history grows one bar at a time and the
        # strategy can't peek ahead.
        report = run_backtest(
            "def decide(history):\n    return 1.0 if len(history) >= 2 else 0.0\n",
            _bars_from_closes([100, 110, 121]),
            initial_cash=1000.0,
            cost_rate=0.0,
            return_target=0.0,
            max_drawdown_limit=1.0,
        )
        assert report.ok
        assert abs(report.pnl - 0.10) < 1e-9

    def test_broken_strategy_is_a_failed_episode(self) -> None:
        report = run_backtest(
            'def decide(history):\n    return "buy"\n',
            _bars_from_closes([100, 101, 102]),
            initial_cash=1000.0,
            cost_rate=0.0,
            return_target=0.05,
            max_drawdown_limit=0.5,
        )
        assert not report.ok and not report.passed
        assert "submission failed" in report.reason

    def test_check_success_reads_strategy_from_result_json(
        self, tmp_path: Path
    ) -> None:
        result = _build(tmp_path)
        outcome = TradePnl().check_success(
            result.graph,
            result.tasks[0],
            {"result": {"strategy": "def decide(history):\n    return 0.0\n"}},
        )
        # all-cash on a positive target fails, but the grader ran end to end
        assert not outcome.success
        assert "pnl=" in outcome.reason
        assert set(outcome.subgoals) == {"return_target_met", "drawdown_ok"}

    def test_missing_strategy_field_fails_cleanly(self, tmp_path: Path) -> None:
        result = _build(tmp_path)
        outcome = TradePnl().check_success(
            result.graph, result.tasks[0], {"result": {}}
        )
        assert not outcome.success
        assert "strategy" in outcome.reason


class TestFeasibility:
    def test_default_world_is_feasible(self, tmp_path: Path) -> None:
        result = _build(tmp_path)
        assert TradePnl().check_feasibility(result.graph, result.tasks[0]).feasible

    def test_nonpositive_target_is_infeasible(self, tmp_path: Path) -> None:
        result = _build(tmp_path, return_target="0")
        verdict = TradePnl().check_feasibility(result.graph, result.tasks[0])
        assert not verdict.feasible
        assert "trivially passable" in verdict.reason

    def test_target_above_data_ceiling_is_infeasible(self, tmp_path: Path) -> None:
        result = _build(tmp_path, return_target="100")
        verdict = TradePnl().check_feasibility(result.graph, result.tasks[0])
        assert not verdict.feasible
        assert "perfect-foresight" in verdict.reason

    def test_perfect_foresight_captures_only_up_moves(self) -> None:
        ceiling = perfect_foresight_return(_bars_from_closes([100, 110, 99, 110]))
        assert ceiling > 0.2  # the two +10%-ish up-legs, the down-leg skipped


class TestPatchMutations:
    def test_harden_and_soften_adjust_return_target(self, tmp_path: Path) -> None:
        from openrange.core.admit import AdmissionFailure, admit

        _seed_default_window(tmp_path)
        snapshot = admit(TradingPack(), manifest=_manifest(tmp_path), max_repairs=0)
        assert not isinstance(snapshot, AdmissionFailure)
        options = TradePnl().available_mutations(snapshot, [])
        assert {m.direction for m in options} == {"harden", "soften"}
        harden = next(m for m in options if m.direction == "harden")
        (updated,) = harden.patch.nodes_updated
        assert updated.kind == "risk_limit"
        assert updated.attrs["return_target"] == "0.075"  # default 0.05 * 1.5


class TestGrowViaPrior:
    def test_grow_advances_regime_to_a_new_world(self, tmp_path: Path) -> None:
        from openrange.core.admit import AdmissionFailure, admit
        from openrange.core.curriculum import _grow_snapshot

        # five candidate windows of strictly increasing volatility; the regime
        # tournament picks a higher-vol one as the prior's intensity rises.
        cache = tmp_path / "cache"
        windows = _candidate_windows(_DEFAULT_SEED, _DEFAULT_WINDOW_DAYS, 5)
        for swing, (start, end) in enumerate(windows, start=1):
            closes = [100.0, 100.0 + swing, 100.0, 100.0 + swing]
            _seed_cache(
                cache, _DEFAULT_PRODUCT, start, end, _oscillating_bars(start, closes)
            )
        manifest = _manifest(tmp_path, regime_candidates=5, return_target="0.01")
        snapshot = admit(TradingPack(), manifest=manifest, max_repairs=0)
        assert not isinstance(snapshot, AdmissionFailure)

        grown = _grow_snapshot(snapshot, TradingPack(), "harden", max_repairs=0)
        assert grown is not None
        # a genuinely different admitted world (a harder regime), not a patch
        assert grown.snapshot_id != snapshot.snapshot_id
        assert abs(grown.lineage["curriculum_difficulty"]["overall"] - 0.7) < 1e-9
        assert grown.lineage["_evolve"]["kind"] == "grow"

    def test_grow_is_noop_without_a_candidate_pool(self, tmp_path: Path) -> None:
        # regime_candidates defaults to 1, so intensity is moot — the rebuild
        # lands on the same world and grow declines (None), never looping.
        from openrange.core.admit import AdmissionFailure, admit
        from openrange.core.curriculum import _grow_snapshot

        _seed_default_window(tmp_path)
        snapshot = admit(TradingPack(), manifest=_manifest(tmp_path), max_repairs=0)
        assert not isinstance(snapshot, AdmissionFailure)
        assert _grow_snapshot(snapshot, TradingPack(), "harden", max_repairs=0) is None
