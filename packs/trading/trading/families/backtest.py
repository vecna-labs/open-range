"""Look-ahead-safe backtest grader for ``trade.pnl``.

The agent's ``decide`` strategy runs in the SDK sandbox (see
``openrange_pack_sdk.run_submission`` for the trust model). This module turns the
resulting equity curve into P&L and max-drawdown and checks them against the
task's risk limits.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from openrange_pack_sdk import run_submission

_WALL_TIMEOUT = 10.0


def _replay(entry: Callable[..., Any], case: Mapping[str, Any]) -> dict[str, object]:
    # Look-ahead safety lives here: `decide` only ever sees bars[: t + 1], so it
    # cannot peek past the day it is trading.
    bars = case["bars"]
    cash = float(case["cash"])
    cost_rate = float(case["cost_rate"])
    units = 0.0
    curve: list[float] = []
    for t in range(len(bars)):
        price = float(bars[t]["close"])
        equity = cash + units * price
        target = min(1.0, max(0.0, float(entry(bars[: t + 1]))))
        desired_units = (target * equity / price) if price else units
        trade = desired_units - units
        cash -= trade * price + abs(trade) * price * cost_rate
        units = desired_units
        curve.append(cash + units * price)
    return {"equity_curve": curve}


@dataclass(frozen=True)
class BacktestReport:
    ok: bool  # the strategy ran to completion
    pnl: float
    max_drawdown: float
    return_met: bool
    drawdown_ok: bool
    reason: str

    @property
    def passed(self) -> bool:
        return self.ok and self.return_met and self.drawdown_ok


def run_backtest(
    source: str,
    bars: list[dict[str, object]],
    *,
    initial_cash: float,
    cost_rate: float,
    return_target: float,
    max_drawdown_limit: float,
) -> BacktestReport:
    run = run_submission(
        source,
        entry="decide",
        driver=_replay,
        stdin_obj={"bars": bars, "cash": initial_cash, "cost_rate": cost_rate},
        timeout=_WALL_TIMEOUT,
    )
    if not run.ok:
        return BacktestReport(False, 0.0, 0.0, False, False, run.error)
    curve_raw = run.result.get("equity_curve")
    if not isinstance(curve_raw, list) or not curve_raw:
        return BacktestReport(
            False, 0.0, 0.0, False, False, "backtest produced no equity curve"
        )
    try:
        curve = [float(x) for x in curve_raw]
    except (TypeError, ValueError):
        return BacktestReport(False, 0.0, 0.0, False, False, "non-numeric equity curve")
    equity = [initial_cash, *curve]
    pnl = equity[-1] / initial_cash - 1.0 if initial_cash else 0.0
    max_dd = _max_drawdown(equity)
    return_met = pnl >= return_target
    drawdown_ok = max_dd <= max_drawdown_limit
    reason = (
        f"pnl={pnl:.4f} (target {return_target:.4f}), "
        f"max_drawdown={max_dd:.4f} (limit {max_drawdown_limit:.4f})"
    )
    return BacktestReport(True, pnl, max_dd, return_met, drawdown_ok, reason)


def perfect_foresight_return(bars: list[dict[str, object]]) -> float:
    """Upper bound on a long-only [0, 1] strategy: capture every up-move, sit
    out every down-move (perfect hindsight, costs ignored). If even this can't
    reach the target, no admissible strategy can — the task is infeasible."""
    closes = [float(str(b["close"])) for b in bars]
    growth = 1.0
    for i in range(1, len(closes)):
        ret = closes[i] / closes[i - 1] - 1.0
        if ret > 0:
            growth *= 1.0 + ret
    return growth - 1.0


def _max_drawdown(equity: list[float]) -> float:
    peak = equity[0]
    worst = 0.0
    for value in equity:
        peak = max(peak, value)
        if peak > 0:
            worst = max(worst, (peak - value) / peak)
    return worst
