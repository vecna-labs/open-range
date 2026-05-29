# Trading pack

A backtest world over **real** historical price data. The builder fetches a
past daily OHLC window (keyless Coinbase, cached locally; a committed synthetic
fixture keeps CI offline) and embeds it graph-natively — one `bar` node per day.
The `trade.pnl` task asks the solver for a daily allocation strategy, and a
look-ahead-safe sandboxed backtest grades it.

## A worked episode

**What the builder built** (live data):

- `BTC-USD`, 121 daily bars, 2021-06-22 → 2021-10-20
- an account with `$10,000`
- a risk limit: beat **+5%** total return with **≤30%** max drawdown, 10 bps per trade

**The task** the solver receives:

> Implement `decide(history: list[dict]) -> float` for BTC-USD. `history` is
> every bar up to and including today (you never see the future); return your
> target exposure for the next day as a fraction of equity in `[0, 1]`. Beat a
> +5% return without drawing down more than 30%. Write it to `result.json` as
> `{"strategy": "def decide(history): ..."}`.

**What the solver produces** — the source of that function, e.g. buy-and-hold:

```python
def decide(history):
    return 1.0
```

**The judge** replays the function look-ahead-safe over the window (handing it
only `bars[:t+1]` each day, charging the cost), turns the equity curve into P&L
and max-drawdown, and checks them against the risk limit. For the window above:

| strategy                     | verdict                            |
| ---------------------------- | ---------------------------------- |
| buy-and-hold                 | pass — +102.7% P&L, 22.7% drawdown |
| all-cash                     | fail — 0% < 5% target              |
| momentum (buy after up-days) | pass — +35.2% P&L, 21.6% drawdown  |
| `return "buy"`               | fail — submission error            |

## Run it

```python
from openrange.core.admit import admit
from trading import TradingPack
from trading.families.pnl import TradePnl

snap = admit(TradingPack(), manifest={"seed": 2025, "window_days": 120})
task = snap.tasks[0]
print(task.instruction)  # the task text above

verdict = TradePnl().check_success(
    snap.graph, task,
    {"result": {"strategy": "def decide(history):\n    return 1.0\n"}},
)
print(verdict.success, verdict.reason)
```

The window (and so the numbers) follows from the build seed. Feasibility uses
the window's perfect-foresight ceiling, so a return target no long-only strategy
could reach never admits. The data-regime axis (calmer ↔ more volatile window)
is the curriculum's grow move, driven from core via the prior.
