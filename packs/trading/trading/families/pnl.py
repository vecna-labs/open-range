"""``trade.pnl`` — the agent writes a daily allocation strategy; a
look-ahead-safe sandboxed backtest replays it over the instrument's real price
window and scores P&L against the risk limits.

Feasibility uses the *real* window: a task is admissible only if its return
target is positive (not trivially passable) and reachable by some long-only
strategy (the perfect-foresight ceiling). ``available_mutations`` is the
patch-path curriculum — it tightens/loosens the success criteria in place; the
data-regime (grow) axis is driven from core via the prior.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from graphschema import Node, WorldGraph
from openrange_pack_sdk import (
    EpisodeReportLike,
    EpisodeResult,
    FeasibilityVerdict,
    Manifest,
    Mutation,
    PackPrior,
    TaskFamily,
    TaskSpec,
)

from trading.families.backtest import perfect_foresight_return, run_backtest

if TYPE_CHECKING:
    from openrange_pack_sdk import Snapshot


@dataclass(frozen=True)
class _Target:
    account: Node
    instrument: Node
    risk_limit: Node


class TradePnl(TaskFamily):
    id = "trade.pnl"
    pack_id = "trading"

    def generate(
        self,
        graph: WorldGraph,
        manifest: Manifest,
        prior: PackPrior | None,
    ) -> list[TaskSpec]:
        del manifest, prior
        target = self._pick_target(graph)
        if target is None:
            return []
        return [
            self.make_task(
                instruction=_instruction(target),
                entrypoints=target.account.id,
                goal_nodes=target.instrument.id,
                difficulty=_difficulty(target),
                meta={
                    "symbol": str(target.instrument.attrs.get("symbol")),
                    "return_target": str(target.risk_limit.attrs.get("return_target")),
                    "max_drawdown": str(target.risk_limit.attrs.get("max_drawdown")),
                },
            ),
        ]

    def check_feasibility(
        self,
        graph: WorldGraph,
        task: TaskSpec,
    ) -> FeasibilityVerdict:
        target = self._resolve_target(graph, task)
        if isinstance(target, FeasibilityVerdict):
            return target
        return_target = _float(target.risk_limit.attrs.get("return_target"), -1.0)
        if return_target <= 0:
            return FeasibilityVerdict(
                False, "return_target must be positive (else trivially passable)"
            )
        ceiling = perfect_foresight_return(_series(graph, target.instrument))
        if ceiling < return_target:
            return FeasibilityVerdict(
                False,
                f"no long-only strategy reaches {return_target:.4f}; "
                f"perfect-foresight ceiling is {ceiling:.4f}",
            )
        return FeasibilityVerdict(True)

    def check_success(
        self,
        graph: WorldGraph,
        task: TaskSpec,
        final_state: Mapping[str, Any],
    ) -> EpisodeResult:
        target = self._resolve_target(graph, task)
        if isinstance(target, FeasibilityVerdict):
            return EpisodeResult(
                success=False, reason=f"task target unresolvable: {target.reason}"
            )
        result = final_state.get("result")
        if not isinstance(result, Mapping):
            return EpisodeResult(
                success=False, reason="agent did not write result.json"
            )
        source = result.get("strategy")
        if not isinstance(source, str) or not source.strip():
            return EpisodeResult(
                success=False, reason="result.json missing non-empty 'strategy' string"
            )
        report = run_backtest(
            source,
            _series(graph, target.instrument),
            initial_cash=_float(target.account.attrs.get("cash"), 10000.0),
            cost_rate=_float(target.risk_limit.attrs.get("cost_bps"), 10.0) / 10000.0,
            return_target=_float(target.risk_limit.attrs.get("return_target"), 0.05),
            max_drawdown_limit=_float(
                target.risk_limit.attrs.get("max_drawdown"), 0.30
            ),
        )
        if not report.ok:
            return EpisodeResult(
                success=False, reason=f"strategy failed: {report.reason}"
            )
        return EpisodeResult(
            success=report.passed,
            subgoals={
                "return_target_met": report.return_met,
                "drawdown_ok": report.drawdown_ok,
            },
            reason=report.reason,
        )

    def available_mutations(
        self,
        snapshot: Snapshot,
        reports: Sequence[EpisodeReportLike],
        *,
        llm: object | None = None,
    ) -> tuple[Mutation, ...]:
        # Procedural patch-path: tighten/loosen the return target in place.
        # Harden beyond the data's reach simply fails admission and the evolve
        # loop falls through to soften.
        del reports, llm
        target = self._pick_target(snapshot.graph)
        if target is None:
            return ()
        return_target = _float(target.risk_limit.attrs.get("return_target"), 0.0)
        if return_target <= 0:
            return ()
        options: list[Mutation] = []
        harder = round(return_target * 1.5, 6)
        options.append(
            self.bump_scalar_attr(
                target.risk_limit,
                "return_target",
                str(harder),
                direction="harden",
                relevance=0.5,
                note=f"return target -> {harder}",
            )
        )
        softer = round(return_target * 0.5, 6)
        if softer > 0:
            options.append(
                self.bump_scalar_attr(
                    target.risk_limit,
                    "return_target",
                    str(softer),
                    direction="soften",
                    relevance=0.05,
                    note=f"return target -> {softer}",
                )
            )
        return tuple(options)

    def _pick_target(self, graph: WorldGraph) -> _Target | None:
        accounts = graph.by_kind("account")
        if not accounts:
            return None
        account = accounts[0]
        instrument = _first_dst(graph, account.id, "trades", "instrument")
        risk_limit = _first_src(graph, account.id, "limits", "risk_limit")
        if instrument is None or risk_limit is None:
            return None
        return _Target(account, instrument, risk_limit)

    def _resolve_target(
        self,
        graph: WorldGraph,
        task: TaskSpec,
    ) -> _Target | FeasibilityVerdict:
        if not task.entrypoints or not task.goal_nodes:
            return FeasibilityVerdict(False, "missing entrypoint or goal")
        account = graph.nodes.get(task.entrypoints[0])
        if account is None or account.kind != "account":
            return FeasibilityVerdict(False, "entrypoint is not an account")
        instrument = graph.nodes.get(task.goal_nodes[0])
        if instrument is None or instrument.kind != "instrument":
            return FeasibilityVerdict(False, "goal is not an instrument")
        if not any(
            e.dst == instrument.id for e in graph.out_edges(account.id, "trades")
        ):
            return FeasibilityVerdict(
                False, "account does not trade the goal instrument"
            )
        if len(graph.out_edges(instrument.id, "has_bar")) < 2:
            return FeasibilityVerdict(False, "instrument has too few bars to backtest")
        risk_limit = _first_src(graph, account.id, "limits", "risk_limit")
        if risk_limit is None:
            return FeasibilityVerdict(False, "account has no governing risk_limit")
        return _Target(account, instrument, risk_limit)


def _series(graph: WorldGraph, instrument: Node) -> list[dict[str, object]]:
    bars = sorted(
        (
            graph.nodes[e.dst]
            for e in graph.out_edges(instrument.id, "has_bar")
            if e.dst in graph.nodes
        ),
        key=lambda b: _as_int(b.attrs.get("seq")),
    )
    series: list[dict[str, object]] = []
    for bar in bars:
        series.append(
            {
                "seq": _as_int(bar.attrs.get("seq")),
                "day": str(bar.attrs.get("day")),
                "open": str(bar.attrs.get("open")),
                "high": str(bar.attrs.get("high")),
                "low": str(bar.attrs.get("low")),
                "close": str(bar.attrs.get("close")),
                "volume": str(bar.attrs.get("volume")),
            }
        )
    return series


def _first_dst(
    graph: WorldGraph, src: str, edge_kind: str, dst_kind: str
) -> Node | None:
    for edge in graph.out_edges(src, edge_kind):
        node = graph.nodes.get(edge.dst)
        if node is not None and node.kind == dst_kind:
            return node
    return None


def _first_src(
    graph: WorldGraph, dst: str, edge_kind: str, src_kind: str
) -> Node | None:
    for edge in graph.in_edges(dst, edge_kind):
        node = graph.nodes.get(edge.src)
        if node is not None and node.kind == src_kind:
            return node
    return None


def _instruction(target: _Target) -> str:
    symbol = target.instrument.attrs.get("symbol")
    cash = target.account.attrs.get("cash")
    return_target = target.risk_limit.attrs.get("return_target")
    max_drawdown = target.risk_limit.attrs.get("max_drawdown")
    cost_bps = target.risk_limit.attrs.get("cost_bps")
    return f"""Implement a daily allocation strategy for {symbol}:

    def decide(history: list[dict]) -> float

`history` is every bar observed UP TO AND INCLUDING today, oldest first — you
never see the future. Each bar is a dict with integer "seq", string "day", and
string decimal "open"/"high"/"low"/"close"/"volume". Return your target
exposure for the next day as a fraction of equity in [0, 1] (0 = all cash, 1 =
fully invested); out-of-range values are clamped.

You start with {cash} cash. The backtest rebalances to your target each day at
that day's close, charging {cost_bps} bps on the traded notional. Beat a
{return_target} total return without ever drawing down more than {max_drawdown}
from your equity peak.

Write your function to result.json as
{{"strategy": "def decide(history): ..."}}. The episode ends when result.json
appears; it is replayed over the instrument's price window in a sandbox.
"""


def _difficulty(target: _Target) -> float:
    return_target = _float(target.risk_limit.attrs.get("return_target"), 0.05)
    return min(1.0, max(0.05, return_target / 0.5))


def _float(value: object, default: float) -> float:
    try:
        return float(str(value))
    except TypeError, ValueError:
        return default


def _as_int(value: object) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else -1
