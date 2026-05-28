"""Pack invariants for the trading world — admission layer 3.

Structurally-valid graphs that are still semantically broken: a bar whose
prices don't parse or violate low ≤ {open,close} ≤ high, an instrument whose
series isn't a contiguous ordered window, or an account with no instrument it
can actually trade. None of these are expressible as ``AttrSpec``s — they need
to walk the graph and reason about the embedded series.
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation

from graphschema import Issue, WorldGraph

_PRICE_FIELDS = ("open", "high", "low", "close")


def ohlc_sane(graph: WorldGraph) -> list[Issue]:
    """Every bar's prices parse as non-negative decimals with low <=
    {open, close} <= high, and volume parses as non-negative."""
    issues: list[Issue] = []
    for bar in graph.by_kind("bar"):
        prices: dict[str, Decimal] = {}
        bad = False
        for field in (*_PRICE_FIELDS, "volume"):
            try:
                value = Decimal(str(bar.attrs.get(field)))
            except InvalidOperation:
                issues.append(
                    Issue(
                        "error",
                        "bar_unparseable",
                        f"bar {bar.id!r} attr {field!r} is not a decimal: "
                        f"{bar.attrs.get(field)!r}",
                        bar.id,
                    )
                )
                bad = True
                continue
            if value < 0:
                issues.append(
                    Issue(
                        "error",
                        "bar_negative",
                        f"bar {bar.id!r} attr {field!r} is negative: {value}",
                        bar.id,
                    )
                )
                bad = True
            prices[field] = value
        if bad:
            continue
        low, high = prices["low"], prices["high"]
        if low > high:
            issues.append(
                Issue(
                    "error",
                    "bar_low_gt_high",
                    f"bar {bar.id!r} low {low} > high {high}",
                    bar.id,
                )
            )
            continue
        for field in ("open", "close"):
            if not low <= prices[field] <= high:
                issues.append(
                    Issue(
                        "error",
                        "bar_ohlc_range",
                        f"bar {bar.id!r} {field} {prices[field]} outside "
                        f"[low {low}, high {high}]",
                        bar.id,
                    )
                )
    return issues


def bars_contiguous(graph: WorldGraph) -> list[Issue]:
    """Each instrument exposes >=2 bars whose seq are a contiguous 0..n-1 and
    whose days strictly increase."""
    issues: list[Issue] = []
    for inst in graph.by_kind("instrument"):
        bars = [
            graph.nodes[e.dst]
            for e in graph.out_edges(inst.id, "has_bar")
            if e.dst in graph.nodes
        ]
        if len(bars) < 2:
            issues.append(
                Issue(
                    "error",
                    "instrument_too_few_bars",
                    f"instrument {inst.id!r} has {len(bars)} bars; need >=2",
                    inst.id,
                )
            )
            continue
        ordered = sorted(bars, key=lambda b: _as_int(b.attrs.get("seq")))
        if [_as_int(b.attrs.get("seq")) for b in ordered] != list(range(len(ordered))):
            issues.append(
                Issue(
                    "error",
                    "bar_seq_noncontiguous",
                    f"instrument {inst.id!r} bar seqs are not 0..{len(ordered) - 1}",
                    inst.id,
                )
            )
        days = [str(b.attrs.get("day")) for b in ordered]
        if days != sorted(set(days)):
            issues.append(
                Issue(
                    "error",
                    "bar_days_unordered",
                    f"instrument {inst.id!r} bar days are not strictly increasing",
                    inst.id,
                )
            )
    return issues


def account_can_trade(graph: WorldGraph) -> list[Issue]:
    """Each account trades at least one instrument that actually has bars."""
    bars_per_instrument: dict[str, int] = {}
    for edge in graph.edges.values():
        if edge.kind == "has_bar":
            bars_per_instrument[edge.src] = bars_per_instrument.get(edge.src, 0) + 1
    issues: list[Issue] = []
    for account in graph.by_kind("account"):
        tradable = [
            e.dst
            for e in graph.out_edges(account.id, "trades")
            if bars_per_instrument.get(e.dst, 0) > 0
        ]
        if not tradable:
            issues.append(
                Issue(
                    "error",
                    "account_no_tradable_instrument",
                    f"account {account.id!r} trades no instrument with bars",
                    account.id,
                )
            )
    return issues


def _as_int(value: object) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else -1
