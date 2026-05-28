"""Graph sampling for the trading procedural builder.

Selects a price window (regime-biased by the prior), loads its bars, and
embeds them graph-natively: instrument + one bar node per day + account +
risk_limit, wired by has_bar / trades / limits.
"""

from __future__ import annotations

import random
import statistics
from pathlib import Path

from graphschema import Role, WorldGraph
from openrange_pack_sdk import (
    Manifest,
    PackPrior,
    add_edge,
    add_node,
    manifest_int,
    manifest_str,
)

from trading import data
from trading.ontology import ONTOLOGY_ID


def sample_graph(
    rng: random.Random,
    manifest: Manifest,
    prior: PackPrior | None,
    *,
    cache_dir: Path,
    fixture: Path,
) -> WorldGraph:
    product = manifest_str(manifest, "product", default="BTC-USD")
    window_days = manifest_int(manifest, "window_days", default=180)
    candidates = max(1, manifest_int(manifest, "regime_candidates", default=1))
    intensity = _regime_intensity(prior)
    window = _select_window(
        rng,
        product,
        window_days,
        candidates,
        intensity,
        cache_dir=cache_dir,
        fixture=fixture,
    )
    return _build_graph(product, window, intensity, manifest)


def _regime_intensity(prior: PackPrior | None) -> float:
    if prior is None:
        return 0.5
    raw = prior.difficulty.get("overall", 0.5)
    return min(1.0, max(0.0, float(raw)))


def _select_window(
    rng: random.Random,
    product: str,
    window_days: int,
    candidates: int,
    intensity: float,
    *,
    cache_dir: Path,
    fixture: Path,
) -> data.Window:
    # >1 candidate turns volatility into a curriculum axis: pick the window at
    # the intensity quantile of realized vol (calm at 0, most volatile at 1).
    windows: list[data.Window] = []
    for _ in range(candidates):
        start, end = data.select_window(rng.getrandbits(32), window_days)
        windows.append(
            data.load_window(product, start, end, cache_dir=cache_dir, fixture=fixture)
        )
    if len(windows) == 1:
        return windows[0]
    windows.sort(key=lambda w: _realized_vol(w.bars))
    index = min(len(windows) - 1, round(intensity * (len(windows) - 1)))
    return windows[index]


def _realized_vol(bars: list[data.Bar]) -> float:
    # Ranking proxy only — never enters the graph hash, so float math is fine.
    closes = [float(b.close) for b in bars]
    returns = [
        closes[i] / closes[i - 1] - 1.0 for i in range(1, len(closes)) if closes[i - 1]
    ]
    return statistics.pstdev(returns) if len(returns) > 1 else 0.0


def _build_graph(
    product: str,
    window: data.Window,
    intensity: float,
    manifest: Manifest,
) -> WorldGraph:
    bars = window.bars
    graph = WorldGraph(
        ontology=ONTOLOGY_ID,
        meta={
            "data_source": window.source,
            "product": product,
            "regime_intensity": intensity,
            "realized_vol": _realized_vol(bars),
        },
    )
    instrument_id = f"instrument:{product}"
    add_node(
        graph,
        kind="instrument",
        id=instrument_id,
        roles={Role.EXTERNAL},
        attrs={
            "symbol": product,
            "venue": manifest_str(manifest, "venue", default="coinbase"),
            "quote_currency": manifest_str(manifest, "quote_currency", default="USD"),
            "window_start": bars[0].day,
            "window_end": bars[-1].day,
        },
    )
    for seq, bar in enumerate(bars):
        bar_id = f"bar:{product}:{seq:04d}"
        add_node(
            graph,
            kind="bar",
            id=bar_id,
            roles={Role.EXTERNAL},
            attrs={
                "seq": seq,
                "day": bar.day,
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
            },
        )
        add_edge(graph, kind="has_bar", src=instrument_id, dst=bar_id)
    account_id = "account:main"
    add_node(
        graph,
        kind="account",
        id=account_id,
        roles={Role.ACTOR},
        attrs={
            "cash": manifest_str(manifest, "cash", default="10000"),
            "base_currency": manifest_str(manifest, "base_currency", default="USD"),
        },
    )
    add_edge(graph, kind="trades", src=account_id, dst=instrument_id)
    risk_id = "risk_limit:main"
    add_node(
        graph,
        kind="risk_limit",
        id=risk_id,
        attrs={
            "return_target": manifest_str(manifest, "return_target", default="0.05"),
            "max_drawdown": manifest_str(manifest, "max_drawdown", default="0.30"),
            "cost_bps": manifest_str(manifest, "cost_bps", default="10"),
        },
    )
    add_edge(graph, kind="limits", src=risk_id, dst=account_id)
    return graph
