"""Ontology contract for the trading pack — a backtest world over real OHLC.

The world's identity *is* its price series, so the series is embedded
graph-natively: one ``bar`` node per day (not an opaque blob attr), linked to
its ``instrument``. The agent's ``account`` and the ``risk_limit`` the
curriculum tunes round it out. Prices are exact source strings so the graph's
content hash is byte-stable across machines.
"""

from __future__ import annotations

from graphschema import AttrSpec, AttrType, EdgeKind, NodeKind, Ontology

ONTOLOGY_ID = "trading.market@v1"


def market_ontology() -> Ontology:
    # fresh instance per call so callers can mutate without leaking
    s = AttrSpec
    return Ontology(
        id=ONTOLOGY_ID,
        node_kinds={
            "instrument": NodeKind(
                "instrument",
                attrs={
                    "symbol": s(AttrType.STRING, required=True),
                    "venue": s(
                        AttrType.STRING,
                        required=True,
                        description="data source the bars were fetched from",
                    ),
                    "quote_currency": s(AttrType.STRING, default="USD"),
                    "window_start": s(
                        AttrType.STRING,
                        required=True,
                        description="ISO date of the first bar",
                    ),
                    "window_end": s(
                        AttrType.STRING,
                        required=True,
                        description="ISO date of the last bar",
                    ),
                },
                description="a tradeable instrument and the window its bars span",
            ),
            "bar": NodeKind(
                "bar",
                attrs={
                    "seq": s(
                        AttrType.INT,
                        required=True,
                        description="0-based index in the series",
                    ),
                    "day": s(AttrType.STRING, required=True),
                    "open": s(AttrType.STRING, required=True),
                    "high": s(AttrType.STRING, required=True),
                    "low": s(AttrType.STRING, required=True),
                    "close": s(AttrType.STRING, required=True),
                    "volume": s(AttrType.STRING, required=True),
                },
                description="one daily OHLCV bar; prices are exact source strings "
                "so the graph hash is byte-stable",
            ),
            "account": NodeKind(
                "account",
                attrs={
                    "cash": s(
                        AttrType.STRING,
                        required=True,
                        description="starting cash, exact decimal",
                    ),
                    "base_currency": s(AttrType.STRING, default="USD"),
                },
                description="the agent's trading account",
            ),
            "risk_limit": NodeKind(
                "risk_limit",
                attrs={
                    "return_target": s(
                        AttrType.STRING,
                        required=True,
                        description="fractional total return to beat, e.g. 0.05",
                    ),
                    "max_drawdown": s(
                        AttrType.STRING,
                        required=True,
                        description="max tolerated peak-to-trough equity fraction",
                    ),
                    "cost_bps": s(
                        AttrType.STRING,
                        default="10",
                        description="per-trade transaction cost, basis points",
                    ),
                },
                description="success + risk thresholds the curriculum tunes",
            ),
        },
        edge_kinds={
            "has_bar": EdgeKind(
                "has_bar",
                endpoints=[("instrument", "bar")],
                description="this bar belongs to the instrument's series",
            ),
            "trades": EdgeKind(
                "trades",
                endpoints=[("account", "instrument")],
                description="this account may trade this instrument",
            ),
            "limits": EdgeKind(
                "limits",
                endpoints=[("risk_limit", "account")],
                description="these thresholds govern this account",
            ),
        },
    )
