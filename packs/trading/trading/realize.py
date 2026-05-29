"""TradingRuntime — a thin OnDemandRuntime for the backtest world.

No persistent process: the agent writes a ``decide`` strategy to ``result.json``
and the trade.pnl grader replays it in a sandbox. The runtime just hands the
agent a clean workspace plus ``bars.json`` (the price window) to study — the
look-ahead guard lives in the grader, so seeing the full window at authoring
time is fine. Bars come from the graph, never a re-fetch, so realize is
offline-safe.
"""

from __future__ import annotations

import json
from collections.abc import Mapping

from graphschema import Node, WorldGraph
from openrange_pack_sdk import Backing, OnDemandRuntime, OpenRangeError


class TradingRuntimeError(OpenRangeError):
    pass


class TradingRuntime(OnDemandRuntime):
    def __init__(self, graph: WorldGraph, backing: Backing) -> None:
        if backing is not Backing.PROCESS:
            raise NotImplementedError(
                f"TradingRuntime does not support backing={backing!r}; only "
                "Backing.PROCESS is wired (the backtest has no live process)"
            )
        super().__init__(graph)

    def prepare_env_files(self, graph: WorldGraph) -> Mapping[str, str]:
        # No persistent process → nothing to stage under pack_root.
        del graph
        return {}

    def reset(self) -> None:
        super().reset()
        assert self.solver_root is not None
        (self.solver_root / "bars.json").write_text(
            _bars_json(self._graph), encoding="utf-8"
        )


def _bars_json(graph: WorldGraph) -> str:
    instruments = graph.by_kind("instrument")
    if not instruments:
        return "[]"
    instrument = instruments[0]
    bars = sorted(
        (
            graph.nodes[e.dst]
            for e in graph.out_edges(instrument.id, "has_bar")
            if e.dst in graph.nodes
        ),
        key=_seq,
    )
    series = [
        {
            "seq": _seq(bar),
            "day": str(bar.attrs.get("day")),
            "open": str(bar.attrs.get("open")),
            "high": str(bar.attrs.get("high")),
            "low": str(bar.attrs.get("low")),
            "close": str(bar.attrs.get("close")),
            "volume": str(bar.attrs.get("volume")),
        }
        for bar in bars
    ]
    return json.dumps(series, indent=2)


def _seq(node: Node) -> int:
    raw = node.attrs.get("seq")
    return raw if isinstance(raw, int) and not isinstance(raw, bool) else -1
