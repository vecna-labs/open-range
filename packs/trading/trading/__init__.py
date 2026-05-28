from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from graphschema import Issue, Ontology, WorldGraph
from openrange_pack_sdk import (
    Backing,
    Builder,
    Pack,
    PackPrior,
    RuntimeHandle,
    TaskFamily,
)

from trading.builder import TradingBuilder
from trading.families import TradePnl
from trading.invariants import account_can_trade, bars_contiguous, ohlc_sane
from trading.ontology import ONTOLOGY_ID, market_ontology
from trading.priors import default_prior as _default_prior
from trading.realize import TradingRuntime


class TradingPack(Pack):
    id = "trading"
    version = "v1"

    def __init__(self, dir: Path | None = None) -> None:
        # accepted for parity with path-loaded packs; nothing on disk to load
        del dir
        self.dir = None

    def ontology(self) -> Ontology:
        return market_ontology()

    def invariants(self) -> list[Callable[[WorldGraph], list[Issue]]]:
        return [ohlc_sane, bars_contiguous, account_can_trade]

    def make_builder(self, prior: PackPrior | None) -> Builder:
        return TradingBuilder(prior)

    def default_prior(self) -> PackPrior | None:
        return _default_prior()

    def realize(self, graph: WorldGraph, backing: Backing) -> RuntimeHandle:
        return TradingRuntime(graph, backing)

    def task_families(self) -> list[TaskFamily]:
        return [TradePnl()]


__all__ = [
    "ONTOLOGY_ID",
    "TradePnl",
    "TradingBuilder",
    "TradingPack",
    "TradingRuntime",
    "account_can_trade",
    "bars_contiguous",
    "market_ontology",
    "ohlc_sane",
]
