"""Hand-authored default prior so the builder always has a signal to read.

``difficulty["overall"]`` ∈ [0, 1] is a *generic* intensity knob — core's grow
path steps it up/down without knowing what it means. The trading builder
interprets it as a volatility quantile when selecting the price window (calm at
0, the most volatile candidate at 1). The shape stays generic (DESIGN §9), so
an external producer can supply the same prior without the builder changing
code paths.
"""

from __future__ import annotations

from openrange_pack_sdk import PackPrior

from trading.ontology import market_ontology


def default_prior() -> PackPrior:
    return PackPrior(
        source="trading.default",
        ontology=market_ontology(),
        topology={},
        difficulty={"overall": 0.5},
    )
