"""Keyless historical-OHLC sourcing for the trading pack.

The pack never redistributes market data — it *fetches* a past (immutable)
daily window from a keyless endpoint and caches it locally. Offline (and CI)
falls back to a committed synthetic fixture so tests stay hermetic. Prices are
carried as their exact source strings, never via float, so the world graph's
content hash is byte-stable across machines.
"""

from __future__ import annotations

import json
import logging
import urllib.request
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from random import Random
from urllib.error import URLError

_log = logging.getLogger(__name__)

_CANDLES_URL = "https://api.exchange.coinbase.com/products/{product}/candles"

# A fixed span of settled history to sample windows from, so a given seed
# always maps to the same dates. Kept comfortably in the past.
_HISTORY_START = date(2019, 1, 1)
_HISTORY_END = date(2024, 12, 31)

# Coinbase caps a request at 300 daily candles; an inclusive [start, end] spans
# length+1 bars, so the window length must stay at or below 299.
_MAX_WINDOW_DAYS = 299


@dataclass(frozen=True)
class Bar:
    day: str  # ISO date, UTC
    open: str
    high: str
    low: str
    close: str
    volume: str


@dataclass(frozen=True)
class Window:
    """Bars plus where they came from, so a fixture-backed (synthetic) world is
    never silently mistaken for one built from real market data."""

    bars: list[Bar]
    source: str  # "cache" | "live" | "fixture"


def select_window(seed: int, length_days: int) -> tuple[date, date]:
    """Map a seed to a deterministic ``[start, end]`` daily window that always
    lands inside the settled-history span."""
    if not 1 <= length_days <= _MAX_WINDOW_DAYS:
        raise ValueError(
            f"length_days must be in 1..{_MAX_WINDOW_DAYS}; got {length_days}"
        )
    # length_days <= 299 << span, so latest_offset is always >= 0 and end never
    # runs past _HISTORY_END.
    latest_offset = (_HISTORY_END - _HISTORY_START).days - length_days
    offset = Random(seed).randint(0, latest_offset)
    start = _HISTORY_START + timedelta(days=offset)
    return start, start + timedelta(days=length_days)


def fetch_daily(product: str, start: date, end: date) -> list[Bar]:
    """Fetch settled daily bars for ``product`` over ``[start, end]`` from the
    keyless Coinbase candles endpoint, validated to cover the window."""
    url = (
        f"{_CANDLES_URL.format(product=product)}"
        f"?granularity=86400&start={start.isoformat()}&end={end.isoformat()}"
    )
    # Coinbase 403s the default urllib User-Agent; send an explicit one.
    request = urllib.request.Request(url, headers={"User-Agent": "openrange-trading"})
    with urllib.request.urlopen(request, timeout=20) as resp:  # noqa: S310 — fixed https host
        # parse_float/int=str keeps every number in its exact source text, so
        # the bars (and the graph hash built from them) are byte-stable.
        rows = json.loads(resp.read(), parse_float=str, parse_int=str)
    bars = [
        Bar(
            day=datetime.fromtimestamp(int(ts), tz=UTC).date().isoformat(),
            open=opn,
            high=high,
            low=low,
            close=close,
            volume=vol,
        )
        # Coinbase row order is [time, low, high, open, close, volume].
        for ts, low, high, opn, close, vol in sorted(rows, key=lambda r: int(r[0]))
    ]
    _validate(bars, start, end)
    return bars


def load_window(
    product: str,
    start: date,
    end: date,
    *,
    cache_dir: Path,
    fixture: Path,
) -> Window:
    """Real bars for the window: local cache → live fetch (then cache) →
    committed synthetic fixture when offline. The fixture is the only branch
    that doesn't reflect ``product``/dates, so it is tagged as such."""
    cached = cache_dir / f"{_safe(product)}_{start.isoformat()}_{end.isoformat()}.json"
    if cached.exists():
        try:
            bars = _bars_from_json(cached.read_text(encoding="utf-8"))
            _validate(bars, start, end)
            return Window(bars, "cache")
        except (ValueError, TypeError, OSError) as exc:
            # A corrupt or schema-drifted cache is a miss, not a crash.
            _log.warning("ignoring unreadable cache %s: %s", cached, exc)
    try:
        bars = fetch_daily(product, start, end)
    except (URLError, OSError, TimeoutError, ValueError) as exc:
        _log.warning(
            "live fetch failed for %s [%s..%s]: %s — using synthetic fixture",
            product,
            start,
            end,
            exc,
        )
        return Window(_bars_from_json(fixture.read_text(encoding="utf-8")), "fixture")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached.write_text(_bars_to_json(bars), encoding="utf-8")
    return Window(bars, "live")


def _validate(bars: list[Bar], start: date, end: date) -> None:
    # A truncated or malformed series must degrade to the fixture, not poison
    # the graph hash with partial data.
    if not bars:
        raise ValueError("empty bar series")
    days = [b.day for b in bars]
    if days != sorted(set(days)):
        raise ValueError("bar days are not strictly increasing")
    if days[0] < start.isoformat() or days[-1] > end.isoformat():
        raise ValueError(
            f"bars [{days[0]}..{days[-1]}] fall outside window "
            f"[{start.isoformat()}..{end.isoformat()}]"
        )


def _safe(product: str) -> str:
    # product ids are exchange-issued but still land on the filesystem.
    return "".join(c if c.isalnum() or c in "-." else "_" for c in product)


def _bars_to_json(bars: list[Bar]) -> str:
    return json.dumps([asdict(b) for b in bars], indent=2)


def _bars_from_json(text: str) -> list[Bar]:
    return [Bar(**row) for row in json.loads(text)]
