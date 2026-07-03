"""Per-persona scoped memory for NPCs.

Training-free and dependency-free by default: ``DictMemory`` is a keyword-ranked
note store. The ``ScopedMemory`` protocol is the seam to drop in a vector
backend later without touching the NPC.

The ``scope`` is a stable ``"{run}:{actor}"`` string INJECTED by the NPC wrapper,
never chosen by the model, so one persona can never read another persona's (or
the SUT's) notes. Ranking is deterministic (term overlap, then recency) so a
fixed seed replays identically.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

_WORD = re.compile(r"[a-z0-9]+")


@runtime_checkable
class ScopedMemory(Protocol):
    """A note store partitioned by an opaque ``scope`` string."""

    def store(self, scope: str, content: str) -> None: ...

    def retrieve(self, scope: str, query: str, k: int = 5) -> list[str]: ...


def _terms(text: str) -> set[str]:
    # Word tokens, so adjacent punctuation ("finance,") still matches "finance".
    return set(_WORD.findall(text.lower()))


@dataclass
class DictMemory:
    """In-process scoped notes with keyword-overlap retrieval.

    Retrieval returns notes sharing at least one query term, most-overlapping
    first; when nothing overlaps it falls back to the most recent notes, so a
    bare ``recall`` still surfaces context.
    """

    _items: dict[str, list[str]] = field(default_factory=dict)

    def store(self, scope: str, content: str) -> None:
        if content:
            self._items.setdefault(scope, []).append(content)

    def retrieve(self, scope: str, query: str, k: int = 5) -> list[str]:
        items = self._items.get(scope, [])
        if not items:
            return []
        q = _terms(query)
        if q:
            scored = sorted(
                enumerate(items),
                key=lambda pair: (-len(q & _terms(pair[1])), -pair[0]),
            )
            hits = [c for i, c in scored if q & _terms(c)]
            if hits:
                return hits[:k]
        return list(reversed(items))[:k]

    def scopes(self) -> Iterable[str]:
        return tuple(self._items)
