"""Per-persona scoped memory for NPCs.

Training-free and dependency-free: ``DictMemory`` is a keyword-ranked note store.
The ``scope`` is a stable ``"{run}:{actor}"`` string INJECTED by the NPC wrapper,
never chosen by the model, so one persona can never read another persona's (or
the SUT's) notes. Ranking is deterministic (term overlap, then recency) so a
fixed seed replays identically.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_WORD = re.compile(r"[a-z0-9]+")

# Bound a persona's note store so recall can't grow the prompt without limit.
_MAX_NOTE_CHARS = 2000
_MAX_NOTES = 500


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
            items = self._items.setdefault(scope, [])
            items.append(content[:_MAX_NOTE_CHARS])
            del items[:-_MAX_NOTES]  # keep only the most recent notes per scope

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
