"""Pure-Python Aho-Corasick multi-pattern substring matcher.

Used by :mod:`cyber_webapp.consequence` to scan responses for every guarded
value's encodings in a single pass, instead of one substring search per secret
per response (which grows O(secrets) — see OpenRange #262). One automaton is
built from all patterns, then each body is scanned once.

An Aho-Corasick match is exactly ``pattern in body`` (an unanchored substring
match reported at every occurrence), so swapping the per-secret
``any(var in body)`` loop for this automaton yields *identical* leak results —
only faster as the pattern count grows.

The generated runtime (``codegen/templates/app.py.j2``) inlines a matcher of
the same shape in ``_scan_leaks`` because the template renders into a
standalone container and cannot import this pack. The two MUST stay
behavior-identical: any change here should be mirrored there.
"""

from __future__ import annotations

from collections import deque


class AhoCorasick:
    """Multi-pattern substring matcher keyed by an opaque string payload.

    Add ``(pattern, payload)`` pairs, :meth:`build` the failure links once, then
    :meth:`scan` bodies. A scan returns the set of payloads whose pattern occurs
    as a substring of the scanned text. The same payload may be registered under
    several patterns (a value's encodings) and several payloads may share one
    pattern (distinct nodes with a colliding value); both are handled.
    """

    __slots__ = ("_children", "_fail", "_out")

    def __init__(self) -> None:
        # Parallel arrays indexed by state id; state 0 is the root. ``_children``
        # is the trie/goto function, ``_fail`` the failure links (filled by
        # :meth:`build`), ``_out`` the payloads reported on reaching a state.
        self._children: list[dict[str, int]] = [{}]
        self._fail: list[int] = [0]
        self._out: list[list[str]] = [[]]

    def add(self, pattern: str, payload: str) -> None:
        """Register ``payload`` to fire wherever ``pattern`` occurs.

        An empty pattern is ignored: callers only feed non-empty encodings of
        real secrets, and a zero-length needle has no meaningful substring
        match to report.
        """
        if not pattern:
            return
        node = 0
        for ch in pattern:
            child = self._children[node]
            nxt = child.get(ch)
            if nxt is None:
                nxt = len(self._children)
                self._children.append({})
                self._fail.append(0)
                self._out.append([])
                child[ch] = nxt
            node = nxt
        self._out[node].append(payload)

    def build(self) -> None:
        """Wire failure links and propagate outputs (breadth-first over depth)."""
        queue: deque[int] = deque(self._children[0].values())
        while queue:
            node = queue.popleft()
            for ch, nxt in self._children[node].items():
                queue.append(nxt)
                fail = self._fail[node]
                while fail and ch not in self._children[fail]:
                    fail = self._fail[fail]
                dest = self._children[fail].get(ch, 0)
                # A depth-1 state's goto sits on the root, whose own goto[ch] is
                # that same state; its failure link must fall back to the root.
                self._fail[nxt] = dest if dest != nxt else 0
                self._out[nxt].extend(self._out[self._fail[nxt]])

    def scan(self, text: str) -> set[str]:
        """Return every payload whose pattern occurs as a substring of ``text``."""
        found: set[str] = set()
        children = self._children
        fail = self._fail
        out = self._out
        node = 0
        for ch in text:
            while node and ch not in children[node]:
                node = fail[node]
            node = children[node].get(ch, 0)
            if out[node]:
                found.update(out[node])
        return found
