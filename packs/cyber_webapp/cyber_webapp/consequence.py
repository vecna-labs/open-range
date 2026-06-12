"""Consequence verification — did a guarded (HIDDEN) value reach observed output.

A world's HIDDEN ``value_ref`` nodes are the values an observer must not be able to
read off the wire. This scans observed responses for any of them and reports which
leaked. It judges by *content alone*: a benign response and an exploit response are
treated the same way, so it answers only *that* a guarded value crossed into output
— not whether the path that produced it was the intended exploit (a separate
question the mutually-exclusive injection contexts settle).

In a planted world the guarded set is just the flag, so this agrees with the
planted-flag verdict by construction; the generalization earns its keep when a world
holds secrets beyond the one designated goal.

Limitations (a raw-substring oracle, honest about what it does NOT catch — these are
safe today because the sole guarded value is the long, random flag, but each must be
addressed before many/short HIDDEN values land):
  - Encoded exfil: a value returned base64/hex/url-encoded/gzipped does not contain
    its literal form, so it reads as no-leak. Canonicalizing bodies would widen this.
  - Containment: two guarded values in a substring relationship over-report (leaking
    the longer flags both); there is no containment de-duplication.
A length floor (below) removes the worst false-positive — a short value colliding
with benign text.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from graphschema import Visibility, WorldGraph

# Guarded values are matched by unanchored substring search, so a short value_ref
# would collide with ordinary response text (HTML, openapi.json, decoys). Real
# secrets clear this comfortably; a degenerate one is excluded rather than allowed
# to report a leak on every response.
_MIN_GUARDED_LEN = 8


@dataclass(frozen=True)
class LeakVerdict:
    """The guarded nodes whose value appeared in observed output."""

    leaked: frozenset[str]

    @property
    def occurred(self) -> bool:
        return bool(self.leaked)


def guarded_values(graph: WorldGraph) -> dict[str, str]:
    """Map each HIDDEN node id to the ``value_ref`` that must not leak."""
    guarded: dict[str, str] = {}
    for node in graph.nodes.values():
        if node.visibility is not Visibility.HIDDEN:
            continue
        ref = node.attrs.get("value_ref")
        if isinstance(ref, str) and len(ref) >= _MIN_GUARDED_LEN:
            guarded[node.id] = ref
    return guarded


def detect_leak(graph: WorldGraph, responses: Iterable[str]) -> LeakVerdict:
    """Return the guarded nodes whose value appears in any observed response."""
    guarded = guarded_values(graph)
    if not guarded:
        return LeakVerdict(frozenset())
    bodies = list(responses)
    leaked = {
        node_id
        for node_id, value in guarded.items()
        if any(value in body for body in bodies)
    }
    return LeakVerdict(frozenset(leaked))
