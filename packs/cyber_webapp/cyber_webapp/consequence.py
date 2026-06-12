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

It catches a value returned literally or under a cheap reversible text encoding
(base64, hex, percent-encoding) — by searching for those *encodings of the value*,
which finds it even as a substring of a larger body. Still out (would need decoding
the body, not encoding the needle): gzip/binary transforms, multibyte splits, bespoke
schemes. Containment: when several guarded values leak and one is a substring of
another, only the maximal value is reported. A length floor excludes a short value_ref
that would otherwise collide with benign text.

Note the live runtime signal (``final_state["leaked_secret_ids"]``) is per-response
and does not apply containment — the scanner logs node ids, not values, so it cannot
compare them — so this offline verifier (and the grader, which hold the graph) is the
de-duped verdict.
"""

from __future__ import annotations

import base64
import urllib.parse
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


def value_variants(value: str) -> set[str]:
    """The forms a value might take in a response: literal + cheap text encodings.

    Encoding the *needle* (rather than decoding the body) finds the value even when
    it is a substring of a larger response. The rendered app's scanner mirrors this,
    so the live and offline verdicts agree.
    """
    raw = value.encode()
    b64 = base64.b64encode(raw).decode()
    return {value, b64, b64.rstrip("="), raw.hex(), urllib.parse.quote(value, safe="")}


def _drop_contained(leaked: set[str], guarded: dict[str, str]) -> frozenset[str]:
    # Only the maximal value genuinely leaked; a shorter value that is a proper
    # substring of another leaked value is an artifact of the unanchored match.
    # Distinct nodes sharing one value are both kept.
    return frozenset(
        node_id
        for node_id in leaked
        if not any(
            other != node_id
            and guarded[node_id] != guarded[other]
            and guarded[node_id] in guarded[other]
            for other in leaked
        )
    )


def detect_leak(graph: WorldGraph, responses: Iterable[str]) -> LeakVerdict:
    """Return the guarded nodes whose value appears in any observed response."""
    guarded = guarded_values(graph)
    if not guarded:
        return LeakVerdict(frozenset())
    bodies = list(responses)
    leaked = {
        node_id
        for node_id, value in guarded.items()
        if any(var in body for var in value_variants(value) for body in bodies)
    }
    return LeakVerdict(_drop_contained(leaked, guarded))
