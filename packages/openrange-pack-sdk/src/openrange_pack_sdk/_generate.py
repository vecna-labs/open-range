"""Self-verifying world generation: author a part, keep it only if a verifier accepts.

An author — LLM-backed in practice — proposes part of a world, but the part ships
only if an independent verifier accepts it: the verifier, not the author, decides
what is real. The loop is domain-agnostic; a pack supplies the domain pieces through
:class:`WorldAuthor`. Booting the world to verify needs the runtime, so that transport
lives inside the author's ``verify`` and the loop stays runtime-free.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from openrange_pack_sdk._types import Snapshot


class WorldAuthor(Protocol):
    """A pack's hook for authoring self-verified parts of a world.

    For each key in ``authorable``, :func:`realize_verified` calls ``apply`` (which
    authors the part — typically via an LLM — and mutates ``snapshot.graph``,
    returning whether it took), then keeps it iff ``verify`` accepts it, else
    ``revert`` undoes it. ``verify`` owns the consequence check — booting the world
    and running the same gate that grades the agent — so a faked or unsolvable
    authoring is rejected here, not in training.
    """

    def authorable(self, snapshot: Snapshot) -> Sequence[str]: ...

    def apply(self, snapshot: Snapshot, key: str) -> bool: ...

    def verify(self, snapshot: Snapshot, key: str) -> bool: ...

    def revert(self, snapshot: Snapshot, key: str) -> None: ...


def realize_verified(snapshot: Snapshot, author: WorldAuthor) -> Snapshot:
    """Author each authorable key, keeping only the verifier-accepted ones, and
    re-freeze to a content-addressed snapshot recording them in ``lineage["realized"]``.
    Mutates ``snapshot.graph`` — use the returned snapshot.
    """
    realized: list[str] = []
    for key in author.authorable(snapshot):
        if not author.apply(snapshot, key):
            continue
        if author.verify(snapshot, key):
            realized.append(key)
        else:
            author.revert(snapshot, key)
    return _refrozen(snapshot, tuple(realized))


def _refrozen(snapshot: Snapshot, realized: tuple[str, ...]) -> Snapshot:
    graph = snapshot.graph
    return Snapshot(
        snapshot_id=graph.content_hash(),
        ontology_id=snapshot.ontology_id,
        graph=graph,
        tasks=snapshot.tasks,
        lineage={**dict(snapshot.lineage), "realized": realized},
        history=snapshot.history,
    )
