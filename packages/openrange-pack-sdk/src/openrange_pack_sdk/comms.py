"""Environment-mediated communication channels for NPCs.

Comms flow THROUGH THE WORLD, never through a direct agent-to-agent handle: an
NPC writes a shared store via a pack-surfaced callable, and other NPCs (and a
grader) perceive the message only by reading the same store. A pack surfaces
these closures from ``surface_extras()`` so they land in every NPC's per-tick
``interface``, and drains the store in ``collect_extras()`` so a grader can read
what was said and attribute it by sender.

Runtime hands the SAME frozen surface mapping to every NPC, so the surface
closures are IDENTITY-NEUTRAL: ``sender`` is an explicit argument. Identity is
bound one layer up, NPC-side, where each persona injects its own ``actor_id``
(and never exposes ``sender`` to the model) — so cover traffic is attributable
by construction and a persona cannot forge another's identity.

Everything here is deterministic (monotonic ids, insertion order, no clock or
randomness) so a fixed seed replays identically. The stores are NOT thread-safe;
the harness steps NPCs on a single thread today, so a lock would be dead weight
(add one if parallel/async stepping ever lands).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Message:
    """One delivered message. ``recipient`` is a mailbox name or a chat channel;
    the empty string is a mailbox broadcast."""

    id: int
    sender: str
    recipient: str
    subject: str
    body: str

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "sender": self.sender,
            "recipient": self.recipient,
            "subject": self.subject,
            "body": self.body,
        }


@dataclass
class MailboxStore:
    """Directed mail: ``send(to=...)`` lands in one box; a recipient of ``""``
    is a broadcast every box also reads. ``read`` takes a ``since`` cursor so a
    reader only sees new mail."""

    _messages: list[Message] = field(default_factory=list)
    _counter: int = 0

    def send(self, *, sender: str, to: str, subject: str = "", body: str = "") -> int:
        self._counter += 1
        self._messages.append(Message(self._counter, sender, to, subject, body))
        return self._counter

    def read(self, box: str, since: int = 0) -> list[Message]:
        return [m for m in self._messages if m.id > since and m.recipient in (box, "")]

    def all(self) -> list[Message]:
        """Every message, for ``collect_extras()`` / grading."""
        return list(self._messages)


@dataclass
class ChatStore:
    """Channel chat: ``post(channel=...)`` is visible to everyone who reads that
    channel. ``since`` lets a reader poll only new lines."""

    _messages: list[Message] = field(default_factory=list)
    _counter: int = 0

    def post(self, *, sender: str, channel: str, text: str) -> int:
        self._counter += 1
        self._messages.append(Message(self._counter, sender, channel, "", text))
        return self._counter

    def read(self, channel: str, since: int = 0) -> list[Message]:
        return [m for m in self._messages if m.recipient == channel and m.id > since]

    def all(self) -> list[Message]:
        return list(self._messages)


def surface_mailbox(store: MailboxStore) -> dict[str, Callable[..., object]]:
    """The identity-neutral ``mail_send`` / ``mail_read`` callables a pack merges
    into ``surface_extras()``. ``sender``/``box`` are explicit because one shared
    surface serves every NPC; the persona binds its own identity NPC-side."""

    def mail_send(sender: str, to: str, subject: str = "", body: str = "") -> str:
        mid = store.send(sender=sender, to=to, subject=subject, body=body)
        return f"sent#{mid}"

    def mail_read(box: str, since: int = 0) -> list[dict[str, object]]:
        return [m.as_dict() for m in store.read(box, since)]

    return {"mail_send": mail_send, "mail_read": mail_read}


def surface_chat(store: ChatStore) -> dict[str, Callable[..., object]]:
    """The identity-neutral ``chat_post`` / ``chat_read`` callables a pack merges
    into ``surface_extras()``."""

    def chat_post(sender: str, channel: str, text: str) -> str:
        mid = store.post(sender=sender, channel=channel, text=text)
        return f"posted#{mid}"

    def chat_read(channel: str, since: int = 0) -> list[dict[str, object]]:
        return [m.as_dict() for m in store.read(channel, since)]

    return {"chat_post": chat_post, "chat_read": chat_read}
