"""The cyber webapp wires the shared NPC comms channels end-to-end.

Proves the seam the generic PersonaAgent relies on: the pack surfaces the
identity-neutral mail/chat callables into ``surface_extras()`` and drains them
(attributed by sender) in ``collect_extras()``, and clears them per episode.
"""

from __future__ import annotations

from pathlib import Path

from cyber_webapp import WebappPack
from cyber_webapp.realize import WebappRuntime
from graphschema import WorldGraph
from openrange_pack_sdk import ChatStore, MailboxStore


def _sample_graph(seed: int = 0) -> WorldGraph:
    return WebappPack().make_builder(None).build({"seed": seed}).graph


def test_webapp_surfaces_and_drains_npc_comms(tmp_path: Path) -> None:
    runtime = WebappRuntime(_sample_graph())
    assert isinstance(runtime._mailbox, MailboxStore)
    assert isinstance(runtime._chat, ChatStore)

    # surface_extras merges the identity-neutral comms callables alongside http
    runtime._base_url = "http://test.local"
    surface = runtime.surface_extras()
    assert {"http_get", "mail_send", "mail_read", "chat_post", "chat_read"} <= set(
        surface
    )

    # a message written through the surface lands in the world store...
    surface["mail_send"](sender="dana", to="sam", subject="q3", body="reconcile please")
    surface["chat_post"](sender="exec", channel="office", text="standup at 10")

    # ...and collect_extras drains it, attributed by sender, for grading
    runtime._base_url = None
    runtime._solver_root = tmp_path
    collected = runtime.collect_extras()
    assert collected["npc_mail"][0]["sender"] == "dana"
    assert collected["npc_mail"][0]["body"] == "reconcile please"
    assert collected["npc_chat"][0]["sender"] == "exec"

    # per-episode reset clears comms so a warm-pooled world doesn't leak
    runtime.reset_episode()
    assert runtime._mailbox.all() == []
    assert runtime._chat.all() == []
