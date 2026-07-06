"""The cyber webapp wires the shared NPC comms channels end-to-end.

Proves the seam the generic PersonaAgent relies on: the pack surfaces the
identity-neutral mail/chat callables into ``surface_extras()`` and drains them
(attributed by sender) in ``collect_extras()``, and clears them per episode.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from cyber_webapp import WebappPack
from cyber_webapp.realize import WebappRuntime
from graphschema import WorldGraph
from openrange_pack_sdk import ChatStore, MailboxStore, PersonaAgent


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


class _RecBackend:
    """Records prompts; stands in for the LLM (satisfies the AgentBackend
    protocol structurally)."""

    def __init__(self) -> None:
        self.tools: tuple[Callable[..., Any], ...] = ()
        self.prompts: list[str] = []

    def preflight(self) -> None:
        pass

    def build_agent(
        self, *, system_prompt: str, tools: Sequence[Callable[..., Any]] = ()
    ) -> Callable[[str], object]:
        self.tools = tuple(tools)
        return self

    def __call__(self, prompt: str) -> object:
        self.prompts.append(prompt)
        return {"message": "ok"}


def test_persona_drives_the_real_runtime_surface_and_is_graded(tmp_path: Path) -> None:
    """End-to-end over the REAL WebappRuntime surface: a PersonaAgent binds the
    pack's surfaced mail tool, sends through it, and the runtime's collect_extras
    attributes the message by the persona's actor id — the grading path."""
    runtime = WebappRuntime(_sample_graph())
    runtime._base_url = "http://world.local"
    surface = dict(runtime.surface_extras())

    npc = PersonaAgent(
        config={"name": "Dana", "tools": ["mail_send"], "cadence_ticks": 1}
    )
    backend = _RecBackend()
    npc._backend_override = backend
    npc.start({"episode_id": "ep", "agent_backend": backend})
    npc._cooldown = 0
    npc.step(surface)  # builds the mail_send tool over the runtime's real surface

    mail_send = next(
        t for t in backend.tools if getattr(t, "__name__", "") == "mail_send"
    )
    # invoke the raw adapter (the model would call the decorated form the same way)
    raw = next(f for f in npc._tool_functions(surface) if f.__name__ == "mail_send")
    raw(to="Sam", subject="Q3", body="approve the batch")
    assert getattr(mail_send, "__name__", "") == "mail_send"

    runtime._base_url = None
    runtime._solver_root = tmp_path
    graded = runtime.collect_extras()
    assert graded["npc_mail"][0]["sender"] == "Dana"  # attributed to the persona
    assert graded["npc_mail"][0]["body"] == "approve the batch"
