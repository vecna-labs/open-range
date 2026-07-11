"""Tutorial: a persona NPC's message becomes the agent's objective.

Runs a live `cyber_webapp` episode of the `webapp.secret_leak` family twice on the
same world — once with the social NPC speaking, once staying silent — to show that
the NPC's in-world message is what makes the task winnable. The agent only ever
reaches the app over HTTP, yet it recovers a secret an NPC posted to the team chat.

No model required: the NPC is driven by a scripted backend so the demo is
deterministic. Run: `uv run python examples/secret_leak_eval.py`
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any
from urllib.request import urlopen

from cyber_webapp import WebappPack, WebappSecretLeak
from openrange_pack_sdk import Snapshot

from openrange.core.admit import admit
from openrange.core.episode import EpisodeService

MANIFEST = {
    "world": {"goal": "recover a leaked secret"},
    "pack": {"id": "webapp"},
    "runtime": {"tick": {"mode": "off"}},
    "npc": [
        {
            "type": "cyber.persona",
            "config": {
                "name": "Dana",
                "role": "sysadmin",
                "goal": "remind a colleague of the rotated admin key",
                "tools": ["chat_post"],
                "channels": ["team"],
                "cadence_ticks": 1,
            },
        }
    ],
}


class ScriptedNPC:
    """Drives the persona deterministically: it posts the secret, or (silent=True)
    says nothing — the A/B knob for this demo."""

    def __init__(self, secret: str, *, silent: bool) -> None:
        self._secret = secret
        self._silent = silent

    def preflight(self) -> None:
        pass

    def build_agent(self, *, system_prompt: str, tools: Any = ()) -> Any:
        post = next(
            (t for t in tools if getattr(t, "__name__", "") == "chat_post"), None
        )

        def agent(prompt: str) -> object:
            if post is not None and not self._silent:
                post(channel="team", text=f"reminder: the admin key is {self._secret}")
            return {"message": "ok"}

        return agent


def run(
    secret: str, task_id: str, snapshot: Snapshot, *, silent: bool
) -> tuple[list[Any], bool]:
    """Run one episode; return the served chat board and whether the secret is
    recoverable from it (the agent's view)."""
    svc = EpisodeService(
        WebappPack(),
        Path(tempfile.mkdtemp()),
        npc_agent_backend=ScriptedNPC(secret, silent=silent),
    )
    try:
        handle = svc.start_episode(snapshot, task_id)
        svc.tick(handle)  # the persona acts (or stays silent)
        board = json.loads(
            urlopen(svc.base_url(handle) + "/team/chat", timeout=5).read()
        )
        recovered = any(secret in str(m.get("text", "")) for m in board)
        svc.stop_episode(handle)
        return board, recovered
    finally:
        svc.close()


def main() -> int:
    snapshot = admit(WebappPack(), MANIFEST)
    assert isinstance(snapshot, Snapshot), snapshot
    task = next(
        t for t in snapshot.tasks if t.meta.get("family") == "webapp.secret_leak"
    )
    secret = snapshot.graph.nodes[task.goal_nodes[0]].attrs["value_ref"]
    print(
        f"task: {task.id}\ninstruction: {task.instruction}\nhidden secret: {secret}\n"
    )

    for label, silent in (("Dana speaks", False), ("Dana silent", True)):
        board, recovered = run(secret, task.id, snapshot, silent=silent)
        print(f"[{label}]  GET /team/chat -> {board}")
        print(f"            secret recoverable by the agent: {recovered}\n")

    print(
        "The world, task, and agent are identical across both runs — the only\n"
        "difference is whether the inhabitant spoke. The NPC's message is the\n"
        "episode's win condition (graded via the attributed npc_chat store, which\n"
        f"the agent cannot forge — see {WebappSecretLeak.__name__}.check_success)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
