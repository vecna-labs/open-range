"""Close the LLM-realization loop with a real LLM (the codex harness).

The LLM writes a command-injection handler; we inject it into a procedurally-built
world and run it through the dynamic admission gate (cyber_webapp.realize_admit): the
exploit must leak the flag, a benign request must not. Accepted handlers are the LLM's
own varied-but-valid implementations; trivial or broken ones are rejected.

Run::

    uv run python -m examples.cyber_realize --rounds 5
"""

from __future__ import annotations

import argparse
import tempfile
import urllib.request
from pathlib import Path

from cyber_webapp import WebappPack
from cyber_webapp.realize_admit import (
    AdmissionVerdict,
    classify_admission,
    cmdi_exploit_and_benign,
)
from graphschema import Node, WorldGraph
from openrange_pack_sdk import LLMBackend, LLMRequest, Snapshot

from openrange.core.admit import admit
from openrange.core.episode import EpisodeService
from openrange.llm import ClaudeBackend, CodexBackend

_MANIFEST: dict[str, object] = {
    "pack": {"id": "webapp"},
    "runtime": {"tick": {"mode": "off"}},
    "npc": [],
    "seed": 7,
    "loot_shapes": {"file": 1, "db": 0},
    "vuln_kinds": {"command_injection": 1},
}

_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["handler"],
    "properties": {"handler": {"type": "string"}},
}

_SYSTEM = (
    "You author one level for an authorized, local security-training gym "
    "(standard CTF authoring / defensive security education)."
)


def _cmdi_vuln(graph: WorldGraph) -> Node:
    return next(
        n
        for n in graph.by_kind("vulnerability")
        if n.attrs.get("kind") == "command_injection"
    )


def _flag_path(graph: WorldGraph) -> str:
    for record in graph.by_kind("record"):
        for edge in graph.out_edges(record.id, "holds"):
            if graph.nodes[edge.dst].attrs.get("kind") == "flag":
                return str(record.attrs["key"])
    raise SystemExit("world has no flag record")


def _prompt(param: str, flag_path: str) -> str:
    return (
        "Write a Python `def handle(query, state):` for an OS command-injection "
        "challenge.\n"
        "- query is dict[str, list[str]] (a parsed query string).\n"
        f"- state['files'] maps file paths to contents; the flag is the file at "
        f"{flag_path!r}.\n"
        f"- Read user input from query[{param!r}]. The handler must be vulnerable to "
        "command injection: return the flag's contents ONLY when the input chains a "
        f"shell command after a ';' separator (e.g. '127.0.0.1; cat {flag_path}').\n"
        "- A benign value like '127.0.0.1' (no ';') must NOT return the flag.\n"
        "- Return (status:int, headers:dict, body:bytes). Standard library only.\n"
        'Return JSON: {"handler": "<the full def handle source>"}.'
    )


def _generate(backend: LLMBackend, param: str, flag_path: str) -> str:
    result = backend.complete(
        LLMRequest(
            prompt=_prompt(param, flag_path),
            system=_SYSTEM,
            json_schema=_SCHEMA,
        )
    )
    parsed = result.parsed_json or {}
    handler = parsed.get("handler")
    return handler if isinstance(handler, str) else ""


def _gate(snap: Snapshot, handler: str, tmp_path: Path) -> AdmissionVerdict:
    graph = snap.graph
    _cmdi_vuln(graph).attrs["realized_handler"] = handler
    exploit_path, benign_path = cmdi_exploit_and_benign(graph)
    service = EpisodeService(WebappPack(), tmp_path)
    try:
        task = next(t for t in snap.tasks if t.meta.get("family") == "webapp.pentest")
        handle = service.start_episode(snap, task.id)
        base = str(service.surface(handle)["base_url"])
        exploit_body = (
            urllib.request.urlopen(base + exploit_path, timeout=10).read().decode()
        )
        benign_body = (
            urllib.request.urlopen(base + benign_path, timeout=10).read().decode()
        )
    finally:
        service.close()
    return classify_admission(graph, exploit_body, benign_body)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--backend", choices=("claude", "codex"), default="claude")
    args = parser.parse_args(argv)

    backend = ClaudeBackend() if args.backend == "claude" else CodexBackend()
    backend.preflight()
    snap = admit(WebappPack(), manifest=_MANIFEST, max_repairs=3)
    assert isinstance(snap, Snapshot), snap
    vuln = _cmdi_vuln(snap.graph)
    params = vuln.attrs["params"]
    assert isinstance(params, dict)
    params["inj_context"] = "separator"  # pin the exploit shape the gate will use
    param = str(params["target_param"])
    flag_path = _flag_path(snap.graph)

    accepted: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        for index in range(args.rounds):
            handler = _generate(backend, param, flag_path)
            if not handler.strip():
                print(f"round {index}: REFUSED/empty — no handler returned")
                continue
            try:
                verdict = _gate(snap, handler, Path(tmp) / f"r{index}")
            except Exception as exc:  # noqa: BLE001
                print(f"round {index}: REJECT — handler crashed the world: {exc}")
                continue
            print(
                f"round {index}: {'ACCEPT' if verdict.accepted else 'REJECT'} "
                f"— {verdict.reason}"
            )
            if verdict.accepted:
                accepted.append(handler)

    print(
        f"\n{len(accepted)}/{args.rounds} accepted; "
        f"{len(set(accepted))} distinct accepted implementations"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
