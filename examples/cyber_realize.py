"""Close the LLM-realization loop with a real LLM (DESIGN.md §9, #260).

For each realizable class, the LLM writes the vuln handler; we inject it into a
procedurally-built world and run it through the dynamic admission gate (realize_admit +
reference_solver): the intended exploit must leak the flag, a benign request must not.
Accepted handlers are the LLM's own varied-but-valid implementations; trivial or broken
ones are rejected.

Run::

    uv run python -m examples.cyber_realize --rounds 3            # all classes
    uv run python -m examples.cyber_realize --kind sql_injection  # just one
"""

from __future__ import annotations

import argparse
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

from cyber_webapp import WebappPack
from cyber_webapp.llm_realize import (
    REALIZABLE_KINDS,
    handler_from_result,
    realization_request,
)
from cyber_webapp.realize_admit import AdmissionVerdict, classify_admission
from cyber_webapp.reference_solver import _vuln_of_kind, exploit_and_benign
from openrange_pack_sdk import Snapshot

from openrange.core.admit import admit
from openrange.core.episode import EpisodeService
from openrange.llm import ClaudeBackend, CodexBackend

_LOOT = {
    "command_injection": "file",
    "path_traversal": "file",
    "sql_injection": "db",
}


def _admit(kind: str) -> Snapshot:
    loot = _LOOT[kind]
    snap = admit(
        WebappPack(),
        manifest={
            "pack": {"id": "webapp"},
            "runtime": {"tick": {"mode": "off"}},
            "npc": [],
            "seed": 7,
            "loot_shapes": {loot: 1, "db" if loot == "file" else "file": 0},
            "vuln_kinds": {kind: 1},
        },
        max_repairs=3,
    )
    assert isinstance(snap, Snapshot), snap
    return snap


def _fetch(url: str) -> str:
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            return str(resp.read().decode())
    except urllib.error.HTTPError as exc:
        return exc.read().decode()


def _gate(snap: Snapshot, kind: str, handler: str, workdir: Path) -> AdmissionVerdict:
    graph = snap.graph
    _vuln_of_kind(graph, kind).attrs["realized_handler"] = handler
    exploit_path, benign_path = exploit_and_benign(graph, kind)
    service = EpisodeService(WebappPack(), workdir)
    try:
        task = next(t for t in snap.tasks if t.meta.get("family") == "webapp.pentest")
        handle = service.start_episode(snap, task.id)
        base = str(service.surface(handle)["base_url"])
        exploit_body = _fetch(base + exploit_path)
        benign_body = _fetch(base + benign_path)
    finally:
        service.close()
    return classify_admission(graph, exploit_body, benign_body)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--backend", choices=("claude", "codex"), default="claude")
    parser.add_argument("--kind", choices=REALIZABLE_KINDS, default=None)
    args = parser.parse_args(argv)

    backend = ClaudeBackend() if args.backend == "claude" else CodexBackend()
    backend.preflight()
    kinds = [args.kind] if args.kind else list(REALIZABLE_KINDS)

    total = 0
    with tempfile.TemporaryDirectory() as tmp:
        for kind in kinds:
            snap = _admit(kind)
            request = realization_request(snap.graph, kind)
            accepted: list[str] = []
            for index in range(args.rounds):
                handler = handler_from_result(backend.complete(request).parsed_json)
                if not handler.strip():
                    print(f"{kind} r{index}: REFUSED/empty")
                    continue
                try:
                    verdict = _gate(snap, kind, handler, Path(tmp) / f"{kind}{index}")
                except Exception as exc:  # noqa: BLE001
                    print(f"{kind} r{index}: REJECT — handler crashed: {exc}")
                    continue
                tag = "ACCEPT" if verdict.accepted else "REJECT"
                print(f"{kind} r{index}: {tag} — {verdict.reason}")
                if verdict.accepted:
                    accepted.append(handler)
            print(
                f"  {kind}: {len(accepted)}/{args.rounds} accepted, "
                f"{len(set(accepted))} distinct"
            )
            total += len(accepted)

    print(f"\ntotal accepted: {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
