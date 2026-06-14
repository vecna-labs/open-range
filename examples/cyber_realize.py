"""Produce an LLM-realized world with a real LLM (DESIGN.md §9, #260).

The build pipeline: procedural admit -> the LLM realizes each vuln's handler ->
each is dynamically admitted (the intended exploit must leak the flag, a benign
request must not) -> the result is re-frozen to a content-addressed snapshot. This
is `cyber_webapp.llm_realize.realize_world`; here we just inject the LLM and the
episode runner.

Run::

    uv run python -m examples.cyber_realize              # vuln handlers, all classes
    uv run python -m examples.cyber_realize --kind ssti  # one class
    uv run python -m examples.cyber_realize --service --kind sql_injection  # a service

With ``--service`` the LLM realizes a whole service's BENIGN endpoint bodies
(`realize_service_surface`, #212) instead of the vuln handler — admitted iff the oracle
still fires and no realized endpoint leaks the flag.
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
    benign_endpoints_of,
    handler_from_result,
    realization_request,
    realize_service_surface,
    realize_world,
    service_handlers_from_result,
    service_realization_request,
)
from cyber_webapp.reference_solver import control_request, exploit_and_benign
from graphschema import WorldGraph
from openrange_pack_sdk import LLMBackend, Snapshot

from openrange.core.admit import admit
from openrange.core.episode import EpisodeService
from openrange.llm import ClaudeBackend, CodexBackend

_LOOT = {
    "command_injection": "file",
    "path_traversal": "file",
    "xxe": "file",
    "ssti": "file",
    "sql_injection": "db",
    "idor": "db",
    "broken_authz": "db",
    "weak_credentials": "db",
    "ssrf": "db",
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


def _realize(backend: LLMBackend, kind: str, base_dir: Path) -> Snapshot:
    snap = _admit(kind)
    task = next(t for t in snap.tasks if t.meta.get("family") == "webapp.pentest")
    counter = iter(range(1000))

    def propose(graph: WorldGraph, k: str) -> str:
        return handler_from_result(
            backend.complete(realization_request(graph, k)).parsed_json
        )

    def run_probes(k: str) -> tuple[str, str, str | None]:
        svc = EpisodeService(WebappPack(), base_dir / f"{kind}{next(counter)}")
        try:
            handle = svc.start_episode(snap, task.id)
            base = str(svc.surface(handle)["base_url"])
            exploit_path, benign_path = exploit_and_benign(snap.graph, k)
            control = control_request(snap.graph, k)
            control_body = _fetch(base + control.request) if control else None
            return _fetch(base + exploit_path), _fetch(base + benign_path), control_body
        finally:
            svc.close()

    return realize_world(snap, propose, run_probes)


def _realize_service(
    backend: LLMBackend, kind: str, base_dir: Path
) -> tuple[Snapshot, str]:
    snap = _admit(kind)
    service_id = max(
        (s.id for s in snap.graph.by_kind("service")),
        key=lambda sid: len(benign_endpoints_of(snap.graph, sid)),
    )
    task = next(t for t in snap.tasks if t.meta.get("family") == "webapp.pentest")
    counter = iter(range(1000))

    def propose_service(graph: WorldGraph, sid: str) -> dict[str, str]:
        return service_handlers_from_result(
            backend.complete(service_realization_request(graph, sid)).parsed_json
        )

    def run_service_probes(sid: str) -> tuple[str, str, dict[str, str], bool]:
        svc = EpisodeService(WebappPack(), base_dir / f"{kind}{next(counter)}")
        try:
            handle = svc.start_episode(snap, task.id)
            base = str(svc.surface(handle)["base_url"])
            exploit_path, benign_path = exploit_and_benign(snap.graph, kind)
            bodies = {
                str(ep.attrs.get("path")): _fetch(base + str(ep.attrs["public_url"]))
                for ep in benign_endpoints_of(snap.graph, sid)
            }
            try:
                with urllib.request.urlopen(base + "/", timeout=10) as resp:
                    root_ok = resp.status == 200
            except urllib.error.HTTPError:
                root_ok = False
            return (
                _fetch(base + exploit_path),
                _fetch(base + benign_path),
                bodies,
                root_ok,
            )
        finally:
            svc.close()

    out = realize_service_surface(snap, service_id, propose_service, run_service_probes)
    return out, str(snap.graph.nodes[service_id].attrs.get("name", service_id))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("claude", "codex"), default="claude")
    parser.add_argument("--kind", choices=REALIZABLE_KINDS, default=None)
    parser.add_argument(
        "--service",
        action="store_true",
        help="realize a whole service's benign surface instead of vuln handlers",
    )
    args = parser.parse_args(argv)

    backend = ClaudeBackend() if args.backend == "claude" else CodexBackend()
    backend.preflight()
    kinds = [args.kind] if args.kind else list(REALIZABLE_KINDS)

    with tempfile.TemporaryDirectory() as tmp:
        for kind in kinds:
            if args.service:
                realized, svc_name = _realize_service(backend, kind, Path(tmp))
                eps = realized.lineage["realized_endpoints"]
                status = (
                    f"realized {len(eps)} endpoint(s) on {svc_name}: {list(eps)}"
                    if eps
                    else "fell back to stubs"
                )
                print(f"{kind} (service): {status} -> {realized.snapshot_id[:19]}")
            else:
                realized = _realize(backend, kind, Path(tmp))
                done = kind in realized.lineage["realized_handlers"]
                status = "realized" if done else "fell back to template"
                print(f"{kind}: {status} -> snapshot {realized.snapshot_id[:19]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
