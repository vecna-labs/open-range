"""A consequence gate for ``auto_evolve`` (cyber webapp): accept an evolved world
only if it is genuinely solvable.

Structural re-admission only proves a *path* exists on the graph. This realizes
the evolved world, runs the reference breach over HTTP, and requires the flag to
actually leak via the exploit while a benign request does not — the consequence
verifier's verdict ([#312](https://github.com/vecna-labs/open-range/issues/312)).
It composes pack pieces (reference solver + verifier) with host realization
(``EpisodeService``), so it lives here, not in the pack (which must not import
``openrange``) or core (which must not import a pack).
"""

from __future__ import annotations

import urllib.error
import urllib.request
from pathlib import Path

from cyber_webapp import _is_networked
from cyber_webapp.realize_admit import AdmissionVerdict, classify_admission
from cyber_webapp.reference_solver import exploit_and_benign, solve_chain
from graphschema import WorldGraph
from openrange_pack_sdk import Mutation, Pack, Snapshot

from openrange.core.curriculum import EvolutionGate
from openrange.core.episode import EpisodeService


def _fetch(url: str) -> str:
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            raw: bytes = resp.read()
    except urllib.error.HTTPError as exc:
        raw = exc.read()
    except Exception:  # noqa: BLE001 — an unreachable surface just can't leak
        return ""
    return raw.decode("utf-8", "replace")


def verdict(graph: WorldGraph, base_url: str, entry_path: str) -> AdmissionVerdict:
    """Realize-free verdict: drive the reference breach against an already-running
    world at ``base_url`` and classify whether the exploit leaks while a benign
    request to ``entry_path`` does not."""

    def fetch(path: str) -> str:
        return _fetch(base_url + str(path))

    benign = fetch(entry_path)
    if _is_networked(graph):
        try:
            terminal = solve_chain(graph, fetch).terminal
        except Exception:  # noqa: BLE001 — a chain that won't drive isn't solvable
            return AdmissionVerdict(False, False, False, "reference breach failed")
        return classify_admission(graph, terminal, benign)
    for vuln in graph.by_kind("vulnerability"):
        try:
            exploit_path, benign_path = exploit_and_benign(
                graph, str(vuln.attrs["kind"])
            )
        except Exception:  # noqa: BLE001 — no reference exploit for this kind
            continue
        return classify_admission(graph, fetch(exploit_path), fetch(benign_path))
    return AdmissionVerdict(False, False, False, "no reference exploit to verify")


def consequence_gate(pack: Pack, workdir: str | Path) -> EvolutionGate:
    """Build a gate for ``auto_evolve(..., gate=…)`` / ``WorldPool`` that accepts an
    evolved world only when its reference breach actually leaks the flag (and a
    benign request does not). Anything it can't assess passes through."""
    root = Path(workdir)

    def gate(evolved: Snapshot, mutation: Mutation) -> bool:
        del mutation  # the world is verified regardless of which move produced it
        task = next(
            (t for t in evolved.tasks if t.meta.get("family") == "webapp.pentest"),
            None,
        )
        if task is None or not task.entrypoints:
            return True
        entry = str(evolved.graph.nodes[task.entrypoints[0]].attrs["public_url"])
        svc = EpisodeService(pack, root)
        try:
            handle = svc.start_episode(evolved, task.id)
            base = str(svc.surface(handle)["base_url"])
            return verdict(evolved.graph, base, entry).accepted
        finally:
            svc.close()

    return gate
