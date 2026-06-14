"""The reference solver proves each generated world is really solvable (DESIGN.md §8).

Admission today is *structural* — a graph path exists. The reference solver upgrades
that to a *behavioural* proof: it runs the intended exploit and a benign control
against the live world, and the consequence verifier confirms the exploit leaks the
flag while the benign request does not. Running it across a frozen seed corpus yields
the per-shape solve rate (here it must be 1.0 — every world is solvable by
construction; a regression shows up as a number below 1.0).
"""

from __future__ import annotations

import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path

import pytest
from cyber_webapp import WebappPack
from cyber_webapp.ontology import ONTOLOGY_ID
from cyber_webapp.realize_admit import AdmissionVerdict, classify_admission
from cyber_webapp.reference_solver import (
    _company_pivot,
    exploit_and_benign,
    solve_chain,
)
from graphschema import WorldGraph
from openrange_pack_sdk import PackError, Snapshot

from openrange.core.admit import admit
from openrange.core.episode import EpisodeService

# (loot shape, exploit shape): each shape forced as the oracle on its compatible loot.
_CLASS_CASES = [
    ("file", "path_traversal"),
    ("file", "command_injection"),
    ("file", "xxe"),
    ("file", "ssti"),
    ("db", "sql_injection"),
    ("db", "ssrf"),
    ("db", "broken_authz"),
    ("db", "idor"),
    ("db", "weak_credentials"),
]


def _admit(loot: str, kind: str, seed: int = 7) -> Snapshot:
    snap = admit(
        WebappPack(),
        manifest={
            "pack": {"id": "webapp"},
            "runtime": {"tick": {"mode": "off"}},
            "npc": [],
            "seed": seed,
            "loot_shapes": {loot: 1, "db" if loot == "file" else "file": 0},
            "vuln_kinds": {kind: 1},
        },
        max_repairs=3,
    )
    assert isinstance(snap, Snapshot), snap
    return snap


def _get(base: str, path: str) -> str:
    try:
        with urllib.request.urlopen(f"{base}{path}", timeout=15) as resp:
            return str(resp.read().decode())
    except urllib.error.HTTPError as exc:
        return exc.read().decode()


def _fetcher(base: str) -> Callable[[str], str]:
    return lambda path: _get(base, path)


def _solve(snap: Snapshot, kind: str, workdir: Path) -> AdmissionVerdict:
    exploit_path, benign_path = exploit_and_benign(snap.graph, kind)
    pentest = next(t for t in snap.tasks if t.meta.get("family") == "webapp.pentest")
    svc = EpisodeService(WebappPack(), workdir)
    try:
        handle = svc.start_episode(snap, pentest.id)
        base = str(svc.surface(handle)["base_url"])
        exploit_body, benign_body = _get(base, exploit_path), _get(base, benign_path)
    finally:
        svc.close()
    return classify_admission(snap.graph, exploit_body, benign_body)


@pytest.mark.parametrize(("loot", "kind"), _CLASS_CASES)
def test_reference_exploit_admits_each_shape(
    loot: str, kind: str, tmp_path: Path
) -> None:
    # The intended exploit leaks the flag; a benign request to the same endpoint does
    # not — so the world is solvable and not trivial.
    verdict = _solve(_admit(loot, kind), kind, tmp_path)
    assert verdict.solvable, f"{kind}: the intended exploit did not leak the flag"
    assert not verdict.trivial, f"{kind}: a benign request leaked the flag"
    assert verdict.accepted


def test_reference_solver_corpus_solve_rate(tmp_path: Path) -> None:
    # The paper's first measured number: per-shape solve rate over a frozen corpus.
    # Solvable-by-construction means it is 1.0; anything less is a generator regression.
    seeds = (7, 8)
    rate: dict[str, float] = {}
    for loot, kind in _CLASS_CASES:
        admitted = 0
        for seed in seeds:
            snap = _admit(loot, kind, seed=seed)
            verdict = _solve(snap, kind, tmp_path / f"{kind}_{seed}")
            admitted += int(verdict.accepted)
        rate[kind] = admitted / len(seeds)
    table = "\n".join(f"  {kind:18s} {r:.2f}" for kind, r in rate.items())
    print(f"\nreference-solver per-shape solve rate ({len(seeds)} seeds):\n{table}")
    assert all(r == 1.0 for r in rate.values()), rate


def _admit_lateral(seed: int) -> Snapshot:
    snap = admit(
        WebappPack(),
        manifest={
            "pack": {"id": "webapp"},
            "runtime": {"tick": {"mode": "off"}},
            "npc": [],
            "seed": seed,
            "lateral_movement": True,
        },
        max_repairs=3,
    )
    assert isinstance(snap, Snapshot), snap
    return snap


def _chain_depth(snap: Snapshot) -> int:
    return sum(
        1
        for n in snap.graph.by_kind("vulnerability")
        if n.attrs.get("kind") in ("credential_gated_relay", "credential_gated_flag")
    )


def test_reference_solver_walks_lateral_chains_across_depths(tmp_path: Path) -> None:
    # The chain walker recovers the flag across the synthesized depth distribution —
    # every lateral world is solvable, whatever its sampled hop count.
    depths: set[int] = set()
    for seed in range(6):
        snap = _admit_lateral(seed)
        depths.add(_chain_depth(snap))
        graph = snap.graph
        pentest = next(
            t for t in snap.tasks if t.meta.get("family") == "webapp.pentest"
        )
        svc = EpisodeService(WebappPack(), tmp_path / f"lat_{seed}")
        try:
            handle = svc.start_episode(snap, pentest.id)
            base = str(svc.surface(handle)["base_url"])
            trace = solve_chain(graph, _fetcher(base))
            verdict = classify_admission(graph, trace.terminal, "\n".join(trace.probes))
        finally:
            svc.close()
        assert verdict.accepted, f"seed {seed}: {verdict.reason}"
    assert (
        len(depths) >= 2
    )  # the preset synthesizes a distribution, not one fixed shape


def test_reference_solver_defensive_branches() -> None:
    # exploit_and_benign rejects a kind it has no recipe for.
    with pytest.raises(PackError):
        exploit_and_benign(WorldGraph(ontology=ONTOLOGY_ID), "totally_unknown")

    # _company_pivot builds the right SSRF payload for each filter (both branches).
    seen: list[str] = []

    def fake(url: str) -> str:
        seen.append(url)
        return ""

    _company_pivot(
        {"internal_host": "db", "internal_path": "/m", "ssrf_filter": "scheme_block"},
        fake,
    )
    _company_pivot(
        {
            "internal_host": "db",
            "internal_path": "/m",
            "ssrf_filter": "host_allowlist",
            "allowed_host": "ok",
        },
        fake,
    )
    assert seen[0] == "gopher://db/m"  # scheme_block -> gopher
    assert seen[2] == "http://ok@db/m"  # host_allowlist -> credentials-in-userinfo
