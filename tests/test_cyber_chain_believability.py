"""A believability / solvability sweep over credential-reuse chain worlds: across seeds
and the sampled depth spread, every chain must be reachable from the public foothold,
solvable by the reference walk, benign-safe (no probe before the terminal leaks the
flag), recon-discoverable, and admit cleanly. Guards against sampling or evolving a
degenerate chain that is unsolvable, trivially short-circuitable, or undiscoverable."""

from __future__ import annotations

import urllib.error
import urllib.request
from pathlib import Path

import pytest
from cyber_webapp import WebappPack
from cyber_webapp.reference_solver import solve_chain
from graphschema import WorldGraph, validate
from openrange_pack_sdk import Backing, Snapshot, TaskSpec

from openrange.core.admit import admit
from openrange.core.episode import EpisodeService

_SEEDS = list(range(12))


def _admit(seed: int) -> Snapshot:
    snap = admit(
        WebappPack(),
        manifest={
            "pack": {"id": "webapp"},
            "runtime": {"tick": {"mode": "off"}},
            "npc": [],
            "seed": seed,
            "topology": "chain",
        },
        max_repairs=3,
    )
    assert isinstance(snap, Snapshot), snap
    return snap


def _pentest(snap: Snapshot) -> TaskSpec:
    return next(t for t in snap.tasks if t.meta.get("family") == "webapp.pentest")


def _depth(graph: WorldGraph) -> int:
    return sum(
        1
        for v in graph.by_kind("vulnerability")
        if v.attrs.get("kind") in ("credential_gated_relay", "credential_gated_flag")
    )


@pytest.mark.parametrize("seed", _SEEDS)
def test_chain_world_is_believable_and_solvable(tmp_path: Path, seed: int) -> None:
    pack = WebappPack()
    snap = _admit(seed)
    graph = snap.graph
    flag = str(graph.nodes["secret_flag"].attrs["value_ref"])

    # (f) admits cleanly — no invariant errors.
    assert not [
        i
        for i in validate(graph, pack.ontology(), pack.invariants())
        if i.severity == "error"
    ]

    # (a) reachable — a public foothold endpoint exists.
    task = _pentest(snap)
    assert task.entrypoints
    entry = str(graph.nodes[task.entrypoints[0]].attrs["public_url"])

    # (d) recon — the SSRF is proxy-mode, so a blind agent can enumerate internal hosts.
    ssrf = next(
        v for v in graph.by_kind("vulnerability") if v.attrs.get("kind") == "ssrf"
    )
    assert ssrf.attrs["params"].get("internal_hosts")

    svc = EpisodeService(pack, tmp_path, backing=Backing.PROCESS)
    try:
        base = svc.base_url(svc.start_episode(snap, task.id))

        def fetch(path: str) -> str:
            try:
                with urllib.request.urlopen(f"{base}{path}", timeout=15) as resp:
                    return str(resp.read().decode())
            except urllib.error.HTTPError as exc:
                return exc.read().decode()

        # (c) benign-safe — hitting the public foothold alone never serves the flag.
        assert flag not in fetch(entry)
        trace = solve_chain(graph, fetch)
    finally:
        svc.close()

    # (b) solvable — the full walk recovers the flag.
    assert flag in trace.terminal
    # (e) not short-circuitable — no probe before the terminal leaks the flag.
    assert all(flag not in probe for probe in trace.probes)


def test_sweep_exercises_varied_chain_depths() -> None:
    depths = {_depth(_admit(seed).graph) for seed in _SEEDS}
    assert len(depths) >= 2  # the sweep is not all one trivial depth
    assert max(depths) >= 2  # and it reaches genuine multi-hop chains
