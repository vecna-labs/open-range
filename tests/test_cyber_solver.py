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
from cyber_webapp.reference_solver import exploit_and_benign, solve_chain
from cyber_webapp.verify import perform
from graphschema import Edge, Node, WorldGraph
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
            "loot": {loot: 1, "db" if loot == "file" else "file": 0},
            "vuln": {"pin": [{"kind": kind}]},
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
    exploit_req, benign_req = exploit_and_benign(snap.graph, kind)
    pentest = next(t for t in snap.tasks if t.meta.get("family") == "webapp.pentest")
    svc = EpisodeService(WebappPack(), workdir)
    try:
        handle = svc.start_episode(snap, pentest.id)
        base = str(svc.surface(handle)["base_url"])
        exploit_body = perform(base, exploit_req)
        benign_body = perform(base, benign_req)
    finally:
        svc.close()
    return classify_admission(snap.graph, exploit_body, benign_body)


@pytest.mark.parametrize(("loot", "kind"), _CLASS_CASES)
def test_reference_exploit_admits_each_shape(
    loot: str, kind: str, tmp_path: Path
) -> None:
    verdict = _solve(_admit(loot, kind), kind, tmp_path)
    assert verdict.solvable, f"{kind}: the intended exploit did not leak the flag"
    assert not verdict.trivial, f"{kind}: a benign request leaked the flag"
    assert verdict.accepted


def test_reference_solver_corpus_solve_rate(tmp_path: Path) -> None:
    # Solvable-by-construction means every world solves: a 1.0 rate, or a regression.
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
            "topology": "chain",
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


def test_reference_solver_walks_company_pivots(tmp_path: Path) -> None:
    # The networked direct pivot recovers the flag for both SSRF-filter shapes — a
    # gopher scheme-block and a host-allowlist userinfo bypass both occur across seeds.
    for seed in range(6):
        snap = admit(
            WebappPack(),
            manifest={
                "pack": {"id": "webapp"},
                "runtime": {"tick": {"mode": "off"}},
                "npc": [],
                "seed": seed,
                "topology": "company",
            },
            max_repairs=3,
        )
        assert isinstance(snap, Snapshot), snap
        graph = snap.graph
        pentest = next(
            t for t in snap.tasks if t.meta.get("family") == "webapp.pentest"
        )
        svc = EpisodeService(WebappPack(), tmp_path / f"co_{seed}")
        try:
            handle = svc.start_episode(snap, pentest.id)
            base = str(svc.surface(handle)["base_url"])
            trace = solve_chain(graph, _fetcher(base))
            verdict = classify_admission(graph, trace.terminal, "\n".join(trace.probes))
        finally:
            svc.close()
        assert verdict.accepted, f"seed {seed}: {verdict.reason}"


def test_reference_solver_rejects_bad_inputs() -> None:
    empty = WorldGraph(ontology=ONTOLOGY_ID)
    with pytest.raises(PackError):  # no recipe for this kind
        exploit_and_benign(empty, "totally_unknown")
    with pytest.raises(PackError):  # a supported kind, but no such vuln in the graph
        exploit_and_benign(empty, "ssrf")

    # An SSRF that is neither proxy-mode nor a networked pivot is not a chain world.
    graph = WorldGraph(ontology=ONTOLOGY_ID)
    graph.add_node(Node(id="web", kind="service", attrs={"exposure": "public"}))
    graph.add_node(Node(id="ep", kind="endpoint", attrs={"public_url": "/x"}))
    graph.add_edge(Edge(id="x", kind="exposes", src="web", dst="ep", attrs={}))
    graph.add_node(
        Node(
            id="v",
            kind="vulnerability",
            attrs={"kind": "ssrf", "params": {"target_param": "url"}},
        )
    )
    graph.add_edge(Edge(id="a", kind="affects", src="v", dst="ep", attrs={}))
    with pytest.raises(PackError):
        solve_chain(graph, _fetcher("http://unused"))
