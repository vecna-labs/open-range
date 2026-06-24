"""``recon_disclosure`` gates whether a company world hands the agent its internal
estate (``full``) or withholds it so the agent must discover it (``none``). A blind
world stays solvable by construction, and the SSRF turns a named host into recon
feedback (it exists vs a typo) without ever leaking the flag."""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import cast

import pytest
from cyber_webapp import WebappPack
from cyber_webapp.codegen.discovery import build_discovery
from cyber_webapp.mutation import available_mutations
from cyber_webapp.reference_solver import solve_chain
from cyber_webapp.sampling import _flag_service_id
from graphschema import Node, WorldGraph
from openrange_pack_sdk import Snapshot

from openrange.core.admit import admit
from openrange.core.episode import EpisodeService

_COMPANY = {
    "pack": {"id": "webapp"},
    "runtime": {"tick": {"mode": "off"}},
    "npc": [],
    "seed": 3,
    "topology": "company",
}
_NONE = {**_COMPANY, "recon": "none"}
_LATERAL_NONE = {**_COMPANY, "topology": "chain", "recon": "none"}
_DEFAULT = {k: v for k, v in _COMPANY.items() if k != "topology"}


def _admit(manifest: dict[str, object]) -> Snapshot:
    snap = admit(WebappPack(), manifest=manifest, max_repairs=3)
    assert isinstance(snap, Snapshot), snap
    return snap


def _vuln(graph: WorldGraph, kind: str) -> Node | None:
    return next(
        (n for n in graph.by_kind("vulnerability") if n.attrs.get("kind") == kind),
        None,
    )


def _internal_names(graph: WorldGraph) -> set[str]:
    return {
        str(n.attrs.get("name"))
        for n in graph.by_kind("service")
        if n.attrs.get("exposure") != "public"
    }


def _fetch(base: str, path: str) -> tuple[int, str]:
    try:
        with urllib.request.urlopen(base + path, timeout=15) as resp:
            return resp.status, resp.read().decode()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode()


def _ssrf_request(graph: WorldGraph, base: str, host: str) -> tuple[int, str]:
    ssrf = _vuln(graph, "ssrf")
    assert ssrf is not None
    params = ssrf.attrs.get("params", {})
    endpoint = graph.nodes[next(e.dst for e in graph.out_edges(ssrf.id, "affects"))]
    if "internal_hosts" in params or params.get("ssrf_filter") == "scheme_block":
        target = "gopher://" + host + "/"
    else:
        target = "http://" + str(params["allowed_host"]) + "@" + host + "/"
    query = urllib.parse.urlencode({str(params.get("target_param", "url")): target})
    return _fetch(base, str(endpoint.attrs["path"]) + "?" + query)


def _flag(graph: WorldGraph) -> str:
    return str(graph.nodes["secret_flag"].attrs["value_ref"])


def _services(graph: WorldGraph) -> list[dict[str, object]]:
    return cast("list[dict[str, object]]", build_discovery(graph)["services"])


@pytest.mark.parametrize("level", ["full", "none"])
@pytest.mark.parametrize("seed", [1, 2, 3, 5, 7])
def test_every_recon_level_admits(level: str, seed: int) -> None:
    _admit({**_COMPANY, "seed": seed, "recon": level})


@pytest.mark.parametrize("manifest", [_NONE, _LATERAL_NONE])
def test_a_blind_world_still_solves(
    manifest: dict[str, object], tmp_path: Path
) -> None:
    snap = _admit(manifest)
    flag = _flag(snap.graph)
    task = next(t for t in snap.tasks if t.meta.get("family") == "webapp.pentest")
    svc = EpisodeService(WebappPack(), tmp_path)
    try:
        handle = svc.start_episode(snap, task.id)
        base = str(svc.surface(handle)["base_url"])
        terminal = solve_chain(snap.graph, lambda path: _fetch(base, path)[1]).terminal
        assert flag in terminal  # the reference exploit reaches the flag with no recon
        _, doc = _fetch(base, "/openapi.json")
        assert flag not in doc  # a benign request never carries the flag
    finally:
        svc.close()


def test_full_discloses_the_estate_none_withholds_it() -> None:
    full = _admit(_COMPANY).graph
    blind = _admit(_NONE).graph

    # The public discovery doc never lists the internal estate, at either level —
    # disclosure is the recon vuln's job, which the knob gates.
    for graph in (full, blind):
        listed = {str(s["name"]) for s in _services(graph)}
        assert _internal_names(graph).isdisjoint(listed)

    recon = _vuln(full, "config_disclosure")
    assert recon is not None
    disclosed = set(recon.attrs["params"]["internal_services"])
    # Every real internal host is named, padded with decoy hostnames so the page is a
    # candidate set to triage rather than a perfect oracle.
    assert _internal_names(full) <= disclosed
    assert disclosed - _internal_names(full)
    assert _vuln(blind, "config_disclosure") is None  # none has no recon vuln at all


@pytest.mark.parametrize("manifest", [_NONE, _LATERAL_NONE])
def test_blind_ssrf_confirms_a_real_host_and_rejects_a_typo(
    manifest: dict[str, object], tmp_path: Path
) -> None:
    snap = _admit(manifest)
    graph = snap.graph
    flag = _flag(graph)
    ssrf = _vuln(graph, "ssrf")
    assert ssrf is not None
    pool = (
        ssrf.attrs["params"].get("internal_hosts")
        or ssrf.attrs["params"]["internal_inventory"]
    )
    flag_host = ssrf.attrs["params"].get("internal_host")
    real = next(h for h in pool if h != flag_host)
    task = next(t for t in snap.tasks if t.meta.get("family") == "webapp.pentest")
    svc = EpisodeService(WebappPack(), tmp_path)
    try:
        handle = svc.start_episode(snap, task.id)
        base = str(svc.surface(handle)["base_url"])
        real_status, real_body = _ssrf_request(graph, base, real)
        fake_status, fake_body = _ssrf_request(graph, base, "nonexistent-zzz-typo")
    finally:
        svc.close()

    assert json.loads(real_body).get("service") == real  # the host resolves
    assert (real_status, real_body) != (fake_status, fake_body)  # a typo is told apart
    assert flag not in real_body and flag not in fake_body  # neither leaks the flag


def test_a_non_company_world_has_no_recon_and_a_public_doc() -> None:
    graph = _admit(_DEFAULT).graph
    assert _vuln(graph, "config_disclosure") is None  # recon is a company-only vuln
    listed = {str(s["name"]) for s in _services(graph)}
    public = {
        str(n.attrs.get("name"))
        for n in graph.by_kind("service")
        if n.attrs.get("exposure") == "public"
    }
    assert listed == public  # the doc is the public vantage, internals never listed


def test_evolution_does_not_re_add_recon_to_a_blind_world() -> None:
    graph = _admit(_NONE).graph
    moves = available_mutations(graph, "webapp.pentest", [])
    added = [
        node.attrs.get("kind")
        for move in moves
        if move.direction == "harden"
        for node in move.patch.nodes_added
        if node.kind == "vulnerability"
    ]
    assert "config_disclosure" not in added


def test_evolution_leaves_the_credential_chain_intact_but_still_deepens_it() -> None:
    # Internal-only kinds (recon + the synthesized credential-reuse chain) must never
    # be dropped, swapped, or added as a decoy -- only the append-hop may extend the
    # chain. Without this, admission rejects the move after the fact, wasting budget.
    internal_only = {
        "config_disclosure",
        "metadata_credential_leak",
        "credential_leak",
        "credential_gated_relay",
        "credential_gated_flag",
    }
    chain_kinds = {"credential_leak", "credential_gated_relay", "credential_gated_flag"}
    graph = _admit({**_COMPANY, "topology": "chain"}).graph
    moves = available_mutations(graph, "webapp.pentest", [])

    for move in moves:
        if move.direction in ("soften", "diversify"):
            assert not any(ck in move.note for ck in chain_kinds), move.note

    appended = False
    for move in moves:
        if move.direction != "harden":
            continue
        if move.note.startswith("append a credential hop"):
            appended = True
            continue
        for node in move.patch.nodes_added:
            if node.kind == "vulnerability":
                assert node.attrs.get("kind") not in internal_only, move.note
    assert appended  # the legitimate chain-deepening frontier move is still offered


def test_evolution_keeps_the_networked_foothold() -> None:
    # In a networked world the SSRF is the only public entry; soften/diversify must
    # neither remove it nor swap it away (that strips the foothold -> unsolvable, which
    # admission rejects), but the world must still be evolvable by other moves.
    graph = _admit(_COMPANY).graph
    moves = available_mutations(graph, "webapp.pentest", [])
    for move in moves:
        if move.direction in ("soften", "diversify"):
            assert "ssrf" not in move.note, move.note  # the foothold is protected
    assert any(m.direction in ("soften", "diversify") for m in moves)  # not frozen


@pytest.mark.parametrize("seed", [1, 2, 3, 5, 7])
def test_a_blind_world_always_names_the_flag_host_in_the_ssrf_pool(seed: int) -> None:
    graph = _admit({**_NONE, "seed": seed}).graph
    flag_service_id = _flag_service_id(graph)
    assert flag_service_id is not None
    flag_host = str(graph.nodes[flag_service_id].attrs.get("name"))
    ssrf = _vuln(graph, "ssrf")
    assert ssrf is not None
    params = ssrf.attrs["params"]
    pool = params.get("internal_hosts") or [params.get("internal_host")]
    assert flag_host in pool  # a blind agent can always reach the flag host by name
