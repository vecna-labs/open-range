"""The flag-only re-roll primitive (`replant_flag`) behind the #317 integrity check:
a world byte-identical except for the secret, so the SAME exploit recovers the fresh
flag (genuine) while a memorized old value does not. `reseed_chain` extends it to the
credential-reuse chain (flag + every per-hop token), proving the chain is
memorization-proof: the response-driven breach loots the live values."""

from __future__ import annotations

import random
import urllib.error
import urllib.request
from pathlib import Path

from cyber_webapp import WebappPack
from cyber_webapp.reference_solver import Request, exploit_and_benign, solve_chain
from cyber_webapp.reseed import replant_flag, reseed_chain
from cyber_webapp.verify import perform
from graphschema import validate
from openrange_pack_sdk import Backing, Snapshot

from openrange.core.admit import admit
from openrange.core.episode import EpisodeService

_SQLI = {
    "pack": {"id": "webapp"},
    "runtime": {"tick": {"mode": "off"}},
    "npc": [],
    "seed": 7,
    "vuln": {"pin": [{"kind": "sql_injection"}]},
    "loot": {"db": 1, "file": 0},
}

_LATERAL = {
    "pack": {"id": "webapp"},
    "runtime": {"tick": {"mode": "off"}},
    "npc": [],
    "seed": 3,
    "topology": "chain",
}


def _admit() -> Snapshot:
    snap = admit(WebappPack(), manifest=_SQLI, max_repairs=3)
    assert isinstance(snap, Snapshot), snap
    return snap


def _admit_lateral() -> Snapshot:
    snap = admit(WebappPack(), manifest=_LATERAL, max_repairs=3)
    assert isinstance(snap, Snapshot), snap
    return snap


def _tokens(snap: Snapshot) -> set[str]:
    return {
        str(n.attrs["value_ref"])
        for n in snap.graph.by_kind("credential")
        if n.attrs.get("kind") == "token"
    }


def _pentest_task(snap: Snapshot) -> str:
    return next(t.id for t in snap.tasks if t.meta.get("family") == "webapp.pentest")


def _run(pack: WebappPack, work_dir: Path, snap: Snapshot, request: Request) -> str:
    svc = EpisodeService(pack, work_dir)
    try:
        handle = svc.start_episode(snap, _pentest_task(snap))
        return perform(svc.base_url(handle), request)
    finally:
        svc.close()


def test_replant_flag_changes_only_the_flag() -> None:
    snap = _admit()
    old = str(snap.graph.nodes["secret_flag"].attrs["value_ref"])
    fresh = "ghp_reseededFLAG01"
    snap2 = replant_flag(snap, fresh)

    assert snap2.snapshot_id != snap.snapshot_id
    assert snap2.graph.nodes["secret_flag"].attrs["value_ref"] == fresh
    values = {
        n.attrs.get("fields", {}).get("value") for n in snap2.graph.by_kind("record")
    }
    assert fresh in values and old not in values  # the loot record moved too
    # replanting the old value back restores the id -> nothing but the flag changed
    assert replant_flag(snap2, old).snapshot_id == snap.snapshot_id


def test_same_exploit_recovers_the_fresh_flag(tmp_path: Path) -> None:
    pack = WebappPack()
    snap = _admit()
    old = str(snap.graph.nodes["secret_flag"].attrs["value_ref"])
    fresh = "ghp_reseededFLAG01"
    snap2 = replant_flag(snap, fresh)

    exploit, _benign = exploit_and_benign(snap.graph, "sql_injection")
    body_a = _run(pack, tmp_path / "a", snap, exploit)
    body_b = _run(pack, tmp_path / "b", snap2, exploit)

    # The one exploit string recovers each world's own flag -> a genuine exploit; a
    # memorized old value would be absent from world B.
    assert old in body_a and fresh not in body_a
    assert fresh in body_b and old not in body_b


def test_reseed_chain_refreshes_flag_and_tokens_and_still_admits() -> None:
    pack = WebappPack()
    snap = _admit_lateral()
    old_flag = str(snap.graph.nodes["secret_flag"].attrs["value_ref"])
    old_tokens = _tokens(snap)
    assert old_tokens, "a lateral world has per-hop tokens"

    fresh = reseed_chain(snap, random.Random(99))

    assert fresh.snapshot_id != snap.snapshot_id
    assert fresh.graph.nodes["secret_flag"].attrs["value_ref"] != old_flag
    assert _tokens(fresh).isdisjoint(old_tokens)  # every token re-rolled
    # The re-seed rewrote the node value_ref and the param mirrors together, so the
    # value-consistency invariant still holds and the fresh world admits.
    assert not validate(fresh.graph, pack.ontology(), pack.invariants())


def test_reseeded_chain_loots_live_not_memorized(tmp_path: Path) -> None:
    pack = WebappPack()
    snap = _admit_lateral()
    old_flag = str(snap.graph.nodes["secret_flag"].attrs["value_ref"])
    fresh = reseed_chain(snap, random.Random(123))
    fresh_flag = str(fresh.graph.nodes["secret_flag"].attrs["value_ref"])
    assert fresh_flag != old_flag

    svc = EpisodeService(pack, tmp_path, backing=Backing.PROCESS)
    try:
        base = svc.base_url(svc.start_episode(fresh, _pentest_task(fresh)))

        def fetch(path: str) -> str:
            try:
                with urllib.request.urlopen(f"{base}{path}", timeout=15) as resp:
                    return str(resp.read().decode())
            except urllib.error.HTTPError as exc:
                return exc.read().decode()

        trace = solve_chain(fresh.graph, fetch)
    finally:
        svc.close()
    # The walk loots the FRESH flag live; a memorized old value never appears.
    assert fresh_flag in trace.terminal
    assert old_flag not in trace.terminal
