"""The flag-only re-roll primitive (`replant_flag`) behind the #317 integrity check:
a world byte-identical except for the secret, so the SAME exploit recovers the fresh
flag (genuine) while a memorized old value does not."""

from __future__ import annotations

from pathlib import Path

from cyber_webapp import WebappPack
from cyber_webapp.reference_solver import Request, exploit_and_benign
from cyber_webapp.reseed import replant_flag
from cyber_webapp.verify import perform
from openrange_pack_sdk import Snapshot

from openrange.core.admit import admit
from openrange.core.episode import EpisodeService

_SQLI = {
    "pack": {"id": "webapp"},
    "runtime": {"tick": {"mode": "off"}},
    "npc": [],
    "seed": 7,
    "vuln_kinds": {"sql_injection": 1},
    "loot_shapes": {"db": 1, "file": 0},
}


def _admit() -> Snapshot:
    snap = admit(WebappPack(), manifest=_SQLI, max_repairs=3)
    assert isinstance(snap, Snapshot), snap
    return snap


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
