"""The manifest is the gym's one auto<->specific control surface (docs/manifest.md):
a knob absent means "auto" (the seeded RNG samples it), a knob present is a constraint
merged onto the defaults. This pins the contract's two halves -- valid manifests admit
and place what they ask for, malformed or misapplied knobs are rejected up front with a
``PackError`` rather than silently ignored or surfaced as an opaque crash later."""

from __future__ import annotations

import pytest
from cyber_webapp import WebappBuilder, WebappPack
from graphschema import WorldGraph
from openrange_pack_sdk import PackError, Snapshot

from openrange.core.admit import admit

_REJECTED = {
    # old keys renamed in the control-surface migration -- each rejects with a hint
    "renamed-company": {"company": True},
    "renamed-lateral_movement": {"lateral_movement": True},
    "renamed-vuln_kinds": {"vuln_kinds": {"idor": 1}},
    "renamed-loot_shapes": {"loot_shapes": {"db": 1}},
    "renamed-recon_disclosure": {"recon_disclosure": "none"},
    "renamed-difficulty": {"difficulty": "hard"},
    # bad enum values
    "bad-topology": {"topology": "mesh"},
    "bad-recon": {"topology": "company", "recon": "loud"},
    # knobs the chosen topology does not own
    "vuln-under-company": {"topology": "company", "vuln": {"pin": [{"kind": "idor"}]}},
    "loot-under-chain": {"topology": "chain", "loot": {"db": 1}},
    "recon-on-flat": {"recon": "none"},
    "chain-on-flat": {"chain": {"depth": {"min": 2, "max": 3}}},
    # malformed vuln knob
    "vuln-not-mapping": {"vuln": [1, 2]},
    "vuln-empty": {"vuln": {}},
    "pin-empty": {"vuln": {"pin": []}},
    "pin-duplicate": {"vuln": {"pin": [{"kind": "idor"}, {"kind": "idor"}]}},
    "pin-missing-kind": {"vuln": {"pin": [{"endpoint": "/x"}]}},
    "pin-unknown-kind": {"vuln": {"pin": [{"kind": "no_such_vuln"}]}},
    "weights-not-mapping": {"vuln": {"weights": [1]}},
    "weights-non-int": {"vuln": {"weights": {"idor": "five"}}},
    # malformed loot / chain
    "loot-not-mapping": {"loot": [1]},
    "loot-non-int": {"loot": {"db": "lots"}},
    "loot-bool": {"loot": {"db": True}},
    "chain-no-depth": {"topology": "chain", "chain": {}},
    "chain-depth-zero": {"topology": "chain", "chain": {"depth": {"min": 0, "max": 2}}},
    "chain-depth-inverted": {
        "topology": "chain",
        "chain": {"depth": {"min": 5, "max": 2}},
    },
}


@pytest.mark.parametrize("extra", _REJECTED.values(), ids=list(_REJECTED))
def test_malformed_manifest_is_rejected(extra: dict[str, object]) -> None:
    builder = WebappBuilder(None)
    with pytest.raises(PackError):
        builder._effective_prior({"pack": {"id": "webapp"}, "npc": [], **extra})


def _admit(extra: dict[str, object]) -> Snapshot:
    snap = admit(
        WebappPack(),
        manifest={"pack": {"id": "webapp"}, "npc": [], "seed": 7, **extra},
        max_repairs=3,
    )
    assert isinstance(snap, Snapshot), snap
    return snap


def _vuln_kinds(graph: WorldGraph) -> list[str]:
    return [str(n.attrs.get("kind")) for n in graph.by_kind("vulnerability")]


@pytest.mark.parametrize(
    "extra",
    [
        {},  # fully-auto: just the seed
        {"topology": "flat", "vuln": {"weights": {"xxe": 5}}},  # partial: bias one kind
        {"topology": "company"},
        {"topology": "chain"},
        {"topology": "company", "recon": "none"},
        {"topology": "chain", "chain": {"depth": {"min": 2, "max": 3}}},
    ],
)
def test_the_documented_manifests_admit(extra: dict[str, object]) -> None:
    _admit(extra)


def test_pin_places_exactly_those_kinds() -> None:
    pin = [{"kind": "sql_injection"}, {"kind": "idor"}]
    snap = _admit({"topology": "flat", "vuln": {"pin": pin}})
    kinds = _vuln_kinds(snap.graph)
    assert sorted(kinds) == ["idor", "sql_injection"]  # exactly the pinned set
