"""The networked multi-service backing: per-service realization, then (docker-gated)
the runtime that runs one container per service on a real network with real SSRF."""

from __future__ import annotations

import ast

from cyber_webapp import WebappPack
from cyber_webapp.container import realize_services
from openrange_pack_sdk import Snapshot

from openrange.core.admit import admit

_SSRF_MANIFEST = {
    "pack": {"id": "webapp"},
    "runtime": {"tick": {"mode": "off"}},
    "npc": [],
    "seed": 3,
    "vuln_kinds": {"ssrf": 1},
}


def _admit_ssrf() -> Snapshot:
    snap = admit(WebappPack(), manifest=_SSRF_MANIFEST, max_repairs=3)
    assert isinstance(snap, Snapshot), snap
    return snap


def test_realize_services_splits_per_service_and_confines_the_flag() -> None:
    snap = _admit_ssrf()
    graph = snap.graph
    flag = str(graph.nodes["secret_flag"].attrs["value_ref"])
    images = realize_services(graph)

    assert len(images) == sum(1 for _ in graph.by_kind("service"))
    assert len(images) >= 2  # a networked world has multiple services
    owners = []
    for image in images:
        assert set(image.build_files) == {"Dockerfile", "app.py", "seed.json"}
        ast.parse(image.build_files["app.py"])  # each per-service app is valid Python
        if flag in image.build_files["seed.json"]:
            owners.append(image.name)
    assert len(owners) == 1  # the flag lives in exactly one service's image

    public = next(im for im in images if im.exposure == "public")
    internal = next(im for im in images if im.exposure == "internal")
    assert flag not in public.build_files["seed.json"]  # never in the public image
    # Per-service apps route on bare paths (their own container/port), not the
    # single-app `/svc/<name>` namespace.
    assert "/svc/" not in internal.build_files["app.py"]
