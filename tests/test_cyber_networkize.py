"""The ``_networkize_ssrf`` sampler transform.

It turns an SSRF world into a real networked chain — the SSRF on a public service
pivots to an internal ``metadata_credential_leak`` that serves the flag. These
drive the guard branches: it only rewires when there's an actual public-service
pivot to make, and is a no-op otherwise.
"""

from __future__ import annotations

import pytest
from cyber_webapp import WebappPack
from cyber_webapp.sampling import _flag_service_id, _networkize_ssrf
from graphschema import Edge, Node, Visibility, WorldGraph
from openrange_pack_sdk import PackError


def _graph() -> WorldGraph:
    return WorldGraph(ontology="cyber.webapp@v2")


def _ssrf(g: WorldGraph) -> None:
    g.add_node(
        Node(
            id="ssrf",
            kind="vulnerability",
            attrs={"kind": "ssrf", "params": {}},
            visibility=Visibility.HIDDEN,
        )
    )


def _flag_on(g: WorldGraph, service_id: str) -> None:
    g.add_node(Node(id="store", kind="data_store"))
    g.add_node(Node(id="record", kind="record"))
    g.add_node(
        Node(
            id="secret_flag",
            kind="secret",
            attrs={"kind": "flag"},
            visibility=Visibility.HIDDEN,
        )
    )
    g.add_edge(Edge(id="b", kind="backed_by", src=service_id, dst="store"))
    g.add_edge(Edge(id="c", kind="contains", src="store", dst="record"))
    g.add_edge(Edge(id="h", kind="holds", src="record", dst="secret_flag"))


def test_flag_service_id_none_without_flag() -> None:
    assert _flag_service_id(_graph()) is None


def test_metadata_kind_is_filtered_from_general_sampling() -> None:
    # The internal-only metadata leak is never planted by general sampling: on a
    # reachable endpoint it would hand over the flag with no exploit. The new contract
    # rejects it at the manifest boundary -- you can't even request it as a decoy -- and
    # a flat world built without it carries zero metadata vulns.
    with pytest.raises(PackError):
        WebappPack().make_builder(None).build(
            {"seed": 1, "vuln": {"weights": {"metadata_credential_leak": 3}}}
        )
    graph = (
        WebappPack()
        .make_builder(None)
        .build(
            {
                "seed": 1,
                "vuln": {"weights": {"sql_injection": 3}},
                "scale": {"vuln_count": {"min": 4, "max": 4}},
            }
        )
        .graph
    )
    metas = [
        v
        for v in graph.by_kind("vulnerability")
        if v.attrs["kind"] == "metadata_credential_leak"
    ]
    assert metas == []


def test_networkize_noop_without_public_service() -> None:
    g = _graph()
    g.add_node(Node(id="svc", kind="service", attrs={"exposure": "internal"}))
    _ssrf(g)
    _flag_on(g, "svc")
    before = set(g.nodes)
    _networkize_ssrf(g, "db")
    assert set(g.nodes) == before  # no metadata endpoint/vuln added


def test_networkize_noop_when_flag_lives_on_the_public_service() -> None:
    g = _graph()
    g.add_node(Node(id="svc", kind="service", attrs={"exposure": "public"}))
    g.add_node(Node(id="ep", kind="endpoint", attrs={"path": "/"}))
    g.add_edge(Edge(id="x", kind="exposes", src="svc", dst="ep"))
    _ssrf(g)
    g.add_edge(Edge(id="a", kind="affects", src="ssrf", dst="ep"))
    _flag_on(g, "svc")  # the flag's own service IS the public one — nothing to pivot to
    before = set(g.nodes)
    _networkize_ssrf(g, "db")
    assert set(g.nodes) == before


def test_networkize_noop_when_public_has_no_endpoint() -> None:
    g = _graph()
    g.add_node(
        Node(id="pub", kind="service", attrs={"exposure": "public"})
    )  # no exposes
    g.add_node(Node(id="int", kind="service", attrs={"exposure": "internal"}))
    _ssrf(g)
    _flag_on(g, "int")
    before = set(g.nodes)
    _networkize_ssrf(g, "db")
    assert set(g.nodes) == before


def test_networkize_noop_for_file_backed_flag() -> None:
    # A file-backed flag lives in the file map, not secrets["flag"], which the metadata
    # pivot serves — networkizing would strand it. The world stays flat (the file-read
    # oracle solves it), so the pivot half is never built.
    g = _graph()
    g.add_node(
        Node(id="pub", kind="service", attrs={"exposure": "public", "name": "web"})
    )
    g.add_node(Node(id="pub_ep", kind="endpoint", attrs={"path": "/search"}))
    g.add_edge(Edge(id="e", kind="exposes", src="pub", dst="pub_ep"))
    _ssrf(g)
    g.add_edge(Edge(id="a", kind="affects", src="ssrf", dst="pub_ep"))
    g.add_node(
        Node(id="int", kind="service", attrs={"exposure": "internal", "name": "db"})
    )
    _flag_on(g, "int")
    before = set(g.nodes)
    _networkize_ssrf(g, "file")
    assert set(g.nodes) == before  # no metadata endpoint/vuln added


def test_networkize_builds_the_pivot_half() -> None:
    g = _graph()
    g.add_node(
        Node(id="pub", kind="service", attrs={"exposure": "public", "name": "web"})
    )
    g.add_node(Node(id="pub_ep", kind="endpoint", attrs={"path": "/search"}))
    g.add_edge(Edge(id="e", kind="exposes", src="pub", dst="pub_ep"))
    _ssrf(g)
    # No affects edge from the SSRF yet: the re-home loop finds nothing to move, but
    # the internal pivot half is still added.
    g.add_node(
        Node(id="int", kind="service", attrs={"exposure": "internal", "name": "db"})
    )
    _flag_on(g, "int")
    _networkize_ssrf(g, "db")

    assert "ep_db_metadata" in g.nodes
    assert "vuln_metadata_credential_leak_0" in g.nodes
    params = g.nodes["ssrf"].attrs["params"]
    assert params["internal_host"] == "db"
    assert params["internal_path"] == "/latest/meta-data/credential"
    enables = [(e.src, e.dst) for e in g.edges.values() if e.kind == "enables"]
    assert ("ssrf", "vuln_metadata_credential_leak_0") in enables


def test_networkize_switches_decimal_ip_filter_to_host_allowlist() -> None:
    # A service-name target has no decimal-IP form, so the decimal_ip evasion can't
    # apply; generation swaps it for the http allowlist trick that does.
    g = _graph()
    g.add_node(
        Node(id="pub", kind="service", attrs={"exposure": "public", "name": "web"})
    )
    g.add_node(Node(id="pub_ep", kind="endpoint", attrs={"path": "/search"}))
    g.add_edge(Edge(id="e", kind="exposes", src="pub", dst="pub_ep"))
    g.add_node(
        Node(
            id="ssrf",
            kind="vulnerability",
            attrs={"kind": "ssrf", "params": {"ssrf_filter": "decimal_ip"}},
            visibility=Visibility.HIDDEN,
        )
    )
    g.add_edge(Edge(id="a", kind="affects", src="ssrf", dst="pub_ep"))
    g.add_node(
        Node(id="int", kind="service", attrs={"exposure": "internal", "name": "db"})
    )
    _flag_on(g, "int")
    _networkize_ssrf(g, "db")

    ssrf = g.nodes["ssrf"]
    assert ssrf.attrs["params"]["ssrf_filter"] == "host_allowlist"
    affected = [
        e.dst for e in g.edges.values() if e.kind == "affects" and e.src == "ssrf"
    ]
    assert affected == ["pub_ep"]
