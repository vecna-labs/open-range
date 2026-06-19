"""Build the ``/openapi.json`` discovery payload from a webapp world graph."""

from __future__ import annotations

from graphschema import Node, WorldGraph


def build_discovery(
    graph: WorldGraph, only_services: frozenset[str] | None = None
) -> dict[str, object]:
    # A per-service container advertises its own endpoints; the single app only its
    # public services -- internals are reached by pivoting, not read off the public doc.
    if only_services is not None:
        services = [n for n in graph.by_kind("service") if n.id in only_services]
        url_key = "path"
    else:
        services = [
            n for n in graph.by_kind("service") if n.attrs.get("exposure") == "public"
        ]
        url_key = "public_url"

    service_ids = {s.id for s in services}
    endpoints_by_service: dict[str, list[Node]] = {sid: [] for sid in service_ids}
    for edge in graph.edges.values():
        if edge.kind == "exposes" and edge.src in service_ids:
            endpoint = graph.nodes.get(edge.dst)
            if endpoint is not None and endpoint.kind == "endpoint":
                endpoints_by_service[edge.src].append(endpoint)

    services_payload: list[dict[str, object]] = [
        {
            "name": str(service.attrs.get("name", service.id)),
            "kind": str(service.attrs.get("kind", "unknown")),
            "exposure": str(service.attrs.get("exposure", "internal")),
            "paths": [
                {
                    "url": str(endpoint.attrs[url_key]),
                    "method": str(endpoint.attrs.get("method", "GET")),
                }
                for endpoint in endpoints_by_service[service.id]
            ],
        }
        for service in services
    ]
    return {"title": _discovery_title(graph), "services": services_payload}


def _discovery_title(graph: WorldGraph) -> str:
    title = graph.meta.get("discovery_title")
    if isinstance(title, str) and title:
        return title
    return "Internal Services"
