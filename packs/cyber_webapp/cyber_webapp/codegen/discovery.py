"""Build the ``/openapi.json`` discovery payload from a webapp world graph."""

from __future__ import annotations

from graphschema import Node, WorldGraph


def build_discovery(
    graph: WorldGraph, only_services: frozenset[str] | None = None
) -> dict[str, object]:
    # ``only_services`` scopes the discovery doc to those services — a per-service app
    # advertises only its own endpoints, not the rest of the (internal) estate.
    services_by_id: dict[str, Node] = {
        n.id: n
        for n in graph.nodes.values()
        if n.kind == "service" and (only_services is None or n.id in only_services)
    }
    endpoints_by_service: dict[str, list[Node]] = {sid: [] for sid in services_by_id}
    for edge in graph.edges.values():
        if edge.kind != "exposes":
            continue
        if edge.src in endpoints_by_service:
            endpoint = next(
                (
                    n
                    for n in graph.nodes.values()
                    if n.kind == "endpoint" and n.id == edge.dst
                ),
                None,
            )
            if endpoint is not None:
                endpoints_by_service[edge.src].append(endpoint)

    services_payload: list[dict[str, object]] = []
    for service_id, service in services_by_id.items():
        name = str(service.attrs.get("name", service_id))
        kind = str(service.attrs.get("kind", "unknown"))
        exposure = str(service.attrs.get("exposure", "internal"))
        paths: list[dict[str, str]] = []
        for endpoint in endpoints_by_service[service_id]:
            # Per-service: advertise the bare path the service serves on its own
            # container (the single app uses the /svc/<name> namespace instead).
            url_key = "path" if only_services is not None else "public_url"
            paths.append(
                {
                    "url": str(endpoint.attrs[url_key]),
                    "method": str(endpoint.attrs.get("method", "GET")),
                },
            )
        services_payload.append(
            {
                "name": name,
                "kind": kind,
                "exposure": exposure,
                "paths": paths,
            },
        )

    return {
        "title": _discovery_title(graph),
        "services": services_payload,
    }


def _discovery_title(graph: WorldGraph) -> str:
    title = graph.meta.get("discovery_title")
    if isinstance(title, str) and title:
        return title
    return "Internal Services"
