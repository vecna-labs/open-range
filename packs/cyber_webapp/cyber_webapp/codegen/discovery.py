"""Build the ``/openapi.json`` discovery payload from a webapp world graph.

The realized app exposes this at ``/openapi.json`` so agents (and the
``cyber.admin_audit`` NPC) can enumerate routes without having to
guess. The shape is intentionally lightweight — not actual OpenAPI 3,
just a small JSON document describing services + paths + which
endpoints are public.
"""

from __future__ import annotations

from openrange.world_ir import Node, WorldGraph


def build_discovery(graph: WorldGraph) -> dict[str, object]:
    """Return a ``{services: [{name, kind, exposure, paths}]}`` summary.

    Each service lists the paths it exposes. Public paths (``web``
    service) are mounted at the root; internal paths only at
    ``/svc/<name><path>``. The payload is fed to the codegen template
    as ``discovery`` and embedded in the generated app.
    """
    services_by_id: dict[str, Node] = {
        n.id: n for n in graph.nodes.values() if n.kind == "service"
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
            ep_path = str(endpoint.attrs.get("path", "/"))
            paths.append(
                {
                    "internal": f"/svc/{name}{ep_path}",
                    "public": ep_path if kind == "web" else "",
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
    """Read the per-build display title from the network node's attrs.

    The sampler stashes a ``display_title`` on the main network node so
    the discovery payload doesn't telegraph the scenario name. Falls
    back to a generic string if the graph wasn't built by the v1
    sampler (e.g. tests assembling minimal graphs).
    """
    for node in graph.nodes.values():
        if node.kind == "network":
            title = node.attrs.get("display_title")
            if isinstance(title, str) and title:
                return title
    return "Internal Services"
