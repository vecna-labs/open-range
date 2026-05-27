"""Snapshot topology normalization.

KNOWN DOMAIN LEAK: cyber kind-name branching (`host`/`service`/
`endpoint`/`vulnerability`/`account`/`runs_on`/`exposes`/`affects`/
`backed_by`) — follow-up moves it onto a per-pack `Pack.topology_view`
hook.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

from graphschema import WorldGraph

from openrange.core.admit import Snapshot
from openrange.dashboard.events import json_safe


def empty_runtime_topology() -> dict[str, object]:
    return {
        "services": [],
        "edges": [],
        "zones": [],
        "users": [],
        "green_personas": [],
    }


def normalized_runtime_topology(snapshot: Snapshot) -> dict[str, object]:
    raw = embedded_topology(snapshot)
    services = normalized_rows(raw.get("services"))
    known_services = {str(service.get("id", "")) for service in services}

    # Promote any task entrypoint that doesn't already exist as a service
    # row into one — the dashboard needs a row to render the agent's
    # starting surface even when the pack didn't ship a topology block.
    graph = snapshot.graph
    for task in snapshot.tasks:
        for node_id in task.entrypoints:
            if node_id in known_services:
                continue
            node = graph.nodes.get(node_id)
            kind = node.kind if node is not None else ""
            services.append(
                {
                    "id": node_id,
                    "kind": kind,
                    "zone": "episode",
                    "ports": [],
                },
            )
            known_services.add(node_id)

    zones = normalized_strings(raw.get("zones"))
    service_zones = sorted(
        {
            str(service["zone"])
            for service in services
            if isinstance(service.get("zone"), str)
        },
    )
    if not zones:
        zones = service_zones
    else:
        zones.extend(zone for zone in service_zones if zone not in zones)

    # Fall back to manifest NPC entries so the scene renders before any
    # tick has fired.
    personas = normalized_rows(raw.get("green_personas"))
    if not personas:
        manifest = _manifest_from_snapshot(snapshot)
        npc_entries = manifest.get("npc", []) if isinstance(manifest, Mapping) else []
        if isinstance(npc_entries, Sequence) and not isinstance(
            npc_entries, str | bytes
        ):
            personas = personas_from_manifest(
                [entry for entry in npc_entries if isinstance(entry, Mapping)],
            )

    return {
        "services": services,
        "edges": normalized_rows(raw.get("edges")),
        "zones": zones,
        "users": normalized_rows(raw.get("users")),
        "green_personas": personas,
    }


def personas_from_manifest(
    npc_entries: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Seat NPCs at desks without waiting for the first tick to land an event."""
    rows: list[dict[str, object]] = []
    for entry in npc_entries:
        config = entry.get("config", {})
        if not isinstance(config, Mapping):
            continue
        name = config.get("name")
        if not isinstance(name, str) or not name:
            continue
        count_raw = entry.get("count", 1)
        count = count_raw if isinstance(count_raw, int) and count_raw > 0 else 1
        role = config.get("role", "engineer")
        title = config.get("title", "")
        tone = config.get("tone", "warm, professional")
        colleagues = config.get("colleagues", ())
        home = config.get("home")
        for index in range(count):
            suffix = "" if count == 1 else f"-{index + 1}"
            rows.append(
                {
                    "id": f"{name}{suffix}",
                    "display_name": f"{name}{suffix}",
                    "role": str(role) if isinstance(role, str) else "engineer",
                    "title": str(title) if isinstance(title, str) else "",
                    "tone": str(tone) if isinstance(tone, str) else "",
                    "colleagues": (
                        [str(c) for c in colleagues if isinstance(c, str)]
                        if isinstance(colleagues, list | tuple)
                        else []
                    ),
                    "home": str(home) if isinstance(home, str) else None,
                },
            )
    return rows


def embedded_topology(snapshot: Snapshot) -> dict[str, object]:
    """Collect a topology dict from manifest hints + the graph fallback.

    If the manifest carries a ``world`` block (rare; a pack pre-baking
    topology) it is honored; otherwise the topology is projected from
    the graph.
    """
    raw: dict[str, object] = {}
    manifest = _manifest_from_snapshot(snapshot)
    world_block = manifest.get("world") if isinstance(manifest, Mapping) else None
    if isinstance(world_block, Mapping):
        topology_block = world_block.get("topology")
        if isinstance(topology_block, Mapping):
            raw.update(topology_block)
        for key in ("services", "edges", "zones", "users", "green_personas"):
            value = world_block.get(key)
            if value is not None:
                raw[key] = value

    if not raw.get("services"):
        graph_view = topology_from_world_graph(snapshot.graph)
        for key, value in graph_view.items():
            raw.setdefault(key, value)
    return raw


def topology_from_world_graph(graph: WorldGraph) -> dict[str, object]:
    """Coupled to cyber-pack ontology kind names; returns ``{}`` when no
    service nodes so the fallback chain stays a no-op."""
    if not graph.nodes:
        return {}
    services = _services_from_graph(graph)
    if not services:
        return {}
    return {
        "services": services,
        "edges": _edges_from_graph(graph),
        "zones": sorted({str(s["zone"]) for s in services if s.get("zone")}),
        "users": _users_from_graph(graph),
    }


def _services_from_graph(graph: WorldGraph) -> list[dict[str, object]]:
    host_zone = {
        n.id: str(n.attrs.get("zone", ""))
        for n in graph.nodes.values()
        if n.kind == "host"
    }
    service_host: dict[str, str] = {}
    endpoints_by_service: dict[str, list[str]] = {}
    for edge in graph.edges.values():
        if edge.kind == "runs_on":
            service_host[edge.src] = edge.dst
        elif edge.kind == "exposes":
            endpoints_by_service.setdefault(edge.src, []).append(edge.dst)
    endpoint_path = {
        n.id: str(n.attrs.get("path", ""))
        for n in graph.nodes.values()
        if n.kind == "endpoint"
    }
    vuln_kind = {
        n.id: str(n.attrs.get("kind", ""))
        for n in graph.nodes.values()
        if n.kind == "vulnerability"
    }
    vuln_target: dict[str, str] = {}
    for edge in graph.edges.values():
        if edge.kind == "affects":
            vuln_target[edge.src] = edge.dst

    services: list[dict[str, object]] = []
    for node in graph.nodes.values():
        if node.kind != "service":
            continue
        endpoints = endpoints_by_service.get(node.id, [])
        zone = host_zone.get(service_host.get(node.id, ""), "")
        vulns = sorted(
            {
                vuln_kind[vid]
                for vid, target in vuln_target.items()
                if target == node.id or target in endpoints
                if vid in vuln_kind
            },
        )
        services.append(
            {
                "id": str(node.attrs.get("name", node.id)),
                "kind": str(node.attrs.get("kind", "")),
                "zone": zone or "default",
                "exposure": str(node.attrs.get("exposure", "")),
                "ports": [],
                "paths": sorted(endpoint_path.get(ep, "") for ep in endpoints),
                "vulns": vulns,
            },
        )
    return services


def _edges_from_graph(graph: WorldGraph) -> list[dict[str, object]]:
    service_name = {
        n.id: str(n.attrs.get("name", n.id))
        for n in graph.nodes.values()
        if n.kind == "service"
    }
    edges: list[dict[str, object]] = []
    for edge in graph.edges.values():
        if edge.kind != "backed_by":
            continue
        source = service_name.get(edge.src)
        if source is None:
            continue
        edges.append(
            {"source": source, "target": str(edge.dst), "relation": "backed_by"},
        )
    return edges


def _users_from_graph(graph: WorldGraph) -> list[dict[str, object]]:
    return [
        {
            "id": str(n.attrs.get("username", n.id)),
            "role": str(n.attrs.get("role", "user")),
        }
        for n in graph.nodes.values()
        if n.kind == "account"
    ]


def normalized_rows(value: object) -> list[dict[str, object]]:
    if isinstance(value, Mapping):
        iterable = tuple(value.items())
    elif isinstance(value, Sequence) and not isinstance(value, str | bytes):
        iterable = tuple((None, item) for item in value)
    else:
        return []

    rows: list[dict[str, object]] = []
    for key, item in iterable:
        if isinstance(item, Mapping):
            row = dict(cast(Mapping[str, object], json_safe(item)))
            if "id" not in row:
                row["id"] = "" if key is None else str(key)
            rows.append(row)
        elif isinstance(item, str):
            rows.append({"id": item})
    return rows


def normalized_strings(value: object) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        return []
    return [item for item in value if isinstance(item, str)]


def public_world(world: Mapping[str, object]) -> dict[str, object]:
    redacted: dict[str, object] = {}
    for key, value in world.items():
        if sensitive_world_key(key):
            redacted[key] = "[redacted]"
        else:
            redacted[key] = value
    return redacted


def sensitive_world_key(key: str) -> bool:
    normalized = key.lower()
    return normalized == "flag" or any(
        marker in normalized for marker in ("secret", "password", "token")
    )


def stored_entrypoints(tasks: Sequence[object]) -> list[dict[str, object]]:
    entrypoints: list[dict[str, object]] = []
    for task in tasks:
        if not isinstance(task, Mapping):
            continue
        task_id = task.get("id")
        for node_id in stored_task_entrypoints(task):
            entrypoints.append(
                {"task_id": str(task_id), "node_id": node_id, "node_kind": ""},
            )
    return entrypoints


def stored_missions(tasks: Sequence[object]) -> list[dict[str, object]]:
    missions: list[dict[str, object]] = []
    for task in tasks:
        if not isinstance(task, Mapping):
            continue
        missions.append(
            {
                "task_id": str(task.get("id", "")),
                "instruction": str(task.get("instruction", "")),
            },
        )
    return missions


def stored_task_entrypoints(task: Mapping[str, object]) -> list[str]:
    """Read a stored task's entrypoints, returning [] on malformed input."""
    rows = task.get("entrypoints")
    if not isinstance(rows, list):
        return []
    return [str(row) for row in rows if isinstance(row, str)]


def _manifest_from_snapshot(snapshot: Snapshot) -> Mapping[str, object]:
    manifest = snapshot.lineage.get("manifest")
    if isinstance(manifest, Mapping):
        return cast(Mapping[str, object], manifest)
    return {}
