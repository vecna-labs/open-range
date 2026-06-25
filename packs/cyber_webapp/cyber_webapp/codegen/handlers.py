from __future__ import annotations

import ast
import textwrap
from collections.abc import Mapping

from graphschema import Node, WorldGraph
from openrange_pack_sdk import PackError

from cyber_webapp.sampling import _is_networked
from cyber_webapp.vulnerabilities import (
    CATALOG as VULN_CATALOG,
)
from cyber_webapp.vulnerabilities import render_vulnerability


def build_handlers_and_routes(
    graph: WorldGraph,
    only_services: frozenset[str] | None = None,
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    services_by_id: dict[str, Node] = {
        n.id: n for n in graph.nodes.values() if n.kind == "service"
    }
    endpoints_by_id: dict[str, Node] = {
        n.id: n for n in graph.nodes.values() if n.kind == "endpoint"
    }
    vulns_by_id: dict[str, Node] = {
        n.id: n for n in graph.nodes.values() if n.kind == "vulnerability"
    }
    service_for_endpoint: dict[str, str] = {}
    for edge in graph.edges.values():
        if edge.kind == "exposes":
            service_for_endpoint[edge.dst] = edge.src
    vuln_for_target: dict[str, str] = {}
    for edge in graph.edges.values():
        if edge.kind == "affects":
            vuln_for_target.setdefault(edge.dst, edge.src)

    handlers: list[dict[str, str]] = []
    routes: list[dict[str, str]] = []
    internal_routes: list[dict[str, str]] = []
    # In a single-process networked world every service shares one server, so a direct
    # request to an internal ``/svc/<name>`` route would hand over the flag with no SSRF
    # pivot. Those routes go into ``internal_routes`` (reachable only via the in-process
    # pivot), making network position the gate as under CONTAINER. A flat world has no
    # such boundary — every service answers directly, which is the intended solve.
    segmented = only_services is None and _is_networked(graph)

    for endpoint_id, endpoint in endpoints_by_id.items():
        service_id = service_for_endpoint.get(endpoint_id)
        if service_id is None:
            continue
        if only_services is not None and service_id not in only_services:
            continue
        service = services_by_id[service_id]
        service_name = str(service.attrs.get("name", service_id))
        path = str(endpoint.attrs.get("path", "/"))
        public_url = str(endpoint.attrs["public_url"])
        handler_name = _handler_name(service_name, endpoint_id)
        vuln_id = vuln_for_target.get(endpoint_id)
        if vuln_id is None:
            vuln_id = vuln_for_target.get(service_id)
        if vuln_id is not None and vuln_id in vulns_by_id:
            vuln_node = vulns_by_id[vuln_id]
            body = _render_vuln_body(vuln_node)
        else:
            realized = endpoint.attrs.get("realized_handler")
            if isinstance(realized, str) and realized.strip():
                body = _extract_handle_body(realized)
            else:
                kind = str(service.attrs.get("kind", ""))
                body = _default_handler_body(service_name, path, kind)
        docstring = f"Endpoint {service_name}{path}."
        handlers.append(
            {"name": handler_name, "body": body, "docstring": docstring},
        )
        # Single app: route on namespaced ``public_url`` (/svc/<name>). Per-service
        # container: route on bare ``path`` (reached at http://<service-name><path>).
        route_path = path if only_services is not None else public_url
        method = str(endpoint.attrs.get("method", "GET"))
        route = {"path": route_path, "handler": handler_name, "method": method}
        internal = segmented and service.attrs.get("exposure") != "public"
        (internal_routes if internal else routes).append(route)
    return handlers, routes, internal_routes


def _render_vuln_body(vuln_node: Node) -> str:
    realized = vuln_node.attrs.get("realized_handler")
    if isinstance(realized, str) and realized.strip():
        return _extract_handle_body(realized)
    kind = str(vuln_node.attrs.get("kind", ""))
    catalog_entry = VULN_CATALOG.get(kind)
    if catalog_entry is None:
        return _default_handler_body("", "/", "")
    params = vuln_node.attrs.get("params", {})
    if not isinstance(params, Mapping):
        params = {}
    rendered = render_vulnerability(catalog_entry, params)
    return _extract_handle_body(rendered)


def _extract_handle_body(rendered: str) -> str:
    try:
        module = ast.parse(rendered)
    except SyntaxError as exc:
        raise PackError(
            f"rendered vuln template is not valid Python: {exc}",
        ) from exc

    pre_statements: list[ast.stmt] = []
    handle_func: ast.FunctionDef | None = None
    for index, stmt in enumerate(module.body):
        if isinstance(stmt, ast.FunctionDef) and stmt.name == "handle":
            handle_func = stmt
            continue
        if (
            index == 0
            and isinstance(stmt, ast.Expr)
            and isinstance(stmt.value, ast.Constant)
            and isinstance(stmt.value.value, str)
        ):
            continue
        if isinstance(stmt, ast.Import | ast.ImportFrom):
            continue
        pre_statements.append(stmt)
    if handle_func is None:
        raise PackError("rendered vuln template has no def handle(...)")

    body = list(handle_func.body)
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]
    if not body:
        body = [ast.Pass()]

    statements = pre_statements + body
    rendered_body = "\n".join(ast.unparse(stmt) for stmt in statements)
    return textwrap.indent(rendered_body, "    ") + "\n"


def _default_handler_body(service_name: str, path: str, kind: str) -> str:
    if kind == "api":
        body = (
            f'payload = {{"items": [], "next_cursor": None, '
            f'"resource": "{path}"}}\n'
            'return 200, {"Content-Type": "application/json"}, '
            "json.dumps(payload).encode()\n"
        )
    elif kind == "db":
        body = (
            'schema = state["schema"]\n'
            'table = schema["table"]\n'
            'key_col = schema["key_column"]\n'
            'value_col = schema["value_column"]\n'
            'requested = (query.get("key", [""]) or [""])[0]\n'
            'cursor = state["db"].cursor()\n'
            "if requested:\n"
            '    sql = f"SELECT {key_col}, {value_col} FROM {table} WHERE '
            '{key_col} = ?"\n'
            "    rows = [\n"
            "        dict(r) for r in cursor.execute(sql, (requested,)).fetchall()\n"
            "    ]\n"
            "else:\n"
            '    sql = f"SELECT {key_col} FROM {table} ORDER BY {key_col}"\n'
            "    rows = [{key_col: r[0]} for r in cursor.execute(sql).fetchall()]\n"
            'payload = {"rows": rows, "count": len(rows)}\n'
            'return 200, {"Content-Type": "application/json"}, '
            "json.dumps(payload).encode()\n"
        )
    elif kind == "auth":
        body = (
            'payload = {"session": None, "authenticated": False}\n'
            'return 401, {"Content-Type": "application/json"}, '
            "json.dumps(payload).encode()\n"
        )
    elif kind == "web":
        body = (
            f'html = b"<h1>{service_name}</h1>'
            f'<p>route: {path}</p>"\n'
            'return 200, {"Content-Type": "text/html"}, html\n'
        )
    else:
        body = (
            f'payload = {{"service": "{service_name}", '
            f'"path": "{path}", "status": "ok"}}\n'
            'return 200, {"Content-Type": "application/json"}, '
            "json.dumps(payload).encode()\n"
        )
    return textwrap.indent(body, "    ")


def _handler_name(service_name: str, endpoint_id: str) -> str:
    safe_service = service_name.replace(".", "_").replace("-", "_")
    safe_endpoint = endpoint_id.replace(".", "_").replace("-", "_")
    return f"handle__{safe_service}__{safe_endpoint}"
