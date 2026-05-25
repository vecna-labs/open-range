"""Codegen-based source generation for the cyber webapp pack.

Walks a webapp-ontology ``WorldGraph`` and produces the files that
``WebappRuntimeHandle`` materializes into a per-episode workspace.
Each ``service`` becomes a path namespace (``/svc/<name>/...``); the
public ``web`` service also mounts at ``/`` for convenience. Each
``endpoint`` becomes a route. Each ``vulnerability`` with an
``affects`` edge to an endpoint has its template body inlined as that
endpoint's handler.

Pipeline:

  1. ``seeding.project_seed`` — graph → seed dicts (flag, accounts,
     secrets, records) baked into ``seed.json``
  2. ``handlers.build_handlers_and_routes`` — graph → handler funcs
     and route table, with vuln templates inlined per endpoint
  3. ``discovery.build_discovery`` — graph → ``/openapi.json``
     payload embedded in the generated app
  4. Render the Jinja template under ``templates/app.py.j2``

This module's ONLY public consumer is the pack's
``WebappRuntimeHandle`` (in ``cyber_webapp.realize``); the
``_realize_graph`` function is therefore module-private and returns a
plain ``{relative_path: source}`` mapping that the handle writes to
disk.

Multi-process / docker-compose isolation is a deferred concern;
``Backing.PROCESS`` is the only supported backing today. Every service
is reachable on the same single Python process — vulns fire
end-to-end, but network-level service isolation is simulated, not
real.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from jinja2 import Environment, FileSystemLoader, StrictUndefined, select_autoescape

from cyber_webapp.codegen.discovery import build_discovery
from cyber_webapp.codegen.entrypoint import APP_FILE_NAME, SEED_FILE_NAME
from cyber_webapp.codegen.handlers import build_handlers_and_routes
from cyber_webapp.codegen.seeding import project_seed
from openrange.world_ir import WorldGraph

_TEMPLATES_DIR = Path(__file__).parent / "templates"


def _realize_graph(graph: WorldGraph) -> dict[str, str]:
    """Render ``graph`` into a ``{relative_path: source}`` file mapping.

    Module-private — the only intended caller is
    ``WebappRuntimeHandle.__init__``. The returned mapping always
    contains ``app.py`` (executable; no secrets) plus ``seed.json``
    (accounts/secrets/records + SQL schema). At startup ``app.py``
    reads the seed into an in-memory SQLite db and unlinks the file,
    so the agent never sees the secret on disk.
    """
    seed = project_seed(graph)
    handlers, routes = build_handlers_and_routes(graph)
    discovery = build_discovery(graph)

    template = _jinja_env().get_template("app.py.j2")
    source = template.render(
        handlers=handlers,
        routes=routes,
        discovery=discovery,
    )

    accounts = cast("Mapping[str, Mapping[str, object]]", seed["accounts"])
    secrets = cast("Mapping[str, object]", seed["secrets"])
    records = cast("Mapping[str, Mapping[str, object]]", seed["records"])
    schema = cast("Mapping[str, object]", seed["schema"])
    seed_payload = {
        "accounts": {k: dict(v) for k, v in accounts.items()},
        "secrets": dict(secrets),
        "records": {k: dict(v) for k, v in records.items()},
        "schema": dict(schema),
    }
    seed_json = json.dumps(seed_payload, sort_keys=True, indent=2)

    return {
        APP_FILE_NAME: source,
        SEED_FILE_NAME: seed_json,
    }


def _jinja_env() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(_TEMPLATES_DIR)),
        undefined=StrictUndefined,
        autoescape=select_autoescape(disabled_extensions=("py",), default=False),
        keep_trailing_newline=True,
    )
