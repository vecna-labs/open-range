from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from graphschema import WorldGraph
from jinja2 import Environment, FileSystemLoader, StrictUndefined, select_autoescape

from cyber_webapp.codegen.discovery import build_discovery
from cyber_webapp.codegen.entrypoint import APP_FILE_NAME, SEED_FILE_NAME
from cyber_webapp.codegen.handlers import build_handlers_and_routes
from cyber_webapp.codegen.seeding import project_seed

_TEMPLATES_DIR = Path(__file__).parent / "templates"


def _realize_graph(graph: WorldGraph) -> dict[str, str]:
    # Secret must never land on disk — app.py loads seed into memory then unlinks.
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
    files = cast("Mapping[str, object]", seed["files"])
    schema = cast("Mapping[str, object]", seed["schema"])
    guarded = cast("Mapping[str, object]", seed["guarded"])
    seed_payload = {
        "accounts": {k: dict(v) for k, v in accounts.items()},
        "secrets": dict(secrets),
        "records": {k: dict(v) for k, v in records.items()},
        "files": dict(files),
        "schema": dict(schema),
        "guarded": dict(guarded),
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
