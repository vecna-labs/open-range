"""Container build context for a webapp world (M1 — DESIGN.md §9, #252).

The same rendered app the ``PROCESS`` backing runs as a subprocess, packaged to run in
a real container — real filesystem and shell, so file-read / RCE exploits eventually
hit the real thing instead of the in-memory emulation. This first brick produces the
build context; the runtime that builds and runs it lands on top.

Caveat (first brick): the seed (with the flag) is COPYed into the image, so it lives in
an image layer until the app unlinks it at startup. Mounting it at run time — keeping
the flag out of the image entirely — is the follow-up.
"""

from __future__ import annotations

from graphschema import WorldGraph

from cyber_webapp.codegen import _realize_graph
from cyber_webapp.codegen.entrypoint import APP_FILE_NAME, SEED_FILE_NAME

# A fixed in-container port (the host maps it to an ephemeral port at run time, the way
# the PROCESS backing binds port 0). jinja2 is the app's one third-party import.
CONTAINER_PORT = 8000
BASE_IMAGE = "python:3.13-slim"

_DOCKERFILE = f"""\
FROM {BASE_IMAGE}
WORKDIR /app
RUN pip install --no-cache-dir jinja2
COPY {APP_FILE_NAME} {SEED_FILE_NAME} ./
EXPOSE {CONTAINER_PORT}
CMD ["python", "{APP_FILE_NAME}", "--host", "0.0.0.0", \
"--port", "{CONTAINER_PORT}", "--log", "/app/requests.jsonl"]
"""


def image_files(graph: WorldGraph) -> dict[str, str]:
    """The build context for the world's container: the Dockerfile + the rendered app
    + its seed. Same content the PROCESS backing renders, plus the packaging."""
    rendered = _realize_graph(graph)
    return {
        "Dockerfile": _DOCKERFILE,
        APP_FILE_NAME: rendered[APP_FILE_NAME],
        SEED_FILE_NAME: rendered[SEED_FILE_NAME],
    }
