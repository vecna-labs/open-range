"""Container build context for a webapp world (M1 — DESIGN.md §9, #252).

The same rendered app the ``PROCESS`` backing runs as a subprocess, packaged to run in a
real container. The container sets ``OPENRANGE_REALFS``, so the app's surfaces go real:
the file-read shape (path_traversal, xxe) does a real ``open()`` and a traversal escape
is real OS path resolution, and command_injection runs a real ``sh -c`` — genuine RCE /
file-read across the nine classes on the ONE generated app, not a bespoke app per class.

Caveat: the seed (with the flag) is COPYed into the image, so it lives in an image layer
until the app unlinks it at startup. Mounting it at run time — keeping the flag out of
the image entirely — is the follow-up (the isolation increment, #202).
"""

from __future__ import annotations

from graphschema import WorldGraph

from cyber_webapp.codegen import _realize_graph
from cyber_webapp.codegen.entrypoint import APP_FILE_NAME, SEED_FILE_NAME

# A fixed in-container port (the host maps it to an ephemeral port at run time, the way
# the PROCESS backing binds port 0).
CONTAINER_PORT = 8000
BASE_IMAGE = "python:3.13-slim"

# OPENRANGE_REALFS flips the rendered app's surfaces to the real container: the file map
# becomes a real filesystem (real open() / traversal) and command_injection a real
# `sh -c`. The PROCESS backing never sets it and stays the in-memory emulation.
#
# The diagnostic tools command_injection's base_command samples from (ping / nslookup /
# dig / host / traceroute) are installed so the real shell acts like a real vulnerable
# endpoint: a chained `; cat` reads the flag, and `$(cat flag)` leaks it too since each
# tool echoes the (flag-as-)hostname in its resolver error. jinja2 is the one pip dep.
_DOCKERFILE = f"""\
FROM {BASE_IMAGE}
WORKDIR /app
ENV OPENRANGE_REALFS=1
RUN apt-get update \
&& apt-get install -y --no-install-recommends iputils-ping dnsutils traceroute \
&& rm -rf /var/lib/apt/lists/* \
&& pip install --no-cache-dir jinja2
COPY {APP_FILE_NAME} {SEED_FILE_NAME} ./
EXPOSE {CONTAINER_PORT}
CMD ["python", "{APP_FILE_NAME}", "--host", "0.0.0.0", \
"--port", "{CONTAINER_PORT}", "--log", "/app/requests.jsonl"]
"""


def image_files(graph: WorldGraph) -> dict[str, str]:
    """The build context for the world's container: the Dockerfile + the rendered app
    + its seed. Same rendered app the PROCESS backing runs, but the container sets
    OPENRANGE_REALFS so its surfaces are real (real open() / traversal, real `sh -c`),
    not the in-memory emulation."""
    rendered = _realize_graph(graph)
    return {
        "Dockerfile": _DOCKERFILE,
        APP_FILE_NAME: rendered[APP_FILE_NAME],
        SEED_FILE_NAME: rendered[SEED_FILE_NAME],
    }
