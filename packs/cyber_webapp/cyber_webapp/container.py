"""Container build context for a webapp world (M1 — DESIGN.md §9, #252).

The same rendered app the ``PROCESS`` backing runs as a subprocess, packaged to run in
a real container. The container sets ``OPENRANGE_REALFS``, so the file surface is a REAL
filesystem: the file-read shape (path_traversal, xxe) does a real ``open()`` and a
traversal escape is real OS path resolution, not a dict lookup — across all nine classes
on the one generated app. (Real-shell code-exec — a real ``sh -c`` for command_injection
in the generated app — is tracked separately; the stdlib ``image_files_realfs`` variant
below is the standalone proof of that until it folds in.)

Caveat: the seed (with the flag) is COPYed into the image, so it lives in an image layer
until the app unlinks it at startup. Mounting it at run time — keeping the flag out of
the image entirely — is the follow-up (the realfs variant already injects it via env).
"""

from __future__ import annotations

from collections.abc import Mapping

from graphschema import WorldGraph
from openrange_pack_sdk import PackError

from cyber_webapp.codegen import _realize_graph
from cyber_webapp.codegen.entrypoint import APP_FILE_NAME, SEED_FILE_NAME

# A fixed in-container port (the host maps it to an ephemeral port at run time, the way
# the PROCESS backing binds port 0). jinja2 is the app's one third-party import.
CONTAINER_PORT = 8000
BASE_IMAGE = "python:3.13-slim"

# OPENRANGE_REALFS flips the rendered app's file surface to a REAL filesystem, so the
# file-read shape (path_traversal, xxe) and the readers cmdi chains hit the container fs
# instead of the in-memory dict. The PROCESS backing never sets it and stays in-memory.
_DOCKERFILE = f"""\
FROM {BASE_IMAGE}
WORKDIR /app
ENV OPENRANGE_REALFS=1
RUN pip install --no-cache-dir jinja2
COPY {APP_FILE_NAME} {SEED_FILE_NAME} ./
EXPOSE {CONTAINER_PORT}
CMD ["python", "{APP_FILE_NAME}", "--host", "0.0.0.0", \
"--port", "{CONTAINER_PORT}", "--log", "/app/requests.jsonl"]
"""


def image_files(graph: WorldGraph) -> dict[str, str]:
    """The build context for the world's container: the Dockerfile + the rendered app
    + its seed. Same rendered app the PROCESS backing runs, but the container sets
    OPENRANGE_REALFS so its file surface is a REAL filesystem (real open() / traversal),
    not the in-memory dict."""
    rendered = _realize_graph(graph)
    return {
        "Dockerfile": _DOCKERFILE,
        APP_FILE_NAME: rendered[APP_FILE_NAME],
        SEED_FILE_NAME: rendered[SEED_FILE_NAME],
    }


# --- Real fs/shell variant (M1 step 1) ----------------------------------------------
#
# At CONTAINER backing the command-injection handler runs a REAL shell against a REAL
# flag file, so `; cat <path>` actually executes against the container filesystem —
# genuine RCE/file-read instead of the in-memory emulation. The flag arrives via the
# OPENRANGE_FLAG env var at run time (never an image layer) and the app writes it to
# its real path at startup. Stdlib only, so no jinja2 and a faster build.
#
# Scope: command_injection only, and a plain `; cat` injection (the mutually-exclusive
# contexts of §6 are a refinement to re-apply over the real shell). The next increment
# wires this in as the real Backing.CONTAINER runtime for every class.

_DOCKERFILE_REALFS = f"""\
FROM {BASE_IMAGE}
WORKDIR /app
COPY {APP_FILE_NAME} ./
EXPOSE {CONTAINER_PORT}
CMD ["python", "{APP_FILE_NAME}", "--host", "0.0.0.0", "--port", "{CONTAINER_PORT}"]
"""

_REALFS_APP_HEADER = """\
import os
import re
import subprocess
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

"""

_REALFS_APP_BODY = """

def _init_flag():
    flag = os.environ.get("OPENRANGE_FLAG", "")
    parent = os.path.dirname(FLAG_PATH)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(FLAG_PATH, "w", encoding="utf-8") as handle:
        handle.write(flag)


def _run_cmdi(target):
    # REAL command injection over a real shell. The same naive, context-specific filter
    # the in-memory emulation uses keeps the mutually-exclusive contexts: each strips
    # the OTHER vectors, leaving exactly one live, then a REAL shell runs the result.
    if INJ_CONTEXT == "quoted":
        command = BASE + " " + QUOTE + target + QUOTE
    else:
        command = BASE + " " + target
    if INJ_CONTEXT == "substitution":
        command = command.translate(str.maketrans(";|&\\n", "    "))
    else:
        command = re.sub(r"\\$\\([^()]*\\)|`[^`]*`", "", command)
    try:
        completed = subprocess.run(
            ["sh", "-c", command], capture_output=True, timeout=5, check=False
        )
    except subprocess.SubprocessError:
        return b"error"
    return completed.stdout + completed.stderr


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == ENDPOINT:
            query = parse_qs(parsed.query, keep_blank_values=True)
            body = _run_cmdi((query.get(PARAM) or [""])[0])
        elif parsed.path == "/":
            body = b"ok"
        else:
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        return


def main():
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    _init_flag()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(json.dumps({"host": args.host, "port": args.port}), flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
"""


def _flag_record_key(graph: WorldGraph) -> str:
    for record in graph.by_kind("record"):
        for edge in graph.out_edges(record.id, "holds"):
            if graph.nodes[edge.dst].attrs.get("kind") == "flag":
                return str(record.attrs["key"])
    raise PackError("no record holds the flag")


def realfs_cmdi_app(graph: WorldGraph) -> str:
    """A stdlib real-shell command-injection app for the container backing, built from
    the world's command_injection vuln (its endpoint + injected parameter) and the
    flag's path. The flag itself is supplied at run time via OPENRANGE_FLAG."""
    vuln = next(
        n
        for n in graph.by_kind("vulnerability")
        if n.attrs.get("kind") == "command_injection"
    )
    params = vuln.attrs["params"]
    if not isinstance(params, Mapping):
        raise PackError("command_injection vuln has no params mapping")
    endpoint_id = next(e.dst for e in graph.out_edges(vuln.id, "affects"))
    constants = (
        f"PARAM = {str(params['target_param'])!r}\n"
        f"ENDPOINT = {str(graph.nodes[endpoint_id].attrs['public_url'])!r}\n"
        f"FLAG_PATH = {_flag_record_key(graph)!r}\n"
        f"INJ_CONTEXT = {str(params.get('inj_context', 'separator'))!r}\n"
        f"QUOTE = {str(params.get('quote', chr(39)))!r}\n"
        "BASE = 'echo pinging'\n"
    )
    return _REALFS_APP_HEADER + constants + _REALFS_APP_BODY


def image_files_realfs(graph: WorldGraph) -> dict[str, str]:
    """Build context for the real fs/shell command-injection container (no seed in the
    image — the flag is an env var at run time)."""
    return {"Dockerfile": _DOCKERFILE_REALFS, APP_FILE_NAME: realfs_cmdi_app(graph)}
