"""Container build context for a webapp world.

The same rendered app the ``PROCESS`` backing runs as a subprocess, packaged to run in a
real container. The container sets ``OPENRANGE_REALFS``, so the app's surfaces go real:
the file-read shape (path_traversal, xxe) does a real ``open()`` and a traversal escape
is real OS path resolution, and command_injection runs a real ``sh -c`` — genuine RCE /
file-read across the nine classes on the one generated app.

A world is the *target* the agent attacks, reached only over its HTTP surface — it is
not the agent's toolbox. So it carries only what its OWN vulns run server-side: the
diagnostic tool command_injection shells out to (ping / nslookup / …) is installed ONLY
when the world has that vuln, and only the one its ``base_command`` names. A world with
no command_injection installs no OS tools. The attacking agent's own recon/exploit
tooling lives in a separate sandbox the harness brings, not in here.

The seed (with the flag) is COPYed into the image, so the flag lives in an image layer
until the app unlinks it at startup; run-time mounting would keep it out of the image
entirely.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from graphschema import WorldGraph

from cyber_webapp.codegen import _realize_graph
from cyber_webapp.codegen.entrypoint import APP_FILE_NAME, SEED_FILE_NAME

# A fixed in-container port (the host maps it to an ephemeral port at run time, the way
# the PROCESS backing binds port 0).
CONTAINER_PORT = 8000
BASE_IMAGE = "python:3.13-slim"

# command_injection's base_command (sampling._COMMAND_INJECTION_BASE) → the apt package
# that puts that diagnostic tool in the image, so the real `sh -c` endpoint can run it.
# Each tool echoes the (flag-as-)hostname back in its resolver error, so a `$(cat flag)`
# substitution leaks too — confirmed empirically on python:3.13-slim for all five.
_CMDI_APT_PACKAGES: dict[str, str] = {
    "ping": "iputils-ping",
    "nslookup": "dnsutils",
    "dig": "dnsutils",
    "host": "dnsutils",
    "traceroute": "traceroute",
}


def required_apt_packages(graph: WorldGraph) -> set[str]:
    """The apt packages this world's container actually needs, based ONLY on its
    command_injection vulns and each one's base_command (union across vulns). A world
    with no command_injection returns an empty set — its image installs no OS tools."""
    packages: set[str] = set()
    for vuln in graph.by_kind("vulnerability"):
        if vuln.attrs.get("kind") != "command_injection":
            continue
        params = vuln.attrs.get("params")
        if not isinstance(params, Mapping):
            continue
        package = _CMDI_APT_PACKAGES.get(str(params.get("base_command")))
        if package is not None:
            packages.add(package)
    return packages


def hardening_run_args() -> list[str]:
    """``docker run`` flags that contain a world running attacker-controlled code:
    drop every Linux capability, forbid gaining privileges (setuid), and cap memory /
    CPU / pid count, so an exploit can't escalate, fork-bomb, or exhaust the host. The
    world stays reachable over its published HTTP port; blocking outbound egress is a
    separate concern.

    Not read-only-root: the app writes the materialized files + request log and unlinks
    the seed at startup, so a read-only rootfs would need those writes redirected to a
    writable mount first."""
    return [
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--memory",
        "512m",
        "--cpus",
        "1.0",
        "--pids-limit",
        "256",
    ]


def _dockerfile(apt_packages: set[str]) -> str:
    # OPENRANGE_REALFS flips the app's surfaces to the real container. jinja2 is the one
    # pip dep (the ssti handler imports it); OS tools are added only when a
    # command_injection vuln needs them, else the apt layer is skipped entirely.
    if apt_packages:
        names = " ".join(sorted(apt_packages))
        run = (
            "RUN apt-get update \\\n"
            f"&& apt-get install -y --no-install-recommends {names} \\\n"
            "&& rm -rf /var/lib/apt/lists/* \\\n"
            "&& pip install --no-cache-dir jinja2\n"
        )
    else:
        run = "RUN pip install --no-cache-dir jinja2\n"
    return (
        f"FROM {BASE_IMAGE}\n"
        "WORKDIR /app\n"
        "ENV OPENRANGE_REALFS=1\n"
        f"{run}"
        f"COPY {APP_FILE_NAME} {SEED_FILE_NAME} ./\n"
        f"EXPOSE {CONTAINER_PORT}\n"
        f'CMD ["python", "{APP_FILE_NAME}", "--host", "0.0.0.0", '
        f'"--port", "{CONTAINER_PORT}", "--log", "/app/requests.jsonl"]\n'
    )


def image_files(graph: WorldGraph) -> dict[str, str]:
    """The build context for the world's container: the Dockerfile + the rendered app
    + its seed. Same rendered app the PROCESS backing runs, but the container sets
    OPENRANGE_REALFS so its surfaces are real (real open() / traversal, real `sh -c`),
    not the in-memory emulation. The Dockerfile installs only the OS tools this world's
    own vulns run server-side (see :func:`required_apt_packages`)."""
    rendered = _realize_graph(graph)
    return {
        "Dockerfile": _dockerfile(required_apt_packages(graph)),
        APP_FILE_NAME: rendered[APP_FILE_NAME],
        SEED_FILE_NAME: rendered[SEED_FILE_NAME],
    }


@dataclass(frozen=True)
class ServiceImage:
    """One service's container build context (the networked CONTAINER backing builds
    one image per service node). ``name`` is the container/DNS name services reach each
    other by; ``exposure`` decides whether the host publishes it (public) or it stays
    reachable only on the container network (internal)."""

    service_id: str
    name: str
    exposure: str
    build_files: dict[str, str]


def realize_services(graph: WorldGraph) -> list[ServiceImage]:
    """Per-service build contexts: one image per service, each carrying only its own
    endpoints + its own state (so the flag stays in the internal service that owns it
    and never enters the public image). The networked runtime wires these on a container
    network and publishes only the public service."""
    dockerfile = _dockerfile(required_apt_packages(graph))
    images: list[ServiceImage] = []
    for service in graph.by_kind("service"):
        rendered = _realize_graph(graph, frozenset({service.id}))
        images.append(
            ServiceImage(
                service_id=service.id,
                name=str(service.attrs.get("name", service.id)),
                exposure=str(service.attrs.get("exposure", "internal")),
                build_files={
                    "Dockerfile": dockerfile,
                    APP_FILE_NAME: rendered[APP_FILE_NAME],
                    SEED_FILE_NAME: rendered[SEED_FILE_NAME],
                },
            )
        )
    return images
