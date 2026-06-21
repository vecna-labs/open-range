"""The POST/body request channel (#258).

Body-shaped classes (an XML document, a login form, an injected command) are
delivered as a POST body, not a query string, so an agent learns the real request
shape. A body-shaped endpoint is generated as POST and rejects GET; the runtime
accepts both form-encoded and JSON bodies; the SSRF pivot stays a GET URL param.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
from cyber_webapp import WebappPack
from cyber_webapp.reference_solver import Request, exploit_and_benign
from cyber_webapp.verify import perform
from cyber_webapp.vulnerabilities import BODY_SHAPED_KINDS
from openrange_pack_sdk import Backing, Snapshot
from openrange_trl import EpisodeEnv

from openrange.core.admit import admit
from openrange.core.episode import EpisodeService


def shell(surface: Mapping[str, Any], command: str) -> str:
    """Run a shell command on your machine and return its output.

    Args:
        command: The shell command line to run (e.g. a curl invocation).
    """
    return str(surface["run"](command).output)


def submit(surface: Mapping[str, Any], content: str) -> str:
    """Submit your final answer; the grader reads result.json.

    Args:
        content: A JSON object carrying the recovered field, e.g. {"flag": "..."}.
    """
    (Path(str(surface["solver_root"])) / "result.json").write_text(
        content, encoding="utf-8"
    )
    return f"submitted {len(content)} byte(s)"


_CLASS_CASES = [
    ("file", "path_traversal"),
    ("file", "command_injection"),
    ("file", "xxe"),
    ("file", "ssti"),
    ("db", "sql_injection"),
    ("db", "ssrf"),
    ("db", "broken_authz"),
    ("db", "idor"),
    ("db", "weak_credentials"),
]


def _admit(loot: str, kind: str) -> Snapshot:
    snap = admit(
        WebappPack(),
        manifest={
            "pack": {"id": "webapp"},
            "runtime": {"tick": {"mode": "off"}},
            "npc": [],
            "seed": 7,
            "loot": {loot: 1, "db" if loot == "file" else "file": 0},
            "vuln": {"pin": [{"kind": kind}]},
        },
        max_repairs=3,
    )
    assert isinstance(snap, Snapshot), snap
    return snap


def _oracle_endpoint_method(snap: Snapshot, kind: str) -> str:
    vuln = next(
        v for v in snap.graph.by_kind("vulnerability") if v.attrs.get("kind") == kind
    )
    endpoint = snap.graph.nodes[
        next(e.dst for e in snap.graph.out_edges(vuln.id, "affects"))
    ]
    return str(endpoint.attrs.get("method", "GET"))


def _flag(snap: Snapshot) -> str:
    return str(snap.graph.nodes["secret_flag"].attrs["value_ref"])


def _base(snap: Snapshot, svc: EpisodeService) -> str:
    task = next(t for t in snap.tasks if t.meta.get("family") == "webapp.pentest")
    return str(svc.surface(svc.start_episode(snap, task.id))["base_url"])


@pytest.mark.parametrize(("loot", "kind"), _CLASS_CASES)
def test_each_class_is_generated_in_its_request_shape(loot: str, kind: str) -> None:
    expected = "POST" if kind in BODY_SHAPED_KINDS else "GET"
    assert _oracle_endpoint_method(_admit(loot, kind), kind) == expected


def test_a_post_endpoint_rejects_a_get(tmp_path: Path) -> None:
    snap = _admit("db", "sql_injection")
    flag = _flag(snap)
    exploit, _benign = exploit_and_benign(snap.graph, "sql_injection")
    assert exploit.method == "POST"
    svc = EpisodeService(WebappPack(), tmp_path)
    try:
        base = _base(snap, svc)
        try:
            urllib.request.urlopen(base + exploit.path, timeout=10)
            get_status = 200
        except urllib.error.HTTPError as exc:
            get_status = exc.code
        leaked = perform(base, exploit)
    finally:
        svc.close()
    assert get_status == 405  # a query-string delivery is refused outright
    assert flag in leaked  # the body delivery is the one that works


def test_runtime_accepts_both_form_and_json_bodies(tmp_path: Path) -> None:
    snap = _admit("db", "sql_injection")
    flag = _flag(snap)
    form_exploit, _benign = exploit_and_benign(snap.graph, "sql_injection")
    fields = {k: v[0] for k, v in urllib.parse.parse_qs(str(form_exploit.body)).items()}
    json_exploit = Request(
        form_exploit.path, "POST", json.dumps(fields), "application/json"
    )
    svc = EpisodeService(WebappPack(), tmp_path)
    try:
        base = _base(snap, svc)
        via_form = perform(base, form_exploit)
        via_json = perform(base, json_exploit)
    finally:
        svc.close()
    assert flag in via_form  # application/x-www-form-urlencoded
    assert flag in via_json  # application/json — same handler, same inputs


def test_the_ssrf_pivot_stays_a_get_url_param() -> None:
    snap = admit(
        WebappPack(),
        manifest={
            "pack": {"id": "webapp"},
            "runtime": {"tick": {"mode": "off"}},
            "npc": [],
            "seed": 3,
            "topology": "company",
        },
        max_repairs=3,
    )
    assert isinstance(snap, Snapshot), snap
    assert _oracle_endpoint_method(snap, "ssrf") == "GET"


def _docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        probe = subprocess.run(
            ["docker", "info"], capture_output=True, timeout=10, check=False
        )
    except Exception:  # noqa: BLE001 - a best-effort probe; any failure means "no"
        return False
    return probe.returncode == 0


gated = pytest.mark.skipif(
    not _docker_available(), reason="docker engine not reachable"
)


def _pentest_task_id(snap: Snapshot) -> str:
    task = next(t for t in snap.tasks if t.meta.get("family") == "webapp.pentest")
    return task.id


@gated
def test_http_post_example_tool_reaches_a_post_endpoint(tmp_path: Path) -> None:
    # The agent, in its own sandbox, delivers a body-shaped exploit with its own curl: a
    # POST body leaks the flag, and the same endpoint refuses a GET (a query string).
    snap = _admit("file", "command_injection")
    flag = _flag(snap)
    exploit, _benign = exploit_and_benign(snap.graph, "command_injection")
    assert exploit.method == "POST"
    service = EpisodeService(WebappPack(), tmp_path / "svc", backing=Backing.CONTAINER)
    env = EpisodeEnv(
        service=service,
        snapshots={snap.snapshot_id: snap},
        tools=[shell, submit],
        sandbox=True,
    )
    try:
        env.reset(snapshot_id=snap.snapshot_id, task_id=_pentest_task_id(snap))
        target = f"http://target:8000{exploit.path}"
        leaked = env.shell(
            f"curl -s -X POST -H 'Content-Type: {exploit.content_type}' "
            f"--data '{exploit.body}' '{target}'"
        )
        get_status = env.shell(f"curl -s -o /dev/null -w '%{{http_code}}' '{target}'")
    finally:
        service.close()
    assert flag in leaked  # the body delivery is the one that leaks
    assert "405" in get_status  # a query-string GET to the same endpoint is refused
