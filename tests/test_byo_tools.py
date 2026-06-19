"""The gym is bring-your-own-tools: a world is exposed only over HTTP, and the
example ``http_get``/``submit`` helpers are conveniences, not part of OpenRange.

These pin that contract so it can't regress: an episode is solvable and graded
with nothing but a plain HTTP client and the answer file, and the gym core names
no example tool.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from pathlib import Path

from cyber_webapp import WebappPack
from cyber_webapp.reference_solver import exploit_and_benign
from openrange_pack_sdk import Snapshot

import openrange
from openrange.core.admit import admit
from openrange.core.episode import EpisodeService

_FLAT_MANIFEST = {
    "pack": {"id": "webapp"},
    "runtime": {"tick": {"mode": "off"}},
    "npc": [],
    "seed": 3,
}


def _http(url: str) -> str:
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            raw: bytes = resp.read()
    except urllib.error.HTTPError as exc:
        raw = exc.read()
    return raw.decode("utf-8", "replace")


def _flag_from(body: str) -> str:
    try:
        obj = json.loads(body)
    except json.JSONDecodeError:
        obj = None
    if isinstance(obj, dict):
        for key in ("credential", "data", "flag", "secret"):
            value = obj.get(key)
            if isinstance(value, str):
                return value
    found = re.search(r"(?:ghp_|sk_live_|AKIA)[A-Za-z0-9_]+|[0-9a-f-]{32,36}", body)
    return found.group(0) if found else ""


def test_episode_is_solvable_with_a_plain_http_client(tmp_path: Path) -> None:
    pack = WebappPack()
    snap = admit(pack, manifest=_FLAT_MANIFEST, max_repairs=3)
    assert isinstance(snap, Snapshot), snap
    task = next(t for t in snap.tasks if t.meta.get("family") == "webapp.pentest")
    kind = next(v.attrs["kind"] for v in snap.graph.by_kind("vulnerability"))
    exploit, benign = exploit_and_benign(snap.graph, kind)

    svc = EpisodeService(pack, tmp_path)
    episode = svc.start_episode(snap, task.id)
    try:
        base = svc.base_url(episode)
        leaked = _http(base + exploit)
        control = _http(base + benign)
        flag = _flag_from(leaked)
        assert flag, f"exploit over plain HTTP leaked nothing: {leaked[:200]!r}"
        assert flag not in control
        (svc.solver_root(episode) / "result.json").write_text(
            json.dumps({"flag": flag}), encoding="utf-8"
        )
        report = svc.stop_episode(episode)
        assert report.passed
    finally:
        svc.close()


def test_gym_core_names_no_example_tool() -> None:
    core = Path(openrange.__file__).parent
    forbidden = (
        "http_get",
        "WEB_TOOLS",
        "FILE_TOOLS",
        "examples.tools",
        "from examples",
    )
    offenders = [
        f"{py.relative_to(core)}: {tok}"
        for py in core.rglob("*.py")
        for tok in forbidden
        if tok in py.read_text(encoding="utf-8")
    ]
    assert not offenders, f"the gym core must stay tool-agnostic, found: {offenders}"
