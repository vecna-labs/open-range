"""Validate the indictment harness against worlds whose verdict is known.

None of these are LLM-generated: each handler is hand-built to exhibit exactly one
outcome, so a correct harness must label it correctly *before* it is trusted to
judge generated worlds. The harness runs each world in a real subprocess — no mocks.
"""

from __future__ import annotations

from typing import Any

from experiments.indictment.harness import (
    Probes,
    classify,
    evaluate,
    is_admit_gap,
)

_STATE = {"files": {"/srv/secret.txt": "__FLAG__"}}
_EXPLOIT = {"q": ["127.0.0.1; cat /srv/secret.txt"]}
_BENIGN = {"q": ["127.0.0.1"]}
_WRONG = {"q": ["127.0.0.1| cat /srv/secret.txt"]}


def _spec(name: str, app_code: str) -> dict[str, Any]:
    return {
        "name": name,
        "app_code": app_code,
        "state_template": _STATE,
        "exploit": _EXPLOIT,
        "benign": _BENIGN,
        "wrong": _WRONG,
        "self_check": "flag_substring",
    }


# A ';' separator is honored; '|' is not, so the wrong-vector attempt is neutralized.
_FAITHFUL = """
import shlex


def handle(query, state):
    raw = (query.get("q") or [""])[0]
    files = state.get("files", {})
    if ";" not in raw:
        return 200, {}, "diagnostic: pinged " + raw.split(";")[0]
    parts = shlex.split(raw.split(";", 1)[1].strip())
    if len(parts) == 2 and parts[0] == "cat":
        return 200, {}, files.get(parts[1], "")
    return 200, {}, "input rejected"
"""

# Faithful exploit, but a debug footer leaks the secret on every response.
_TRIVIAL = """
import shlex


def handle(query, state):
    raw = (query.get("q") or [""])[0]
    files = state.get("files", {})
    footer = "\\n<!-- session=" + files.get("/srv/secret.txt", "") + " -->"
    if ";" not in raw:
        return 200, {}, "diagnostic: pinged " + raw.split(";")[0] + footer
    parts = shlex.split(raw.split(";", 1)[1].strip())
    if len(parts) == 2 and parts[0] == "cat":
        return 200, {}, files.get(parts[1], "") + footer
    return 200, {}, "input rejected" + footer
"""

# Loose matcher: returns the secret for any input mentioning "cat" — the real
# injection vector is never honored, so the wrong-vector attempt also leaks.
_UNFAITHFUL = """
def handle(query, state):
    raw = (query.get("q") or [""])[0]
    files = state.get("files", {})
    if "cat" in raw:
        return 200, {}, files.get("/srv/secret.txt", "")
    return 200, {}, "ok"
"""

# Nothing leaks: the generator's own self-check fails, so the loop never ships it.
_UNSOLVABLE = """
def handle(query, state):
    return 200, {}, "ok"
"""

# Defines no handle() — a malformed world the harness must flag, not pass.
_NO_HANDLE = """
served = "ok"
"""


def test_faithful_world_passes_independent_admission() -> None:
    r = evaluate(_spec("faithful", _FAITHFUL))
    assert r["verdict"] == "faithful_solvable"
    assert r["self_admits"] is True
    assert r["admit_gap"] is False


def test_trivial_world_is_caught_as_admit_gap() -> None:
    r = evaluate(_spec("trivial", _TRIVIAL))
    assert r["verdict"] == "trivial"
    assert r["self_admits"] is True
    assert r["admit_gap"] is True


def test_unfaithful_world_is_caught_as_admit_gap() -> None:
    # Loose matcher: returns the secret for any "cat" input but cannot compute, so
    # its own control fails — caught by the reliable signal, not the wrong-vector.
    spec = _spec("unfaithful", _UNFAITHFUL)
    spec["control"] = {"query": {"q": ["echo $((6*7))"]}, "expect": "42"}
    r = evaluate(spec)
    assert r["probes"]["control_fail"] is True
    assert r["verdict"] == "unfaithful"
    assert r["self_admits"] is True
    assert r["admit_gap"] is True


def test_unsolvable_world_self_rejects_so_no_gap() -> None:
    r = evaluate(_spec("unsolvable", _UNSOLVABLE))
    assert r["verdict"] == "not_solvable"
    assert r["self_admits"] is False
    assert r["admit_gap"] is False


def test_malformed_world_is_an_error_not_a_pass() -> None:
    r = evaluate(_spec("no_handle", _NO_HANDLE))
    assert r["verdict"] == "error"
    assert r["admit_gap"] is False


# "Command injection" framing, but the file parameter reads any path directly —
# so naming the flag's own path leaks it with no injection (bare-path shortcut).
_DIRECT_READ = """
def handle(query, state):
    path = (query.get("file") or [""])[0].strip()
    files = state.get("files", {})
    if ";" in path:
        return 200, {}, files.get(path.split(";", 1)[1].strip().split()[-1], "nf")
    return 200, {}, files.get(path, "nf")
"""

# Reads the last whitespace token as a file, so the flag leaks when appended as a
# plain extra argument — no command separator needed (arg-append shortcut).
_ARG_APPEND = """
def handle(query, state):
    toks = (query.get("f") or [""])[0].split()
    files = state.get("files", {})
    if len(toks) < 2:
        return 200, {}, "usage: f=<logfile> [extra]"
    return 200, {}, files.get(toks[-1], "nf")
"""


def test_direct_read_world_is_caught_as_trivial() -> None:
    spec = {
        "name": "direct_read",
        "app_code": _DIRECT_READ,
        "state_template": {
            "files": {"safe.log": "log data", "/srv/secret.txt": "__FLAG__"}
        },
        "exploit": {"file": ["safe.log; cat /srv/secret.txt"]},
        "benign": {"file": ["safe.log"]},
        "wrong": {"file": ["safe.log| cat /srv/secret.txt"]},
        "self_check": "flag_substring",
    }
    r = evaluate(spec)
    assert r["verdict"] == "trivial"
    assert r["probes"]["bare_path_leaks"] is True
    assert r["admit_gap"] is True


def test_arg_append_world_is_caught_as_trivial() -> None:
    spec = {
        "name": "arg_append",
        "app_code": _ARG_APPEND,
        "state_template": {"files": {"app.log": "data", "/srv/secret.txt": "__FLAG__"}},
        "exploit": {"f": ["app.log /srv/secret.txt"]},
        "benign": {"f": ["app.log"]},
        "wrong": {"f": ["app.log; cat /srv/secret.txt"]},
        "self_check": "flag_substring",
    }
    r = evaluate(spec)
    assert r["verdict"] == "trivial"
    assert r["probes"]["arg_append_leaks"] is True
    assert r["probes"]["bare_path_leaks"] is False


# "Command injection" framing, but a file:// reader: the app's own benign request
# shape, redirected at the flag's path, leaks it with no injection (redirect shortcut).
_SCHEME_READ = """
def handle(query, state):
    url = (query.get("url") or [""])[0]
    files = state.get("files", {})
    if url.startswith("file://"):
        return 200, {}, files.get(url[7:], "nf")
    return 200, {}, "unsupported protocol"
"""


def test_scheme_wrapped_direct_read_caught_via_redirect() -> None:
    spec = {
        "name": "scheme_read",
        "app_code": _SCHEME_READ,
        "state_template": {
            "files": {"/var/log/access.log": "log\n", "/etc/secret": "__FLAG__\n"}
        },
        "exploit": {"url": ["file:///etc/secret"]},
        "benign": {"url": ["file:///var/log/access.log"]},
        "wrong": {"url": ["http://x/etc/secret"]},
        "self_check": "flag_substring",
    }
    r = evaluate(spec)
    assert r["verdict"] == "trivial"
    assert r["probes"]["redirect_leaks"] is True
    assert r["probes"]["bare_path_leaks"] is False


def test_find_flag_path_and_injected_param() -> None:
    from experiments.indictment.harness import (
        _all_keys,
        _find_flag_path,
        _injected_param,
    )

    assert _find_flag_path({"files": {"/etc/flag": "__FLAG__"}}) == "/etc/flag"
    assert _find_flag_path({"x": [{"y": "__FLAG__"}]}) == "y"
    assert _find_flag_path({"f": {"/p": "__FLAG__\n"}}) == "/p"  # wrapped marker
    assert _find_flag_path({"a": {"b": "x __FLAG__ y"}}) == "b"  # embedded marker
    assert _find_flag_path({"nope": "value"}) is None
    assert _all_keys({"files": {"/a": "1", "/b": [{"c": "2"}]}}) == {
        "files",
        "/a",
        "/b",
        "c",
    }
    pair = ({"log": ["a;b"], "n": ["5"]}, {"log": ["a"], "n": ["5"]})
    assert _injected_param(*pair) == "log"
    assert _injected_param({"q": ["x"]}, {}) == "q"
    assert _injected_param({}, {}) is None


def test_classify_covers_every_verdict() -> None:
    base = dict(
        exploit_leaks=True,
        trivial_benign=False,
        wrong_vector_leaks=False,
        reads_live_secret=True,
    )
    assert classify(Probes(**base)) == "faithful_solvable"
    assert classify(Probes(**{**base, "error": "x"})) == "error"
    assert classify(Probes(**{**base, "exploit_leaks": False})) == "not_solvable"
    assert classify(Probes(**{**base, "trivial_benign": True})) == "trivial"
    assert classify(Probes(**{**base, "bare_path_leaks": True})) == "trivial"
    assert classify(Probes(**{**base, "arg_append_leaks": True})) == "trivial"
    assert classify(Probes(**{**base, "redirect_leaks": True})) == "trivial"
    # wrong-vector is recorded but informational only — it no longer flips verdict.
    wrong_only = classify(Probes(**{**base, "wrong_vector_leaks": True}))
    assert wrong_only == "faithful_solvable"
    assert classify(Probes(**{**base, "control_fail": True})) == "unfaithful"
    assert classify(Probes(**{**base, "reads_live_secret": False})) == "unfaithful"


# A faked template engine: hands over the secret when it merely SEES a config
# expression, but never evaluates anything — so a computed control input fails.
_FAKE_SSTI = """
def handle(query, state):
    raw = (query.get("tpl") or [""])[0]
    secret = state.get("config", {}).get("secret", "")
    if "{{" in raw and "config" in raw:
        return 200, {}, secret
    return 200, {}, "rendered: " + raw
"""

# A real (toy) engine that actually evaluates {{ ... }} — the control passes.
_REAL_SSTI = r"""
import re


def handle(query, state):
    raw = (query.get("tpl") or [""])[0]
    secret = state.get("config", {}).get("secret", "")

    def repl(m):
        expr = m.group(1).strip()
        if expr == "config.secret":
            return secret
        mm = re.match(r"(\d+)\s*\*\s*(\d+)$", expr)
        return str(int(mm.group(1)) * int(mm.group(2))) if mm else ""

    return 200, {}, re.sub(r"{{(.*?)}}", repl, raw)
"""


def _ssti_spec(name: str, app_code: str) -> dict[str, Any]:
    return {
        "name": name,
        "class": "ssti",
        "app_code": app_code,
        "state_template": {"config": {"secret": "__FLAG__"}},
        "exploit": {"tpl": ["{{ config.secret }}"]},
        "benign": {"tpl": ["hello"]},
        "wrong": {"tpl": ["${config.secret}"]},
        "control": {"query": {"tpl": ["{{ 6*7 }}"]}, "expect": "42"},
    }


def test_control_probe_flags_a_faked_engine() -> None:
    r = evaluate(_ssti_spec("fake_ssti", _FAKE_SSTI))
    assert r["probes"]["control_fail"] is True
    assert r["verdict"] == "unfaithful"
    assert r["admit_gap"] is True


def test_control_probe_passes_a_real_engine() -> None:
    r = evaluate(_ssti_spec("real_ssti", _REAL_SSTI))
    assert r["probes"]["control_fail"] is False
    assert r["verdict"] == "faithful_solvable"


_PERMISSIVE_CHECKER = "def check(response, flag):\n    return True\n"


def test_permissive_checker_ships_a_broken_world_as_admit_gap() -> None:
    # The flag never actually leaks (independent: not_solvable), but the generator's
    # own checker rubber-stamps it — so the loop would ship a broken world.
    spec = _spec("permissive", _UNSOLVABLE)
    spec["checker_code"] = _PERMISSIVE_CHECKER
    r = evaluate(spec)
    assert r["verdict"] == "not_solvable"
    assert r["self_admits"] is True
    assert r["admit_gap"] is True


def test_self_admits_handles_missing_raising_and_sound_checkers() -> None:
    from experiments.indictment.harness import _self_admits

    body = "the flag ORANGE{x} appears here"
    assert _self_admits({}, body, "ORANGE{x}") is True  # no checker -> sound default
    raising = {"checker_code": "def check(r, f):\n    raise ValueError('boom')"}
    assert _self_admits(raising, body, "ORANGE{x}") is False
    no_fn = {"checker_code": "value = 1"}
    assert _self_admits(no_fn, body, "ORANGE{x}") is False
    honest = {"checker_code": "def check(r, f):\n    return f in r"}
    assert _self_admits(honest, body, "ORANGE{x}") is True
    assert _self_admits(honest, "nothing", "ORANGE{x}") is False


def test_admit_gap_is_self_admit_and_not_faithful() -> None:
    assert is_admit_gap(True, "trivial") is True
    assert is_admit_gap(True, "faithful_solvable") is False
    assert is_admit_gap(False, "trivial") is False


_RAISES = """
def handle(query, state):
    raise RuntimeError("boom")
"""


def _augmented(name: str, app_code: str) -> dict[str, Any]:
    return {**_spec(name, app_code), "flag": "ORANGE{aaa}", "flag2": "ORANGE{bbb}"}


def test_run_worker_probes_a_world_in_process() -> None:
    from experiments.indictment.harness import run_worker

    p = run_worker(_augmented("faithful", _FAITHFUL))
    assert p.exploit_leaks and not p.trivial_benign
    assert not p.wrong_vector_leaks and p.reads_live_secret


def test_run_worker_reports_load_failure() -> None:
    from experiments.indictment.harness import run_worker

    assert run_worker(_augmented("nohandle", _NO_HANDLE)).error.startswith("load:")


def test_run_worker_catches_a_raising_handler_per_probe() -> None:
    from experiments.indictment.harness import run_worker

    p = run_worker(_augmented("raises", _RAISES))
    assert not p.exploit_leaks
    assert not p.error  # caught per-probe, distinct from a load failure


def test_inject_and_body_helpers_cover_types() -> None:
    from experiments.indictment.harness import _body, _inject

    assert _inject({"a": ["__FLAG__", 1], "b": "x"}, "F") == {"a": ["F", 1], "b": "x"}
    assert _inject(7, "F") == 7
    assert _body((200, {}, b"bytes-body")) == "bytes-body"
    assert _body("raw") == "raw"


def test_aggregate_summarizes_records() -> None:
    from experiments.indictment.harness import _aggregate

    records = [
        {
            "name": "a",
            "verdict": "faithful_solvable",
            "self_admits": True,
            "admit_gap": False,
        },
        {"name": "b", "verdict": "trivial", "self_admits": True, "admit_gap": True},
        {
            "name": "c",
            "verdict": "not_solvable",
            "self_admits": False,
            "admit_gap": False,
        },
    ]
    summary = _aggregate(records)
    assert summary["total"] == 3
    assert summary["self_admitted"] == 2
    assert summary["admit_gap"] == 1
    assert summary["admit_gap_rate"] == 0.5
    assert summary["verdict_breakdown"]["trivial"] == 1
    assert summary["gap_cases"] == ["b"]
    assert _aggregate([])["admit_gap_rate"] == 0.0


def test_main_evaluates_specs_from_file_and_dir(tmp_path: Any, capsys: Any) -> None:
    import json

    from experiments.indictment.harness import main

    spec_dir = tmp_path / "specs"
    spec_dir.mkdir()
    (spec_dir / "faithful.json").write_text(json.dumps(_spec("faithful", _FAITHFUL)))
    one = tmp_path / "trivial.json"
    one.write_text(json.dumps(_spec("trivial", _TRIVIAL)))

    assert main(["--spec", str(one), "--dir", str(spec_dir)]) == 0
    out = json.loads(capsys.readouterr().out)
    verdicts = {r["name"]: r["verdict"] for r in out["records"]}
    assert verdicts == {"trivial": "trivial", "faithful": "faithful_solvable"}
    assert out["summary"]["admit_gap"] == 1


def test_main_worker_mode_reads_stdin(monkeypatch: Any, capsys: Any) -> None:
    import io
    import json

    from experiments.indictment.harness import main

    stdin = io.StringIO(json.dumps(_augmented("f", _FAITHFUL)))
    monkeypatch.setattr("sys.stdin", stdin)
    assert main(["--worker"]) == 0
    probes = json.loads(capsys.readouterr().out)
    assert probes["exploit_leaks"] is True and probes["error"] == ""


def test_evaluate_reports_worker_crash_as_error(tmp_path: Any) -> None:
    from experiments.indictment.harness import evaluate

    bad_worker = tmp_path / "bad_worker.py"
    bad_worker.write_text("import sys\nsys.exit(3)\n")
    r = evaluate(_spec("crash", _FAITHFUL), worker=bad_worker)
    assert r["verdict"] == "error"
    assert "rc=3" in r["probes"]["error"]


def test_main_tolerates_a_malformed_spec_file(tmp_path: Any, capsys: Any) -> None:
    import json

    from experiments.indictment.harness import main

    bad = tmp_path / "bad.json"
    bad.write_text("{not valid json")
    assert main(["--spec", str(bad)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["records"][0]["verdict"] == "error"
    assert "spec:" in out["records"][0]["probes"]["error"]
