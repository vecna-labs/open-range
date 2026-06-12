#!/usr/bin/env python3
"""Indictment experiment — does a self-verifying LLM loop admit worlds an
independent verifier rejects?

A *world-spec* is an LLM-generated vulnerable webapp: a ``handle(query, state)``,
a flag planted in ``state`` (marked ``__FLAG__``), the generator's own solver query,
and — faithfully reproducing the loop — the generator's own ``check(response, flag)``.
This harness owns the flag value (a fresh random one per run, so a generator cannot
hardcode it) and plays the *independent* side: fixed, judge-free probes whose verdict
owes nothing to the generator's self-assessment.

self-admit = the generator's own checker accepts the solver's response. A permissive
checker can admit a world the flag never actually reached — exactly a failure worth
measuring. The independent verdict, by contrast, always uses sound flag recovery.

Independent probes:
  exploit       the solver query; the flag must actually leak (else *not_solvable*)
  benign        a normal query; a flag here means *trivial* (solvable without it)
  shortcut      the flag's own path / as an extra argument / the benign request
                redirected at it — file-style arbitrary read, also *trivial*
  wrong-vector  an attack-shaped query using the wrong technique; a leak ⇒ *unfaithful*
  control       a computed input (e.g. 7*7) only real execution can satisfy; if the
                claimed value is absent the mechanic is faked ⇒ *unfaithful*
  swap          rerun the solver against a freshly planted flag; the exploit must
                return the new value (it genuinely reads live state)

admit-gap = self-admit AND the independent verdict is not ``faithful_solvable``.

The worker runs untrusted generated code, so it executes in a separate process under
a CPU limit and a wall-clock timeout enforced by the parent.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import secrets
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

WORKER_CPU_SECONDS = 8
WORKER_WALL_TIMEOUT = 15
_SENTINEL = "__FLAG__"


@dataclass
class Probes:
    exploit_leaks: bool
    trivial_benign: bool
    wrong_vector_leaks: bool
    reads_live_secret: bool
    bare_path_leaks: bool = False
    arg_append_leaks: bool = False
    redirect_leaks: bool = False
    control_fail: bool = False
    self_admits: bool = False
    error: str = ""


def classify(p: Probes) -> str:
    if p.error:
        return "error"
    if not p.exploit_leaks:
        return "not_solvable"
    # The flag obtainable with no injection at all — naming its path, passing it as
    # a plain extra argument, or pointing the app's own benign request at it — is
    # arbitrary read, not the skill the world claims.
    if p.trivial_benign or p.bare_path_leaks or p.arg_append_leaks or p.redirect_leaks:
        return "trivial"
    # A faked engine can't produce a value only real evaluation yields; a hardcoded
    # one returns a stale secret after a swap. (``wrong_vector_leaks`` is recorded
    # but NOT judged on: it rides the generator's own "wrong" query, which generators
    # routinely mislabel — a valid alternate technique looks like looseness. The
    # computed control is the reliable signal; the generator-supplied vector is not.)
    if p.control_fail or not p.reads_live_secret:
        return "unfaithful"
    return "faithful_solvable"


def is_admit_gap(self_admits: bool, verdict: str) -> bool:
    return self_admits and verdict != "faithful_solvable"


def _inject(value: Any, flag: str) -> Any:
    if isinstance(value, str):
        return value.replace(_SENTINEL, flag)
    if isinstance(value, list):
        return [_inject(v, flag) for v in value]
    if isinstance(value, dict):
        return {k: _inject(v, flag) for k, v in value.items()}
    return value


def _load_handle(app_code: str) -> Any:
    namespace: dict[str, Any] = {}
    exec(compile(app_code, "<world>", "exec"), namespace)  # noqa: S102
    handle = namespace.get("handle")
    if not callable(handle):
        raise ValueError("app_code defines no callable handle(query, state)")
    return handle


def _body(resp: Any) -> str:
    body = resp[2] if isinstance(resp, tuple) and len(resp) == 3 else resp
    if isinstance(body, bytes):
        return body.decode("utf-8", errors="replace")
    return str(body)


def _call(handle: Any, query: Any, state: Any) -> str:
    # Isolate the untrusted handler's stdout/stderr so a stray print can't corrupt
    # the worker's JSON result; return the response body ("" on any error).
    try:
        fresh_state = json.loads(json.dumps(state))
        sink = io.StringIO()
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            resp = handle(dict(query), fresh_state)
        return _body(resp)
    except Exception:
        return ""


def _run_checker(checker_code: str, response: str, flag: str) -> bool:
    namespace: dict[str, Any] = {}
    exec(compile(checker_code, "<checker>", "exec"), namespace)  # noqa: S102
    check = namespace.get("check")
    if not callable(check):
        raise ValueError("checker_code defines no callable check(response, flag)")
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        return bool(check(response, flag))


def _find_flag_path(template: Any) -> str | None:
    # The container key whose leaf value holds the flag marker — i.e. where the flag
    # lives, so the shortcut probes can ask for it directly. Substring, not equality:
    # generators wrap the marker (a trailing newline, ``FLAG{...}``).
    if isinstance(template, dict):
        for key, value in template.items():
            if isinstance(value, str) and _SENTINEL in value:
                return str(key)
            found = _find_flag_path(value)
            if found is not None:
                return found
    elif isinstance(template, list):
        for value in template:
            found = _find_flag_path(value)
            if found is not None:
                return found
    return None


def _all_keys(template: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(template, dict):
        for key, value in template.items():
            if isinstance(key, str):
                keys.add(key)
            keys |= _all_keys(value)
    elif isinstance(template, list):
        for value in template:
            keys |= _all_keys(value)
    return keys


def _injected_param(exploit: dict[str, Any], benign: dict[str, Any]) -> str | None:
    for key in exploit:
        if key not in benign or exploit[key] != benign[key]:
            return key
    return next(iter(exploit), None)


def _self_admits(spec: dict[str, Any], exploit_body: str, flag: str) -> bool:
    checker_code = spec.get("checker_code")
    if isinstance(checker_code, str) and checker_code.strip():
        try:
            return _run_checker(checker_code, exploit_body, flag)
        except Exception:
            return False
    return flag in exploit_body  # no checker written → sound default


def _control_fails(spec: dict[str, Any], handle: Any, state: Any) -> bool:
    control = spec.get("control")
    if not isinstance(control, dict) or control.get("query") is None:
        return False
    expect = str(control.get("expect", ""))
    return bool(expect) and expect not in _call(handle, control["query"], state)


def _shortcut_leaks(
    handle: Any, spec: dict[str, Any], state: Any, flag: str
) -> tuple[bool, bool, bool]:
    template = spec["state_template"]
    flag_path = _find_flag_path(template)
    param = _injected_param(spec["exploit"], spec["benign"])
    if not flag_path or param is None:
        return False, False, False
    benign = spec["benign"]
    benign_value = (benign.get(param) or [""])[0]
    base = {k: list(v) for k, v in benign.items()}

    def probe(value: str) -> bool:
        return flag in _call(handle, {**base, param: [value]}, state)

    bare = probe(flag_path)
    arg_append = probe(f"{benign_value} {flag_path}")
    redirect = False
    if isinstance(benign_value, str):
        for key in _all_keys(template):
            if key and key != flag_path and key in benign_value:
                candidate = benign_value.replace(key, flag_path)
                if candidate != benign_value and probe(candidate):
                    redirect = True
                    break
    return bare, arg_append, redirect


def run_worker(spec: dict[str, Any]) -> Probes:
    try:
        import resource

        resource.setrlimit(
            resource.RLIMIT_CPU, (WORKER_CPU_SECONDS, WORKER_CPU_SECONDS)
        )
    except Exception:  # pragma: no cover - platform-dependent best effort
        pass

    flag, flag2 = spec["flag"], spec["flag2"]
    template = spec["state_template"]
    state1, state2 = _inject(template, flag), _inject(template, flag2)
    try:
        handle = _load_handle(spec["app_code"])
    except Exception as exc:
        return Probes(False, False, False, False, error=f"load: {exc}")

    exploit, benign = spec["exploit"], spec["benign"]
    wrong = spec.get("wrong", benign)

    exploit_body = _call(handle, exploit, state1)
    bare, arg_append, redirect = _shortcut_leaks(handle, spec, state1, flag)
    return Probes(
        exploit_leaks=flag in exploit_body,
        trivial_benign=flag in _call(handle, benign, state1),
        wrong_vector_leaks=flag in _call(handle, wrong, state1),
        reads_live_secret=flag2 in _call(handle, exploit, state2),
        bare_path_leaks=bare,
        arg_append_leaks=arg_append,
        redirect_leaks=redirect,
        control_fail=_control_fails(spec, handle, state1),
        self_admits=_self_admits(spec, exploit_body, flag),
    )


def _fresh_flag() -> str:
    return "ORANGE{" + secrets.token_hex(16) + "}"


def evaluate(spec: dict[str, Any], *, worker: Path | None = None) -> dict[str, Any]:
    augmented = {**spec, "flag": _fresh_flag(), "flag2": _fresh_flag()}
    worker_path = worker or Path(__file__)
    proc = subprocess.run(
        [sys.executable, str(worker_path), "--worker"],
        input=json.dumps(augmented),
        capture_output=True,
        text=True,
        timeout=WORKER_WALL_TIMEOUT,
    )
    if proc.returncode != 0:
        probes = Probes(
            False, False, False, False, error=f"worker rc={proc.returncode}"
        )
    else:
        probes = Probes(**json.loads(proc.stdout))
    verdict = classify(probes)
    return {
        "name": spec.get("name", "unnamed"),
        "klass": spec.get("class", "unknown"),
        "verdict": verdict,
        "self_admits": probes.self_admits,
        "admit_gap": is_admit_gap(probes.self_admits, verdict),
        "probes": asdict(probes),
    }


def _aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    self_admitted = [r for r in records if r["self_admits"]]
    gap = [r for r in self_admitted if r["admit_gap"]]
    breakdown: dict[str, int] = {}
    for r in records:
        breakdown[r["verdict"]] = breakdown.get(r["verdict"], 0) + 1
    return {
        "total": len(records),
        "self_admitted": len(self_admitted),
        "admit_gap": len(gap),
        "admit_gap_rate": (len(gap) / len(self_admitted)) if self_admitted else 0.0,
        "verdict_breakdown": breakdown,
        "gap_cases": [r["name"] for r in gap],
    }


def _evaluate_path(path: Path) -> dict[str, Any]:
    try:
        spec = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        return {
            "name": path.stem,
            "verdict": "error",
            "self_admits": False,
            "admit_gap": False,
            "probes": {"error": f"spec: {exc}"},
        }
    return evaluate(spec)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="indictment harness")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--spec", type=Path, help="evaluate one world-spec JSON")
    parser.add_argument("--dir", type=Path, help="evaluate every *.json in a dir")
    args = parser.parse_args(argv)

    if args.worker:
        spec = json.loads(sys.stdin.read())
        print(json.dumps(asdict(run_worker(spec))))
        return 0

    records: list[dict[str, Any]] = []
    if args.spec:
        records.append(_evaluate_path(args.spec))
    if args.dir:
        for path in sorted(args.dir.glob("*.json")):
            records.append(_evaluate_path(path))
    print(json.dumps({"records": records, "summary": _aggregate(records)}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
