"""Tests for the submission sandbox.

Exercise the generic primitive directly — both the single-call shape (cyber's
grader) and the replay-loop shape (a trading-style backtest driver that calls
the submission once per step inside one subprocess). Drivers are real
module-level functions, loaded and run in the child like a pack's would be.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import pytest
from openrange_pack_sdk import SandboxResult, run_submission


def _out(entry: Callable[..., Any], case: Mapping[str, Any]) -> dict[str, object]:
    return {"out": entry(case["n"])}


def _replay(entry: Callable[..., Any], case: Mapping[str, Any]) -> dict[str, object]:
    acc = 0
    trail: list[int] = []
    for x in case["series"]:
        acc = entry(acc, x)
        trail.append(acc)
    return {"final": acc, "trail": trail}


def _call(entry: Callable[..., Any], case: Mapping[str, Any]) -> dict[str, object]:
    return {"v": entry()}


def _empty(entry: Callable[..., Any], case: Mapping[str, Any]) -> dict[str, object]:
    return {}


def _bare(entry: Callable[..., Any], case: Mapping[str, Any]) -> Any:
    return entry()


def test_single_call_returns_driver_payload() -> None:
    run = run_submission(
        "def compute(n):\n    return n + 1\n",
        entry="compute",
        driver=_out,
        stdin_obj={"n": 41},
    )
    assert isinstance(run, SandboxResult)
    assert run.ok, run.error
    assert run.result == {"out": 42}


def test_loop_driver_replays_submission_in_one_subprocess() -> None:
    run = run_submission(
        "def step(prev, x):\n    return prev + x\n",
        entry="step",
        driver=_replay,
        stdin_obj={"series": [1, 2, 3, 4]},
    )
    assert run.ok, run.error
    assert run.result == {"final": 10, "trail": [1, 3, 6, 10]}


def test_missing_entry_callable() -> None:
    run = run_submission("x = 1\n", entry="handle", driver=_empty, stdin_obj={})
    assert not run.ok
    assert "no callable" in run.error


def test_source_load_error() -> None:
    run = run_submission("def f(: pass", entry="f", driver=_empty, stdin_obj={})
    assert not run.ok
    assert "source did not load" in run.error


def test_submission_exception_is_caught() -> None:
    run = run_submission(
        "def f():\n    raise RuntimeError('boom')\n",
        entry="f",
        driver=_call,
        stdin_obj={},
    )
    assert not run.ok
    assert "submission failed" in run.error
    assert "RuntimeError" in run.error


def test_non_dict_payload_rejected() -> None:
    run = run_submission(
        "def f():\n    return 1\n",
        entry="f",
        driver=_bare,
        stdin_obj={},
    )
    assert not run.ok
    assert "expected object" in run.error


def test_forged_stdout_result_does_not_leak() -> None:
    # A submission that writes a passing-looking JSON to stdout and hard-exits
    # must NOT be read as a result — the real result lives in a private file.
    forge = (
        "import os, sys\n"
        'sys.__stdout__.write(\'{"ok": true, "result": {"forged": true}}\')\n'
        "sys.__stdout__.flush()\n"
        "os._exit(0)\n"
        "def f():\n    return 1\n"
    )
    run = run_submission(forge, entry="f", driver=_call, stdin_obj={})
    assert not run.ok
    assert "without a result" in run.error


def test_non_identifier_entry_rejected() -> None:
    with pytest.raises(ValueError, match="identifier"):
        run_submission("x = 1", entry="not an id", driver=_empty, stdin_obj={})


def test_infinite_loop_times_out() -> None:
    run = run_submission(
        "def f():\n    while True:\n        pass\n",
        entry="f",
        driver=_call,
        stdin_obj={},
        timeout=0.5,
    )
    assert not run.ok
    assert "timed out" in run.error
