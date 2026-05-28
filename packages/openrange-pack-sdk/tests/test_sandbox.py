"""Tests for the submission sandbox.

Exercise the generic primitive directly — both the single-call shape (cyber's
grader) and the replay-loop shape (a trading-style backtest driver that calls
the submission once per step inside one subprocess).
"""

from __future__ import annotations

import pytest
from openrange_pack_sdk import SandboxResult, run_submission


def test_single_call_returns_driver_payload() -> None:
    run = run_submission(
        "def compute(n):\n    return n + 1\n",
        entry="compute",
        driver='payload = {"out": entry(case["n"])}',
        stdin_obj={"n": 41},
    )
    assert isinstance(run, SandboxResult)
    assert run.ok, run.error
    assert run.result == {"out": 42}


def test_loop_driver_replays_submission_in_one_subprocess() -> None:
    # The trading pattern: the trusted driver drives a stateful submission over
    # a series, calling it once per step — all inside a single sandboxed run.
    driver = (
        "acc = 0\n"
        "trail = []\n"
        "for x in case['series']:\n"
        "    acc = entry(acc, x)\n"
        "    trail.append(acc)\n"
        "payload = {'final': acc, 'trail': trail}\n"
    )
    run = run_submission(
        "def step(prev, x):\n    return prev + x\n",
        entry="step",
        driver=driver,
        stdin_obj={"series": [1, 2, 3, 4]},
    )
    assert run.ok, run.error
    assert run.result == {"final": 10, "trail": [1, 3, 6, 10]}


def test_missing_entry_callable() -> None:
    run = run_submission("x = 1\n", entry="handle", driver="payload = {}", stdin_obj={})
    assert not run.ok
    assert "no callable" in run.error


def test_source_load_error() -> None:
    run = run_submission("def f(: pass", entry="f", driver="payload = {}", stdin_obj={})
    assert not run.ok
    assert "source did not load" in run.error


def test_submission_exception_is_caught() -> None:
    run = run_submission(
        "def f():\n    raise RuntimeError('boom')\n",
        entry="f",
        driver="payload = {'v': entry()}",
        stdin_obj={},
    )
    assert not run.ok
    assert "submission failed" in run.error
    assert "RuntimeError" in run.error


def test_non_dict_payload_rejected() -> None:
    run = run_submission(
        "def f():\n    return 1\n",
        entry="f",
        driver="payload = entry()",
        stdin_obj={},
    )
    assert not run.ok
    assert "expected object" in run.error


def test_non_identifier_entry_rejected() -> None:
    with pytest.raises(ValueError, match="identifier"):
        run_submission("x = 1", entry="not an id", driver="payload = {}", stdin_obj={})


def test_infinite_loop_times_out() -> None:
    run = run_submission(
        "def f():\n    while True:\n        pass\n",
        entry="f",
        driver="payload = {'v': entry()}",
        stdin_obj={},
        timeout=0.5,
    )
    assert not run.ok
    assert "timed out" in run.error
