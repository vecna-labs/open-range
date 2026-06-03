"""Test-runner sandbox — run a repo's own pytest suite over a candidate tree.

This is the behavioral grader: materialize a candidate working tree plus the
held-out test files into a throwaway temp dir and run the repo's own pytest suite,
recording per-nodeid pass/fail. It runs the whole requested id set in **one**
pytest invocation (SWE-bench's harness shape — shared conftest, session fixtures,
and import state) and parses the JUnit XML for per-test outcomes, rather than one
subprocess per id.

Importability, not installation. The candidate tree's own packages are made
importable by ``swe.sandbox`` prepending the repo root (and ``root/src``) to
``PYTHONPATH``, so both flat and ``src/`` layouts resolve without a build step.
The grader deliberately does *not* ``pip install`` the repo or provision
third-party dependencies — repos whose tests need external packages or a compiled
build are the per-instance container-image milestone (the model SWE-bench uses),
documented in ``DESIGN.md``.

Trust model — read before deploying. Running an agent's patched code plus arbitrary
repo tests *is* arbitrary code execution. Isolation is delegated to
``swe.sandbox.run_sandboxed``: a wall-clock timeout always, plus OS-level
filesystem / network / pid isolation where the host supports it (Linux bwrap). On
a host without those, grading runs in a bare subprocess and is safe only for
*trusted* submissions on a disposable machine. See ``swe/sandbox.py`` and the SWE
pack design doc.
"""

from __future__ import annotations

import tempfile
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from openrange_pack_sdk import write_tree

from swe.sandbox import SandboxResult, run_sandboxed

_DEFAULT_TIMEOUT = 30.0
_REPORT = "openrange_report.xml"


@dataclass(frozen=True, slots=True)
class TestReport:
    """Per-test-id pass/fail from one suite run. ``True`` == the test passed."""

    results: Mapping[str, bool]

    def all_pass(self, ids: Sequence[str]) -> bool:
        return bool(ids) and all(self.results.get(i, False) for i in ids)

    def all_fail(self, ids: Sequence[str]) -> bool:
        return bool(ids) and all(self.results.get(i, False) is False for i in ids)

    @property
    def passed(self) -> int:
        return sum(1 for ok in self.results.values() if ok)


def run_tests(
    tree: Mapping[str, str],
    test_files: Mapping[str, str],
    test_ids: Sequence[str],
    *,
    timeout: float = _DEFAULT_TIMEOUT,
) -> TestReport:
    """Materialize ``tree`` + ``test_files`` in a temp dir, run the suite once.

    ``tree`` is the candidate working tree (``{path: contents}``); ``test_files``
    are overlaid on top (held-out, so they always win over an agent that tried to
    edit them). Each id is a pytest nodeid (``"path::test"`` /
    ``"path::Class::test"``). One pytest invocation runs the whole id set so a
    shared ``conftest`` and session fixtures behave as the repo intends.
    """
    if not test_ids:
        return TestReport(results={})
    files = {**dict(tree), **dict(test_files)}
    with _temp_tree(files) as root:
        report_path = root / _REPORT
        proc = run_sandboxed(
            [
                "-m",
                "pytest",
                *test_ids,
                "-q",
                "-p",
                "no:cacheprovider",
                "--tb=no",
                f"--junit-xml={report_path}",
                "-o",
                # xunit1 keeps the per-case ``file`` attribute that _nodeid needs
                # to reconstruct the path; xunit2 drops it.
                "junit_family=xunit1",
            ],
            root=root,
            timeout=timeout,
        )
        results = _parse_report(report_path, test_ids, proc)
    return TestReport(results=results)


def _parse_report(
    report_path: Path,
    test_ids: Sequence[str],
    proc: SandboxResult,
) -> dict[str, bool]:
    """Map each requested nodeid to pass/fail from the JUnit XML.

    A test passes iff it ran with no failure / error / skip. Ids missing from the
    report (collection error, timeout, crash) are failures by construction.
    """
    outcomes: dict[str, bool] = dict.fromkeys(test_ids, False)
    if proc.timed_out or not report_path.exists():
        return outcomes
    try:
        tree = ET.parse(report_path)
    except ET.ParseError:
        return outcomes
    by_id: dict[str, bool] = {}
    for case in tree.iter("testcase"):
        nodeid = _nodeid(case)
        if nodeid is None:
            continue
        bad = any(case.find(tag) is not None for tag in ("failure", "error", "skipped"))
        by_id[nodeid] = not bad
    for tid in test_ids:
        if tid in by_id:
            outcomes[tid] = by_id[tid]
    return outcomes


def _nodeid(case: ET.Element) -> str | None:
    """Reconstruct a pytest nodeid (``file::[Class::]name``) from a JUnit
    ``<testcase>``.

    JUnit gives ``file`` (the path — authoritative for the location), ``name``
    (the test, with any ``[param]``), and ``classname`` (dotted module, plus any
    enclosing test class). The path comes straight from ``file``; the class part
    is whatever trails the module in ``classname``. We anchor on the file's stem
    rather than assuming ``classname`` carries the full directory prefix, because
    that prefix is present under some pytest import modes and absent under others.
    """
    file = case.get("file")
    name = case.get("name")
    if not file or not name:
        return None
    file = file.replace("\\", "/")
    classname = (case.get("classname") or "").replace("\\", "/").replace("/", ".")
    stem = file.rsplit("/", 1)[-1].removesuffix(".py")
    parts = classname.split(".") if classname else []
    anchors = [i for i, p in enumerate(parts) if p == stem]
    class_parts = parts[anchors[-1] + 1 :] if anchors else []
    if class_parts:
        return f"{file}::{'::'.join(class_parts)}::{name}"
    return f"{file}::{name}"


def _temp_tree(files: Mapping[str, str]):  # type: ignore[no-untyped-def]
    class _Tree:
        def __enter__(self) -> Path:
            self._tmp = tempfile.TemporaryDirectory(prefix="openrange-swe-")
            root = Path(self._tmp.name)
            write_tree(root, dict(files))
            return root

        def __exit__(self, *exc: object) -> None:
            self._tmp.cleanup()

    return _Tree()
