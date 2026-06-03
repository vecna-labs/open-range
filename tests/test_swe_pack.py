"""SWE pack: the instance recipe is embedded graph-natively (repo + held-out
suite + hidden solution), structural invariants catch malformed instances, the
world admits only when its own tests prove it well-posed (gold greens the suite,
base fails FAIL_TO_PASS), and the realizer materializes the buggy tree while
keeping the tests and gold fix off disk.

Tests are hermetic: the fixture ships a self-contained micro-repo, and grading
shells out to the same interpreter's pytest — no network, no clone.
"""

from __future__ import annotations

import json
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import replace
from pathlib import Path

import pytest
from graphschema import Visibility, WorldGraph, validate
from openrange_pack_sdk import Backing, BuildResult, TaskSpec, write_tree
from swe import SweFix, SwePack, repo_ontology
from swe.builder import SweBuilder
from swe.grading import _nodeid
from swe.instances import SweInstance, load_instance, to_graph
from swe.realize import SweRuntime
from swe.sandbox import _bwrap_wrap, run_sandboxed
from swe.swebench import instance_from_row

_INSTANCE = "calc_sum"
_MULTI = "shapes_area"
_BAD_GOLD = "def add(a, b):\n    return 0\n\n\ndef subtract(a, b):\n    return a - b\n"


def _graph_and_task() -> tuple[WorldGraph, TaskSpec]:
    instance = load_instance(_INSTANCE)
    graph = to_graph(instance)
    task = SweFix().generate(graph, {}, None)[0]
    return graph, task


def _error_codes(graph: WorldGraph) -> set[str]:
    return {i.code for i in validate(graph, repo_ontology(), SwePack().invariants())}


class TestBuilderGraph:
    def test_recipe_is_graph_native(self) -> None:
        build: BuildResult = SweBuilder().build({"instance": _INSTANCE})
        kinds = {n.kind for n in build.graph.nodes.values()}
        assert kinds == {"repo", "test_suite", "solution"}
        assert len(build.tasks) == 1

    def test_solution_is_hidden(self) -> None:
        graph = to_graph(load_instance(_INSTANCE))
        solution = graph.by_kind("solution")[0]
        assert solution.visibility is Visibility.HIDDEN

    def test_content_hash_is_deterministic(self) -> None:
        a = to_graph(load_instance(_INSTANCE))
        b = to_graph(load_instance(_INSTANCE))
        assert a.content_hash() == b.content_hash()

    def test_task_wired_to_repo_and_suite(self) -> None:
        graph, task = _graph_and_task()
        repo = graph.by_kind("repo")[0]
        suite = graph.by_kind("test_suite")[0]
        assert task.entrypoints == (repo.id,)
        assert task.goal_nodes == (suite.id,)


class TestInvariants:
    def test_clean_world_passes_all(self) -> None:
        graph = to_graph(load_instance(_INSTANCE))
        issues = validate(graph, repo_ontology(), SwePack().invariants())
        assert [i for i in issues if i.severity == "error"] == []

    def test_dangling_test_id_is_flagged(self) -> None:
        graph = to_graph(load_instance(_INSTANCE))
        suite = graph.by_kind("test_suite")[0]
        suite.attrs["fail_to_pass"] = ["nonexistent.py::test_x"]
        assert "suite_test_id_dangling" in _error_codes(graph)

    def test_f2p_p2p_overlap_is_flagged(self) -> None:
        graph = to_graph(load_instance(_INSTANCE))
        suite = graph.by_kind("test_suite")[0]
        suite.attrs["pass_to_pass"] = list(suite.attrs["fail_to_pass"])
        assert "suite_f2p_p2p_overlap" in _error_codes(graph)


class TestAdmission:
    def test_world_admits_through_all_layers(self) -> None:
        from openrange.core.admit import AdmissionFailure, admit

        result = admit(SwePack(), manifest={"instance": _INSTANCE}, max_repairs=0)
        assert not isinstance(result, AdmissionFailure)
        assert result.ontology_id == "swe.repo@v1"
        assert [t.id for t in result.tasks] == ["swe.fix.calc"]


class TestFeasibility:
    def test_default_world_is_feasible(self) -> None:
        graph, task = _graph_and_task()
        verdict = SweFix().check_feasibility(graph, task)
        assert verdict.feasible, verdict.reason

    def test_world_with_no_real_bug_is_infeasible(self) -> None:
        # gold == base: the FAIL_TO_PASS test already passes on base, so the
        # base-must-fail half of the self-test rejects the world.
        instance = load_instance(_INSTANCE)
        no_bug = replace(
            instance, base_files={**instance.base_files, **instance.gold_files}
        )
        graph = to_graph(no_bug)
        task = SweFix().generate(graph, {}, None)[0]
        verdict = SweFix().check_feasibility(graph, task)
        assert not verdict.feasible
        assert "base state does not fail" in verdict.reason

    def test_wrong_gold_is_infeasible(self) -> None:
        instance = load_instance(_INSTANCE)
        bad_gold = replace(instance, gold_files={"calc/core.py": _BAD_GOLD})
        graph = to_graph(bad_gold)
        task = SweFix().generate(graph, {}, None)[0]
        verdict = SweFix().check_feasibility(graph, task)
        assert not verdict.feasible
        assert "gold fix does not green" in verdict.reason


class TestGrading:
    def test_gold_tree_resolves(self) -> None:
        instance = load_instance(_INSTANCE)
        graph, task = _graph_and_task()
        gold_tree = {**instance.base_files, **instance.gold_files}
        result = SweFix().check_success(
            graph, task, {"workspace_files": gold_tree, "result": {"done": True}}
        )
        assert result.success
        assert all(result.subgoals.values())

    def test_unfixed_base_does_not_resolve(self) -> None:
        instance = load_instance(_INSTANCE)
        graph, task = _graph_and_task()
        result = SweFix().check_success(
            graph,
            task,
            {"workspace_files": dict(instance.base_files), "result": {"done": True}},
        )
        assert not result.success
        assert result.subgoals["test_calc.py::test_add"] is False
        assert result.subgoals["test_calc.py::test_subtract"] is True

    def test_empty_workspace_fails_cleanly(self) -> None:
        graph, task = _graph_and_task()
        result = SweFix().check_success(graph, task, {"result": {"done": True}})
        assert not result.success
        assert "no workspace" in result.reason


class TestRealizer:
    def test_materializes_base_tree_without_tests_or_gold(self) -> None:
        graph = to_graph(load_instance(_INSTANCE))
        runtime = SweRuntime(graph, Backing.PROCESS)
        runtime.reset()
        try:
            root = runtime.solver_root
            assert root is not None
            on_disk = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}
            assert on_disk == {"calc/__init__.py", "calc/core.py"}
            assert "test_calc.py" not in on_disk
            workspace = runtime.collect_extras()["workspace_files"]
            assert set(workspace) == {"calc/__init__.py", "calc/core.py"}
        finally:
            runtime.stop()

    def test_rejects_non_process_backing(self) -> None:
        graph = to_graph(load_instance(_INSTANCE))
        with pytest.raises(NotImplementedError):
            SwePack().realize(graph, Backing.CONTAINER)


class TestRunTestsSurface:
    """The multi-turn loop: the agent writes a reproduction, runs it against the
    live workspace via the surfaced ``run_tests`` tool (red on the buggy tree),
    applies a fix, and re-runs (green) — all without the held-out grading suite
    ever touching disk.
    """

    _REPRO = (
        "from calc.core import add\n\n\ndef test_repro():\n    assert add(2, 3) == 5\n"
    )

    def test_run_tests_reports_red_then_green(self) -> None:
        graph = to_graph(load_instance(_INSTANCE))
        runtime = SweRuntime(graph, Backing.PROCESS)
        runtime.reset()
        try:
            run_tests = runtime.surface()["run_tests"]
            assert callable(run_tests)
            root = runtime.solver_root
            assert root is not None
            (root / "repro_test.py").write_text(self._REPRO, encoding="utf-8")

            red = run_tests(["repro_test.py"])
            assert not red["ok"]
            assert red["returncode"] != 0
            assert red["isolation"] in {"subprocess", "bwrap", "bwrap+netns"}

            for rel, contents in load_instance(_INSTANCE).gold_files.items():
                (root / rel).write_text(contents, encoding="utf-8")

            green = run_tests(["repro_test.py"])
            assert green["ok"]
            assert green["returncode"] == 0
        finally:
            runtime.stop()

    def test_run_tests_does_not_expose_held_out_suite(self) -> None:
        # Collecting everything in the workspace must not pick up the held-out
        # test file — it is never written to disk.
        graph = to_graph(load_instance(_INSTANCE))
        runtime = SweRuntime(graph, Backing.PROCESS)
        runtime.reset()
        try:
            root = runtime.solver_root
            assert root is not None
            assert not (root / "test_calc.py").exists()
            collected = runtime.surface()["run_tests"]()
            # No tests on disk yet → pytest exit code 5 (nothing collected),
            # never the held-out suite running behind the agent's back.
            assert collected["returncode"] == 5
        finally:
            runtime.stop()


class TestInstanceLoader:
    def test_load_instance_shape(self) -> None:
        instance = load_instance(_INSTANCE)
        assert isinstance(instance, SweInstance)
        assert instance.fail_to_pass == ("test_calc.py::test_add",)
        assert instance.pass_to_pass == ("test_calc.py::test_subtract",)


class TestMultiFileInstance:
    """A realistic instance: a package with intra-package imports, a shared
    ``conftest`` fixture, a nested ``tests/`` dir, and a class-based test —
    exercising the real-repo execution path (not just the flat calc micro-repo).
    """

    def _graph_and_task(self) -> tuple[WorldGraph, TaskSpec]:
        graph = to_graph(load_instance(_MULTI))
        return graph, SweFix().generate(graph, {}, None)[0]

    def test_world_is_feasible(self) -> None:
        graph, task = self._graph_and_task()
        verdict = SweFix().check_feasibility(graph, task)
        assert verdict.feasible, verdict.reason

    def test_gold_resolves_every_subgoal(self) -> None:
        instance = load_instance(_MULTI)
        graph, task = self._graph_and_task()
        gold_tree = {**instance.base_files, **instance.gold_files}
        result = SweFix().check_success(
            graph, task, {"workspace_files": gold_tree, "result": {"done": True}}
        )
        assert result.success
        assert all(result.subgoals.values())
        # the class-based, nested-dir nodeid was reconstructed and graded:
        assert result.subgoals["tests/test_geometry.py::TestRectangle::test_large"]

    def test_base_fails_f2p_and_keeps_p2p(self) -> None:
        instance = load_instance(_MULTI)
        graph, task = self._graph_and_task()
        result = SweFix().check_success(
            graph,
            task,
            {"workspace_files": dict(instance.base_files), "result": {"done": True}},
        )
        assert not result.success
        assert all(result.subgoals[tid] is False for tid in instance.fail_to_pass)
        assert all(result.subgoals[tid] is True for tid in instance.pass_to_pass)


class TestNodeidReconstruction:
    @pytest.mark.parametrize(
        ("classname", "name", "file", "expected"),
        [
            (
                "tests.test_geometry",
                "test_rectangle_area",
                "tests/test_geometry.py",
                "tests/test_geometry.py::test_rectangle_area",
            ),
            (
                "tests.test_geometry.TestRectangle",
                "test_large",
                "tests/test_geometry.py",
                "tests/test_geometry.py::TestRectangle::test_large",
            ),
            # An import mode that drops the directory prefix still recovers the
            # class by anchoring on the file stem.
            (
                "test_geometry.TestRectangle",
                "test_large",
                "tests/test_geometry.py",
                "tests/test_geometry.py::TestRectangle::test_large",
            ),
            ("test_mod", "test_p[2-3]", "test_mod.py", "test_mod.py::test_p[2-3]"),
        ],
    )
    def test_reconstructs(
        self, classname: str, name: str, file: str, expected: str
    ) -> None:
        case = ET.fromstring(
            f'<testcase classname="{classname}" name="{name}" file="{file}"/>'
        )
        assert _nodeid(case) == expected

    def test_missing_file_is_unmatchable(self) -> None:
        case = ET.fromstring('<testcase classname="m" name="t"/>')
        assert _nodeid(case) is None


class TestSandbox:
    def test_runs_python_and_captures_stdout(self, tmp_path: Path) -> None:
        res = run_sandboxed(["-c", "print('hello-sbx')"], root=tmp_path, timeout=10)
        assert res.ok
        assert res.returncode == 0
        assert "hello-sbx" in res.stdout
        assert res.isolation in {"subprocess", "bwrap", "bwrap+netns"}

    def test_nonzero_exit_is_reported_not_raised(self, tmp_path: Path) -> None:
        res = run_sandboxed(["-c", "raise SystemExit(3)"], root=tmp_path, timeout=10)
        assert res.returncode == 3
        assert not res.ok
        assert not res.timed_out

    def test_wall_clock_timeout(self, tmp_path: Path) -> None:
        res = run_sandboxed(
            ["-c", "import time; time.sleep(10)"], root=tmp_path, timeout=2
        )
        assert res.timed_out

    def test_bwrap_argv_isolates_net_only_when_disabled(self) -> None:
        with_net = _bwrap_wrap(["x"], root=Path("/w"), network=True)
        without_net = _bwrap_wrap(["x"], root=Path("/w"), network=False)
        assert "--unshare-net" in without_net
        assert "--unshare-net" not in with_net
        assert "--unshare-pid" in without_net
        assert without_net[-1] == "x"


# --- "pull a world from a git repo" pipeline (deterministic, offline) --------
#
# A SWE-bench row points at a repo@base_commit and ships the fix + held-out
# tests as diffs. Rather than hit GitHub, we author a tiny git repo locally and
# feed its path as ``source`` — the same ``git clone`` code path the GitHub
# default takes — so the whole adapter is exercised hermetically.

_BASE_CALC = (
    "def add(a, b):\n    return a - b\n\n\ndef subtract(a, b):\n    return a - b\n"
)
_GOLD_CALC = (
    "def add(a, b):\n    return a + b\n\n\ndef subtract(a, b):\n    return a - b\n"
)
_P2P_TEST = (
    "from pkg.calc import subtract\n\n\n"
    "def test_subtract():\n    assert subtract(5, 3) == 2\n"
)
_F2P_TEST = "from pkg.calc import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n"


def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", *args], cwd=str(repo), capture_output=True, text=True, check=True
    )
    return proc.stdout


def _author_origin(root: Path) -> tuple[Path, str]:
    """Init a repo whose base commit has the bug + a pre-existing P2P test."""
    repo = root / "origin"
    repo.mkdir()
    _git(repo, "init", "--quiet", "-b", "main")
    write_tree(
        repo,
        {
            "pkg/__init__.py": "",
            "pkg/calc.py": _BASE_CALC,
            "tests/test_calc.py": _P2P_TEST,
        },
    )
    _git(repo, "add", "-A")
    _git(
        repo,
        "-c",
        "user.name=test",
        "-c",
        "user.email=test@example.com",
        "-c",
        "commit.gpgsign=false",
        "commit",
        "--quiet",
        "-m",
        "base: add() has the bug",
    )
    return repo, _git(repo, "rev-parse", "HEAD").strip()


def _author_patches(repo: Path) -> tuple[str, str]:
    """Return ``(gold_patch, test_patch)`` as unified diffs against base."""
    (repo / "pkg/calc.py").write_text(_GOLD_CALC, encoding="utf-8")
    gold_patch = _git(repo, "diff")
    _git(repo, "checkout", "--", "pkg/calc.py")

    (repo / "tests/test_added.py").write_text(_F2P_TEST, encoding="utf-8")
    _git(repo, "add", "-N", "tests/test_added.py")
    test_patch = _git(repo, "diff")
    _git(repo, "reset", "--quiet", "--", "tests/test_added.py")
    (repo / "tests/test_added.py").unlink()
    return gold_patch, test_patch


class TestPullFromGit:
    @pytest.fixture(scope="class")
    def pulled(
        self, tmp_path_factory: pytest.TempPathFactory
    ) -> tuple[dict[str, str], str]:
        """Author a local origin repo and return its SWE-bench row + source.

        ``source`` is the local path the builder/adapter clones from, so the
        whole "pull a world from GitHub" path runs offline.
        """
        root = tmp_path_factory.mktemp("pullgit")
        repo, base_commit = _author_origin(root)
        gold_patch, test_patch = _author_patches(repo)
        row = {
            "instance_id": "local-calc-001",
            "repo": "acme/calc",
            "base_commit": base_commit,
            "problem_statement": "add() returns the difference instead of the sum",
            "patch": gold_patch,
            "test_patch": test_patch,
            "FAIL_TO_PASS": json.dumps(["tests/test_added.py::test_add"]),
            "PASS_TO_PASS": json.dumps(["tests/test_calc.py::test_subtract"]),
        }
        return row, str(repo)

    @pytest.fixture(scope="class")
    def instance(
        self,
        pulled: tuple[dict[str, str], str],
        tmp_path_factory: pytest.TempPathFactory,
    ) -> SweInstance:
        row, source = pulled
        work = tmp_path_factory.mktemp("pullwork")
        return instance_from_row(row, workdir=work, source=source)

    def test_recovers_held_out_tests_and_gold(self, instance: SweInstance) -> None:
        assert instance.instance_id == "local-calc-001"
        assert instance.name == "calc"
        assert instance.fail_to_pass == ("tests/test_added.py::test_add",)
        assert instance.pass_to_pass == ("tests/test_calc.py::test_subtract",)
        # The whole graded suite is held out: the new F2P file (from the test
        # patch) and the pre-existing P2P file both land in test_files, leaving
        # base_files as source only — the agent can't see or edit its scorer.
        assert set(instance.test_files) == {
            "tests/test_added.py",
            "tests/test_calc.py",
        }
        assert set(instance.base_files) == {"pkg/__init__.py", "pkg/calc.py"}
        # The gold tree carries the fix; the base tree carries the defect.
        assert "return a + b" in instance.gold_files["pkg/calc.py"]
        assert "return a - b" in instance.base_files["pkg/calc.py"]

    def test_reconstructed_world_admits_and_discriminates(
        self, instance: SweInstance
    ) -> None:
        graph = to_graph(instance)
        task = SweFix().generate(graph, {}, None)[0]
        verdict = SweFix().check_feasibility(graph, task)
        assert verdict.feasible, verdict.reason

        gold_tree = {**instance.base_files, **instance.gold_files}
        gold = SweFix().check_success(
            graph, task, {"workspace_files": gold_tree, "result": {"done": True}}
        )
        assert gold.success
        assert all(gold.subgoals.values())

        base = SweFix().check_success(
            graph,
            task,
            {"workspace_files": dict(instance.base_files), "result": {"done": True}},
        )
        assert not base.success
        assert base.subgoals["tests/test_added.py::test_add"] is False
        assert base.subgoals["tests/test_calc.py::test_subtract"] is True

    def test_row_admits_through_the_builder(
        self, pulled: tuple[dict[str, str], str]
    ) -> None:
        # The whole "pull a world from GitHub" path, reachable from a manifest:
        # admit() -> SweBuilder clones the row's repo and lays it over the
        # ontology -> the same self-test that gates any world admits it.
        from openrange.core.admit import AdmissionFailure, admit

        row, source = pulled
        result = admit(
            SwePack(),
            manifest={"swebench": row, "source": source},
            max_repairs=0,
        )
        assert not isinstance(result, AdmissionFailure)
        assert result.ontology_id == "swe.repo@v1"
        assert result.lineage["source"] == "github"
        assert result.lineage["instance"] == "local-calc-001"
