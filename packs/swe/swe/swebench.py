"""Ingest a SWE-bench-shaped row into a :class:`~swe.instances.SweInstance`.

A SWE-bench row references its repo by ``repo`` + ``base_commit`` and ships the
gold fix and the held-out tests as unified diffs (``patch`` / ``test_patch``),
with the graded nodeids in ``FAIL_TO_PASS`` / ``PASS_TO_PASS`` (JSON-encoded
lists). This is the *imported* world-source made real: clone the repo at the base
commit, read its working tree, and recover the held-out test files and the gold
post-fix files by applying the two diffs with ``git apply``. The result lays out
over the same ``swe.repo@v1`` ontology and is checked by the same admission
twin-test as any hand-authored world — so "pull a world from GitHub" reduces to
"build an instance, then admit it."

Materialization is *trusted*, build-time git plumbing in a throwaway clone — it
is not the agent sandbox (that is :mod:`swe.grading` / :mod:`swe.sandbox`). The
git source may be a remote URL (the GitHub default) or a local path; both flow
through the same ``git clone``.

Scale ceiling: this inlines the repo's working tree into the instance, and hence
into the world graph. That is fine for small / medium repos; large monorepos are
the lazy-clone-at-realize milestone (#212), where the graph stores only the repo
reference and the realizer materializes on demand.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from swe.instances import SweInstance

_MAX_FILE_BYTES = 1024 * 1024  # skip blobs larger than this when inlining a tree.
_GIT_TIMEOUT = 600.0


class SweBenchError(RuntimeError):
    """A SWE-bench row could not be materialized into an instance."""


def instance_from_row(
    row: Mapping[str, Any],
    *,
    workdir: Path,
    source: str | None = None,
) -> SweInstance:
    """Clone ``row``'s repo at its base commit and build the instance.

    ``source`` overrides where to clone from (a local path in tests); it defaults
    to the row's GitHub URL. ``workdir`` is a caller-owned scratch dir that holds
    the throwaway clone — discard it after.
    """
    repo = str(row["repo"])
    base_commit = str(row["base_commit"])
    src = source or f"https://github.com/{repo}.git"
    workdir.mkdir(parents=True, exist_ok=True)
    checkout = workdir / "repo"
    _materialize_repo(src, base_commit, checkout)

    base_tree = _read_tree(checkout)
    patch_test_files = _apply_and_read(checkout, str(row.get("test_patch") or ""))
    gold_files = _apply_and_read(checkout, str(row.get("patch") or ""))

    f2p = parse_test_ids(row.get("FAIL_TO_PASS"))
    p2p = parse_test_ids(row.get("PASS_TO_PASS"))

    # Hold the *entire* graded suite out of the agent's view: the files the test
    # patch introduces/modifies, plus any pre-existing file a graded id names
    # (PASS_TO_PASS tests usually already live in the repo). Post-patch content
    # wins where the test patch touched a file; otherwise we take the base-tree
    # copy. base_files is then the buggy tree with every graded test removed, so
    # the agent can neither see nor edit the suite that scores it — and the
    # gold-modified source stays buggy there (that *is* the defect).
    test_files = _collect_test_files(base_tree, patch_test_files, (*f2p, *p2p))
    base_files = {p: c for p, c in base_tree.items() if p not in test_files}
    return SweInstance(
        instance_id=str(row["instance_id"]),
        name=repo.rsplit("/", 1)[-1] or repo,
        language="python",
        problem_statement=str(row.get("problem_statement") or ""),
        base_files=base_files,
        gold_files=gold_files,
        test_files=test_files,
        fail_to_pass=tuple(f2p),
        pass_to_pass=tuple(p2p),
    )


def _collect_test_files(
    base_tree: Mapping[str, str],
    patch_test_files: Mapping[str, str],
    test_ids: Sequence[str],
) -> dict[str, str]:
    """The held-out suite: test-patch files + every graded id's pre-existing file.

    A graded id whose file is neither in the test patch nor the base tree is left
    out, so the dangling-id invariant rejects the world at admission rather than
    grading against a phantom test.
    """
    held: dict[str, str] = dict(patch_test_files)
    for tid in test_ids:
        path = tid.split("::", 1)[0]
        if path not in held and path in base_tree:
            held[path] = base_tree[path]
    return held


def parse_test_ids(value: Any) -> list[str]:
    """Normalize a FAIL_TO_PASS / PASS_TO_PASS field (JSON string or list)."""
    if value is None:
        return []
    if isinstance(value, str):
        value = json.loads(value) if value.strip() else []
    if not isinstance(value, list | tuple):
        raise SweBenchError(f"test-id field is {type(value).__name__}, expected list")
    return [str(v) for v in value]


def _materialize_repo(source: str, commit: str, dest: Path) -> None:
    _git(["clone", "--quiet", source, str(dest)], cwd=dest.parent)
    _git(["checkout", "--quiet", commit], cwd=dest)


def _apply_and_read(repo: Path, patch: str) -> dict[str, str]:
    """Apply ``patch`` to ``repo``, read the files it touched, then reset.

    Returns ``{path: post-patch contents}`` for each existing target (deletions
    drop out). The clone is reset to the base commit afterward so the next patch
    applies cleanly against a pristine tree.
    """
    if not patch.strip():
        return {}
    paths = _patched_paths(repo, patch)
    _git(["apply", "--whitespace=nowarn", "-"], cwd=repo, stdin=patch)
    out: dict[str, str] = {}
    for rel in paths:
        text = _read_text(repo / rel)
        if text is not None:
            out[rel] = text
    _git(["reset", "--hard", "--quiet"], cwd=repo)
    _git(["clean", "-fdq"], cwd=repo)
    return out


def _patched_paths(repo: Path, patch: str) -> list[str]:
    res = _git(["apply", "--numstat", "-"], cwd=repo, stdin=patch)
    paths: list[str] = []
    for line in res.stdout.splitlines():
        cols = line.split("\t")  # "<added>\t<removed>\t<path>"
        if len(cols) == 3:
            paths.append(cols[2])
    return paths


def _read_tree(repo: Path) -> dict[str, str]:
    tree: dict[str, str] = {}
    for path in sorted(repo.rglob("*")):
        if not path.is_file() or ".git" in path.relative_to(repo).parts:
            continue
        text = _read_text(path)
        if text is not None:
            tree[path.relative_to(repo).as_posix()] = text
    return tree


def _read_text(path: Path) -> str | None:
    try:
        if not path.is_file() or path.stat().st_size > _MAX_FILE_BYTES:
            return None
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def _git(
    args: Sequence[str], *, cwd: Path, stdin: str | None = None
) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        input=stdin,
        capture_output=True,
        text=True,
        timeout=_GIT_TIMEOUT,
        check=False,
    )
    if proc.returncode != 0:
        raise SweBenchError(
            f"git {args[0]} failed ({proc.returncode}): {proc.stderr.strip()[:300]}"
        )
    return proc
