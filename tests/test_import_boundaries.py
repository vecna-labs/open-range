"""Repo-wide import-boundary invariants:

1. ``packs/`` MUST NOT import from ``openrange`` (except ``openrange_pack_sdk``).
2. ``src/openrange/`` MUST NOT import from any pack.
3. ``packages/openrange-pack-sdk/`` MUST NOT import from any pack.
4. ``openrange.__init__`` MUST NOT re-export ``openrange_pack_sdk`` symbols
   (no migration shim — callers import from the SDK directly).
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _imports_in(file: Path) -> set[str]:
    tree = ast.parse(file.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def _py_files(root: Path) -> list[Path]:
    return [p for p in root.rglob("*.py") if "__pycache__" not in p.parts]


def _pack_module_names() -> set[str]:
    return {
        sub.name
        for pack_dir in (REPO_ROOT / "packs").iterdir()
        if pack_dir.is_dir()
        for sub in pack_dir.iterdir()
        if sub.is_dir() and (sub / "__init__.py").exists()
    }


def _imports_under(imports: set[str], prefix: str) -> set[str]:
    return {i for i in imports if i == prefix or i.startswith(prefix + ".")}


def test_import_boundaries() -> None:
    pack_modules = _pack_module_names()
    assert pack_modules, "expected at least one pack module under packs/"

    violations: list[str] = []

    for file in _py_files(REPO_ROOT / "packs"):
        imports = _imports_in(file)
        leaks = _imports_under(imports, "openrange") - _imports_under(
            imports, "openrange_pack_sdk"
        )
        for leak in sorted(leaks):
            violations.append(f"pack→openrange  {file.relative_to(REPO_ROOT)} → {leak}")

    for file in _py_files(REPO_ROOT / "src" / "openrange"):
        imports = _imports_in(file)
        for pack in pack_modules:
            for leak in sorted(_imports_under(imports, pack)):
                violations.append(
                    f"openrange→pack  {file.relative_to(REPO_ROOT)} → {leak}"
                )

    sdk_src = REPO_ROOT / "packages" / "openrange-pack-sdk" / "src"
    for file in _py_files(sdk_src):
        imports = _imports_in(file)
        for pack in pack_modules:
            for leak in sorted(_imports_under(imports, pack)):
                violations.append(f"sdk→pack  {file.relative_to(REPO_ROOT)} → {leak}")

    openrange_init = REPO_ROOT / "src" / "openrange" / "__init__.py"
    init_imports = _imports_in(openrange_init)
    for leak in sorted(_imports_under(init_imports, "openrange_pack_sdk")):
        violations.append(
            f"openrange-reexports-sdk  src/openrange/__init__.py → {leak}"
        )

    assert not violations, "import-boundary violations:\n  " + "\n  ".join(violations)
