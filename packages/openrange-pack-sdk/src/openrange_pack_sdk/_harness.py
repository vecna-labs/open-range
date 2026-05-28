"""Child-side runner for ``openrange_pack_sdk.run_submission``; not a public API.

Reads a JSON envelope from stdin, runs the untrusted submission ``source`` and
the trusted ``driver`` function in one isolated process, and writes the JSON
result to the envelope's ``result_path`` — never stdout, so a submission that
scribbles on stdout or exits early cannot forge a result it has no handle to.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import sys
from typing import Any


def main() -> None:
    envelope = json.loads(sys.stdin.read())
    try:
        text = json.dumps(_run(envelope))
    except BaseException as exc:  # noqa: BLE001 — never leave the parent without a result
        text = json.dumps({"ok": False, "error": _fmt("harness error", exc)})
    with open(envelope["result_path"], "w", encoding="utf-8") as handle:
        handle.write(text)


def _run(envelope: dict[str, Any]) -> dict[str, Any]:
    _apply_limits(envelope["memory_bytes"], envelope["cpu_seconds"])
    sink = io.StringIO()
    real_stdout = sys.stdout
    namespace: dict[str, Any] = {}
    sys.stdout = sink
    try:
        exec(envelope["source"], namespace)
    except BaseException as exc:  # noqa: BLE001 — any load failure is a failed case
        return {"ok": False, "error": _fmt("source did not load", exc)}
    finally:
        sys.stdout = real_stdout

    entry = namespace.get(envelope["entry"])
    if not callable(entry):
        return {
            "ok": False,
            "error": f"submission defines no callable {envelope['entry']}",
        }

    driver = _load_driver(
        envelope["driver_module"], envelope["driver_file"], envelope["driver_qualname"]
    )
    sys.stdout = sink
    try:
        result = driver(entry, envelope["case"])
    except BaseException as exc:  # noqa: BLE001 — a misbehaving submission is a failed case
        return {"ok": False, "error": _fmt("submission failed", exc)}
    finally:
        sys.stdout = real_stdout
    return {"ok": True, "result": result}


def _apply_limits(memory_bytes: int, cpu_seconds: int) -> None:
    try:
        import resource
    except ImportError:  # not POSIX (e.g. Windows); nothing to cap
        return
    for name, limit in (("RLIMIT_AS", memory_bytes), ("RLIMIT_CPU", cpu_seconds)):
        with contextlib.suppress(ValueError, OSError):
            resource.setrlimit(getattr(resource, name), (limit, limit))


def _load_driver(module_name: str, file: str, qualname: str) -> Any:
    # Load by file, not import, so the child runs only the driver's module —
    # never its whole pack — which is fast and sidesteps the circular import a
    # pack __init__ would cause. (Driver modules must stay import-light.)
    spec = importlib.util.spec_from_file_location(module_name, file)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load driver {module_name!r} from {file!r}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    obj: Any = module
    for part in qualname.split("."):
        obj = getattr(obj, part)
    return obj


def _fmt(prefix: str, exc: BaseException) -> str:
    return f"{prefix}: {type(exc).__name__}: {exc}"[:500]


if __name__ == "__main__":
    main()
