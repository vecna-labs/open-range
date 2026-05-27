"""OpenRange CLI."""

from __future__ import annotations

import argparse
import json
import webbrowser
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from openrange.core.admit import AdmissionFailure, admit, snapshot_to_dict
from openrange.core.errors import EpisodeRuntimeError
from openrange.core.pack import PACKS
from openrange.core.store import SnapshotStore
from openrange.dashboard import (
    DashboardHTTPServer,
    DashboardView,
    RunsRegistry,
)


def main() -> None:
    parser = argparse.ArgumentParser(prog="openrange")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("manifest", type=Path)
    build_parser.add_argument("--output", type=Path)
    build_parser.add_argument("--max-repairs", type=int, default=2)

    inspect_parser = subparsers.add_parser("inspect")
    inspect_parser.add_argument("snapshot", type=Path)

    dashboard_parser = subparsers.add_parser("dashboard")
    dashboard_parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("or-runs"),
        help="Directory containing run subdirs. Default: ./or-runs",
    )
    dashboard_parser.add_argument(
        "--run-root",
        type=Path,
        help="Single-run mode. Opens just this directory and skips multi-run "
        "discovery.",
    )
    dashboard_parser.add_argument(
        "--store-dir",
        type=Path,
        default=Path("snapshots"),
        help="Snapshot store for legacy --snapshot-id mode (no live events).",
    )
    dashboard_parser.add_argument("--snapshot-id")
    dashboard_parser.add_argument("--host", default="127.0.0.1")
    dashboard_parser.add_argument("--port", type=int, default=8000)
    dashboard_parser.add_argument("--no-browser", action="store_true")

    args = parser.parse_args()
    if args.command == "build":
        _run_build(args)
        return
    if args.command == "dashboard":
        _run_dashboard(args, parser)
        return
    _run_inspect(args)


def _run_build(args: argparse.Namespace) -> None:
    raw = json.loads(args.manifest.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise EpisodeRuntimeError("manifest must be a JSON object")
    manifest = cast(Mapping[str, object], raw)
    pack_field = manifest.get("pack")
    if isinstance(pack_field, Mapping):
        pack_id = pack_field.get("id")
    elif isinstance(pack_field, str):
        pack_id = pack_field
    else:
        pack_id = None
    if not isinstance(pack_id, str) or not pack_id:
        raise EpisodeRuntimeError(
            "manifest must declare a pack via 'pack.id' or 'pack' (string)",
        )
    try:
        pack = PACKS.resolve(pack_id)
    except Exception as exc:
        raise EpisodeRuntimeError(f"unknown pack {pack_id!r}") from exc
    result = admit(pack, manifest, max_repairs=args.max_repairs)
    if isinstance(result, AdmissionFailure):
        raise EpisodeRuntimeError(
            f"admission failed after {result.attempts} attempt(s): "
            f"{len(result.issues)} error(s), "
            f"{len(result.infeasible_tasks)} infeasible task(s)",
        )
    if args.output is not None:
        path = SnapshotStore(args.output).save(result)
        print(path)
    else:
        print(json.dumps(snapshot_to_dict(result), sort_keys=True))


def _run_inspect(args: argparse.Namespace) -> None:
    data = json.loads(args.snapshot.read_text(encoding="utf-8"))
    lineage = data.get("lineage", {})
    if not isinstance(lineage, Mapping):
        lineage = {}
    tasks_raw = data.get("tasks", [])
    if not isinstance(tasks_raw, list):
        tasks_raw = []
    task_ids = [
        task["id"]
        for task in tasks_raw
        if isinstance(task, Mapping) and isinstance(task.get("id"), str)
    ]
    print(
        json.dumps(
            {
                "snapshot_id": data.get("snapshot_id"),
                "tasks": task_ids,
                "pack": lineage.get("pack"),
                "pack_version": lineage.get("pack_version"),
            },
            sort_keys=True,
        ),
    )


def _run_dashboard(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    server: DashboardHTTPServer
    if args.run_root is not None:
        events = args.run_root / "dashboard.events.jsonl"
        state = args.run_root / "dashboard.json"
        if not events.exists() or not state.exists():
            parser.error(
                f"{args.run_root} is missing dashboard.events.jsonl or dashboard.json",
            )
        view = DashboardView(
            event_log_path=events,
            state_path=state,
            reset_artifacts=False,
            tail=True,
        )
        server = DashboardHTTPServer((args.host, args.port), view=view)
    elif args.snapshot_id is not None:
        snapshot = SnapshotStore(args.store_dir).load(args.snapshot_id)
        view = DashboardView(snapshot)
        server = DashboardHTTPServer((args.host, args.port), view=view)
    else:
        args.runs_dir.mkdir(parents=True, exist_ok=True)
        registry = RunsRegistry(args.runs_dir)
        server = DashboardHTTPServer((args.host, args.port), runs=registry)

    host = str(server.server_address[0])
    url = f"http://{host}:{server.server_address[1]}"
    print(url, flush=True)
    if server.runs is not None:
        print(f"watching: {server.runs.runs_dir}", flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if server.runs is not None:
            server.runs.close()
        elif server.view is not None:
            server.view.close()
        server.server_close()


def latest_snapshot_id(store_dir: Path) -> str | None:
    snapshots = sorted(store_dir.glob("*.json"))
    if not snapshots:
        return None
    return snapshots[-1].stem


if __name__ == "__main__":
    main()
