"""DashboardView: aggregates snapshot data and event history for the UI."""

from __future__ import annotations

import json
import threading
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import cast

from openrange_pack_sdk import Snapshot, TaskSpec

from openrange.core import ActorTurn
from openrange.dashboard.events import (
    DashboardEvent,
    EventBridge,
    dashboard_event_from_mapping,
    fallback_narrate,
    json_safe,
    read_dashboard_events,
    read_dashboard_state,
    write_dashboard_state,
)
from openrange.dashboard.summaries import (
    activity_summary,
    actor_summaries,
    health_summary,
)
from openrange.dashboard.topology import (
    empty_runtime_topology,
    graph_projection,
    normalized_runtime_topology,
    public_world,
    stored_entrypoints,
    stored_missions,
)


class _EventLogTail:
    # Polls ``dashboard.events.jsonl`` and pushes new lines into the
    # bridge. Polling (not inotify/fsevents) keeps it dependency-free
    # and cross-platform; 250ms is fast enough that the writer's
    # per-line flush feels instant to a human watching the dashboard.

    POLL_INTERVAL_SEC: float = 0.25

    def __init__(
        self,
        path: Path,
        bridge: EventBridge,
        *,
        initial_offset: int,
    ) -> None:
        self._path = path
        self._bridge = bridge
        self._offset = initial_offset
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2)

    def _loop(self) -> None:
        while not self._stop.wait(self.POLL_INTERVAL_SEC):
            self._poll_once()

    def _poll_once(self) -> None:
        try:
            size = self._path.stat().st_size
        except OSError:
            return
        if size < self._offset:
            self._offset = 0
        if size == self._offset:
            return
        try:
            with self._path.open("r", encoding="utf-8") as handle:
                handle.seek(self._offset)
                chunk = handle.read()
        except OSError:
            return
        last_newline = chunk.rfind("\n")
        if last_newline == -1:
            return
        self._offset += last_newline + 1
        for line in chunk[: last_newline + 1].splitlines():
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(data, Mapping):
                self._bridge.push(dashboard_event_from_mapping(data))


class DashboardView:
    def __init__(
        self,
        snapshot: Snapshot | None = None,
        *,
        bridge: EventBridge | None = None,
        event_log_path: str | Path | None = None,
        state_path: str | Path | None = None,
        reset_artifacts: bool = True,
        tail: bool = False,
    ) -> None:
        self.snapshot = snapshot
        self.bridge = bridge or EventBridge()
        self._running = False
        self._lock = threading.Lock()
        self._lineage_nodes: list[dict[str, object]] = (
            [] if snapshot is None else [_lineage_node(snapshot)]
        )
        self._event_log_path = None if event_log_path is None else Path(event_log_path)
        self._state_path = None if state_path is None else Path(state_path)
        self._stored_dashboard = (
            {}
            if self._state_path is None or reset_artifacts
            else read_dashboard_state(self._state_path)
        )
        if self._event_log_path is not None:
            self._event_log_path.parent.mkdir(parents=True, exist_ok=True)
            if reset_artifacts or not self._event_log_path.exists():
                self._event_log_path.write_text("", encoding="utf-8")
            else:
                for event in read_dashboard_events(self._event_log_path):
                    self.bridge.push(event)
        self._event_count = len(self.bridge.snapshot_buffer())
        if self._state_path is not None:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            self._write_state(self._state_path)
        # Reader-mode views (tail=True) need a poller because the
        # writer is in another process; writer-mode views own the
        # bridge directly and would double-publish if they tailed.
        self._tail: _EventLogTail | None = None
        if tail and self._event_log_path is not None and not reset_artifacts:
            try:
                offset = self._event_log_path.stat().st_size
            except OSError:
                offset = 0
            self._tail = _EventLogTail(
                self._event_log_path,
                self.bridge,
                initial_offset=offset,
            )
            self._tail.start()

    def close(self) -> None:
        if self._tail is not None:
            self._tail.stop()
            self._tail = None
        self.bridge.close()

    def topology(self) -> Mapping[str, object]:
        if self.snapshot is None:
            stored = self._stored_section("topology")
            if stored:
                return stored
            return {
                "snapshot_id": None,
                "world": {},
                "tasks": [],
                **empty_runtime_topology(),
            }
        runtime_topology = normalized_runtime_topology(self.snapshot)
        manifest = _manifest_from_lineage(self.snapshot.lineage)
        world_block = manifest.get("world", {})
        if not isinstance(world_block, Mapping):
            world_block = {}
        return {
            "snapshot_id": self.snapshot.snapshot_id,
            "world": public_world(cast(Mapping[str, object], world_block)),
            "tasks": [_task_to_dict(task) for task in self.snapshot.tasks],
            **runtime_topology,
        }

    def register_snapshot(self, snapshot: Snapshot) -> None:
        """Re-bind to ``snapshot`` and append it to the lineage chain.

        Re-binding keeps topology reflecting the current world, not the one the
        view was created with; the appended node is what makes the evolution
        history viewable. A repeated id is a no-op.
        """
        self.snapshot = snapshot
        last = self._lineage_nodes[-1]["id"] if self._lineage_nodes else None
        if last != snapshot.snapshot_id:
            self._lineage_nodes.append(_lineage_node(snapshot))
        self._write_configured_state()

    def lineage(self) -> Mapping[str, object]:
        if self.snapshot is None:
            stored = self._stored_section("lineage")
            if stored:
                return stored
            return {
                "snapshot_id": None,
                "nodes": [],
                "lineage": {},
                "history": [],
                "parent_snapshot_id": None,
            }
        return {
            "snapshot_id": self.snapshot.snapshot_id,
            "nodes": [dict(node) for node in self._lineage_nodes],
            "lineage": dict(self.snapshot.lineage),
            "history": [event.to_dict() for event in self.snapshot.history],
            "parent_snapshot_id": _parent_snapshot_id(self.snapshot.lineage),
        }

    def state(self) -> Mapping[str, object]:
        events = [event.as_dict() for event in self.bridge.snapshot_buffer()]
        turn_count = sum(1 for event in events if event["type"] == "env_turn")
        return {
            "running": self._running,
            "status": self._status(),
            "snapshot_id": self._snapshot_id(),
            "event_count": len(events),
            "turn_count": turn_count,
            "latest_event": None if not events else events[-1],
            "activity_summary": activity_summary(events),
            "health": health_summary(events),
            "events": events,
        }

    def _status(self) -> str:
        if self.snapshot is None and self._snapshot_id() is None:
            return "waiting_for_snapshot"
        if self._running:
            return "playing"
        return "paused"

    def reset(self, snapshot: Snapshot | None = None) -> Mapping[str, object]:
        if snapshot is not None:
            self.snapshot = snapshot
        self._running = False
        if self.snapshot is None:
            result = {
                "status": "waiting_for_snapshot",
                "snapshot_id": None,
                "topology": self.topology(),
            }
        else:
            result = {
                "status": "ready",
                "snapshot_id": self.snapshot.snapshot_id,
                "topology": self.topology(),
            }
        self._write_configured_state()
        return result

    def play(self) -> Mapping[str, object]:
        if self._running:
            return {"status": "already running"}
        self._running = True
        self._write_configured_state()
        return {"status": "playing"}

    def pause(self) -> Mapping[str, object]:
        self._running = False
        self._write_configured_state()
        return {"status": "paused"}

    def record_event(
        self,
        event_type: str,
        *,
        actor: str,
        target: str,
        data: Mapping[str, object] | None = None,
    ) -> DashboardEvent:
        with self._lock:
            self._event_count += 1
            event = DashboardEvent(
                f"{self._event_count}:{event_type}",
                event_type,
                actor,
                target,
                float(self._event_count - 1),
                MappingProxyType(dict(data or {})),
            )
            self.bridge.push(event)
            self._write_event(event)
        return event

    def record_turn(self, turn: ActorTurn) -> DashboardEvent:
        return self.record_event(
            "env_turn",
            actor=turn.actor_id,
            target=turn.target,
            data=turn.as_dict(),
        )

    def turns(self, task_id: str | None = None) -> list[dict[str, object]]:
        turns = [
            dict(event.data)
            for event in self.bridge.snapshot_buffer()
            if event.type == "env_turn"
        ]
        if task_id is None:
            return turns
        return [turn for turn in turns if turn.get("task_id") == task_id]

    def builder_steps(self) -> list[dict[str, object]]:
        return [
            dict(event.data)
            for event in self.bridge.snapshot_buffer()
            if event.type == "builder_step"
        ]

    def inspect(self) -> Mapping[str, object]:
        return {
            "briefing": self.briefing(),
            "topology": self.topology(),
            "lineage": self.lineage(),
            "state": self.state(),
            "actors": self.actors(),
            "turns": self.turns(),
            "builder": {"steps": self.builder_steps()},
            "narration": self.narration(),
        }

    def actors(self) -> list[dict[str, object]]:
        return actor_summaries(
            [event.as_dict() for event in self.bridge.snapshot_buffer()],
        )

    def briefing(self) -> Mapping[str, object]:
        if self.snapshot is None:
            topology = self.topology()
            snapshot_id = topology.get("snapshot_id")
            world = topology.get("world")
            tasks = topology.get("tasks")
            if isinstance(snapshot_id, str) and isinstance(world, Mapping):
                task_rows = tasks if isinstance(tasks, list) else []
                return {
                    "snapshot_id": snapshot_id,
                    "title": str(world.get("title", "")),
                    "goal": str(world.get("goal", "")),
                    "entrypoints": stored_entrypoints(task_rows),
                    "missions": stored_missions(task_rows),
                }
            return {
                "snapshot_id": None,
                "title": "",
                "goal": "",
                "entrypoints": [],
                "missions": [],
            }
        manifest = _manifest_from_lineage(self.snapshot.lineage)
        world_block = manifest.get("world", {})
        if not isinstance(world_block, Mapping):
            world_block = {}
        graph_nodes = self.snapshot.graph.nodes
        return {
            "snapshot_id": self.snapshot.snapshot_id,
            "title": str(world_block.get("title", "")),
            "goal": str(world_block.get("goal", "")),
            "entrypoints": [
                {
                    "task_id": task.id,
                    "node_id": node_id,
                    "node_kind": (
                        graph_nodes[node_id].kind if node_id in graph_nodes else ""
                    ),
                }
                for task in self.snapshot.tasks
                for node_id in task.entrypoints
            ],
            "missions": [
                {"task_id": task.id, "instruction": task.instruction}
                for task in self.snapshot.tasks
            ],
        }

    def narration(self) -> Mapping[str, object]:
        return {"narration": fallback_narrate(self.bridge.snapshot_buffer())}

    def _write_event(self, event: DashboardEvent) -> None:
        if self._event_log_path is not None:
            with self._event_log_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(json_safe(event.as_dict()), sort_keys=True) + "\n",
                )
        self._write_configured_state()

    def _write_configured_state(self) -> None:
        if self._state_path is not None:
            self._write_state(self._state_path)

    def _write_state(self, state_path: Path) -> None:
        write_dashboard_state(state_path, self.bridge.snapshot_buffer(), self)

    def _stored_section(self, key: str) -> Mapping[str, object]:
        # Reader-mode (no in-memory snapshot): re-read the state file
        # each call so a snapshot the writer lands later surfaces
        # without restarting the dashboard server.
        if self.snapshot is None and self._state_path is not None:
            fresh = read_dashboard_state(self._state_path)
            if fresh:
                self._stored_dashboard = fresh
        section = self._stored_dashboard.get(key)
        if isinstance(section, Mapping):
            return section
        return {}

    def _snapshot_id(self) -> str | None:
        if self.snapshot is not None:
            return self.snapshot.snapshot_id
        topology = self._stored_section("topology")
        snapshot_id = topology.get("snapshot_id")
        return snapshot_id if isinstance(snapshot_id, str) else None


def _manifest_from_lineage(lineage: Mapping[str, object]) -> Mapping[str, object]:
    # ``admit()`` stashes the manifest under ``lineage["manifest"]``.
    manifest = lineage.get("manifest")
    if isinstance(manifest, Mapping):
        return cast(Mapping[str, object], manifest)
    return {}


def _evolve_block(lineage: Mapping[str, object]) -> Mapping[str, object]:
    top = lineage.get("_evolve")
    return cast(Mapping[str, object], top) if isinstance(top, Mapping) else {}


def _parent_snapshot_id(lineage: Mapping[str, object]) -> str | None:
    parent = _evolve_block(lineage).get("parent_snapshot_id")
    return parent if isinstance(parent, str) else None


# "harden"/"soften" collide with security-hardening (the opposite); the UI
# shows the agent-difficulty sense instead.
_DIRECTION_LABEL = {"harden": "harder", "soften": "easier", "diversify": "varied"}


def _evolve_summary(evolve: Mapping[str, object]) -> str:
    if not evolve:
        return "Initial build"
    direction = str(evolve.get("direction") or "evolved")
    word = _DIRECTION_LABEL.get(direction, direction)
    family = evolve.get("family")
    note = evolve.get("note")
    head = f"{word} · {family}" if family else word
    return f"{head} — {note}" if note else head


def _lineage_node(snapshot: Snapshot) -> dict[str, object]:
    lineage = snapshot.lineage
    evolve = _evolve_block(lineage)
    manifest = dict(_manifest_from_lineage(lineage))
    # Drop _evolve so the Build tab's manifest delta shows authoring changes.
    manifest.pop("_evolve", None)
    difficulty = lineage.get("curriculum_difficulty")
    curriculum = dict(difficulty) if isinstance(difficulty, Mapping) else {}
    solve_cost = lineage.get("world_difficulty")
    return {
        "id": snapshot.snapshot_id,
        "parent_id": _parent_snapshot_id(lineage),
        "builder_summary": _evolve_summary(evolve),
        "manifest": manifest,
        "pack": {"id": lineage.get("pack"), "version": lineage.get("pack_version")},
        "curriculum": curriculum,
        "world_difficulty": float(solve_cost)
        if isinstance(solve_cost, int | float)
        else None,
        "evolve": dict(evolve) if evolve else None,
        "graph": graph_projection(snapshot.graph),
    }


def _task_to_dict(task: TaskSpec) -> dict[str, object]:
    # Mirrors ``snapshot_to_dict`` in ``admit.py`` so the dashboard
    # doesn't have to import the snapshot serializer for one helper.
    out: dict[str, object] = {
        "id": task.id,
        "instruction": task.instruction,
        "entrypoints": list(task.entrypoints),
        "goal_nodes": list(task.goal_nodes),
        "feasibility_check": task.feasibility_check,
        "success_check": task.success_check,
    }
    if task.meta:
        out["meta"] = dict(task.meta)
    return out
