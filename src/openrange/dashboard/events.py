"""Dashboard event types, in-memory bridge, and on-disk persistence."""

from __future__ import annotations

import asyncio
import json
import queue
import threading
from collections import deque
from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openrange.dashboard.view import DashboardView


def json_safe(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [json_safe(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class DashboardEvent:
    id: str
    type: str
    actor: str
    target: str
    time: float
    data: Mapping[str, object]

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "type": self.type,
            "actor": self.actor,
            "target": self.target,
            "time": self.time,
            "data": dict(self.data),
        }


class EventBridge:
    def __init__(self, *, max_buffer: int = 200) -> None:
        if max_buffer <= 0:
            raise ValueError("max_buffer must be positive")
        self._events: deque[DashboardEvent] = deque(maxlen=max_buffer)
        self._lock = threading.Lock()
        self._subscribers: list[
            tuple[asyncio.Queue[DashboardEvent | None], asyncio.AbstractEventLoop]
        ] = []
        self._sync_subscribers: list[queue.SimpleQueue[DashboardEvent | None]] = []

    def push(self, event: DashboardEvent) -> None:
        with self._lock:
            self._events.append(event)
            subscribers = tuple(self._subscribers)
            sync_subscribers = tuple(self._sync_subscribers)
        for event_queue, loop in subscribers:
            loop.call_soon_threadsafe(event_queue.put_nowait, event)
        for sync_queue in sync_subscribers:
            sync_queue.put(event)

    def snapshot_buffer(self) -> tuple[DashboardEvent, ...]:
        with self._lock:
            return tuple(self._events)

    async def subscribe(self) -> AsyncIterator[DashboardEvent]:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[DashboardEvent | None] = asyncio.Queue(maxsize=500)
        with self._lock:
            backlog = tuple(self._events)
            self._subscribers.append((queue, loop))
        try:
            for event in backlog:
                yield event
            while True:
                queued = await queue.get()
                if queued is None:
                    return
                yield queued
        finally:
            with self._lock:
                self._subscribers = [
                    (candidate, candidate_loop)
                    for candidate, candidate_loop in self._subscribers
                    if candidate is not queue
                ]

    def subscribe_sync(self) -> Iterator[DashboardEvent]:
        event_queue: queue.SimpleQueue[DashboardEvent | None] = queue.SimpleQueue()
        with self._lock:
            backlog = tuple(self._events)
            self._sync_subscribers.append(event_queue)
        try:
            yield from backlog
            while True:
                event = event_queue.get()
                if event is None:
                    return
                yield event
        finally:
            with self._lock:
                self._sync_subscribers = [
                    candidate
                    for candidate in self._sync_subscribers
                    if candidate is not event_queue
                ]

    def close(self) -> None:
        with self._lock:
            subscribers = tuple(self._subscribers)
            sync_subscribers = tuple(self._sync_subscribers)
        for event_queue, loop in subscribers:
            loop.call_soon_threadsafe(event_queue.put_nowait, None)
        for sync_queue in sync_subscribers:
            sync_queue.put(None)


class DashboardArtifactLog:
    def __init__(
        self,
        event_log_path: str | Path,
        state_path: str | Path,
        *,
        reset: bool = False,
    ) -> None:
        self.event_log_path = Path(event_log_path)
        self.state_path = Path(state_path)
        self.event_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        if reset or not self.event_log_path.exists():
            self.event_log_path.write_text("", encoding="utf-8")
        self._lock = threading.Lock()
        self._event_count = len(self.events())
        self.write_state()

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
            with self.event_log_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(json_safe(event.as_dict()), sort_keys=True) + "\n",
                )
            self.write_state()
            return event

    def record_builder_step(
        self,
        step: str,
        data: Mapping[str, object] | None = None,
    ) -> DashboardEvent:
        return self.record_event(
            "builder_step",
            actor="builder",
            target="snapshot",
            data={**dict(data or {}), "step": step},
        )

    def events(self) -> tuple[DashboardEvent, ...]:
        return tuple(read_dashboard_events(self.event_log_path))

    def write_state(self) -> None:
        write_dashboard_state(self.state_path, self.events(), snapshot=None)


def read_dashboard_events(path: Path) -> list[DashboardEvent]:
    if not path.exists():
        return []
    events: list[DashboardEvent] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(data, Mapping):
            events.append(dashboard_event_from_mapping(data))
    return events


def read_dashboard_state(path: Path) -> Mapping[str, object]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, Mapping) else {}


def dashboard_event_from_mapping(data: Mapping[str, object]) -> DashboardEvent:
    event_id = data.get("id")
    event_type = data.get("type")
    actor = data.get("actor")
    target = data.get("target")
    time = data.get("time")
    event_data = data.get("data", {})
    return DashboardEvent(
        str(event_id),
        str(event_type),
        str(actor),
        str(target),
        float(time) if isinstance(time, int | float) else 0.0,
        MappingProxyType(dict(event_data if isinstance(event_data, Mapping) else {})),
    )


def fallback_narrate(events: Sequence[DashboardEvent]) -> str:
    if not events:
        return "No episode activity yet."
    return "\n".join(
        f"{event.actor} {event.type} {event.target}" for event in events[-5:]
    )


def write_dashboard_state(
    path: Path,
    events: Sequence[DashboardEvent],
    snapshot: DashboardView | None,
) -> None:
    event_rows = [event.as_dict() for event in events]
    turns = [dict(event.data) for event in events if event.type == "env_turn"]
    builder_steps = [
        dict(event.data) for event in events if event.type == "builder_step"
    ]
    state = (
        {
            "running": False,
            "snapshot_id": None,
            "events": event_rows,
        }
        if snapshot is None
        else snapshot.state()
    )
    data: dict[str, object] = {
        "topology": {} if snapshot is None else snapshot.topology(),
        "lineage": {} if snapshot is None else snapshot.lineage(),
        "state": state,
        "turns": turns,
        "builder": {"steps": builder_steps},
        "narration": {"narration": fallback_narrate(events)},
    }
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(json_safe(data), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
