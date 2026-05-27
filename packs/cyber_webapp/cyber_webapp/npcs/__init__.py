from __future__ import annotations

from abc import abstractmethod
from collections.abc import Mapping
from typing import Any, cast

from openrange.npc import NPC


class _HTTPCadenceNPC(NPC):
    def __init__(self, *, cadence_ticks: int) -> None:
        if cadence_ticks < 1:
            raise ValueError("cadence_ticks must be >= 1")
        self._cadence_ticks = cadence_ticks
        self._cooldown = 0

    @abstractmethod
    def _next_path(self) -> str: ...

    def step(self, interface: Mapping[str, Any]) -> None:
        if self._cooldown > 0:
            self._cooldown -= 1
            return
        self._cooldown = self._cadence_ticks - 1
        http_get = interface.get("http_get")
        if http_get is None:
            return
        try:
            cast(Any, http_get)(self._next_path())
        except Exception:  # noqa: BLE001 — NPC failures are silent
            return
