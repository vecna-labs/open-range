from __future__ import annotations

from collections.abc import Mapping

from cyber_webapp.npcs import _HTTPCadenceNPC
from openrange.npc import NPC

_DEFAULT_PATHS: tuple[str, ...] = ("/",)


class BrowsingUser(_HTTPCadenceNPC):
    def __init__(
        self,
        *,
        cadence_ticks: int = 2,
        paths: tuple[str, ...] = _DEFAULT_PATHS,
        timeout_seconds: float = 1.0,
    ) -> None:
        super().__init__(cadence_ticks=cadence_ticks)
        if not paths:
            raise ValueError("paths must be non-empty")
        self._paths = tuple(paths)
        self._timeout = timeout_seconds
        self._index = 0

    def _next_path(self) -> str:
        path = self._paths[self._index % len(self._paths)]
        self._index += 1
        return path


def factory(config: Mapping[str, object]) -> NPC:
    cadence_raw = config.get("cadence_ticks", 2)
    paths_raw = config.get("paths", list(_DEFAULT_PATHS))
    timeout_raw = config.get("timeout_seconds", 1.0)
    if not isinstance(cadence_raw, int):
        raise ValueError("cadence_ticks must be an int")
    if not isinstance(paths_raw, list | tuple) or not all(
        isinstance(p, str) for p in paths_raw
    ):
        raise ValueError("paths must be a list of strings")
    if not isinstance(timeout_raw, int | float):
        raise ValueError("timeout_seconds must be a number")
    return BrowsingUser(
        cadence_ticks=cadence_raw,
        paths=tuple(paths_raw),
        timeout_seconds=float(timeout_raw),
    )
