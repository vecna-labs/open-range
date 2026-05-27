"""NPC registry — discovers NPC factories via entry points.

The ``NPC`` / ``AgentNPC`` ABCs and the ``NPCError`` exception live in
``openrange_pack_sdk``. This module owns only the runtime registry.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

from openrange_pack_sdk import NPC, NPCError

NPC_ENTRY_POINT_GROUP = "openrange.npcs"

NPCFactory = Callable[[Mapping[str, object]], NPC]


class NPCRegistry:
    """Registry of NPC factories by id.

    ``autodiscover=False`` (default) gives tests a clean slate; the
    global ``NPCS`` autodiscovers from the ``openrange.npcs`` entry-point
    group on first access.
    """

    def __init__(self, *, autodiscover: bool = False) -> None:
        self._factories: dict[str, NPCFactory] = {}
        self._autodiscover = autodiscover
        self._discovered = False

    def register(self, npc_id: str, factory: NPCFactory) -> None:
        self._factories[npc_id] = factory

    def resolve(self, npc_id: str, config: Mapping[str, object]) -> NPC:
        self._ensure_discovered()
        try:
            factory = self._factories[npc_id]
        except KeyError as exc:
            raise NPCError(f"unknown NPC {npc_id!r}") from exc
        npc = factory(config)
        if not isinstance(npc, NPC):
            raise NPCError(
                f"NPC factory {npc_id!r} did not return an NPC instance",
            )
        return npc

    def ids(self) -> tuple[str, ...]:
        self._ensure_discovered()
        return tuple(sorted(self._factories))

    def discover(self) -> None:
        self._ensure_discovered(force=True)

    def _ensure_discovered(self, *, force: bool = False) -> None:
        if not self._autodiscover and not force:
            return
        if self._discovered and not force:
            return
        self._discovered = True
        from openrange.core._registry import iter_entry_points

        for name, value in iter_entry_points(
            NPC_ENTRY_POINT_GROUP,
            error_cls=NPCError,
            kind="NPC",
        ):
            if name in self._factories and not force:
                continue
            if not callable(value):
                raise NPCError(
                    f"entry point {name!r} did not yield a callable",
                )
            self._factories[name] = value


NPCS = NPCRegistry(autodiscover=True)


def resolve_manifest_npcs(
    npc_entries: tuple[Mapping[str, object], ...],
    *,
    registry: NPCRegistry | None = None,
) -> list[NPC]:
    reg = registry if registry is not None else NPCS
    npcs: list[NPC] = []
    for entry in npc_entries:
        npc_type = entry.get("type")
        if not isinstance(npc_type, str) or not npc_type:
            raise NPCError("manifest npc entry must carry a non-empty 'type'")
        count_raw = entry.get("count", 1)
        if not isinstance(count_raw, int) or count_raw < 0:
            raise NPCError(
                f"manifest npc entry 'count' must be a non-negative int "
                f"(got {count_raw!r})",
            )
        config_raw = entry.get("config", {})
        if not isinstance(config_raw, Mapping):
            raise NPCError(
                f"manifest npc entry 'config' must be a mapping for {npc_type!r}",
            )
        for index in range(count_raw):
            # ``_replication_suffix`` mirrors the dashboard's row-id
            # convention (``f"{name}-{index+1}"`` when count > 1, else
            # bare): a factory that consumes this aligns the NPC's
            # ``actor_id`` with the dashboard row id for free.
            slot_config: Mapping[str, object]
            if count_raw > 1:
                slot_config = dict(config_raw, _replication_suffix=f"-{index + 1}")
            else:
                slot_config = config_raw
            npcs.append(reg.resolve(npc_type, slot_config))
    return npcs
