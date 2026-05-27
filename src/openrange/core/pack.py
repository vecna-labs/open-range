"""Pack registry — looks up pack instances by id, discovers via entry points.

The Pack / Builder / TaskFamily / RuntimeHandle Protocols and the value
types that cross the boundary live in ``openrange_pack_sdk``. This module
owns only the runtime registry that openrange itself depends on.
"""

from __future__ import annotations

from openrange_pack_sdk import Pack, PackError

PACK_ENTRY_POINT_GROUP = "openrange.packs"


class PackRegistry:
    """Registry of Pack instances by id, discovered via the
    ``openrange.packs`` entry-point group when ``autodiscover=True``."""

    def __init__(self, *, autodiscover: bool = False) -> None:
        self._packs: dict[str, Pack] = {}
        self._autodiscover = autodiscover
        self._discovered = False

    def register(self, pack: Pack) -> None:
        self._packs[pack.id] = pack

    def resolve(self, pack_id: str) -> Pack:
        self._ensure_discovered()
        try:
            return self._packs[pack_id]
        except KeyError as exc:
            raise PackError(f"unknown pack {pack_id!r}") from exc

    def resolve_class(self, pack_id: str) -> type[Pack]:
        return type(self.resolve(pack_id))

    def ids(self) -> tuple[str, ...]:
        self._ensure_discovered()
        return tuple(sorted(self._packs))

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
            PACK_ENTRY_POINT_GROUP,
            error_cls=PackError,
            kind="pack",
        ):
            if name in self._packs and not force:
                continue
            pack = value() if callable(value) else value
            if not isinstance(pack, Pack):
                raise PackError(
                    f"entry point {name!r} did not return a Pack",
                )
            if pack.id != name:
                raise PackError(
                    f"entry point name {name!r} does not match pack.id {pack.id!r}",
                )
            self._packs[pack.id] = pack


PACKS = PackRegistry(autodiscover=True)
