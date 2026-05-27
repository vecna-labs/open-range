from __future__ import annotations

import importlib.metadata
from collections.abc import Iterator

from openrange.core.errors import OpenRangeError


def iter_entry_points(
    group: str,
    *,
    error_cls: type[OpenRangeError],
    kind: str,
) -> Iterator[tuple[str, object]]:
    # ``importlib.metadata.entry_points`` is looked up dynamically (not
    # bound at import) so tests can monkeypatch it.
    for entry_point in importlib.metadata.entry_points(group=group):
        try:
            value = entry_point.load()
        except Exception as exc:  # noqa: BLE001
            raise error_cls(
                f"failed to load {kind} entry point {entry_point.name!r}: {exc}",
            ) from exc
        yield entry_point.name, value
