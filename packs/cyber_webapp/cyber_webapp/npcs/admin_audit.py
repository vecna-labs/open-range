from __future__ import annotations

from collections.abc import Mapping

from openrange_pack_sdk import NPC

from cyber_webapp.npcs import _HTTPCadenceNPC


class AdminAudit(_HTTPCadenceNPC):
    def __init__(
        self,
        *,
        cadence_ticks: int = 5,
        audit_path: str = "/openapi.json",
        timeout_seconds: float = 1.0,
    ) -> None:
        super().__init__(cadence_ticks=cadence_ticks)
        if not audit_path:
            raise ValueError("audit_path must be non-empty")
        self._audit_path = audit_path
        self._timeout = timeout_seconds

    def _next_path(self) -> str:
        return self._audit_path


def factory(config: Mapping[str, object]) -> NPC:
    cadence_raw = config.get("cadence_ticks", 5)
    audit_path_raw = config.get("audit_path", "/openapi.json")
    timeout_raw = config.get("timeout_seconds", 1.0)
    if not isinstance(cadence_raw, int):
        raise ValueError("cadence_ticks must be an int")
    if not isinstance(audit_path_raw, str):
        raise ValueError("audit_path must be a string")
    if not isinstance(timeout_raw, int | float):
        raise ValueError("timeout_seconds must be a number")
    return AdminAudit(
        cadence_ticks=cadence_raw,
        audit_path=audit_path_raw,
        timeout_seconds=float(timeout_raw),
    )
