"""Legacy typed Manifest — a transitional backward-compat shim.

The typed `Manifest` concept is going away. In the new pack/admission
contracts (see `openrange.core.contracts`), `Manifest = Mapping[str, Any]`:
a free-form dict the pack interprets, never branched on by core.

This module still exists for the brief window during which the legacy
runtime modules (`openrange.core.builder`, `openrange.core.snapshot`,
`openrange.core.pack`) — all scheduled for deletion in Phase 4 of the
runtime migration — still depend on the typed structures it ships.
Once those consumers are gone, this file goes with them.

What changed in this slim pass:
  - YAML loading is removed. Per the migration plan, manifest-as-YAML
    is a future-CLI concern, not part of core. `Manifest.load(...)`
    therefore no longer accepts `str | Path`; callers must pass an
    in-memory mapping (or an existing `Manifest`).
  - No public re-exports. New code imports `Manifest` from
    `openrange.core.contracts` (where it is a `Mapping[str, Any]` alias).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Literal, cast

from openrange.core.errors import ManifestError

PackSourceKind = Literal["builtin", "path", "git", "container"]
WorldMode = Literal["simulation", "emulation"]
TickMode = Literal["auto", "manual"]


@dataclass(frozen=True, slots=True)
class TickConfig:
    """How world time advances during episodes (legacy).

    Kept for backward compat with the pre-refactor `RuntimeConfig`
    pathway. New packs decide their own tick semantics — there is no
    longer a single OpenRange-owned tick contract.
    """

    mode: TickMode = "auto"
    rate_hz: float = 1.0

    @classmethod
    def from_value(cls, value: object) -> TickConfig:
        if value is None:
            return cls()
        if not isinstance(value, Mapping):
            raise ManifestError("'runtime.tick' must be a mapping")
        mode = value.get("mode", "auto")
        if mode not in {"auto", "manual"}:
            raise ManifestError("'runtime.tick.mode' must be 'auto' or 'manual'")
        rate = value.get("rate_hz", 1.0)
        if not isinstance(rate, int | float) or rate <= 0:
            raise ManifestError("'runtime.tick.rate_hz' must be a positive number")
        return cls(cast(TickMode, mode), float(rate))

    def as_dict(self) -> dict[str, object]:
        return {"mode": self.mode, "rate_hz": self.rate_hz}


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    """Manifest-level runtime knobs (legacy)."""

    tick: TickConfig = field(default_factory=TickConfig)

    @classmethod
    def from_value(cls, value: object) -> RuntimeConfig:
        if value is None:
            return cls()
        if not isinstance(value, Mapping):
            raise ManifestError("'runtime' must be a mapping")
        return cls(tick=TickConfig.from_value(value.get("tick")))

    def as_dict(self) -> dict[str, object]:
        return {"tick": self.tick.as_dict()}


@dataclass(frozen=True, slots=True)
class PackSource:
    """Where a pack is loaded from (legacy)."""

    kind: PackSourceKind = "builtin"
    uri: str | None = None

    @classmethod
    def from_value(cls, value: object) -> PackSource:
        if value is None:
            return cls()
        if not isinstance(value, Mapping):
            raise ManifestError("'pack.source' must be a mapping")
        kind = value.get("kind", "builtin")
        uri = value.get("uri")
        if kind not in {"builtin", "path", "git", "container"}:
            raise ManifestError("'pack.source.kind' is invalid")
        if uri is not None and not isinstance(uri, str):
            raise ManifestError("'pack.source.uri' must be a string")
        return cls(cast(PackSourceKind, kind), uri)

    def as_dict(self) -> dict[str, object]:
        result: dict[str, object] = {"kind": self.kind}
        if self.uri is not None:
            result["uri"] = self.uri
        return result


@dataclass(frozen=True, slots=True)
class PackRef:
    """Reference to a pack in a manifest (legacy)."""

    id: str
    source: PackSource = field(default_factory=PackSource)
    options: Mapping[str, object] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> PackRef:
        pack_id = data.get("id")
        if not isinstance(pack_id, str) or not pack_id:
            raise ManifestError("'pack.id' must be a non-empty string")
        options = data.get("options", {})
        if not isinstance(options, Mapping):
            raise ManifestError("'pack.options' must be a mapping")
        return cls(
            id=pack_id,
            source=PackSource.from_value(data.get("source")),
            options=MappingProxyType(dict(options)),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "source": self.source.as_dict(),
            "options": dict(self.options),
        }


@dataclass(frozen=True, slots=True)
class Manifest:
    """Legacy typed manifest.

    DEPRECATED — see module docstring. New code annotates against
    `openrange.core.contracts.Manifest`, which is `Mapping[str, Any]`.
    `Manifest.load(...)` no longer accepts `str | Path`; YAML loading
    is no longer part of core.
    """

    world: Mapping[str, object]
    pack: PackRef
    builder: str | None = None
    mode: WorldMode = "simulation"
    npc: tuple[Mapping[str, object], ...] = ()
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    @classmethod
    def load(cls, manifest: str | Path | Mapping[str, object] | Manifest) -> Manifest:
        """Normalize an in-memory manifest to a `Manifest` instance.

        YAML / path loading was removed: this method no longer reads a
        file. The `str | Path` arms remain in the signature only because
        legacy callers (Phase-4-doomed `openrange.core.builder`) still
        spell those types in their own signatures; passing a path-like
        now raises `ManifestError` immediately so the regression is loud
        rather than silent.
        """
        if isinstance(manifest, Manifest):
            return manifest
        if isinstance(manifest, str | Path):
            raise ManifestError(
                "Manifest.load no longer accepts a path — parse the file "
                "into a mapping in a CLI layer and pass the mapping in. "
                "See openrange.core.contracts.Manifest for the new shape."
            )
        return cls.from_mapping(manifest)

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> Manifest:
        world = data.get("world")
        pack = data.get("pack")
        if not isinstance(world, Mapping):
            raise ManifestError("'world' must be a mapping")
        if not isinstance(pack, Mapping):
            raise ManifestError("'pack' must be a mapping")
        mode = data.get("mode", "simulation")
        if mode not in {"simulation", "emulation"}:
            raise ManifestError("'mode' must be 'simulation' or 'emulation'")
        npc = data.get("npc", ())
        if not isinstance(npc, list | tuple) or not all(
            isinstance(item, Mapping) for item in npc
        ):
            raise ManifestError("'npc' must be a list of mappings")
        builder = data.get("builder")
        if builder is not None and (not isinstance(builder, str) or not builder):
            raise ManifestError("'builder' must be a non-empty string when present")
        return cls(
            world=MappingProxyType(dict(world)),
            pack=PackRef.from_mapping(cast(Mapping[str, object], pack)),
            builder=builder,
            mode=cast(WorldMode, mode),
            npc=tuple(MappingProxyType(dict(item)) for item in npc),
            runtime=RuntimeConfig.from_value(data.get("runtime")),
        )

    def as_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "world": dict(self.world),
            "pack": self.pack.as_dict(),
            "mode": self.mode,
            "npc": [dict(item) for item in self.npc],
            "runtime": self.runtime.as_dict(),
        }
        if self.builder is not None:
            result["builder"] = self.builder
        return result
