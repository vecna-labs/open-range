"""Runtime-only errors. Base errors live in ``openrange_pack_sdk``."""

from __future__ import annotations

from openrange_pack_sdk import OpenRangeError


class AdmissionError(OpenRangeError):
    pass


class StoreError(OpenRangeError):
    pass


class EpisodeRuntimeError(OpenRangeError):
    """Raised by the runtime plumbing — distinct from :class:`AdmissionError`
    (a domain signal that a candidate world failed admission) and
    :class:`EpisodeError` (the in-flight episode lifecycle signal)."""
