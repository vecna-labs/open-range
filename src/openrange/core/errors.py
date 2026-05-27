"""OpenRange core errors."""

from __future__ import annotations


class OpenRangeError(Exception):
    pass


class ManifestError(OpenRangeError):
    pass


class PackError(OpenRangeError):
    pass


class AdmissionError(OpenRangeError):
    pass


class StoreError(OpenRangeError):
    pass


class EpisodeRuntimeError(OpenRangeError):
    """Raised by the runtime plumbing — distinct from :class:`AdmissionError`
    (a domain signal that a candidate world failed admission) and
    :class:`EpisodeError` (the in-flight episode lifecycle signal)."""
