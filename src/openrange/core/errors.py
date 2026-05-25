"""OpenRange core errors."""

from __future__ import annotations


class OpenRangeError(Exception):
    """Base OpenRange error."""


class ManifestError(OpenRangeError):
    """Raised when a manifest is invalid."""


class PackError(OpenRangeError):
    """Raised when a pack cannot build an admissible world."""


class AdmissionError(OpenRangeError):
    """Raised when generated world artifacts fail admission."""


class StoreError(OpenRangeError):
    """Raised when snapshots cannot be loaded from storage."""


class EpisodeRuntimeError(OpenRangeError):
    """Raised when the runtime convenience layer cannot proceed.

    Distinct from ``AdmissionError`` (which is a domain-level "the
    candidate world failed admission" signal) and ``EpisodeError``
    (which is the in-flight episode lifecycle signal): this one is
    what ``OpenRangeRun.build`` / ``__main__.py`` raise when the
    pack/admission/runtime plumbing itself misbehaves. The old
    ``runtime_helpers.EpisodeRuntimeError`` lives at this path now
    so the parallel-path module can be deleted in Phase 4 without
    breaking the user-facing seam.
    """
