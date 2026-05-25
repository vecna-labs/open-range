"""OpenRange core errors.

Minimum surface for the v0.1.0 foundation. Subclasses come back when the
runtime layer re-introduces the code paths that raise them — only one
subclass (`PackError`) is needed today, signalling pack-contract bugs
that admission can't recover from.
"""

from __future__ import annotations


class OpenRangeError(Exception):
    """Base for every error OpenRange raises."""


class PackError(OpenRangeError):
    """Raised when a pack violates the Pack/Builder/TaskFamily contract.

    Distinct from `AdmissionFailure` (which is a returned VALUE describing
    a candidate world that didn't admit): a `PackError` is a programming
    bug in the pack, not a recoverable admission failure.
    """
