"""Base errors a pack and the OpenRange runtime both raise/catch."""

from __future__ import annotations


class OpenRangeError(Exception):
    pass


class ManifestError(OpenRangeError):
    pass


class PackError(OpenRangeError):
    pass


class LLMError(OpenRangeError):
    pass


class LLMRequestError(LLMError):
    pass


class LLMBackendError(LLMError):
    """Carries ``returncode`` so callers can distinguish a crash from a
    timeout from a non-zero exit."""

    def __init__(self, message: str, *, returncode: int | None = None) -> None:
        super().__init__(message)
        self.returncode = returncode


class NPCError(OpenRangeError):
    pass


class AgentBackendError(OpenRangeError):
    pass
