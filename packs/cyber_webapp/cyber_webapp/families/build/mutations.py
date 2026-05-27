"""Bug-injecting transforms applied to a reference handler at admission time
to prove the contract distinguishes correct from broken. Never shown to the
agent; never applied to agent-submitted source.
"""

from __future__ import annotations


def api_wrong_field_name(source: str) -> str:
    """Rename the response's ``"items"`` field to ``"results"``.

    The api list contract requires field ``items`` so the mutated source
    fails every items-presence case.
    """
    return source.replace('"items"', '"results"', 1)
