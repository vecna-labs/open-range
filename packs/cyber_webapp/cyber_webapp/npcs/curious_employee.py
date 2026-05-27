from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast

from openrange_pack_sdk import NPC, AgentNPC

_DEFAULT_SYSTEM_PROMPT = (
    "You are an internal employee at a small company, casually browsing "
    "the company's intranet webapp out of curiosity during a slow "
    "afternoon. On each turn pick one or two pages that sound interesting "
    "and visit them with the visit_url tool. Keep the request paths short "
    "and plausible (e.g. /, /search?q=alpha, /openapi.json). Don't try to "
    "break anything — you're a normal user, not an attacker. Respond with "
    "one short sentence describing what you looked at."
)


class CuriousEmployee(AgentNPC):
    def _build_tools(
        self,
        interface: Mapping[str, Any],
    ) -> Sequence[Callable[..., Any]]:
        from strands import tool

        http_get = interface.get("http_get")
        if http_get is None:
            return ()

        @tool
        def visit_url(path: str) -> str:
            """Visit a path on the company webapp and return a short snippet.

            Args:
                path: URL path on the webapp (e.g. ``/`` or
                    ``/search?q=alpha``). Must start with ``/``.
            """
            try:
                body = cast(Any, http_get)(path)
            except Exception as exc:  # noqa: BLE001 — surface to the LLM
                return f"request failed: {exc}"
            if isinstance(body, bytes):
                text = body.decode(errors="replace")
            else:
                text = str(body)
            return text[:1500]

        return [visit_url]


def factory(config: Mapping[str, object]) -> NPC:
    cadence_raw = config.get("cadence_ticks", 5)
    prompt_raw = config.get("system_prompt", _DEFAULT_SYSTEM_PROMPT)
    if not isinstance(cadence_raw, int):
        raise ValueError("cadence_ticks must be an int")
    if not isinstance(prompt_raw, str) or not prompt_raw:
        raise ValueError("system_prompt must be a non-empty string")
    return CuriousEmployee(
        system_prompt=prompt_raw,
        cadence_ticks=cadence_raw,
    )
