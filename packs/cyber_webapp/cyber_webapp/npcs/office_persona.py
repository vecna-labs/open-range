from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import random
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any, ClassVar, cast

from openrange_pack_sdk import NPC, AgentBackend

_log = logging.getLogger(__name__)

_DEFAULT_TONE = "warm, professional"


def _stable_home_index(name: str) -> int:
    # SHA1 because Python's built-in hash is per-process randomized;
    # the dashboard seats personas by ``home_index`` and a stable
    # seating arrangement keeps screen recordings reproducible.
    return int(hashlib.sha1(name.encode()).hexdigest()[:8], 16)


_SYSTEM_TEMPLATE = (
    "You are {name}, {title} on the {role} team. Your tone is {tone}. "
    "You are quietly going about your workday at a small company. "
    "Each turn you reply with a single JSON object — nothing else, no "
    "code fences, no preamble — of the form: "
    '{{"speak": "<one short in-character line>", "visit": "/<plausible page>"}}. '
    "Pick a page that sounds plausible (/, /search?q=alpha, /openapi.json, "
    "/docs, /api/v1/users, etc). Don't try to break anything — you're a "
    "normal user. Keep speech to one sentence."
)


class OfficePersona(NPC):
    """Persona-faithful single-shot LLM-backed office worker."""

    requires_llm: ClassVar[bool] = True

    def __init__(
        self,
        *,
        name: str,
        role: str = "engineer",
        title: str = "",
        tone: str = _DEFAULT_TONE,
        colleagues: Sequence[str] = (),
        home: str | None = None,
        cadence_ticks: int = 6,
        agent_backend: AgentBackend | None = None,
        seed: int | None = None,
    ) -> None:
        if not name:
            raise ValueError("name must be a non-empty string")
        if not role:
            raise ValueError("role must be a non-empty string")
        if cadence_ticks < 1:
            raise ValueError("cadence_ticks must be >= 1")
        self._name = name
        self._role = role
        self._title = title or role.replace("_", " ").title()
        self._tone = tone or _DEFAULT_TONE
        self._colleagues = tuple(c for c in colleagues if c and c != name)
        self._home = home
        self._actor_id = name
        self._home_index = _stable_home_index(name)
        rng_seed = seed if seed is not None else self._home_index
        self._rng = random.Random(rng_seed)
        self._cadence_ticks = cadence_ticks
        self._cooldown = 0
        self._record: Callable[..., None] | None = None
        self._backend_override = agent_backend
        self._runtime_backend: AgentBackend | None = None
        self._agent: Any = None
        self._broken = False
        self._system_prompt = _SYSTEM_TEMPLATE.format(
            name=name,
            role=role,
            title=self._title,
            tone=self._tone,
        )
        # Preflight a constructor-supplied backend immediately so an
        # absent SDK / binary fails at manifest resolution time, not on
        # the first cadence tick.
        if agent_backend is not None:
            try:
                agent_backend.preflight()
            except Exception as exc:
                self._mark_broken(f"backend preflight failed: {exc}", exc=exc)

    def start(self, context: Mapping[str, Any]) -> None:
        record = context.get("record_action")
        self._record = cast(Callable[..., None], record) if callable(record) else None
        # Presence event so the dashboard can seat the persona at its desk
        # before the first acting tick — even if the backend is broken.
        if self._record is not None:
            self._record(
                {
                    "present": True,
                    "actor_kind": "npc",
                    "home_index": self._home_index,
                    "display_name": self._name,
                    "role": self._role,
                    "title": self._title,
                    "tone": self._tone,
                    "colleagues": list(self._colleagues),
                },
            )
        if self._broken:
            return
        runtime_backend = context.get("agent_backend")
        if runtime_backend is not None:
            self._runtime_backend = cast(AgentBackend, runtime_backend)
        backend = self._backend_override or self._runtime_backend
        if backend is None:
            self._mark_broken(
                "no AgentBackend configured "
                "(set RunConfig.npc_agent_backend or pass agent_backend "
                "to the NPC constructor)",
            )
            return
        if self._backend_override is None:
            try:
                backend.preflight()
            except Exception as exc:
                self._mark_broken(
                    f"runtime backend preflight failed: {exc}",
                    exc=exc,
                )

    def step(self, interface: Mapping[str, Any]) -> None:
        if self._broken:
            return
        if self._cooldown > 0:
            self._cooldown -= 1
            return
        self._cooldown = self._cadence_ticks - 1
        backend = self._backend_override or self._runtime_backend
        if backend is None:
            return  # already broken-marked at start; defensive
        if self._agent is None:
            try:
                # Single-shot session: no tools. Every AgentBackend in
                # OpenRange (Codex, Strands, fakes) accepts an empty
                # tool list, so this works against all of them.
                self._agent = backend.build_agent(
                    system_prompt=self._system_prompt,
                    tools=[],
                )
            except Exception as exc:
                self._mark_broken(
                    f"failed to construct agent: {exc}",
                    exc=exc,
                )
                self._agent = None
                return
        try:
            result = self._agent(self._user_prompt())
        except Exception:
            # Transient (rate limits, timeouts) — DEBUG only.
            _log.debug(
                "OfficePersona %s tick failed; will retry next cadence window",
                self._name,
                exc_info=True,
            )
            return
        text = self._extract_text(result)
        speech, visit = self._extract_payload(text)
        if not speech and self._colleagues:
            speech = self._fallback_speech()
        # Do the HTTP visit ourselves — no tool dispatch needed.
        http_get = interface.get("http_get")
        if visit and http_get is not None:
            # Swallow visit failures — a 404 or transient network error
            # shouldn't sink the persona's tick.
            with contextlib.suppress(Exception):
                cast(Any, http_get)(visit)
        # Record the persona event so the dashboard renders the bubble.
        if self._record is None or not speech:
            return
        action: dict[str, object] = {
            "actor_kind": "npc",
            "speak": speech,
            "home_index": self._home_index,
            "display_name": self._name,
            "role": self._role,
        }
        if visit:
            action["visit"] = visit
        if self._colleagues and self._rng.random() < 0.45:
            action["move"] = "wandering"
            action["target_name"] = self._rng.choice(self._colleagues)
        self._record(action, target=self._home)

    def stop(self) -> None:
        self._agent = None

    def _mark_broken(self, reason: str, *, exc: BaseException | None = None) -> None:
        if self._broken:
            return
        self._broken = True
        self.broken_reason = reason
        # exc_info=exc (not =True) — for broken-by-config cases there
        # is no in-flight exception, and ``=True`` would grab whatever
        # ``sys.exc_info`` returns from an unrelated traceback.
        _log.warning(
            "OfficePersona %s is permanently broken (%s); "
            "the rest of the episode runs without it",
            self._name,
            reason,
            exc_info=exc,
        )

    def _user_prompt(self) -> str:
        colleague_hint = (
            f"Nearby colleagues you might address: {', '.join(self._colleagues)}."
            if self._colleagues
            else "You're at your desk; speak to yourself or to no one in particular."
        )
        return (
            f"It's a quiet moment in the workday. {colleague_hint} "
            "Reply with the JSON object as instructed in the system prompt — "
            "nothing else."
        )

    def _extract_text(self, result: object) -> str:
        # Backends return varied shapes: plain string (Codex), AgentResult
        # (Strands), or a dict. Best-effort across them all.
        if isinstance(result, str):
            return result
        if isinstance(result, Mapping):
            for key in ("text", "message", "content", "output"):
                value = result.get(key)
                if isinstance(value, str):
                    return value
        for attr in ("text", "message", "content", "output"):
            value = getattr(result, attr, None)
            if isinstance(value, str):
                return value
        return ""

    def _extract_payload(self, text: str) -> tuple[str, str]:
        if not text:
            return "", ""
        stripped = text.strip()
        if stripped.startswith("```"):
            stripped = re.sub(r"^```[a-zA-Z]*\n?|```$", "", stripped).strip()
        for candidate in self._json_candidates(stripped):
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, Mapping):
                speak_raw = parsed.get("speak") or parsed.get("speech") or ""
                visit_raw = parsed.get("visit") or parsed.get("path") or ""
                speak = str(speak_raw).strip()[:140] if speak_raw else ""
                visit = str(visit_raw).strip()[:200] if visit_raw else ""
                if visit and not visit.startswith("/"):
                    visit = "/" + visit.lstrip()
                if speak or visit:
                    return speak, visit
        # Loose fallback for unstructured replies.
        first = re.split(r"(?<=[.!?])\s", stripped, maxsplit=1)[0].strip()
        speak = first[:140]
        visit_match = re.search(r"(/\S+)", stripped)
        visit = visit_match.group(1)[:200] if visit_match else ""
        return speak, visit

    def _json_candidates(self, text: str) -> Sequence[str]:
        candidates: list[str] = []
        if text.startswith("{"):
            candidates.append(text)
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end > start:
            candidates.append(text[start : end + 1])
        return candidates

    def _fallback_speech(self) -> str:
        return f"Heads-down on {self._role.replace('_', ' ')} stuff."


def factory(config: Mapping[str, object]) -> NPC:
    name_raw = config.get("name")
    role_raw = config.get("role", "engineer")
    title_raw = config.get("title", "")
    tone_raw = config.get("tone", _DEFAULT_TONE)
    colleagues_raw = config.get("colleagues", ())
    home_raw = config.get("home")
    cadence_raw = config.get("cadence_ticks", 6)
    seed_raw = config.get("seed")
    suffix_raw = config.get("_replication_suffix", "")
    if not isinstance(name_raw, str) or not name_raw:
        raise ValueError("name must be a non-empty string")
    if not isinstance(role_raw, str) or not role_raw:
        raise ValueError("role must be a non-empty string")
    if not isinstance(title_raw, str):
        raise ValueError("title must be a string")
    if not isinstance(tone_raw, str):
        raise ValueError("tone must be a string")
    if not isinstance(colleagues_raw, list | tuple) or not all(
        isinstance(item, str) for item in colleagues_raw
    ):
        raise ValueError("colleagues must be a list of strings")
    if home_raw is not None and not isinstance(home_raw, str):
        raise ValueError("home must be a string or unset")
    if not isinstance(cadence_raw, int):
        raise ValueError("cadence_ticks must be an int")
    if seed_raw is not None and not isinstance(seed_raw, int):
        raise ValueError("seed must be an int or unset")
    if not isinstance(suffix_raw, str):
        raise ValueError("_replication_suffix must be a string")
    return OfficePersona(
        name=name_raw + suffix_raw,
        role=role_raw,
        title=title_raw,
        tone=tone_raw,
        colleagues=tuple(colleagues_raw),
        home=home_raw,
        cadence_ticks=cadence_raw,
        seed=seed_raw,
    )
