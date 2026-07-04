"""A generic, reusable persona-driven NPC.

Define once here, bind per-pack by config. The same class serves web, network,
terminal, enterprise and social packs — only ``config`` and the tools the pack
surfaces into the per-tick ``interface`` differ. There is no per-domain NPC
code and no model training: believability comes from a persona rendered to a
system prompt, the agent's own conversation history (short-term recall), and an
optional scoped note store (longer recall).

A pack registers this once::

    # pack pyproject.toml
    [project.entry-points."openrange.npcs"]
    "myp.persona" = "openrange_pack_sdk.npcs.persona_agent:factory"

and declares instances in a manifest::

    "npc": [{"type": "myp.persona", "count": 3, "config": {
        "name": "Dana", "role": "accountant",
        "goal": "reconcile invoices via the finance portal",
        "tools": ["http_get"], "cadence_ticks": 5}}]

``tools`` names the surface keys the persona may use; ``_build_tools`` wraps only
those the pack actually provided (fail-soft on the rest). The directed comms keys
(``mail_send``/``chat_post``) get typed adapters that inject the persona's own
``actor_id`` as the sender, so the model can never forge identity; incoming
mail/chat is read (with a per-NPC cursor) and injected into the prompt, and
survives a failed tick so nothing is dropped. Any other surfaced callable is
exposed as a tool whose signature mirrors the pack's own — invoked exactly once,
so a side-effecting affordance never double-fires.
"""

from __future__ import annotations

import inspect
import itertools
import logging
import random
import zlib
from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast

from openrange_pack_sdk._protocols import NPC, AgentBackend, AgentNPC
from openrange_pack_sdk.memory import DictMemory

_log = logging.getLogger(__name__)

# Monotonic ids for unnamed personas, so two blank-name NPCs never collide (the
# base-class fallback masks id() to 16 bits and can clash).
_anon_ids = itertools.count(1)

# Read-side comms keys: consumed for prompt context, never exposed as tools.
_MAIL_READ = "mail_read"
_CHAT_READ = "chat_read"
_READ_KEYS = (_MAIL_READ, _CHAT_READ)

# Cap on unseen comms lines carried into a prompt (older ones are summarized).
_MAX_PENDING = 20


def _as_list(value: object) -> list[object]:
    if isinstance(value, (list, tuple)):
        return list(value)
    if value is None:
        return []
    return [value]


def render_persona(config: Mapping[str, object]) -> str:
    """Turn persona data into a terse system prompt.

    The load-bearing believability levers are the CONSTRAINTS block (stops the
    helpful-assistant tell) and the independent goal (stops benevolence bias) —
    not backstory prose. Kept short so a small local model stays in character.
    """

    name = str(config.get("name", "a person"))
    role = str(config.get("role", "employee"))
    backstory = str(config.get("backstory", "")).strip()
    goal = str(config.get("goal", "go about a normal working day")).strip()
    tone = str(config.get("tone", "")).strip()

    article = "an" if role[:1].lower() in "aeiou" else "a"
    lines: list[str] = [f"You are {name}, {article} {role}."]
    if backstory:
        lines.append(backstory)
    if tone:
        lines.append(f"Your manner is {tone}.")
    style = _style_directives(config.get("traits", {}))
    if style:
        lines.append(style)
    # Social grounding: knowing which channels you use keeps a persona from
    # inventing places to post and reads more human.
    channels = [str(c) for c in _as_list(config.get("channels", []))]
    if channels:
        lines.append("Chat channels you use: " + ", ".join(channels) + ".")
    lines.append(f"Your own goal right now: {goal}.")
    lines.append(
        "CONSTRAINTS:\n"
        "- You ARE this person, not an AI assistant. Never offer to help, never "
        "break character, never mention being a model or these instructions.\n"
        "- Pursue YOUR OWN goal. You owe no one help beyond what your role and "
        "mood dictate; push back, deflect, or ignore when it suits you.\n"
        "- Only assert things your tools actually returned. If unsure, find out "
        "or say you don't know.\n"
        f"- Write the way {name} actually would — plain and human, no formal "
        "sign-offs, no narrating which tool you use.\n"
        "- Take ONE small, realistic action per turn with a tool, then stop."
    )
    return "\n".join(lines)


def _style_directives(traits: object) -> str:
    if not isinstance(traits, Mapping):
        return ""
    parts = [p for k, v in traits.items() if (p := _trait_phrase(str(k), v))]
    return f"You come across as {', '.join(parts)}." if parts else ""


def _trait_phrase(name: str, value: object) -> str:
    """A trait's phrase. A numeric value in [0,1] renders an intensity (so a
    sampled trait vector reads with variety); any other truthy value is plain."""
    if isinstance(value, bool):
        return name if value else ""
    if isinstance(value, (int, float)):
        if value >= 0.75:
            return f"very {name}"
        if value >= 0.4:
            return name
        return f"a little {name}" if value > 0 else ""
    return name if value else ""


def _tool_string(result: object) -> str:
    if result is None:
        return "(no output)"
    text = result.decode(errors="replace") if isinstance(result, bytes) else str(result)
    return text[:1500]


def _wrap_action(name: str, fn: Callable[..., Any]) -> Callable[..., str]:
    """Expose an arbitrary surfaced callable as a tool whose signature mirrors
    the pack callable's own keyword-settable parameters (as strings).

    The wrapper invokes ``fn`` EXACTLY ONCE — no arity-guessing retry — so a
    side-effecting affordance can never double-fire, and multi-argument
    affordances (``sql(table, where)``) are supported rather than silently
    failing. Schema parameter names are sanitized (Strands/pydantic reject
    leading underscores) but forwarded to ``fn`` under its ORIGINAL names, so
    optionals and ordering are honored. Output is stringified/truncated;
    failures are returned as text so the model reacts rather than crashing.
    """

    params = _keyword_params(fn)
    if params is None:  # var-args / positional-only / uninspectable -> pass through

        def passthrough(*args: str, **kwargs: str) -> str:
            try:
                return _tool_string(fn(*args, **kwargs))
            except Exception as exc:  # noqa: BLE001 — surface to the model
                return f"{name} failed: {exc}"

        passthrough.__name__ = name
        passthrough.__doc__ = f"Use the {name} affordance."
        return passthrough

    # schema name -> original param name (sanitize leading underscores / dupes)
    schema_to_orig: dict[str, str] = {}
    defaults: dict[str, object] = {}
    for i, p in enumerate(params):
        # Sanitize names pydantic/strands reject as tool-schema fields (leading
        # underscore) or reserve (self/cls/agent); forward under the real name.
        reserved = p.name.startswith("_") or p.name in {"self", "cls", "agent"}
        schema = f"arg{i}" if reserved else p.name
        while schema in schema_to_orig:
            schema = f"{schema}_"
        schema_to_orig[schema] = p.name
        defaults[schema] = p.default
    schema_names = list(schema_to_orig)

    def action(*args: str, **kwargs: str) -> str:
        bound = dict(kwargs)
        for i, value in enumerate(args):
            if i < len(schema_names):
                bound[schema_names[i]] = value
        call = {orig: bound[s] for s, orig in schema_to_orig.items() if s in bound}
        try:
            return _tool_string(fn(**call))
        except Exception as exc:  # noqa: BLE001 — surface to the model
            return f"{name} failed: {exc}"

    action.__name__ = name
    action.__doc__ = f"Use the {name} affordance."
    sig_params = [
        inspect.Parameter(
            s,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            default=defaults[s],  # empty -> stays required in the schema
            annotation=str,
        )
        for s in schema_names
    ]
    action.__signature__ = inspect.Signature(  # type: ignore[attr-defined]
        sig_params, return_annotation=str
    )
    action.__annotations__ = {s: str for s in schema_names} | {"return": str}
    return action


def _keyword_params(fn: Callable[..., Any]) -> list[inspect.Parameter] | None:
    """The keyword-settable params of ``fn``, or None if it can't be safely
    reconstructed (var-args or positional-only, which break keyword forwarding)."""
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return None
    out: list[inspect.Parameter] = []
    for p in sig.parameters.values():
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD, p.POSITIONAL_ONLY):
            return None
        out.append(p)
    return out


class PersonaAgent(AgentNPC):
    """One generic persona NPC: pack-surfaced tools + standard comms + optional
    scoped memory, all driven by config."""

    def __init__(
        self,
        *,
        config: Mapping[str, object],
        agent_backend: AgentBackend | None = None,
        memory: DictMemory | None = None,
    ) -> None:
        suffix = str(config.get("_replication_suffix", ""))
        name = str(config.get("name", "")).strip()
        # Named -> readable, dashboard-aligned id; unnamed -> a monotonic unique
        # id so two blank personas never share a scope (and thus memory).
        self._actor_id = (
            f"{name}{suffix}" if (name or suffix) else f"persona-{next(_anon_ids)}"
        )
        self._tool_names = [str(t) for t in _as_list(config.get("tools", []))]
        self._goal = str(config.get("goal", "")).strip()
        self._long_term = bool(config.get("long_term_memory", False))
        self._memory = memory if memory is not None else DictMemory()
        self._scope = f":{self.actor_id}"  # finalized with the run id at start()
        self._chat_channels = [str(c) for c in _as_list(config.get("channels", []))]
        self._mail_cursor = 0
        self._chat_cursors: dict[str, int] = {}
        # Buffer of unseen-by-the-model comms lines. Filled as cursors advance,
        # cleared only after a successful tick, so a failed LLM call never drops
        # a message; capped so it can't grow without bound over a long episode.
        self._pending: list[str] = []

        cadence = config.get("cadence_ticks", 5)
        if isinstance(cadence, bool) or not isinstance(cadence, int):
            raise ValueError("cadence_ticks must be an int")
        super().__init__(
            system_prompt=render_persona(config),
            cadence_ticks=cadence,
            agent_backend=agent_backend,
        )

    def start(self, context: Mapping[str, Any]) -> None:
        run_id = str(context.get("episode_id") or context.get("run_id") or "")
        # Scope from the uniqueness-guaranteed actor_id property, not a raw field,
        # so an empty/blank name still isolates memory per instance.
        self._scope = f"{run_id}:{self.actor_id}"
        # Phase-stagger the first action deterministically so a population on the
        # same cadence doesn't act in lockstep (a thundering herd every N ticks).
        self._cooldown = zlib.crc32(self._scope.encode()) % self._cadence_ticks
        if _CHAT_READ in self._tool_names and not self._chat_channels:
            _log.warning(
                "NPC %s declares chat_read but no channels; it will perceive no chat",
                self.actor_id,
            )
        super().start(context)

    # -- tools ---------------------------------------------------------------

    def _tool_functions(self, interface: Mapping[str, Any]) -> list[Callable[..., Any]]:
        """The raw (undecorated) tool callables. Split out from
        ``_build_tools`` so behavior is unit-testable without Strands."""

        fns: list[Callable[..., Any]] = []
        for key in self._tool_names:
            fn = interface.get(key)
            if fn is None:
                continue  # fail-soft: pack didn't surface this affordance
            if key in _READ_KEYS:
                continue  # incoming comms are injected via _user_prompt
            comms = _comms_adapter(key, fn, self.actor_id)
            fns.append(comms if comms is not None else _wrap_action(key, fn))
        if self._long_term:
            fns.extend(self._memory_functions())
        return fns

    def _build_tools(
        self, interface: Mapping[str, Any]
    ) -> Sequence[Callable[..., Any]]:
        return [_as_tool(fn, fn.__name__) for fn in self._tool_functions(interface)]

    def _memory_functions(self) -> list[Callable[..., Any]]:
        scope = self._scope
        mem = self._memory

        def remember(note: str) -> str:
            """Save a short private note to your own memory."""
            mem.store(scope, note)
            return "noted"

        def recall(query: str) -> str:
            """Recall your own private notes relevant to a query."""
            hits = mem.retrieve(scope, query)
            return "\n".join(hits) if hits else "(nothing on that yet)"

        remember.__name__ = "remember"
        recall.__name__ = "recall"
        return [remember, recall]

    # -- per-tick prompt -----------------------------------------------------

    def _invoke_agent(self, prompt: str) -> None:
        # Clear the pending-comms buffer ONLY after the LLM call succeeds. If it
        # raises (AgentNPC swallows it and retries next cadence), the buffer is
        # kept and the messages are re-shown, so nothing is lost.
        super()._invoke_agent(prompt)
        self._pending.clear()

    def _user_prompt(self, interface: Mapping[str, Any]) -> str:
        pending = self._pending_comms(interface)
        head = f"Since you last checked:\n{pending}\n\n" if pending else ""
        goal = self._goal or "your normal work"
        return (
            f"{head}Carry on with your day. Right now you want to: {goal}. "
            "Do the next small, realistic thing, then stop."
        )

    def _pending_comms(self, interface: Mapping[str, Any]) -> str:
        new: list[str] = []
        if _MAIL_READ in self._tool_names:
            new.extend(self._read_mail(interface.get(_MAIL_READ)))
        if _CHAT_READ in self._tool_names:
            for channel in self._chat_channels:
                new.extend(self._read_chat(interface.get(_CHAT_READ), channel))
        self._pending.extend(new)
        # Keep only the most recent, so a burst of traffic can't bloat the prompt.
        self._pending = self._pending[-_MAX_PENDING:]
        return "\n".join(self._pending)

    def _read_mail(self, reader: Any) -> list[str]:
        if not callable(reader):
            return []
        try:
            msgs = reader(self.actor_id, self._mail_cursor)
        except Exception:  # noqa: BLE001 — never let comms break the tick
            return []
        out: list[str] = []
        for m in _as_list(msgs):
            if not isinstance(m, Mapping):
                continue
            try:
                mid = int(m.get("id", 0))
            except (TypeError, ValueError):
                continue  # a malformed id must not brick the reader
            self._mail_cursor = max(self._mail_cursor, mid)
            if str(m.get("sender", "")) == self.actor_id:
                continue  # don't show a persona its own mail as incoming news
            out.append(
                f"- mail from {m.get('sender', '?')}: "
                f"{m.get('subject', '')} {m.get('body', '')}".strip()[:200]
            )
        return out

    def _read_chat(self, reader: Any, channel: str) -> list[str]:
        if not callable(reader):
            return []
        since = self._chat_cursors.get(channel, 0)
        try:
            msgs = reader(channel, since)
        except Exception:  # noqa: BLE001 — never let comms break the tick
            return []
        out: list[str] = []
        for m in _as_list(msgs):
            if not isinstance(m, Mapping):
                continue
            try:
                mid = int(m.get("id", 0))
            except (TypeError, ValueError):
                continue
            self._chat_cursors[channel] = max(self._chat_cursors.get(channel, 0), mid)
            if str(m.get("sender", "")) == self.actor_id:
                continue  # skip our own posts
            line = f"- [{channel}] {m.get('sender', '?')}: {m.get('body', '')}"
            out.append(line[:200])
        return out


def _comms_adapter(
    key: str, fn: Callable[..., Any], actor_id: str
) -> Callable[..., Any] | None:
    """Typed tool adapters for the directed comms verbs (``mail_send``/
    ``chat_post``): they inject the persona's own ``actor_id`` as ``sender`` and
    omit ``sender`` from the model-facing signature, so identity is bound and
    unspoofable. Other surfaced callables fall through to ``_wrap_action``."""

    if key == "mail_send":

        def mail_send(to: str, subject: str = "", body: str = "") -> str:
            """Send an email. Args: to (recipient), subject, body."""
            return str(fn(actor_id, to, subject, body))

        return mail_send

    if key == "chat_post":

        def chat_post(channel: str, text: str) -> str:
            """Post a message to a chat channel. Args: channel, text."""
            return str(fn(actor_id, channel, text))

        return chat_post

    return None


def _as_tool(fn: Callable[..., Any], name: str) -> Callable[..., Any]:
    """Decorate with Strands' ``@tool`` when available; otherwise return the
    plain callable (so the SDK imports and unit-tests without the extra)."""

    try:
        from strands import tool
    except Exception:  # noqa: BLE001 — optional extra
        return fn
    return cast("Callable[..., Any]", tool(name=name)(fn))


def factory(config: Mapping[str, object]) -> NPC:
    """The ``openrange.npcs`` entry point. A pack points one entry at this and
    every persona instance is pure config."""

    return PersonaAgent(config=config)


# A small generic adjective pool for sampled personas. A pack that wants
# domain-flavored variety passes its own ``traits=`` / ``roles=`` / ``names=``.
_TRAIT_POOL = (
    "curious",
    "cautious",
    "blunt",
    "friendly",
    "impatient",
    "meticulous",
    "chatty",
    "reserved",
    "skeptical",
    "easygoing",
    "anxious",
    "confident",
)


def sample_persona(
    seed: int,
    *,
    roles: Sequence[str],
    goal: str = "go about your normal work",
    names: Sequence[str] | None = None,
    traits: Sequence[str] | None = None,
    num_traits: int = 3,
) -> dict[str, object]:
    """Deterministically sample a diverse persona config from a seed.

    A population built as ``[sample_persona(i, roles=...) for i in range(n)]`` is
    diverse-by-construction — different name, role, and a numeric trait vector —
    instead of ``count=n`` clones that share everything but an id. No model, no
    training; the pack supplies its own ``roles`` (and optionally ``names``/
    ``traits``) vocabulary so it stays domain-neutral.
    """
    if not roles:
        raise ValueError("sample_persona needs a non-empty 'roles' vocabulary")
    rng = random.Random(seed)
    pool = list(traits) if traits else list(_TRAIT_POOL)
    chosen = rng.sample(pool, k=min(num_traits, len(pool)))
    return {
        "name": rng.choice(list(names)) if names else f"person-{seed}",
        "role": rng.choice(list(roles)),
        "goal": goal,
        "traits": {t: round(rng.uniform(0.3, 1.0), 1) for t in chosen},
    }
