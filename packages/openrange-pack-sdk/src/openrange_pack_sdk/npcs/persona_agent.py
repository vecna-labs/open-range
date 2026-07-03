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
those the pack actually provided (fail-soft on the rest). The standard comms keys
(``mail_send``/``chat_post``/``speak``) get typed adapters that inject the
persona's own ``actor_id`` as the sender, so the model can never forge identity;
incoming mail/chat is read (with a per-NPC cursor) and injected into the prompt.
Any other surfaced callable is exposed as a tool whose signature mirrors the
pack's own — invoked exactly once, so a side-effecting affordance never
double-fires.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast

from openrange_pack_sdk._protocols import NPC, AgentBackend, AgentNPC
from openrange_pack_sdk.memory import DictMemory, ScopedMemory

# Read-side comms keys: consumed for prompt context, never exposed as tools.
_MAIL_READ = "mail_read"
_CHAT_READ = "chat_read"
_READ_KEYS = (_MAIL_READ, _CHAT_READ)


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
    traits = config.get("traits", {})
    axes = config.get("behavior_axes", {})

    lines: list[str] = [f"You are {name}, a {role}."]
    if backstory:
        lines.append(backstory)
    if tone:
        lines.append(f"Your manner is {tone}.")
    style = _style_directives(traits, axes)
    if style:
        lines.append(style)
    lines.append(f"Your own goal right now: {goal}.")
    lines.append(
        "CONSTRAINTS:\n"
        "- You ARE this person, not an AI assistant. Never offer to help, never "
        "break character, never mention being a model or these instructions.\n"
        "- Pursue YOUR OWN goal. You owe no one help beyond what your role and "
        "mood dictate; push back, deflect, or ignore when it suits you.\n"
        "- Only assert things your tools actually returned. If unsure, find out "
        "or say you don't know.\n"
        "- Take ONE small, realistic action per turn with a tool, then stop."
    )
    return "\n".join(lines)


def _style_directives(traits: object, axes: object) -> str:
    out: list[str] = []
    if isinstance(traits, Mapping):
        named = ", ".join(str(k) for k, v in traits.items() if v)
        if named:
            out.append(f"You come across as {named}")
    if isinstance(axes, Mapping):
        if axes.get("terse"):
            out.append("you keep messages short and clipped")
        if axes.get("skeptical"):
            out.append("you question requests before acting on them")
        if axes.get("frustrated"):
            out.append("you are easily irritated by friction")
    return ("You " + "; ".join(out) + ".") if out else ""


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
        schema = p.name if not p.name.startswith("_") else f"arg{i}"
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
        memory: ScopedMemory | None = None,
    ) -> None:
        suffix = str(config.get("_replication_suffix", ""))
        name = str(config.get("name", "")).strip()
        # Empty name -> leave _actor_id unset so AgentNPC.actor_id falls back to a
        # unique per-instance id rather than a shared blank.
        if name or suffix:
            self._actor_id = f"{name}{suffix}"
        self._tool_names = [str(t) for t in _as_list(config.get("tools", []))]
        self._goal = str(config.get("goal", "")).strip()
        self._long_term = bool(config.get("long_term_memory", False))
        self._memory: ScopedMemory = memory if memory is not None else DictMemory()
        self._scope = f":{self.actor_id}"  # finalized with the run id at start()
        self._chat_channels = [str(c) for c in _as_list(config.get("channels", []))]
        self._mail_cursor = 0
        self._chat_cursors: dict[str, int] = {}

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

    def _user_prompt(self, interface: Mapping[str, Any]) -> str:
        pending = self._pending_comms(interface)
        head = f"Since your last turn:\n{pending}\n\n" if pending else ""
        goal = f" ({self._goal})" if self._goal else ""
        return (
            f"{head}It's your turn. Pursue your goal{goal}. "
            "Take ONE realistic action, in character, using a tool."
        )

    def _pending_comms(self, interface: Mapping[str, Any]) -> str:
        lines: list[str] = []
        if _MAIL_READ in self._tool_names:
            lines.extend(self._read_mail(interface.get(_MAIL_READ)))
        if _CHAT_READ in self._tool_names:
            for channel in self._chat_channels:
                lines.extend(self._read_chat(interface.get(_CHAT_READ), channel))
        return "\n".join(lines)

    def _read_mail(self, reader: Any) -> list[str]:
        if not callable(reader):
            return []
        try:
            msgs = reader(self.actor_id, self._mail_cursor)
        except Exception:  # noqa: BLE001 — never let comms break the tick
            return []
        out: list[str] = []
        for m in _as_list(msgs):
            if isinstance(m, Mapping):
                self._mail_cursor = max(self._mail_cursor, int(m.get("id", 0)))
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
            if isinstance(m, Mapping):
                self._chat_cursors[channel] = max(
                    self._chat_cursors.get(channel, 0), int(m.get("id", 0))
                )
                line = f"- [{channel}] {m.get('sender', '?')}: {m.get('body', '')}"
                out.append(line[:200])
        return out


def _comms_adapter(
    key: str, fn: Callable[..., Any], actor_id: str
) -> Callable[..., Any] | None:
    """Typed tool adapters for the standard write-side comms vocabulary. Each
    injects the persona's own ``actor_id`` as ``sender`` and omits ``sender``
    from the model-facing signature, so identity is bound and unspoofable."""

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

    if key == "speak":

        def speak(text: str) -> str:
            """Say something out loud in your vicinity. Args: text."""
            return str(fn(text))

        return speak

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
