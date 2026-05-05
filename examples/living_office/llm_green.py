"""LLM-driven green scheduler — PersonaForge, DayPlan, ReactiveBus.

Implements the runtime.green.GreenScheduler Protocol. Replaces ScriptedGreenScheduler
at runtime via OpenRangeRuntime(green_scheduler=LLMGreenScheduler(...)).

Design:
- PersonaForge runs once at reset(). One batched LLM call hydrates every base
  GreenPersona into a full office person — name, tone, backstory, desk location,
  relationships, communication style.
- DayPlan runs once at reset(). One batched LLM call drafts each persona's
  slot-indexed schedule of intents.
- Routine actions are pulled from the DayPlan (persona-faithful, no fresh LLM
  call per tick). Speech strings are LLM-authored.
- ReactiveBus runs on malicious events via record_event(). Affected personas
  get a fresh LLM turn to react — this is where "Tim noticed the portal is
  down and asks the team" actually happens.
- Every LLM call flows through SeededLLM so the demo is deterministic and
  cheap on replay.
"""

from __future__ import annotations

import logging
import random
from collections import defaultdict, deque
from dataclasses import dataclass, field
from math import floor
from typing import Any

from open_range.config import EpisodeConfig
from open_range.contracts.runtime import Action, RuntimeEvent
from open_range.contracts.snapshot import RuntimeSnapshot
from open_range.contracts.world import GreenPersona

from .llm import SeededLLM

logger = logging.getLogger("openrange.living_office.green")


@dataclass
class RichPersona:
    """Hydrated office person — all LLM-generated from the base GreenPersona."""

    persona_id: str
    role: str
    display_name: str
    pronouns: str
    title: str
    tenure_years: float
    tone: str
    quirks: str
    interests: list[str]
    desk_zone: str
    desk_x: float
    desk_y: float
    reports_to: str
    peers: list[str]
    home_host: str
    mailbox: str
    awareness: float
    susceptibility: dict[str, float]
    base_routine: tuple[str, ...]


@dataclass
class DayPlanStep:
    slot: int
    intent: str
    service: str
    kind: str
    speech: str


@dataclass
class ObservableEvent:
    """What frontend sees — not the RuntimeEvent, but the persona-layer story."""

    time: float
    persona_id: str
    display_name: str
    kind: str
    channel: str
    text: str
    activity: str = ""


class LLMGreenScheduler:
    """LLM-driven replacement for ScriptedGreenScheduler."""

    def __init__(
        self,
        llm: SeededLLM,
        *,
        culture_hint: str = "small modern healthcare SaaS, startup-y, ~20 people, warm but professional",
        tick_window: int = 60,
        observable_sink: list[ObservableEvent] | None = None,
    ) -> None:
        self.llm = llm
        self.culture_hint = culture_hint
        self.tick_window = tick_window
        self._snapshot: RuntimeSnapshot | None = None
        self._episode_config = EpisodeConfig()
        self._seed = 0
        self._last_advanced_slot = -1
        self._ready_actions: deque[Action] = deque()
        self._reactive_queue: dict[int, deque[Action]] = defaultdict(deque)
        self._scheduled_reactions: set[tuple[int, str, str]] = set()
        self._rich: dict[str, RichPersona] = {}
        self._day_plan: dict[str, list[DayPlanStep]] = {}
        self._observable: list[ObservableEvent] = (
            observable_sink if observable_sink is not None else []
        )

    # ---- GreenScheduler Protocol ----
    def reset(self, snapshot: RuntimeSnapshot, episode_config: EpisodeConfig) -> None:
        self._snapshot = snapshot
        self._episode_config = episode_config
        self._seed = snapshot.world.seed
        self._last_advanced_slot = -1
        self._ready_actions = deque()
        self._reactive_queue = defaultdict(deque)
        self._scheduled_reactions = set()
        base_personas = tuple(snapshot.world.green_personas)
        if not base_personas:
            self._rich = {}
            self._day_plan = {}
            return
        self._rich = self._forge_personas(base_personas)
        self._day_plan = self._forge_day_plans(base_personas)

    def advance_until(self, sim_time: float) -> None:
        if self._snapshot is None or not self._episode_config.green_enabled:
            return
        goal_slot = max(0, floor(sim_time))
        for slot in range(self._last_advanced_slot + 1, goal_slot + 1):
            self._ready_actions.extend(self._routine_actions(slot))
            self._ready_actions.extend(self._reactive_actions(slot))
            self._last_advanced_slot = slot

    def pop_ready_actions(self) -> tuple[Action, ...]:
        actions = tuple(self._ready_actions)
        self._ready_actions.clear()
        return actions

    def record_event(self, event: RuntimeEvent) -> None:
        if (
            self._snapshot is None
            or not self._episode_config.green_enabled
            or not self._episode_config.green_branch_enabled
            or not event.malicious
        ):
            return
        affected = self._personas_to_notify(event)
        if not affected:
            return
        slot = floor(event.time) + 1
        key = (slot, event.event_type, event.target_entity)
        if key in self._scheduled_reactions:
            return
        self._scheduled_reactions.add(key)
        self._enqueue_llm_reactions(event, affected, slot)

    # ---- Introspection for frontend ----
    def rich_personas(self) -> list[RichPersona]:
        return list(self._rich.values())

    def day_plan(self) -> dict[str, list[DayPlanStep]]:
        return dict(self._day_plan)

    # ---- Persona forging (one batched LLM call) ----
    def _forge_personas(
        self, base_personas: tuple[GreenPersona, ...]
    ) -> dict[str, RichPersona]:
        # Deterministic auto-seating: bucket personas by role, fill a grid
        # per room, yield (role, x, y) tuples. Scales cleanly from 4 → 40+ NPCs.
        seat_iter_by_role: dict[str, list[tuple[str, float, float]]] = {}
        role_counts: dict[str, int] = {}
        for persona in base_personas:
            role_counts[persona.role] = role_counts.get(persona.role, 0) + 1
        seat_iter_by_role = self._auto_seat_plan(role_counts)
        system = (
            "You write concise office persona cards for a living cyber range. "
            "Every persona is a real employee with a name, a voice, and a place "
            "to sit. Output STRICT JSON matching the schema. Never invent fields."
        )
        base_summary = [
            {
                "persona_id": persona.id,
                "role": persona.role,
                "home_host": persona.home_host,
                "mailbox": persona.mailbox,
                "base_routine": list(persona.routine),
            }
            for persona in base_personas
        ]
        prompt = (
            f"Company context: {self.culture_hint}\n\n"
            f"Generate one JSON object under key 'personas', an array with "
            f"{len(base_personas)} entries in the same order as INPUT. Each "
            f"entry must be:\n"
            "{\n"
            '  "persona_id": <echo input>,\n'
            '  "display_name": <distinct human name, no duplicates>,\n'
            '  "pronouns": "she/her"|"he/him"|"they/them",\n'
            '  "title": <specific job title fitting the role>,\n'
            '  "tenure_years": <float 0.2–8.0>,\n'
            '  "tone": <one short phrase — e.g. "dry, precise">,\n'
            '  "quirks": <one short phrase>,\n'
            '  "interests": <array of 2-3 short items>,\n'
            '  "reports_to": <persona_id of manager or "">,\n'
            '  "peers": <array of 1-3 persona_ids they talk to most>\n'
            "}\n\n"
            f"INPUT:\n{base_summary}"
        )
        payload = self.llm.complete_json(
            prompt,
            system=system,
            seed=self._seed,
            max_tokens=2200,
            extra_cache_tag=f"personaforge/{self._snapshot.world.world_id if self._snapshot else ''}",
        )
        entries = _as_list(payload, key="personas")
        if len(entries) != len(base_personas):
            logger.warning(
                "PersonaForge returned %d entries for %d personas, padding",
                len(entries),
                len(base_personas),
            )
        rich: dict[str, RichPersona] = {}
        by_id = {item.get("persona_id"): item for item in entries if isinstance(item, dict)}
        for persona in base_personas:
            card = by_id.get(persona.id) or {}
            pool = seat_iter_by_role.get(persona.role) or []
            if pool:
                zone, x, y = pool.pop(0)
            else:
                zone, x, y = persona.role, 0.0, 0.0
            rich[persona.id] = RichPersona(
                persona_id=persona.id,
                role=persona.role,
                display_name=str(card.get("display_name") or _fallback_name(persona)),
                pronouns=str(card.get("pronouns") or "they/them"),
                title=str(card.get("title") or persona.role.replace("_", " ").title()),
                tenure_years=float(card.get("tenure_years") or 1.5),
                tone=str(card.get("tone") or "friendly"),
                quirks=str(card.get("quirks") or "takes extensive notes"),
                interests=list(card.get("interests") or []),
                desk_zone=zone,
                desk_x=x,
                desk_y=y,
                reports_to=str(card.get("reports_to") or ""),
                peers=[str(item) for item in (card.get("peers") or [])],
                home_host=persona.home_host,
                mailbox=persona.mailbox,
                awareness=persona.awareness,
                susceptibility=dict(persona.susceptibility or {}),
                base_routine=tuple(persona.routine),
            )
        return rich

    # ---- Day plan forging (one batched LLM call) ----
    def _forge_day_plans(
        self, base_personas: tuple[GreenPersona, ...]
    ) -> dict[str, list[DayPlanStep]]:
        if not self._rich:
            return {}
        window = self.tick_window
        system = (
            "You draft short office day plans grounded in each persona's role "
            "and tone. Every step is one concrete action against one service. "
            "Output STRICT JSON. Speech should sound like the persona."
        )
        summary = [
            {
                "persona_id": rp.persona_id,
                "display_name": rp.display_name,
                "role": rp.role,
                "title": rp.title,
                "tone": rp.tone,
                "base_routine": list(rp.base_routine),
            }
            for rp in self._rich.values()
        ]
        service_ids = self._world_service_ids()
        services = ", ".join(service_ids) if service_ids else "svc-web"
        prompt = (
            f"Draft a {window}-tick day plan for each persona. Return JSON "
            f"under 'plans', an array with one entry per persona (same order "
            f"as INPUT).\n\n"
            "Each entry: {\n"
            '  "persona_id": <echo>,\n'
            '  "steps": [\n'
            '     { "slot": 0, "intent": <short>, "service": <one of the services>, "kind": "api"|"mail"|"shell", "speech": <short in-character one-liner or ""> }\n'
            "  ]\n"
            "}\n\n"
            f"Rules: slot 0..{window - 1}. 4-8 steps per persona spread across the window. "
            f"Services available in this world: {services}. "
            "Every persona acts on every few slots, not every slot. "
            "Speech is optional but useful for 1-2 steps per persona (chat/email channel).\n\n"
            f"INPUT:\n{summary}"
        )
        payload = self.llm.complete_json(
            prompt,
            system=system,
            seed=self._seed + 1,
            max_tokens=3200,
            extra_cache_tag=f"dayplan/{self._snapshot.world.world_id if self._snapshot else ''}",
        )
        entries = _as_list(payload, key="plans")
        by_id: dict[str, list[dict]] = {
            item.get("persona_id"): (item.get("steps") or [])
            for item in entries
            if isinstance(item, dict)
        }
        out: dict[str, list[DayPlanStep]] = {}
        for rp in self._rich.values():
            raw_steps = by_id.get(rp.persona_id) or []
            steps: list[DayPlanStep] = []
            for raw in raw_steps:
                if not isinstance(raw, dict):
                    continue
                try:
                    steps.append(
                        DayPlanStep(
                            slot=int(raw.get("slot", 0)),
                            intent=str(raw.get("intent") or "work"),
                            service=str(raw.get("service") or "svc-web"),
                            kind=str(raw.get("kind") or "api"),
                            speech=str(raw.get("speech") or ""),
                        )
                    )
                except (ValueError, TypeError):
                    continue
            if not steps:
                steps = self._fallback_steps(rp)
            out[rp.persona_id] = sorted(steps, key=lambda s: s.slot)
        return out

    def _fallback_steps(self, rp: RichPersona) -> list[DayPlanStep]:
        cycle = rp.base_routine or ("browse_app",)
        steps = []
        for idx, routine in enumerate(cycle * 3):
            steps.append(
                DayPlanStep(
                    slot=idx * 3,
                    intent=routine,
                    service=_routine_service(routine),
                    kind="mail" if "mail" in routine else "api",
                    speech="",
                )
            )
            if idx >= 8:
                break
        return steps

    # ---- Per-slot dispatch ----
    def _routine_actions(self, slot: int) -> tuple[Action, ...]:
        assert self._snapshot is not None
        if not self._episode_config.green_routine_enabled:
            return ()
        if self._episode_config.green_profile == "off":
            return ()
        workload = self._snapshot.world.green_workload
        interval = max(1, workload.routine_interval_ticks)
        if slot % interval != 0:
            return ()
        due_personas = [
            rp
            for rp in self._rich.values()
            if any(step.slot == slot for step in self._day_plan.get(rp.persona_id, []))
        ]
        if not due_personas:
            return ()
        rng = random.Random(self._seed + slot)
        budget = self._parallel_budget(workload.max_parallel_actions)
        if budget < len(due_personas):
            due_personas = rng.sample(due_personas, k=budget)
        due_personas.sort(key=lambda rp: rp.persona_id)
        actions: list[Action] = []
        for rp in due_personas:
            step = next(
                (s for s in self._day_plan.get(rp.persona_id, []) if s.slot == slot),
                None,
            )
            if step is None:
                continue
            payload = {
                "routine": step.intent,
                "service": step.service,
                "host": rp.home_host,
                "mailbox": rp.mailbox,
                "display_name": rp.display_name,
                "role": rp.role,
                "desk_zone": rp.desk_zone,
                "desk_x": rp.desk_x,
                "desk_y": rp.desk_y,
                "speech": step.speech,
                "source": "day_plan",
            }
            actions.append(
                Action(
                    actor_id=rp.persona_id,
                    role="green",
                    kind=step.kind if step.kind in {"api", "mail", "shell"} else "api",
                    payload=payload,
                )
            )
            if step.speech:
                self._observable.append(
                    ObservableEvent(
                        time=float(slot),
                        persona_id=rp.persona_id,
                        display_name=rp.display_name,
                        kind="speech",
                        channel=_channel_for(step.kind, rp.role),
                        text=step.speech,
                        activity=step.intent,
                    )
                )
        return tuple(actions)

    def _reactive_actions(self, slot: int) -> tuple[Action, ...]:
        if not self._episode_config.green_branch_enabled:
            self._reactive_queue.pop(slot, None)
            return ()
        actions: list[Action] = []
        budget = self._reactive_budget()
        while self._reactive_queue[slot] and len(actions) < budget:
            actions.append(self._reactive_queue[slot].popleft())
        if self._reactive_queue[slot]:
            self._reactive_queue[slot + 1].extend(self._reactive_queue[slot])
        self._reactive_queue.pop(slot, None)
        return tuple(actions)

    def _parallel_budget(self, base: int) -> int:
        profile = self._episode_config.green_profile
        if profile == "off":
            return 0
        if profile == "low":
            return max(1, base // 2)
        if profile == "high":
            return base + 1
        return base

    def _reactive_budget(self) -> int:
        assert self._snapshot is not None
        base = self._snapshot.world.green_workload.reactive_branch_budget
        profile = self._episode_config.green_profile
        if profile == "off":
            return 0
        if profile == "low":
            return max(0, base - 1)
        if profile == "high":
            return base + 2
        return base + 1

    # ---- Reactive bus (LLM per event) ----
    def _personas_to_notify(self, event: RuntimeEvent) -> list[RichPersona]:
        rps = list(self._rich.values())
        rps.sort(
            key=lambda rp: (
                -rp.awareness,
                0 if event.target_entity in rp.home_host or rp.role == "it_admin" else 1,
                rp.persona_id,
            )
        )
        return rps[: max(1, min(3, len(rps)))]

    def _enqueue_llm_reactions(
        self, event: RuntimeEvent, affected: list[RichPersona], slot: int
    ) -> None:
        system = (
            "You are simulating genuine employee reactions to an IT incident. "
            "Each persona reacts in-character with ONE concrete action. Output "
            "STRICT JSON."
        )
        affected_summary = [
            {
                "persona_id": rp.persona_id,
                "display_name": rp.display_name,
                "role": rp.role,
                "title": rp.title,
                "tone": rp.tone,
                "awareness": rp.awareness,
            }
            for rp in affected
        ]
        prompt = (
            f"An IT event just happened:\n"
            f"  event_type: {event.event_type}\n"
            f"  target: {event.target_entity}\n"
            f"  time: {event.time}\n\n"
            f"For each persona below, decide their ONE reaction. Return JSON "
            f"under 'reactions', array in same order, each:\n"
            "{\n"
            '  "persona_id": <echo>,\n'
            '  "kind": "shell"|"control"|"mail",\n'
            '  "target_service": <one of the services listed below>,\n'
            '  "command_or_subject": <short action string>,\n'
            '  "speech": <one line the persona says in chat/email>\n'
            "}\n\n"
            f"Services available: {', '.join(self._world_service_ids()) or 'n/a'}. "
            f"IT admins investigate via the SIEM service if one exists. "
            f"Non-admins usually message peers or IT via the email service. "
            f"High-awareness personas react faster and more decisively.\n\n"
            f"INPUT:\n{affected_summary}"
        )
        payload = self.llm.complete_json(
            prompt,
            system=system,
            seed=self._seed + int(event.time * 17) + hash(event.event_type) % 997,
            max_tokens=900,
            extra_cache_tag=f"reaction/{event.event_type}/{event.target_entity}/{slot}",
        )
        entries = _as_list(payload, key="reactions")
        by_id = {
            item.get("persona_id"): item
            for item in entries
            if isinstance(item, dict)
        }
        for rp in affected:
            card = by_id.get(rp.persona_id) or {}
            kind = str(card.get("kind") or "shell")
            if kind not in {"shell", "control", "mail"}:
                kind = "shell"
            target_service = str(card.get("target_service") or self._pick_default_service("siem"))
            cmd = str(card.get("command_or_subject") or "review recent events")
            speech = str(card.get("speech") or "")
            if kind == "shell":
                siem_id = self._pick_default_service("siem")
                payload_dict: dict[str, Any] = {
                    "target": target_service,
                    "command": (
                        f"wget -qO- http://{siem_id}:9200/all.log | tail -n 40"
                        if "siem" in target_service.lower()
                        else f"echo {cmd!r}"
                    ),
                    "branch": "llm_reaction",
                    "reported_target": event.target_entity,
                    "reported_event_type": event.event_type,
                    "speech": speech,
                    "display_name": rp.display_name,
                }
            elif kind == "control":
                payload_dict = {
                    "target": target_service,
                    "action": "recover",
                    "branch": "llm_reaction",
                    "reported_target": event.target_entity,
                    "speech": speech,
                    "display_name": rp.display_name,
                }
            else:  # mail
                payload_dict = {
                    "routine": "notify_peers",
                    "service": self._pick_default_service("email"),
                    "host": rp.home_host,
                    "mailbox": rp.mailbox,
                    "subject": cmd,
                    "branch": "llm_reaction",
                    "reported_target": event.target_entity,
                    "speech": speech,
                    "display_name": rp.display_name,
                }
            self._reactive_queue[slot].append(
                Action(actor_id=rp.persona_id, role="green", kind=kind, payload=payload_dict)
            )
            if speech:
                self._observable.append(
                    ObservableEvent(
                        time=float(slot),
                        persona_id=rp.persona_id,
                        display_name=rp.display_name,
                        kind="reaction",
                        channel="chat" if kind != "mail" else "email",
                        text=speech,
                        activity=f"Reacting to {event.event_type}",
                    )
                )

    # ---- Helpers ----
    def _world_service_ids(self) -> list[str]:
        if self._snapshot is None:
            return []
        return [svc.id for svc in self._snapshot.world.services]

    def _pick_default_service(self, kind_substring: str) -> str:
        """Return a service id whose kind/id contains the hint, or first available."""
        if self._snapshot is None:
            return f"svc-{kind_substring}"
        needle = kind_substring.lower()
        for svc in self._snapshot.world.services:
            if needle in svc.kind.lower() or needle in svc.id.lower():
                return svc.id
        ids = self._world_service_ids()
        return ids[0] if ids else f"svc-{kind_substring}"

    # Canonical role → (center_x, center_z, room_w, room_h)
    # Aliases let manifests use either singular or plural (engineer/engineering)
    ROOM_LAYOUT: dict[str, tuple[float, float, float, float]] = {
        "engineer":    (-5.5, -3.0, 7.5, 5.0),
        "engineering": (-5.5, -3.0, 7.5, 5.0),
        "sales":       ( 5.5, -3.0, 7.5, 5.0),
        "finance":     (-5.5,  4.0, 7.5, 4.5),
        "it_admin":    ( 5.5,  4.0, 7.5, 4.5),
        "it":          ( 5.5,  4.0, 7.5, 4.5),
        "hr":          (-5.5,  9.0, 5.0, 3.0),
        "legal":       ( 5.5,  9.0, 5.0, 3.0),
        "ops":         ( 0.0,  9.0, 5.0, 3.0),
        "marketing":   (-5.5,  9.0, 5.0, 3.0),
        "support":     ( 5.5,  9.0, 5.0, 3.0),
        "executive":   ( 0.0,  9.0, 5.0, 3.0),
    }

    def _auto_seat_plan(
        self, role_counts: dict[str, int]
    ) -> dict[str, list[tuple[str, float, float]]]:
        """Compute desk coords per role; grid-packs seats into the role's room."""
        plan: dict[str, list[tuple[str, float, float]]] = {}
        # Rooms already claimed — so multiple unknown roles don't pile up on one spot
        extra_row_z = 9.0
        extra_col_x = [-2.0, 2.0, -9.0, 9.0]
        for role, n in role_counts.items():
            key = role.lower()
            if key in self.ROOM_LAYOUT:
                cx, cz, w, h = self.ROOM_LAYOUT[key]
            else:
                # unknown role — deterministic hash → a free bay south of the floor
                slot = abs(hash(key)) % len(extra_col_x)
                cx, cz, w, h = extra_col_x[slot], extra_row_z + 3.0, 4.5, 3.0
                logger.info("role %r has no ROOM_LAYOUT; placing at (%.1f, %.1f)", role, cx, cz)
            cols = max(1, min(4, int(n ** 0.5 + 0.99)))
            rows = (n + cols - 1) // cols
            # inset slightly from walls
            inner_w = max(1.0, w - 1.6)
            inner_h = max(1.0, h - 1.6)
            seats: list[tuple[str, float, float]] = []
            for idx in range(n):
                col = idx % cols
                row = idx // cols
                x = cx + (col + 0.5 - cols / 2) * (inner_w / cols)
                y = cz + (row + 0.5 - rows / 2) * (inner_h / max(1, rows))
                seats.append((role, x, y))
            plan[role] = seats
        return plan


def _fallback_name(persona: GreenPersona) -> str:
    return persona.id.replace("_", " ").title()


def _routine_service(routine: str) -> str:
    lowered = routine.lower()
    if "mail" in lowered:
        return "svc-email"
    if "file" in lowered or "share" in lowered:
        return "svc-fileshare"
    if "idp" in lowered or "password" in lowered:
        return "svc-idp"
    if "alert" in lowered or "triage" in lowered:
        return "svc-siem"
    if "payroll" in lowered:
        return "svc-db"
    return "svc-web"


def _channel_for(kind: str, role: str) -> str:
    if kind == "mail":
        return "email"
    if role == "it_admin":
        return "chat-ops"
    return "chat"


def _as_list(payload: Any, *, key: str) -> list:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        inner = payload.get(key)
        if isinstance(inner, list):
            return inner
        for value in payload.values():
            if isinstance(value, list):
                return value
    return []
