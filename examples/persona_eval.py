"""Live demonstration: one PersonaAgent class across three real packs.

Builds a real world per pack, reads its REAL surfaced affordances, and shows the
same class bind them + act — plus through-the-world comms attribution and a
scale timing. Uses a recording backend in place of the LLM so it runs without a
model; the tools it drives are the packs' genuine callables.
"""

from __future__ import annotations

import time
from typing import Any

from graphschema import WorldGraph
from openrange_pack_sdk import PersonaAgent
from openrange_pack_sdk.npcs.persona_agent import _as_tool


def _schema(fn: Any) -> str:
    t: Any = _as_tool(fn, getattr(fn, "__name__", "?"))
    try:
        props = t.tool_spec["inputSchema"]["json"]["properties"]
        return f"{fn.__name__}({', '.join(sorted(props))})"
    except Exception:
        return f"{fn.__name__}(...)"


def _cyber_surface() -> dict[str, Any]:
    from cyber_webapp import WebappPack
    from cyber_webapp.realize import WebappRuntime

    graph: WorldGraph = WebappPack().make_builder(None).build({"seed": 0}).graph
    rt = WebappRuntime(graph)
    rt._base_url = "http://world.local"  # normally set when the subprocess boots
    return dict(rt.surface_extras())


def _swe_surface() -> dict[str, Any]:
    from openrange_pack_sdk import Backing
    from swe import SwePack
    from swe.realize import SweRuntime

    graph = SwePack().make_builder(None).build({"seed": 0}).graph
    return dict(SweRuntime(graph, Backing.PROCESS).surface_extras())


def _trading_surface() -> dict[str, Any]:
    from openrange_pack_sdk import Backing
    from trading import TradingPack
    from trading.realize import TradingRuntime

    graph = TradingPack().make_builder(None).build({"seed": 0}).graph
    return dict(TradingRuntime(graph, Backing.PROCESS).surface_extras())


class _Rec:
    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.tools: tuple[Any, ...] = ()

    def preflight(self) -> None: ...

    def build_agent(self, *, system_prompt: str, tools: Any = ()) -> Any:
        self.tools = tuple(tools)
        return self

    def __call__(self, prompt: str) -> object:
        self.prompts.append(prompt)
        return {"message": "ok"}


def _run(npc: PersonaAgent, iface: dict[str, Any], run_id: str = "demo") -> _Rec:
    b = _Rec()
    npc._backend_override = b
    npc.start({"episode_id": run_id, "agent_backend": b})
    npc._cooldown = 0  # act now (start() phase-staggers; not what this demo shows)
    npc.step(iface)
    return b


PACKS = [
    (
        "cyber_webapp",
        _cyber_surface,
        {
            "name": "Dana",
            "role": "accountant",
            "goal": "browse the finance portal and email a colleague",
            "tools": ["http_get", "mail_send", "mail_read"],
        },
    ),
    (
        "swe",
        _swe_surface,
        {
            "name": "Riley",
            "role": "developer",
            "goal": "run the test suite before lunch",
            "tools": ["run_tests"],
        },
    ),
    (
        "trading",
        _trading_surface,
        {
            "name": "Morgan",
            "role": "analyst",
            "goal": "think about the market open",
            "tools": ["place_order", "http_get"],
        },
    ),  # neither surfaced -> fail-soft
]

print("=" * 74)
print("GENERALIZATION: one PersonaAgent class across three real packs")
print("=" * 74)
for pack_name, get_surface, config in PACKS:
    surface = get_surface()
    npc = PersonaAgent(config=config)
    rec = _run(npc, surface)
    built = [getattr(t, "__name__", "?") for t in rec.tools]
    raw = npc._tool_functions(surface)  # raw callables (schema-inspectable)
    print(f"\n[{pack_name}]  persona={config['name']} ({config['role']})")
    print(f"  pack really surfaces : {sorted(surface)}")
    print(f"  persona declared     : {config['tools']}")
    print(f"  -> bound tools       : {built or '(none — pure ambient presence)'}")
    if raw:
        print(f"  -> tool schemas      : {', '.join(_schema(f) for f in raw)}")
    print(f"  -> acted (1 tick)    : produced a prompt = {len(rec.prompts) == 1}")

print("\n" + "=" * 74)
print("COMMS: two personas talk THROUGH the real cyber world, attributed")
print("=" * 74)
from cyber_webapp import WebappPack  # noqa: E402
from cyber_webapp.realize import WebappRuntime  # noqa: E402

rt = WebappRuntime(WebappPack().make_builder(None).build({"seed": 1}).graph)
rt._base_url = "http://world.local"
shared = dict(rt.surface_extras())  # ONE surface, handed to every NPC

dana = PersonaAgent(config={"name": "Dana", "tools": ["mail_send"], "cadence_ticks": 1})
sam = PersonaAgent(config={"name": "Sam", "tools": ["mail_read"], "cadence_ticks": 1})
bd = _run(dana, shared, "ep7")
bs = _Rec()
sam._backend_override = bs
sam.start({"episode_id": "ep7", "agent_backend": bs})

# Dana emails Sam through the shared surface (identity injected NPC-side)
send = next(t for t in bd.tools if getattr(t, "__name__", "") == "mail_send")
print(f"\n  mail_send is a real Strands tool: {type(send).__name__}")
# invoke the raw adapter to drive the world deterministically
raw_send = next(f for f in dana._tool_functions(shared) if f.__name__ == "mail_send")
raw_send(to="Sam", subject="Q3", body="can you approve the invoice batch?")

sam.step(shared)  # Sam perceives it next tick via the same store
print(
    f"  Dana -> world store -> Sam sees it: "
    f"{'approve the invoice batch' in bs.prompts[-1]}"
)

import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

rt._base_url = None
rt._solver_root = Path(tempfile.mkdtemp())  # so collect_extras() can read result
graded = rt.collect_extras()
print(
    f"  grader drains npc_mail, attributed by sender: "
    f"{[(m['sender'], m['body'][:22]) for m in graded['npc_mail']]}"
)
print(
    f"  a persona CANNOT forge sender (not in the tool signature): "
    f"{'sender' not in _schema(raw_send)}"
)

print("\n" + "=" * 74)
print("SCALE: instantiate + cold-tick many personas; cadence gates the rest")
print("=" * 74)
for n in (1_000, 10_000, 50_000):
    t0 = time.perf_counter()
    roster = [
        PersonaAgent(
            config={
                "name": f"u{i}",
                "role": "worker",
                "tools": ["mail_send"],
                "cadence_ticks": 9,
            }
        )
        for i in range(n)
    ]
    t1 = time.perf_counter()
    uniq = len({p.actor_id for p in roster})
    idle_state = all(p._agent is None for p in roster)  # no LLM state until they act
    print(
        f"  {n:>6} personas: built in {1000 * (t1 - t0):6.0f} ms | "
        f"unique ids={uniq == n} | idle-hold-no-agent-state={idle_state}"
    )


# cadence stagger: a population on one cadence spreads its actions across the
# window instead of acting in lockstep (a thundering herd every N ticks).
class _Count:
    def __init__(self) -> None:
        self.n = 0

    def preflight(self) -> None: ...

    def build_agent(self, *, system_prompt: str, tools: Any = ()) -> Any:
        c = self

        def agent(prompt: str) -> None:
            c.n += 1

        return agent


pop = []
counter = _Count()
for i in range(900):
    p = PersonaAgent(config={"name": f"g{i}", "tools": [], "cadence_ticks": 9})
    p._backend_override = counter
    p.start({"episode_id": "s", "agent_backend": counter})
    pop.append(p)
per_tick = []
for _ in range(9):  # one full cadence window
    before = counter.n
    for p in pop:
        p.step({})
    per_tick.append(counter.n - before)
print(f"\n  900 personas @ cadence 9: acts per tick over one window = {per_tick}")
print(
    f"  -> total={sum(per_tick)} (each acts once), peak={max(per_tick)} "
    f"(spread, not a 900-wide thundering herd)"
)
print("\nDONE.")
