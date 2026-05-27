# Dashboard

The dashboard is the inspection surface for OpenRange worlds.

It shows what the Builder made, what admission accepted, and what an episode is
doing right now.

## What It Shows

The dashboard has three jobs:

- show the generated world and tasks
- show Builder lineage: manifest, pack, prompt, output, admission verdict, and evolution
- show episode activity

For episode viewing, the event feed is the source of truth. Any visual view is
just a view of that event stream.

## Builder Review

After admission accepts a world and its tasks, the dashboard lets you inspect
what it made before you rely on it.

You should be able to inspect:

- generated world state
- generated tasks
- entrypoints
- admission verdict (per-layer: structural, ontology, pack invariants, task
  bindings, feasibility)
- Builder reasoning or summary
- pack files touched by the Builder

If the world is wrong, prompt the LLM Builder for changes from the dashboard.
That starts another Builder pass. It is not a manual edit to an admitted
snapshot.

Use this loop until the pack is stable:

`build -> inspect -> prompt changes -> admit -> inspect again`

Every prompt and Builder result belongs in lineage.

While the Builder is running inside an `OpenRangeRun`, the environment-owned run
contract attaches the dashboard artifact log as the Builder event sink. That
writes `builder_step` rows to `dashboard.events.jsonl` and mirrors them under
`builder.steps` in `dashboard.json`, so a dashboard can show pack loading, world
generation, admission verdicts, and snapshot creation before the episode
starts. The stream intentionally uses public summaries and does not include
generated secret values.

## Controls

The basic controls are:

- `reset` rebuilds the displayed episode from an admitted snapshot
- `play` starts the runtime loop
- `pause` stops streaming and leaves the current state visible

Reset does not mutate the admitted world. It loads a frozen snapshot and starts a
new episode over it.

## Runtime Data

The dashboard reads from the same runtime state as the SDK.

It exposes:

- public episode briefing derived from the loaded snapshot
- topology from the active snapshot
- current episode state
- activity summaries over event type, actor, and actor kind
- actor summaries with recent per-actor history from the event buffer
- environment actor turns from agents, NPCs, and internal actors
- builder steps from the current build
- runtime events over SSE
- a rolling event buffer for late subscribers
- run-root files for external observers:
  `dashboard.events.jsonl` for tailing and `dashboard.json` for polling
- optional narration over the recent event buffer

Hidden world state, private reference traces, and private builder probes stay
hidden. The dashboard can say that admission passed or failed. It must not turn
the private oracle into an agent-visible walkthrough.

The dashboard API intentionally exposes TaskFamily handle ids
(`feasibility_check`, `success_check`) and admission verdict summaries, but
not pack-internal check source code or admission probe final state.

## Lineage

Every admitted world should be inspectable from its lineage.

A lineage node should show:

- manifest input
- pack input
- user prompt to the Builder
- builder changes
- generated tasks
- admission verdict
- admitted snapshot id
- curriculum or evolution input, if this node came from `evolve`

World evolution is just another builder run with ancestry. The dashboard should
make that ancestry obvious and boring: what changed, why it changed, and whether
it passed admission.

## Running It

Launch it against admitted snapshots:

```bash
uv run openrange dashboard --store-dir snapshots
```

The dashboard can also start before any snapshots exist. In that case it shows
an empty state until you point it at a saved snapshot or run a live eval dashboard
through the environment.

Useful flags:

- `--snapshot-id <id>` loads a specific snapshot on reset
- `--no-browser` starts the server without opening a browser

A live eval writes dashboard artifacts to the run root (`dashboard.events.jsonl`
+ `dashboard.json`) as the episode progresses. The dashboard HTTP server is a
separate process — start it against the run root in another shell to inspect
the run live:

```bash
# shell 1: run the eval
uv run python -m examples.codex_eval --runs-dir or-runs

# shell 2: open the live view against the run root
uv run openrange dashboard --run-root or-runs/<run-id>
```

Each eval run gets a unique subdirectory under `or-runs` by default. Use
`--run-root <path>` only when you want to name the exact immutable run
directory yourself.

## API Shape

The current dashboard backend is small:

- `GET /` serves the UI
- `GET /api/briefing` returns public mission and entrypoint briefing
- `GET /api/actors` returns per-actor activity summaries
- `GET /api/topology` returns world structure
- `GET /api/lineage` returns admission status and builder lineage
- `GET /api/state` returns the current episode state
- `GET /api/inspect` returns topology, lineage, state, turns, and narration
- `GET /api/events/stream` streams runtime events
- `POST /api/episode/reset` resets onto a snapshot
- `POST /api/episode/play` starts autoplay
- `POST /api/episode/pause` pauses autoplay
- `GET /api/narrate` returns narration for recent events
- `GET /api/narrate/stream` streams narration updates

The UI should stay disposable. The snapshot store, runtime, episode results,
and lineage records are the durable parts.
