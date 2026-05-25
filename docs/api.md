# API
OpenRange does not provide a gymnasium-style `step / reset / observation / reward` loop. The domain is arbitrary, so the API the agent talks to is the world's own surface — HTTP, MCP, a shell, a file path, a simulator's step function. The pack decides what that surface is, based on the manifest.

What OpenRange provides is the lifecycle around that surface, inspired by [SkyRL-Agent](https://github.com/NovaSky-AI/SkyRL):

- **build** — admission produces a frozen `Snapshot` from manifest + pack. In a run, `OpenRangeRun` owns the dashboard event sink so pack loading, world generation, admission verdicts, and snapshot creation are visible while the build runs. See [main doc](start_here.md).
- **snapshot** — the world is brought to a known initial state for an episode from the admitted snapshot via `Pack.realize(graph, backing) -> RuntimeHandle`.
- **get tasks** — the harness reads `snapshot.tasks` for instructions, entrypoints, and goal nodes. See [Task](start_here.md#task).
- **run** — the agent acts through the entrypoints. OpenRange does not mediate the agent loop, but environment-owned runtimes can record public-interface evidence (e.g. HTTP access logs) and emit environment events for the dashboard. Episode termination is either agent stop (harness) or success event (world).
- **check_success** — the TaskFamily reads the realizer's final state (whatever `RuntimeHandle.collect()` returned) against the world graph + task and returns a structured `EpisodeResult`. See [Episode checks and rewards](start_here.md#episode-checks-and-rewards).
- **report** — outcome, lineage, final state, and environment-owned actor turns are available to the dashboard.

The harness owns the agent loop. OpenRange owns build, reset, success detection, structured result, and report. There is no observation API and no reward API; the agent interacts with the task-specific surface materialized by the pack's `RuntimeHandle`.
