
![OpenRange visual](assets/open-range-visual.png)
<div align="center">

[![License](https://img.shields.io/github/license/vecna-labs/open-range?style=flat-square)](LICENSE)
[![CI](https://img.shields.io/github/actions/workflow/status/vecna-labs/open-range/ci.yml?branch=main&style=flat-square)](https://github.com/vecna-labs/open-range/actions/workflows/ci.yml)
[![Issues](https://img.shields.io/github/issues/vecna-labs/open-range?style=flat-square)](https://github.com/vecna-labs/open-range/issues)
[![Pull Requests](https://img.shields.io/github/issues-pr/vecna-labs/open-range?style=flat-square)](https://github.com/vecna-labs/open-range/pulls)
[![Stars](https://img.shields.io/github/stars/vecna-labs/open-range?style=flat-square)](https://github.com/vecna-labs/open-range/stargazers)
[![Forks](https://img.shields.io/github/forks/vecna-labs/open-range?style=flat-square)](https://github.com/vecna-labs/open-range/forks)
[![Contributors](https://img.shields.io/github/contributors/vecna-labs/open-range?style=flat-square)](https://github.com/vecna-labs/open-range/graphs/contributors)

</div>

OpenRange is a **domain-agnostic environment platform for training and
evaluating agents**. Give it a manifest and a Pack; it produces a
content-addressed `Snapshot` through a layered admission loop, ready to
run agent episodes against.

The library has just been rebuilt around three new ideas — a typed
property-graph meta-model, a Pack/TaskFamily split that lets one world
serve multiple task domains, and a `PackPrior` seam that lets agent
memory (BBG-shaped JSON from any harness) drive world generation. See
[DESIGN.md](DESIGN.md) for the architecture, [CONTRACTS.md](CONTRACTS.md)
for the JSON wire formats.

> [!WARNING]
> **v0.1.0 is the foundation.** The library, admission, distill seam,
> and one reference pack (`webapp`) ship and are green. The runtime
> layer (episode service, HTTP backing, Flask code-gen, dashboard, NPC
> threads, agent backends) was removed during the refactor and is being
> re-wired against the new shape in follow-up PRs. If you need those
> today, pin to a pre-`zen-hoover` commit.

### Why OpenRange

- **Train on realistic scenarios, not static benches.** Each build
  samples a fresh world; admission verifies it before the agent ever
  touches it.
- **Same training setup, different domain.** Pack = world-family,
  TaskFamily = domain. Swap the pack and you go from webapp to trading
  to robotics without rewriting your harness, training loop, or reward
  policy.
- **Solvable by construction.** Five-layer admission: structural +
  ontology + pack invariants + task bindings + per-task feasibility.
  No broken evals, no impossible tasks.
- **Bootstrap to flywheel.** A hand-authored `PackPrior` boots the
  generator; once an agent has run real tasks and a harness has
  recorded a BBG, `distill()` turns that experience into a sharper
  prior. The builder has one code path — it never knows whether the
  prior was learned or hand-authored.

### 📞 Community Call

Join us every **Friday at 12:00 PM CT** for the Open Range Community Call.
- 🎥 [Google Meet](https://meet.google.com/zuj-skfh-xjk)
- 📱 Dial in: [(US) +1 443-671-4919](tel:+14436714919) · PIN: `320 286 452#` · [More numbers](https://tel.meet/zuj-skfh-xjk?pin=6302524387334)
- 💬 [Join our Discord](https://discord.gg/KqDbvm9T5)

## How it works

```text
manifest + Pack
        ↓
   Pack.make_builder(prior)
        ↓
   Builder.build(manifest)  →  WorldGraph + tasks
        ↓
   admit():
     1. structural + ontology validation
     2. pack invariants
     3. task bindings (entrypoints/goals exist; entrypoints not HIDDEN)
     4. per-task feasibility (each TaskFamily.check_feasibility)
        ↓
   Snapshot  (content-addressed, frozen)
        ↓
   run_episode  (agent acts; family.check_success scores)
        ↓
   EpisodeResult  (structured; not a scalar reward)
```

**OpenRange owns the world, not the agent.** Your harness owns the
model, tools, rollout loop, training algorithm, and reward policy.

The agent interacts with whatever surface the world exposes: HTTP
endpoints, files, shells, MCP tools, simulator APIs. The shape of
that surface is the Pack's choice.

## Core concepts

- **Pack** — the reusable starting point for one world-family (e.g.
  `webapp`). Owns the ontology, builder, realizer, and TaskFamilies.
- **TaskFamily** — a *domain* of tasks against a world (e.g.
  `webapp.build`, `webapp.pentest`). Owns task generation,
  entrypoint/goal selection, feasibility, success. **Domain lives
  here, not on Pack.**
- **Ontology** — declarative node/edge kinds with rich `AttrSpec`
  (enums, REFs, required flags). One generic validator checks any
  graph against any ontology.
- **WorldGraph** — a typed property graph: nodes and edges with
  `kind` + `attrs`. World-absolute facts (`Role`,
  `Visibility=HIDDEN`) live on the node; task-relative facts
  (entrypoints, goal_nodes) live on the `TaskSpec`.
- **Snapshot** — an admitted, content-addressed world. `snapshot_id ==
  graph.content_hash()`. Build history rides alongside in
  `Snapshot.history`.
- **PackPrior** — the BBG → Builder seam; generic graph statistics
  emitted by `distill()` and consumed by a builder. The flywheel
  closes through JSON.

## Install

OpenRange uses [`uv`](https://github.com/astral-sh/uv) and requires
Python 3.14.

```bash
uv sync --group dev
```

## Build a world

```python
from openrange import admit
from webapp import WebappPack

pack = WebappPack()
snapshot = admit(pack, manifest={"seed": 0})

# Returns either a Snapshot or an AdmissionFailure.
print(snapshot.snapshot_id)         # sha256:...
for task in snapshot.tasks:
    print(task.id, "→", task.instruction)
    # webapp.build.0    → Implement the POST /login endpoint ...
    # webapp.pentest.0  → Recover the hidden admin flag ...
```

The `webapp` pack ships two TaskFamilies on the same world graph
(`webapp.build` and `webapp.pentest`). The build task entrypoints the
repo; the pentest task entrypoints an exposed endpoint. That
load-bearing demo lives at [tests/test_webapp_pack.py](tests/test_webapp_pack.py).

## Project layout

```text
src/openrange/
  world_ir.py            typed-property-graph meta-model
  ontologies/
    bbg.py               the bbg@0.1.0 ontology (consumed by distill)
  core/
    pack.py              Pack/Builder/TaskFamily protocols + wire shapes
    admit.py             layered admission loop + Snapshot + BuildEvent
    distill.py           graph + status-log → PackPrior

packs/webapp/            reference Pack: one world, two task families
tests/                   the test suite (107 tests, 95% coverage)

DESIGN.md                architecture narrative
CONTRACTS.md             JSON wire formats
```

## Contributing

Contributions welcome across code, docs, examples, pack design, bug
reports, and design discussion. The `scripts/check_boundary.sh` script
enforces two invariants: core is domain-free and never imports a
specific harness library. See [CONTRIBUTING.md](CONTRIBUTING.md) for
local setup.

Please follow the [Code of Conduct](CODE_OF_CONDUCT.md).
Security-sensitive reports can go to **security@vecna-labs.dev**.

## License

OpenRange is released under the [MIT License](LICENSE).
