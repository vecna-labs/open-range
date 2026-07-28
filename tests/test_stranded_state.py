"""A failure must not strand what it touched.

Three places left state behind when something went wrong: the warm pool dropped
the runtime it displaced, the agent loop abandoned a live episode when the
sampler raised, and a registry that failed to sweep reported itself swept. The
oracle in each case is an absence — a handle that was never stopped, an episode
still registered, a pack that vanished — so every test asserts the absence
directly rather than trusting a return value.
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
from graphschema import (
    AttrSpec,
    AttrType,
    Node,
    NodeKind,
    Ontology,
    WorldGraph,
)
from openrange_pack_sdk import (
    Backing,
    Builder,
    BuildResult,
    EpisodeResult,
    FeasibilityVerdict,
    Manifest,
    Pack,
    PackPrior,
    RuntimeHandle,
    Snapshot,
    TaskFamily,
    TaskSpec,
)

from openrange.agent import SampleResult, arun_rollouts
from openrange.core.admit import admit
from openrange.core.episode import EpisodeService
from openrange.core.pack import PackRegistry

_ONTOLOGY = Ontology(
    id="lifetime@1",
    node_kinds={"box": NodeKind("box", attrs={"n": AttrSpec(AttrType.STRING)})},
    edge_kinds={},
)


class _PoolableHandle:
    """A ``PoolableRuntime`` that records whether anyone stopped it."""

    def __init__(self) -> None:
        self.stopped = False

    def reset(self) -> None: ...

    def surface(self) -> Mapping[str, Any]:
        return {"solver_root": "/tmp"}

    def poll_events(self) -> tuple[Mapping[str, Any], ...]:
        return ()

    def terminal(self) -> tuple[bool, str | None]:
        return False, None

    def checkpoint(self) -> Any:
        return None

    def restore(self, state: Any) -> None:
        del state

    def collect(self) -> Mapping[str, Any]:
        return {}

    def stop(self) -> None:
        self.stopped = True

    def poolable(self) -> bool:
        return True

    def reset_episode(self) -> None: ...


class _Family(TaskFamily):
    id = "lifetime.family"
    pack_id = "lifetime"

    def generate(
        self, graph: WorldGraph, manifest: Manifest, prior: PackPrior | None
    ) -> list[TaskSpec]:
        del graph, manifest, prior
        return [
            TaskSpec(
                id="lifetime.t.0",
                instruction="do the thing",
                entrypoints=("box.a",),
                goal_nodes=("box.a",),
                feasibility_check="lifetime.family",
                success_check="lifetime.family",
            )
        ]

    def check_feasibility(
        self, graph: WorldGraph, task: TaskSpec
    ) -> FeasibilityVerdict:
        del graph, task
        return FeasibilityVerdict(True)

    def check_success(
        self, graph: WorldGraph, task: TaskSpec, final_state: Mapping[str, Any]
    ) -> EpisodeResult:
        del graph, task, final_state
        return EpisodeResult(success=True)


class _Builder(Builder):
    def build(self, manifest: Manifest) -> BuildResult:
        del manifest
        graph = WorldGraph(ontology="lifetime@1")
        graph.add_node(Node("box.a", "box", attrs={"n": "a"}))
        return BuildResult(graph=graph, tasks=_Family().generate(graph, {}, None))


class _Pack(Pack):
    id = "lifetime"
    version = "0.1.0"

    def __init__(self) -> None:
        self.handles: list[_PoolableHandle] = []

    def ontology(self) -> Ontology:
        return _ONTOLOGY

    def invariants(self) -> list[Any]:
        return []

    def make_builder(self, prior: PackPrior | None) -> Builder:
        del prior
        return _Builder()

    def realize(self, graph: WorldGraph, backing: Backing) -> RuntimeHandle:
        del graph, backing
        handle = _PoolableHandle()
        self.handles.append(handle)
        return handle

    def task_families(self) -> list[TaskFamily]:
        return [_Family()]


def _snapshot(pack: _Pack) -> Snapshot:
    snap = admit(pack, manifest={"seed": 0, "runtime": {"tick": {"mode": "off"}}})
    assert isinstance(snap, Snapshot), snap
    return snap


def test_the_warm_pool_stops_the_runtime_it_displaces(tmp_path: Path) -> None:
    # Two *overlapping* episodes of one snapshot — the group shape a trainer
    # runs. Neither can reuse the other's warm slot, so both realize a runtime
    # and both park under the same key on the way out. Overwriting silently
    # drops the first, and because the dict never grows the eviction loop cannot
    # reclaim it either, so nothing would ever stop it.
    pack = _Pack()
    snap = _snapshot(pack)
    service = EpisodeService(pack, tmp_path)
    try:
        first_episode = service.start_episode(snap, snap.tasks[0].id)
        second_episode = service.start_episode(snap, snap.tasks[0].id)
        first, second = pack.handles[0], pack.handles[1]
        assert first is not second

        service.stop_episode(first_episode)
        parked_alive = not first.stopped  # warm, not stopped — the precondition

        service.stop_episode(second_episode)
        assert parked_alive, "the first runtime was stopped instead of parked"
        assert first.stopped, "the displaced runtime was never stopped"
        assert not second.stopped, "the survivor should still be warm"
    finally:
        service.close()


class _RaisingSampler:
    def complete(self, prompt: str, *, system: str | None = None) -> SampleResult:
        del prompt, system
        raise RuntimeError("gateway 503")


def _capability(surface: Mapping[str, Any]) -> Any:
    class _Cap:
        def run(self, command: str) -> str:
            del command
            return ""

        def close(self) -> None: ...

    del surface
    return _Cap()


def test_a_raising_sampler_does_not_strand_the_episode(tmp_path: Path) -> None:
    pack = _Pack()
    snap = _snapshot(pack)
    service = EpisodeService(pack, tmp_path)
    try:
        with pytest.raises(RuntimeError, match="gateway 503"):
            asyncio.run(
                arun_rollouts(
                    service,
                    snap,
                    _RaisingSampler(),
                    bind_run=_capability,
                    task_ids=[snap.tasks[0].id],
                )
            )
        assert not service._episodes, "the episode outlived the failure"
    finally:
        service.close()


def test_a_raising_bind_run_does_not_strand_the_episode(tmp_path: Path) -> None:
    # Binding the shell is the caller attaching to an already-realized world —
    # for a container, the docker exec. It is the most failure-prone step in the
    # loop and the episode is registered before it runs.
    pack = _Pack()
    snap = _snapshot(pack)
    service = EpisodeService(pack, tmp_path)

    def _explode(surface: Mapping[str, Any]) -> Any:
        del surface
        raise RuntimeError("docker exec setup failed")

    try:
        with pytest.raises(RuntimeError, match="docker exec setup failed"):
            asyncio.run(
                arun_rollouts(
                    service,
                    snap,
                    _RaisingSampler(),
                    bind_run=_explode,
                    task_ids=[snap.tasks[0].id],
                )
            )
        assert not service._episodes, "the episode outlived the bind failure"
    finally:
        service.close()


def test_a_raising_stop_does_not_mask_the_real_failure(tmp_path: Path) -> None:
    # stop_episode reaches pack code (`poolable()`) and the dashboard write,
    # neither of which it guards. A cleanup step that raises must not demote the
    # error it was cleaning up after.
    class _HostileHandle(_PoolableHandle):
        def poolable(self) -> bool:
            raise RuntimeError("poolable blew up")

    class _HostilePack(_Pack):
        def realize(self, graph: WorldGraph, backing: Backing) -> RuntimeHandle:
            del graph, backing
            handle = _HostileHandle()
            self.handles.append(handle)
            return handle

    pack = _HostilePack()
    snap = _snapshot(pack)
    service = EpisodeService(pack, tmp_path)
    try:
        with pytest.raises(RuntimeError, match="gateway 503"):
            asyncio.run(
                arun_rollouts(
                    service,
                    snap,
                    _RaisingSampler(),
                    bind_run=_capability,
                    task_ids=[snap.tasks[0].id],
                )
            )
    finally:
        service.close()


def test_one_failure_does_not_cancel_its_siblings(tmp_path: Path) -> None:
    # A bare gather propagates the first exception out of the loop, cancelling
    # every sibling mid-episode so none of them is ever stopped.
    pack = _Pack()
    snap = _snapshot(pack)
    service = EpisodeService(pack, tmp_path)
    task_id = snap.tasks[0].id

    class _OneBadSampler:
        def __init__(self) -> None:
            self.calls = 0

        def complete(self, prompt: str, *, system: str | None = None) -> SampleResult:
            del prompt, system
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("gateway 503")
            return SampleResult(text="```bash\nfinish done\n```")

    try:
        with pytest.raises(RuntimeError, match="gateway 503"):
            asyncio.run(
                arun_rollouts(
                    service,
                    snap,
                    _OneBadSampler(),
                    bind_run=_capability,
                    task_ids=[task_id] * 4,
                    max_concurrency=4,
                )
            )
        assert not service._episodes, "siblings were cancelled before stopping"
    finally:
        service.close()


def test_the_shipped_packs_all_resolve() -> None:
    # A pack that refuses to import takes the whole sweep down with it, and the
    # sweep is correctly never marked complete — so one pack's setup failure
    # would make every *other* pack permanently unreachable in this process.
    # PACKS is a module singleton; nothing recovers without a restart.
    from openrange.core.pack import PACKS

    ids = PACKS.ids()
    assert {"swe", "trading", "webapp"} <= set(ids), ids
    assert PACKS.ids() == ids  # and again — discovery is not one-shot-poisoned


def _install_broken_entry_point(root: Path, group: str, name: str) -> None:
    """Put a real distribution on ``sys.path`` whose entry point cannot load."""
    dist = root / "brokenpack-1.0.dist-info"
    dist.mkdir(parents=True)
    (dist / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: brokenpack\nVersion: 1.0\n", encoding="utf-8"
    )
    (dist / "entry_points.txt").write_text(
        f"[{group}]\n{name} = no_such_module_at_all:factory\n", encoding="utf-8"
    )


def test_a_malformed_pack_does_not_mark_the_registry_swept(tmp_path: Path) -> None:
    # Marking the sweep done before it runs makes the first call raise and every
    # call after it return a partial registry, silently, for the whole process.
    import importlib

    _install_broken_entry_point(tmp_path, "openrange.packs", "zzz.broken")
    sys.path.insert(0, str(tmp_path))
    importlib.invalidate_caches()
    try:
        registry = PackRegistry(autodiscover=True)
        with pytest.raises(Exception, match="zzz.broken"):
            registry.ids()
        # The second call must not pretend the sweep succeeded.
        with pytest.raises(Exception, match="zzz.broken"):
            registry.ids()
    finally:
        sys.path.remove(str(tmp_path))
        importlib.invalidate_caches()
