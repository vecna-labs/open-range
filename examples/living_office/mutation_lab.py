"""MutationLab — real self-mutation loop over admitted snapshots.

Uses OpenRange's actual primitives, zero mocks:

  FrontierMutationPolicy    →  score parents, propose mutations
  BuildPipeline.admit_child →  compile + admit child world (oracle-gated)
  FileSnapshotStore         →  persist admitted children
  ValidatorReport           →  expose admission verdicts + rejection reasons

The lab runs as a background asyncio task. After each episode completes, it
looks at the run's winner/continuity to synthesize PopulationStats, picks the
current snapshot as parent, proposes a mutation, and runs it through admission.

Rejections are preserved and streamed — they're evidence the oracle works.
"""

from __future__ import annotations

import asyncio
import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from open_range.config import DEFAULT_BUILD_CONFIG
from open_range.contracts.snapshot import Snapshot
from open_range.store import BuildPipeline, FileSnapshotStore, load_world_ir
from open_range.training.curriculum import (
    FrontierMutationPolicy,
    PopulationStats,
)

logger = logging.getLogger("openrange.living_office.mutation_lab")


@dataclass
class LineageNode:
    snapshot_id: str
    world_id: str
    generation: int
    parent_world_id: str = ""
    mutation_ops: tuple[str, ...] = ()
    status: str = "admitted"  # admitted | rejected | proposed
    reason: str = ""
    red_win_rate: float = 0.0
    blue_win_rate: float = 0.0
    episodes: int = 0
    last_red_reward: float = 0.0
    last_blue_reward: float = 0.0
    regret: float | None = None   # None until first episode observed
    validator_report: dict | None = None  # full ValidatorReport JSON, populated post-admission


@dataclass
class LineageState:
    nodes: list[LineageNode] = field(default_factory=list)

    def as_payload(self) -> list[dict[str, Any]]:
        return [
            {
                "snapshot_id": n.snapshot_id,
                "short_id": n.snapshot_id[-14:] if n.snapshot_id else "",
                "world_id": n.world_id,
                "generation": n.generation,
                "parent_world_id": n.parent_world_id,
                "mutation_ops": list(n.mutation_ops),
                "status": n.status,
                "reason": n.reason,
                "red_win_rate": round(n.red_win_rate, 3),
                "blue_win_rate": round(n.blue_win_rate, 3),
                "episodes": n.episodes,
                "last_red_reward": round(n.last_red_reward, 3),
                "last_blue_reward": round(n.last_blue_reward, 3),
                "regret": None if n.regret is None else round(n.regret, 3),
                "validator_report": n.validator_report,
            }
            for n in self.nodes
        ]


class MutationLab:
    """Runs the frontier-mutation loop against real admitted snapshots."""

    def __init__(
        self,
        store: FileSnapshotStore,
        broadcast: Any,
        *,
        policy: FrontierMutationPolicy | None = None,
    ) -> None:
        self.store = store
        self.broadcast = broadcast
        self.policy = policy or FrontierMutationPolicy()
        self.pipeline = BuildPipeline(store=store)
        self.lineage = LineageState()
        self._running: asyncio.Task | None = None
        self._outdir_root = Path(tempfile.mkdtemp(prefix="openrange-mutation-"))

    # --- seed lineage from an existing snapshot ---
    def seed_from(self, snapshot_id: str) -> None:
        """Load the active snapshot's lineage as the lab's genesis node."""
        try:
            world = load_world_ir(self.store, snapshot_id)
        except Exception as exc:
            logger.warning("seed_from %s failed: %s", snapshot_id, exc)
            return
        gen = getattr(world.lineage, "generation", 0)
        node = LineageNode(
            snapshot_id=snapshot_id,
            world_id=world.world_id,
            generation=gen,
            parent_world_id=getattr(world.lineage, "parent_world_id", "") or "",
            mutation_ops=tuple(getattr(world.lineage, "mutation_ops", ()) or ()),
            status="admitted",
            reason="seeded from store",
        )
        self.lineage.nodes.append(node)

    # --- kick off one mutation attempt ---
    async def propose_once(
        self,
        parent_snapshot_id: str,
        *,
        red_win_rate: float = 0.5,
        blue_win_rate: float = 0.5,
        episodes: int = 1,
        flake_rate: float = 0.0,
    ) -> LineageNode:
        """Run one: score parent → mutate → admit → report."""
        # Population stats from real episode outcomes
        try:
            parent_world = load_world_ir(self.store, parent_snapshot_id)
        except Exception as exc:
            raise RuntimeError(f"cannot load parent {parent_snapshot_id}: {exc}") from exc
        stats = PopulationStats(
            snapshot_id=parent_snapshot_id,
            world_id=parent_world.world_id,
            split="train",
            episodes=episodes,
            red_win_rate=red_win_rate,
            blue_win_rate=blue_win_rate,
            flake_rate=flake_rate,
            novelty=0.5,
            blue_signal_points=max(1, int(blue_win_rate * 6)),
        )

        # Announce the proposal stage so UI shows progress
        await self._emit_event("mutation_pending", {
            "parent_snapshot_id": parent_snapshot_id,
            "red_win_rate": red_win_rate,
            "blue_win_rate": blue_win_rate,
        })

        # Do the heavy work in a thread — keeps the WS loop responsive
        child_world = await asyncio.to_thread(self._mutate_world, parent_world, stats)

        # Emit the PROPOSED node so the researcher sees the planned ops live
        proposed_node = LineageNode(
            snapshot_id="",
            world_id=child_world.world_id,
            generation=child_world.lineage.generation,
            parent_world_id=parent_world.world_id,
            mutation_ops=tuple(child_world.lineage.mutation_ops),
            status="proposed",
            red_win_rate=red_win_rate,
            blue_win_rate=blue_win_rate,
            episodes=episodes,
        )
        self.lineage.nodes.append(proposed_node)
        await self._broadcast_lineage()

        # Inline admit_child so we can keep the ValidatorReport on both paths.
        outdir = self._outdir_root / child_world.world_id
        report = None
        try:
            candidate = await asyncio.to_thread(
                self.pipeline.build, child_world, outdir
            )
            reference_bundle, report = await asyncio.to_thread(
                self.pipeline.admission.admit,
                candidate.world,
                candidate.artifacts,
                candidate.build_config,
            )
            if report.admitted:
                snapshot: Snapshot = await asyncio.to_thread(
                    self.pipeline.store.create,
                    candidate.world,
                    candidate.artifacts,
                    reference_bundle,
                    report,
                    split="train",
                    synth=candidate.synth,
                )
                proposed_node.snapshot_id = snapshot.snapshot_id
                proposed_node.status = "admitted"
                proposed_node.reason = report.summary or "oracle passed all admission checks"
                await self._emit_event("mutation_admitted", {
                    "snapshot_id": snapshot.snapshot_id,
                    "world_id": snapshot.world_id,
                    "generation": proposed_node.generation,
                    "ops": list(proposed_node.mutation_ops),
                })
            else:
                proposed_node.status = "rejected"
                if report.summary:
                    proposed_node.reason = report.summary
                elif report.rejection_reasons:
                    proposed_node.reason = report.rejection_reasons[0]
                else:
                    proposed_node.reason = "candidate rejected"
                await self._emit_event("mutation_rejected", {
                    "world_id": child_world.world_id,
                    "ops": list(proposed_node.mutation_ops),
                    "reason": proposed_node.reason,
                })
        except Exception as exc:
            # pipeline.build (or an unexpected admission crash) blew up — fall
            # back to the prior exception path but synthesize a minimal report
            # so the frontend never sees a sudden None.
            logger.exception("mutation admit failed: %s", exc)
            proposed_node.status = "rejected"
            proposed_node.reason = f"pipeline error: {exc}"
            report = None  # keep None so synthesis below kicks in
            await self._emit_event("mutation_rejected", {
                "world_id": child_world.world_id,
                "ops": list(proposed_node.mutation_ops),
                "reason": str(exc),
            })
            proposed_node.validator_report = {
                "admitted": False,
                "summary": str(exc),
                "stages": [],
            }

        # Populate validator_report on both admitted and rejected paths when
        # we actually have a report. getattr guards against a weird future
        # report type that lacks model_dump.
        if report is not None:
            dump = getattr(report, "model_dump", lambda **kw: {})
            proposed_node.validator_report = dump(mode="json")
            # Focused stage-level event — full report also rides the lineage payload.
            await self._emit_event("mutation_stages", {
                "world_id": child_world.world_id,
                "snapshot_id": proposed_node.snapshot_id,
                "status": proposed_node.status,
                "stages": [s.model_dump(mode="json") for s in report.stages],
                "gates": {
                    "graph_ok": report.graph_ok,
                    "boot_ok": report.boot_ok,
                    "workflow_ok": report.workflow_ok,
                    "telemetry_ok": report.telemetry_ok,
                    "reference_attack_ok": report.reference_attack_ok,
                    "reference_defense_ok": report.reference_defense_ok,
                    "necessity_ok": report.necessity_ok,
                },
                "scores": {
                    "shortcut_risk": report.shortcut_risk,
                    "determinism_score": report.determinism_score,
                    "flakiness": report.flakiness,
                    "red_path_depth": report.red_path_depth,
                    "red_alt_path_count": report.red_alt_path_count,
                    "blue_signal_points": report.blue_signal_points,
                    "business_continuity_score": report.business_continuity_score,
                },
                "rejection_reasons": list(report.rejection_reasons),
            })

        await self._broadcast_lineage()
        return proposed_node

    def _mutate_world(self, parent_world, stats: PopulationStats):
        return self.policy.mutate(parent_world, parent_stats=stats)

    async def _broadcast_lineage(self) -> None:
        await self.broadcast({
            "type": "lineage",
            "nodes": self.lineage.as_payload(),
        })

    async def _emit_event(self, evtype: str, payload: dict[str, Any]) -> None:
        await self.broadcast({"type": evtype, **payload})

    async def record_episode_result(
        self, snapshot_id: str, red_reward: float, blue_reward: float
    ) -> None:
        """Record observed rewards for the snapshot and update its regret."""
        for node in self.lineage.nodes:
            if node.snapshot_id == snapshot_id:
                node.last_red_reward = red_reward
                node.last_blue_reward = blue_reward
                node.regret = abs(red_reward - blue_reward)
                node.episodes += 1
                await self._broadcast_lineage()
                return
        # Silently return if no matching node.

    def next_training_target(self) -> str:
        """Pick the admitted snapshot with regret nearest 0.5. Ties → most recent."""
        admitted = [
            (idx, n)
            for idx, n in enumerate(self.lineage.nodes)
            if n.status == "admitted" and n.snapshot_id
        ]
        if not admitted:
            return ""
        # Prefer nodes with observed regret inside the [0.2, 0.8] band,
        # ranked by closeness to 0.5, then by most-recent position.
        in_band = [
            (idx, n)
            for idx, n in admitted
            if n.regret is not None and 0.2 <= n.regret <= 0.8
        ]
        if in_band:
            in_band.sort(key=lambda pair: (abs(pair[1].regret - 0.5), -pair[0]))
            return in_band[0][1].snapshot_id
        # Fallback: newest admitted node that hasn't been evaluated yet.
        for idx, n in reversed(admitted):
            if n.regret is None:
                return n.snapshot_id
        # Final fallback: newest admitted node regardless of regret.
        return admitted[-1][1].snapshot_id

    def latest_admitted_id(self) -> str:
        """Deprecated alias for next_training_target()."""
        return self.next_training_target()
