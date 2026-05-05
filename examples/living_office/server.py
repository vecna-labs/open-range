"""FastAPI + WebSocket server for the Living Office example.

Boots an OpenRange runtime with LLMGreenScheduler, runs an episode loop with
LLM red + blue agents, and streams every tick, action, event, and speech
bubble to the 3D frontend.

Endpoints:
  POST /session/start      →  admit/load + reset + start episode loop
  POST /session/stop       →  clean close
  GET  /session/summary    →  current score + scene summary
  WS   /stream             →  live events (scene_init, tick, action, event,
                              speech, reward, done)
  GET  /                   →  static 3D dashboard (index.html)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from open_range.config import EpisodeConfig
from open_range.runtime import OpenRangeRuntime
from open_range.sdk.client import OpenRange
from open_range.store import FileSnapshotStore

from .agents import LLMAgent
from .llm import SeededLLM
from .llm_green import LLMGreenScheduler, ObservableEvent
from .mutation_lab import MutationLab

logger = logging.getLogger("openrange.living_office.server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def _docker_info() -> dict[str, Any]:
    """Report the live Kind control-plane container if it's reachable."""
    import subprocess
    try:
        out = subprocess.check_output(
            ["docker", "ps", "--filter", "name=openrange-control-plane",
             "--format", "{{.ID}}|{{.Names}}|{{.Image}}|{{.Status}}"],
            timeout=2.5, stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return {"present": False}
    if not out:
        return {"present": False}
    parts = (out.splitlines()[0].split("|") + ["", "", "", ""])[:4]
    cid, name, image, status = parts
    return {
        "present": True,
        "container_id": cid,
        "short_id": cid[:12],
        "name": name,
        "image": image,
        "status": status,
    }

EXAMPLE_DIR = Path(__file__).parent
STATIC_DIR = EXAMPLE_DIR / "static"
DEFAULT_STORE = Path(os.environ.get("OPENRANGE_STORE_DIR", "/tmp/demo-snaps"))
def _discover_default_snapshot(store_dir: Path) -> str:
    """Pick the newest admitted snapshot under the store — no hardcoded id."""
    env_id = os.environ.get("OPENRANGE_SNAPSHOT_ID")
    if env_id:
        return env_id
    if not store_dir.exists():
        return ""
    candidates = sorted(
        (p for p in store_dir.iterdir() if (p / "snapshot.json").exists()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0].name if candidates else ""


DEFAULT_SNAPSHOT_ID = _discover_default_snapshot(DEFAULT_STORE)
MAX_TURNS = int(os.environ.get("OPENRANGE_MAX_TURNS", "24"))
TICK_INTERVAL_S = float(os.environ.get("OPENRANGE_TICK_INTERVAL_S", "3.0"))


class StartRequest(BaseModel):
    snapshot_id: str | None = None
    max_turns: int = MAX_TURNS


class DemoSession:
    def __init__(self) -> None:
        self.llm = SeededLLM()
        self.store = FileSnapshotStore(store_dir=DEFAULT_STORE)
        self.observable_sink: list[ObservableEvent] = []
        self.green = LLMGreenScheduler(self.llm, observable_sink=self.observable_sink)
        self.runtime = OpenRangeRuntime(green_scheduler=self.green)
        self.live_backend = self._maybe_live_backend()
        self.env = OpenRange(store=self.store, runtime=self.runtime, live_backend=self.live_backend)
        self.red = LLMAgent("red", self.llm)
        self.blue = LLMAgent("blue", self.llm)
        self.ws_clients: set[WebSocket] = set()
        self.loop_task: asyncio.Task | None = None
        self.active = False
        self.snapshot_id: str = ""
        self.last_summary: dict[str, Any] = {}
        self.last_scene_init: dict[str, Any] | None = None
        self.mutation_lab = MutationLab(store=self.store, broadcast=self.broadcast)
        self.last_lineage_payload: dict[str, Any] | None = None
        # Live RL reward history — populated from real episode rewards
        self.rl_history: dict[str, list[Any]] = {"red": [], "blue": [], "steps": [], "actor": []}
        self.last_docker_info: dict[str, Any] = _docker_info()
        self.docker_poll_task: asyncio.Task | None = None
        self.episode_state: dict[str, Any] = {
            "sim_time": 0.0, "turn": 0, "actor": "",
            "red": 0.0, "blue": 0.0, "continuity": 1.0,
            "events_count": 0, "service_health": [],
        }

    async def _run_mutation_cycle(self, red_won: float, blue_won: float) -> None:
        """After an episode finishes, propose one mutation + run admission."""
        if not self.snapshot_id:
            return
        try:
            await self.mutation_lab.propose_once(
                self.snapshot_id,
                red_win_rate=red_won,
                blue_win_rate=blue_won,
                episodes=1,
            )
        except Exception as exc:
            logger.exception("mutation cycle failed: %s", exc)
            await self.broadcast({"type": "error", "message": f"mutation: {exc}"})

    def _maybe_live_backend(self):
        use_live = os.environ.get("OPENRANGE_LIVE", "0") in {"1", "true", "kind"}
        if use_live and KindBackend is not None:
            try:
                return KindBackend()
            except Exception as exc:
                logger.warning("KindBackend unavailable, falling back to offline: %s", exc)
                return None
        return None

    async def broadcast(self, message: dict[str, Any]) -> None:
        dead: list[WebSocket] = []
        for ws in list(self.ws_clients):
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.ws_clients.discard(ws)

    async def start(self, snapshot_id: str | None, max_turns: int) -> dict[str, Any]:
        if self.active:
            await self.stop()
        snap_id = snapshot_id or DEFAULT_SNAPSHOT_ID
        self.snapshot_id = snap_id
        episode_config = EpisodeConfig(
            mode="joint_pool",
            scheduler_mode="strict_turns",
            green_profile="medium",
            green_branch_backend="workflow_orchestrator",
            opponent_red="none",
            opponent_blue="none",
            episode_horizon_minutes=45.0,
        )
        state = self.env.reset(snap_id, episode_config)
        self.red = LLMAgent("red", self.llm, world_id=snap_id)
        self.blue = LLMAgent("blue", self.llm, world_id=snap_id)
        scene = self._scene_init_payload(state)
        self.last_scene_init = {"type": "scene_init", **scene}
        await self.broadcast(self.last_scene_init)

        # Seed the mutation lineage with the active snapshot if not already tracked
        already = any(n.snapshot_id == snap_id for n in self.mutation_lab.lineage.nodes)
        if not already:
            self.mutation_lab.seed_from(snap_id)
            await self.mutation_lab._broadcast_lineage()
        self.active = True
        self.loop_task = asyncio.create_task(self._run_loop(max_turns))
        if self.docker_poll_task is None or self.docker_poll_task.done():
            self.docker_poll_task = asyncio.create_task(self._docker_poll_loop())
        self.last_summary = {
            "snapshot_id": snap_id,
            "personas": len(self.green.rich_personas()),
            "started_at": time.time(),
        }
        return self.last_summary

    async def stop(self) -> None:
        self.active = False
        if self.loop_task and not self.loop_task.done():
            self.loop_task.cancel()
            try:
                await self.loop_task
            except (asyncio.CancelledError, Exception):
                pass
        if self.docker_poll_task and not self.docker_poll_task.done():
            self.docker_poll_task.cancel()
            try:
                await self.docker_poll_task
            except (asyncio.CancelledError, Exception):
                pass
        try:
            self.env.close()
        except Exception:
            pass

    async def _docker_poll_loop(self) -> None:
        """Every 15s, re-check the Kind control-plane container and broadcast changes."""
        try:
            while self.active:
                await asyncio.sleep(15)
                info = _docker_info()
                if info != self.last_docker_info:
                    self.last_docker_info = info
                    await self.broadcast({"type": "docker", **info})
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("docker poll loop crashed: %s", exc)

    def _score_to_payload(self, score) -> dict[str, Any]:
        """Serialize an EpisodeScore for the wire, tolerating missing audit."""
        dump = score.model_dump(mode="json") if hasattr(score, "model_dump") else {}
        return dump

    async def _run_loop(self, max_turns: int) -> None:
        try:
            for turn in range(max_turns):
                if not self.active:
                    break
                await self._flush_observable(where="pre")
                decision = self.env.next_decision()
                obs = decision.obs
                actor = decision.actor
                await self.broadcast(
                    {
                        "type": "turn",
                        "turn": turn,
                        "actor": actor,
                        "sim_time": round(obs.sim_time, 3),
                        "reward_delta": round(obs.reward_delta, 4),
                        "stdout_preview": (obs.stdout or "")[:200],
                        "visible_events": [
                            {
                                "time": round(ev.time, 2),
                                "event_type": ev.event_type,
                                "source": ev.source_entity,
                                "target": ev.target_entity,
                            }
                            for ev in (obs.visible_events or ())[:8]
                        ],
                        "service_health": [
                            {"service_id": e.service_id, "health": round(e.health, 3)}
                            for e in (obs.service_health or ())
                        ],
                    }
                )
                agent = self.red if actor == "red" else self.blue
                decision_out = agent.decide(obs)
                action = decision_out.action
                await self.broadcast(
                    {
                        "type": "agent_action",
                        "actor": actor,
                        "kind": action.kind,
                        "payload": action.payload,
                        "thought": decision_out.thought,
                        "sim_time": round(obs.sim_time, 3),
                    }
                )
                result = self.env.act(actor, action)
                for ev in result.emitted_events or ():
                    await self.broadcast(
                        {
                            "type": "event",
                            "event_type": ev.event_type,
                            "source": ev.source_entity,
                            "target": ev.target_entity,
                            "time": round(ev.time, 2),
                            "actor": ev.actor,
                            "malicious": ev.malicious,
                        }
                    )
                await self._flush_observable(where="post")
                score = self.env.score()
                # Append to rolling RL history for the live chart
                step_idx = len(self.rl_history["steps"])
                self.rl_history["steps"].append(step_idx)
                self.rl_history["actor"].append(actor)
                self.rl_history["red"].append(round(score.red_reward, 4))
                self.rl_history["blue"].append(round(score.blue_reward, 4))
                for k in ("steps", "red", "blue", "actor"):
                    if len(self.rl_history[k]) > 120:
                        self.rl_history[k] = self.rl_history[k][-120:]
                await self.broadcast(
                    {
                        "type": "reward",
                        "red": round(score.red_reward, 4),
                        "blue": round(score.blue_reward, 4),
                        "continuity": round(score.continuity, 3),
                        "winner": score.winner,
                        "done": score.done,
                        "sim_time": round(score.sim_time, 3),
                        "history_index": len(self.rl_history["steps"]) - 1,
                    }
                )
                # Per-turn audit snapshot — tolerate missing audit gracefully.
                aud = getattr(score, "audit", None)
                if aud is not None:
                    await self.broadcast({
                        "type": "audit",
                        "turn": turn,
                        "action_count": aud.action_count,
                        "unique_fingerprints": aud.unique_fingerprints,
                        "action_diversity_score": aud.action_diversity_score,
                        "collapse_warning": aud.collapse_warning,
                        "role_diversity": [
                            {
                                "actor": rd.actor,
                                "unique_fingerprints": rd.unique_fingerprints,
                                "diversity_score": rd.diversity_score,
                                "dominant_share": rd.dominant_share,
                                "collapse_warning": rd.collapse_warning,
                            }
                            for rd in aud.role_diversity
                        ],
                        "binary_integrity_changes": len(
                            getattr(aud.binary_integrity, "deltas", []) or []
                        ),
                    })
                if score.done:
                    await self.broadcast(
                        {
                            "type": "done",
                            "winner": score.winner,
                            "terminal_reason": score.terminal_reason,
                            "red_reward": round(score.red_reward, 4),
                            "blue_reward": round(score.blue_reward, 4),
                        }
                    )
                    await self.mutation_lab.record_episode_result(
                        self.snapshot_id, score.red_reward, score.blue_reward,
                    )
                    await self.broadcast({
                        "type": "episode_final_score",
                        **self._score_to_payload(score),
                    })
                    # Fire the real mutation loop using actual episode outcome.
                    red_won = 1.0 if score.winner == "red" else 0.0
                    blue_won = 1.0 if score.winner == "blue" else 0.0
                    asyncio.create_task(self._run_mutation_cycle(red_won, blue_won))
                    break
                await asyncio.sleep(TICK_INTERVAL_S)
            else:
                # Exhausted max_turns without a terminal score.done — still record
                # the episode so regret gating has a signal, and fire a mutation
                # so the lineage grows even on horizon-truncated episodes.
                try:
                    final_score = self.env.score()
                    await self.mutation_lab.record_episode_result(
                        self.snapshot_id,
                        final_score.red_reward,
                        final_score.blue_reward,
                    )
                    await self.broadcast({
                        "type": "episode_final_score",
                        **self._score_to_payload(final_score),
                    })
                    red_won = 1.0 if final_score.red_reward > final_score.blue_reward else 0.0
                    blue_won = 1.0 - red_won
                    asyncio.create_task(self._run_mutation_cycle(red_won, blue_won))
                except Exception as exc:
                    logger.warning("horizon-end record failed: %s", exc)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("Episode loop crashed: %s", exc)
            await self.broadcast({"type": "error", "message": str(exc)})
        finally:
            self.active = False

    async def _flush_observable(self, *, where: str) -> None:
        if not self.observable_sink:
            return
        pending = list(self.observable_sink)
        self.observable_sink.clear()
        for ev in pending:
            await self.broadcast(
                {
                    "type": "speech",
                    "persona_id": ev.persona_id,
                    "display_name": ev.display_name,
                    "channel": ev.channel,
                    "text": ev.text,
                    "kind": ev.kind,
                    "activity": getattr(ev, "activity", ""),
                    "time": round(ev.time, 2),
                    "phase": where,
                }
            )

    def _scene_init_payload(self, state: Any) -> dict[str, Any]:
        rich = self.green.rich_personas()
        plans = self.green.day_plan()
        personas_payload = []
        for rp in rich:
            steps = plans.get(rp.persona_id, [])
            personas_payload.append(
                {
                    "persona_id": rp.persona_id,
                    "display_name": rp.display_name,
                    "role": rp.role,
                    "title": rp.title,
                    "tone": rp.tone,
                    "quirks": rp.quirks,
                    "tenure_years": rp.tenure_years,
                    "pronouns": rp.pronouns,
                    "interests": rp.interests,
                    "desk_zone": rp.desk_zone,
                    "desk_x": rp.desk_x,
                    "desk_y": rp.desk_y,
                    "reports_to": rp.reports_to,
                    "peers": rp.peers,
                    "steps": [asdict(step) for step in steps],
                }
            )
        services = self._services_payload()
        return {
            "snapshot_id": getattr(state, "snapshot_id", ""),
            "episode_id": getattr(state, "episode_id", ""),
            "execution_mode": getattr(state, "execution_mode", "offline"),
            "sim_time": getattr(state, "sim_time", 0.0),
            "personas": personas_payload,
            "services": services,
            "zones": self._zones_payload(),
            "llm_model": self.llm.model,
            "tick_interval_s": TICK_INTERVAL_S,
            "live_backend": "kind" if self.live_backend is not None else "offline",
            "docker": _docker_info(),
        }

    def _zones_payload(self) -> list[str]:
        snapshot = getattr(self.runtime, "_snapshot", None) or self.green._snapshot
        world = getattr(snapshot, "world", None) if snapshot is not None else None
        if world is None:
            return []
        zones = [h.zone for h in getattr(world, "hosts", []) if h.zone]
        return list(dict.fromkeys(zones))

    def _services_payload(self) -> list[dict[str, Any]]:
        """Derive services from the snapshot's WorldIR — no hardcoded list.

        Each service is placed either on the external street (public-facing
        kinds) or grouped by zone inside the office. Coordinates are hints —
        the frontend can choose its own final placement per room.
        """
        snapshot = getattr(self.runtime, "_snapshot", None) or self.green._snapshot
        world = getattr(snapshot, "world", None) if snapshot is not None else None
        if world is None:
            return []
        public_kinds = {"web_app", "email"}
        host_zone = {h.id: h.zone for h in getattr(world, "hosts", [])}
        out: list[dict[str, Any]] = []
        for svc in world.services:
            svc_zone = host_zone.get(getattr(svc, "host", ""), "corp")
            zone = "external" if svc.kind in public_kinds else svc_zone
            out.append({
                "id": svc.id,
                "kind": svc.kind,
                "zone": zone,
                "ports": list(getattr(svc, "ports", []) or []),
            })
        return out


SESSION = DemoSession()


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    await SESSION.stop()


app = FastAPI(title="OpenRange Living Office Demo", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.post("/session/start")
async def session_start(req: StartRequest):
    info = await SESSION.start(req.snapshot_id, req.max_turns)
    return info


@app.post("/session/stop")
async def session_stop():
    await SESSION.stop()
    return {"stopped": True}


@app.post("/mutation/propose")
async def mutation_propose():
    """Manually trigger a mutation attempt against the active snapshot."""
    if not SESSION.snapshot_id:
        return JSONResponse({"error": "no active snapshot"}, status_code=400)
    asyncio.create_task(SESSION._run_mutation_cycle(red_won=0.5, blue_won=0.5))
    return {"proposed": True, "parent": SESSION.snapshot_id}


@app.post("/mutation/promote")
async def mutation_promote():
    """Start the next episode against the latest admitted child snapshot."""
    sid = SESSION.mutation_lab.next_training_target()
    if not sid:
        return JSONResponse({"error": "no admitted child snapshot"}, status_code=400)
    asyncio.create_task(SESSION.start(sid, MAX_TURNS))
    return {"promoted": sid}


@app.get("/mutation/lineage")
async def mutation_lineage():
    return {"nodes": SESSION.mutation_lab.lineage.as_payload()}


@app.get("/rl/history")
async def rl_history():
    """Live GRPO-style reward history from actual episode turns."""
    return SESSION.rl_history


@app.get("/traces")
async def traces(limit: int = 30):
    """Last N LLM calls from the SeededLLM disk cache — inputs, timing, tokens."""
    cache_dir = SESSION.llm.cache_dir
    if not cache_dir.exists():
        return {"items": []}
    # pick newest N by mtime, decode, trim the 'text' field for wire size
    files = sorted(cache_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]
    items = []
    for path in files:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        items.append({
            "key": path.stem,
            "model": data.get("model", ""),
            "seed": data.get("seed", 0),
            "latency_ms": data.get("latency_ms", 0),
            "prompt_tokens": data.get("prompt_tokens", 0),
            "completion_tokens": data.get("completion_tokens", 0),
            "mtime": int(path.stat().st_mtime),
            "preview": (data.get("text") or "")[:140],
        })
    items.sort(key=lambda x: x["mtime"], reverse=True)
    totals = {
        "calls": len(items),
        "prompt_tokens": sum(i["prompt_tokens"] for i in items),
        "completion_tokens": sum(i["completion_tokens"] for i in items),
        "median_latency_ms": sorted(i["latency_ms"] for i in items)[len(items)//2] if items else 0,
    }
    return {"items": items, "totals": totals}


@app.get("/session/summary")
async def session_summary():
    return SESSION.last_summary or {"active": SESSION.active}


@app.websocket("/stream")
async def websocket_stream(ws: WebSocket):
    await ws.accept()
    SESSION.ws_clients.add(ws)
    await ws.send_json({"type": "hello", "model": SESSION.llm.model, "snapshot_id": SESSION.snapshot_id, "active": SESSION.active})
    # Re-send the current scene + lineage to late joiners so they see the live state.
    if SESSION.last_scene_init is not None:
        await ws.send_json(SESSION.last_scene_init)
    if SESSION.mutation_lab.lineage.nodes:
        await ws.send_json({
            "type": "lineage",
            "nodes": SESSION.mutation_lab.lineage.as_payload(),
        })
    try:
        while True:
            msg = await ws.receive_text()
            try:
                parsed = json.loads(msg)
            except json.JSONDecodeError:
                continue
            if parsed.get("cmd") == "start":
                asyncio.create_task(
                    SESSION.start(parsed.get("snapshot_id"), int(parsed.get("max_turns", MAX_TURNS)))
                )
            elif parsed.get("cmd") == "stop":
                asyncio.create_task(SESSION.stop())
    except WebSocketDisconnect:
        SESSION.ws_clients.discard(ws)
    except Exception:
        SESSION.ws_clients.discard(ws)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "examples.living_office.server:app",
        host=os.environ.get("OPENRANGE_DEMO_HOST", "127.0.0.1"),
        port=int(os.environ.get("OPENRANGE_DEMO_PORT", "8010")),
        reload=False,
        log_level="info",
    )
