"""Real SFT training loop — small model, expected to be bad, proves the stack.

Uses HuggingFaceTB/SmolLM2-135M-Instruct (smallest pragmatic), LoRA, on
live-generated decision-SFT rows from the OpenRange snapshot store.

This does three things end-to-end:
  1. generate a mini SFT dataset from the admitted snapshot
  2. fine-tune with LoRA on CPU/MPS (no CUDA required)
  3. stream per-step loss to the FastAPI WS as "training_step" events

Deliberately small: 40 steps, batch=1, seq=256, LR=3e-4. The point is
for researchers to see a real curve — even a noisy one — not a chart
baked from a static JSON.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger("openrange.living_office.trainer")


@dataclass
class TrainerConfig:
    model_id: str = "HuggingFaceTB/SmolLM2-135M-Instruct"
    max_steps: int = 40
    batch_size: int = 1
    seq_len: int = 256
    lr: float = 3e-4
    lora_r: int = 8
    lora_alpha: int = 16
    gradient_accumulation_steps: int = 2
    report_every: int = 1


class LiveTrainer:
    """Streams real training loss to the WS broadcast."""

    def __init__(self, broadcast: Callable, cfg: TrainerConfig | None = None) -> None:
        self.broadcast = broadcast
        self.cfg = cfg or TrainerConfig()
        self.task: asyncio.Task | None = None
        self.running = False
        self.losses: list[float] = []
        self.steps: list[int] = []

    async def start(self, traces_path: Path | None = None) -> dict[str, Any]:
        if self.running:
            return {"status": "already_running"}
        self.running = True
        self.losses.clear()
        self.steps.clear()
        await self._emit("training_started", {
            "model": self.cfg.model_id,
            "max_steps": self.cfg.max_steps,
            "lr": self.cfg.lr,
            "note": "small-on-purpose · expected to be poor",
        })
        self.task = asyncio.create_task(self._run(traces_path))
        return {"status": "started", "model": self.cfg.model_id}

    async def _run(self, traces_path: Path | None) -> None:
        try:
            await asyncio.to_thread(self._train_sync, traces_path)
        except Exception as exc:
            logger.exception("trainer crashed: %s", exc)
            await self._emit("training_error", {"message": str(exc)})
        finally:
            self.running = False
            await self._emit("training_done", {
                "final_loss": self.losses[-1] if self.losses else None,
                "steps": len(self.losses),
            })

    def _train_sync(self, traces_path: Path | None) -> None:
        """Synchronous training body — runs in a thread."""
        # Lazy imports so the whole server doesn't need torch at import time.
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dataset = _mini_dataset(traces_path)
        device = _pick_device()
        logger.info("trainer: device=%s dataset=%d samples", device, len(dataset))

        tok = AutoTokenizer.from_pretrained(self.cfg.model_id)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            self.cfg.model_id,
            torch_dtype=torch.float32,  # CPU/MPS friendly
        ).to(device)

        # Minimal LoRA: only attention q/v to keep it fast
        try:
            from peft import LoraConfig, get_peft_model
            peft_cfg = LoraConfig(
                r=self.cfg.lora_r, lora_alpha=self.cfg.lora_alpha,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, peft_cfg)
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            self._emit_sync("training_info", {"trainable_params": int(trainable), "device": str(device)})
        except Exception as exc:
            logger.warning("peft unavailable (%s) — training full model", exc)
            self._emit_sync("training_info", {"trainable_params": -1, "device": str(device), "note": "no LoRA"})

        opt = torch.optim.AdamW(
            (p for p in model.parameters() if p.requires_grad),
            lr=self.cfg.lr,
        )
        model.train()

        rng = random.Random(1234)
        accum = 0
        for step in range(1, self.cfg.max_steps + 1):
            if not self.running:
                break
            batch = rng.choice(dataset)
            enc = tok(
                batch,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.cfg.seq_len,
            ).to(device)
            labels = enc.input_ids.clone()
            labels[enc.attention_mask == 0] = -100
            out = model(**enc, labels=labels)
            loss = out.loss / self.cfg.gradient_accumulation_steps
            loss.backward()
            accum += 1
            if accum >= self.cfg.gradient_accumulation_steps:
                opt.step()
                opt.zero_grad(set_to_none=True)
                accum = 0

            loss_val = float(out.loss.detach().cpu())
            self.losses.append(loss_val)
            self.steps.append(step)

            if step % self.cfg.report_every == 0:
                self._emit_sync("training_step", {
                    "step": step,
                    "loss": round(loss_val, 4),
                    "max_steps": self.cfg.max_steps,
                })

    def _emit_sync(self, evtype: str, payload: dict[str, Any]) -> None:
        """Bridge to async broadcast from the training thread."""
        try:
            loop = asyncio.get_event_loop()
            fut = asyncio.run_coroutine_threadsafe(self._emit(evtype, payload), loop)
            fut.result(timeout=2)
        except Exception:
            pass

    async def _emit(self, evtype: str, payload: dict[str, Any]) -> None:
        try:
            await self.broadcast({"type": evtype, **payload})
        except Exception as exc:
            logger.debug("broadcast failed: %s", exc)


# ---- helpers ----

def _pick_device():
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _mini_dataset(traces_path: Path | None) -> list[str]:
    """Tiny, pragmatic SFT corpus: agent-thought triplets.

    If the caller passes a traces file, we use it. Otherwise we synthesize
    a handful of realistic red/blue scratch examples — enough to see the
    loss curve move during the demo without waiting for a trace run.
    """
    if traces_path and traces_path.exists():
        rows: list[str] = []
        with traces_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    text = obj.get("text") or obj.get("messages") or ""
                    if isinstance(text, list):
                        text = "\n".join(str(m.get("content", "")) for m in text if isinstance(m, dict))
                    if text:
                        rows.append(str(text))
                except Exception:
                    rows.append(line[:800])
                if len(rows) >= 64:
                    break
        if rows:
            return rows

    return [
        "<red> Probe the public HTTP service on svc-web to enumerate the application. </red>"
        " <observation> HTTP 200 OK · nginx/1.25 · x-powered-by hidden </observation>"
        " <red> Try common paths: /admin, /login, /api/v1/users </red>",

        "<blue> Poll svc-siem for malicious events since t=0. </blue>"
        " <observation> 1 event matched: InitialAccess on svc-web </observation>"
        " <blue> submit_finding InitialAccess svc-web </blue>",

        "<red> No SQL errors on /api/search. Try LFI via ../../../etc/passwd. </red>"
        " <observation> 400 Bad Request </observation>"
        " <red> Fall back to reflected XSS probe: <script>alert(1)</script></red>",

        "<blue> ContainmentApplied not triggered. Apply control to svc-web before asset_read. </blue>"
        " <observation> path_broken=true, containment=succeeded </observation>"
        " <blue> Monitor continuity — green workflow on svc-fileshare intact </blue>",

        "<red> Credentials obtained (idp_admin_cred). Pivot to svc-fileshare for finance_docs. </red>"
        " <observation> asset_read(finance_docs)=granted </observation>"
        " <red> Mission satisfied. </red>",
    ]
