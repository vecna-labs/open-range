"""Believability eval: drive sampled personas through a REAL local model and
measure how in-character they stay — the model-gated companion to the
deterministic CI checks in `openrange_pack_sdk.npcs.metrics`.

Run:  ollama serve && ollama pull qwen3-4b-40k
      uv run --extra strands python examples/persona_believability_eval.py

It skips cleanly (exit 0) when no Ollama model is reachable, so it's a runnable
artifact rather than a hard dependency. Numbers vary by model; a small *thinking*
model tends to leak reasoning as prose ("the user wants me to act as ...") — the
assistant_tell_rate catches that. Prefer a non-thinking instruct model for real
cover traffic.
"""

from __future__ import annotations

import re
import time
import urllib.request

from openrange_pack_sdk import render_persona, sample_persona
from openrange_pack_sdk.npcs.metrics import assistant_tell_rate, role_entropy

MODEL = "qwen3-4b-40k"
HOST = "http://localhost:11434"
ROLES = ["accountant", "it admin", "sales rep", "office manager", "developer"]
N = 5


def _ollama_ready() -> bool:
    try:
        with urllib.request.urlopen(f"{HOST}/api/tags", timeout=4) as r:
            return MODEL.split(":")[0] in r.read().decode()
    except Exception:
        return False


def _utterance(text: str) -> str:
    # strip any think block, then take the model's final non-empty line as what
    # it "said" (thinking models put the utterance after their reasoning).
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def main() -> int:
    if not _ollama_ready():
        print(f"skip: no Ollama model {MODEL!r} at {HOST}")
        return 0

    try:
        from strands import Agent
        from strands.models.ollama import OllamaModel
        from strands.types.exceptions import MaxTokensReachedException
    except ImportError as exc:
        print(f"skip: needs `uv pip install strands-agents ollama` ({exc})")
        return 0

    model = OllamaModel(
        host=HOST, model_id=MODEL, options={"num_predict": 1000, "temperature": 0.8}
    )
    personas = [sample_persona(i, roles=ROLES) for i in range(N)]
    utterances: list[str] = []

    print(f"model={MODEL}  personas={N}\n" + "=" * 72)
    for cfg in personas:
        agent = Agent(
            model=model, system_prompt=render_persona(cfg), callback_handler=None
        )
        t0 = time.time()
        try:
            out = str(
                agent(
                    "Slow afternoon at work. Say the next thing you'd naturally "
                    "say to a colleague — one short line, in character."
                )
            )
        except MaxTokensReachedException:
            out = ""
        if not out:
            for msg in reversed(agent.messages):
                if msg.get("role") == "assistant":
                    out = "".join(
                        b.get("text", "")
                        for b in msg.get("content", [])
                        if isinstance(b, dict)
                    )
                    break
        line = _utterance(out)
        utterances.append(line)
        print(f"[{cfg['role']:<14}] {time.time() - t0:4.0f}s  {line[:90]!r}")

    entropy = role_entropy([str(c["role"]) for c in personas], len(ROLES))
    unique_ids = len({str(c["name"]) for c in personas})
    tell = assistant_tell_rate(utterances)
    print("=" * 72)
    print(f"population role entropy : {entropy:.2f}")
    print(f"actor-id uniqueness     : {unique_ids}/{N}")
    print(f"assistant-tell rate     : {tell:.2f}  (lower is better)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
