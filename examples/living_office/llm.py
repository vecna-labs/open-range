"""Seeded, cached OpenRouter gateway for deterministic LLM-driven NPCs.

All LLM calls in the demo go through SeededLLM. Cache keys embed (persona_id,
slot, seed, prompt_hash) so replays are byte-identical and admission stays
reproducible. Grok-4.1-fast is the default model — cheap, fast, reliable for
persona chatter and event reactions.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger("openrange.living_office.llm")

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "x-ai/grok-4.1-fast"
DEFAULT_CACHE_DIR = Path(__file__).parent / "cache"


@dataclass
class LLMUsage:
    calls: int = 0
    cache_hits: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0


class SeededLLM:
    """Deterministic OpenRouter client with on-disk cache.

    Cache key is sha256 over (model, seed, prompt, system). Every NPC decision
    that passes through this gateway is replayable with zero LLM cost on second
    run — which is how the demo stays cheap and deterministic.
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        cache_dir: Path | str = DEFAULT_CACHE_DIR,
        api_key: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.model = model
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        if not self.api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY missing — source env_dev.sh before starting"
            )
        self._client = httpx.Client(
            timeout=timeout,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://openrange.ai",
                "X-Title": "OpenRange Living Office Demo",
            },
        )
        self.usage = LLMUsage()

    def _cache_key(
        self, *, prompt: str, system: str, seed: int, extra: str = ""
    ) -> str:
        raw = f"{self.model}|{seed}|{system}|{prompt}|{extra}".encode()
        return hashlib.sha256(raw).hexdigest()[:32]

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def complete(
        self,
        prompt: str,
        *,
        system: str = "",
        seed: int = 0,
        max_tokens: int = 512,
        temperature: float = 0.0,
        response_format: dict | None = None,
        extra_cache_tag: str = "",
    ) -> str:
        key = self._cache_key(
            prompt=prompt, system=system, seed=seed, extra=extra_cache_tag
        )
        path = self._cache_path(key)
        if path.exists():
            self.usage.cache_hits += 1
            return json.loads(path.read_text(encoding="utf-8"))["text"]

        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "seed": seed,
        }
        if response_format:
            body["response_format"] = response_format

        t0 = time.time()
        try:
            response = self._client.post(OPENROUTER_URL, json=body)
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            logger.warning("OpenRouter call failed: %s", exc)
            return ""
        dt_ms = int((time.time() - t0) * 1000)

        choice = (data.get("choices") or [{}])[0]
        text = (choice.get("message") or {}).get("content") or ""
        usage = data.get("usage") or {}
        self.usage.calls += 1
        self.usage.prompt_tokens += int(usage.get("prompt_tokens", 0))
        self.usage.completion_tokens += int(usage.get("completion_tokens", 0))

        payload = {
            "text": text,
            "model": self.model,
            "seed": seed,
            "latency_ms": dt_ms,
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return text

    def complete_json(
        self,
        prompt: str,
        *,
        system: str = "",
        seed: int = 0,
        max_tokens: int = 1024,
        extra_cache_tag: str = "",
    ) -> dict | list:
        text = self.complete(
            prompt,
            system=system,
            seed=seed,
            max_tokens=max_tokens,
            temperature=0.0,
            response_format={"type": "json_object"},
            extra_cache_tag=extra_cache_tag,
        )
        return _extract_json(text)


def _extract_json(text: str) -> dict | list:
    if not text:
        return {}
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(stripped[start : end + 1])
            except json.JSONDecodeError:
                return {}
        start = stripped.find("[")
        end = stripped.rfind("]")
        if start >= 0 and end > start:
            try:
                return json.loads(stripped[start : end + 1])
            except json.JSONDecodeError:
                return {}
        return {}
