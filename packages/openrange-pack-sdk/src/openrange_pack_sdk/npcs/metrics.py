"""Deterministic, model-free metrics for a persona population.

These make the feature's claims measurable in CI (no model needed): a sampled
population is diverse (role entropy), and the persona prompt doesn't read like an
assistant (tell rate). Live believability against a real model — an LLM-judge
adherence score, a non-degeneracy A/B — is a separate, model-gated follow-up.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable

# Phrases a helpful-assistant or a model reasoning out loud emits, but an
# in-character persona should not. The meta-reasoning group (the second block)
# was added after a small local model leaked its chain of thought as prose
# ("the user wants me to act as Dana...") — a real believability failure the
# helpful-assistant patterns alone missed.
_ASSISTANT_TELLS = (
    re.compile(r"\bas an ai\b"),
    re.compile(r"\blanguage model\b"),
    re.compile(r"\bhow (can|may) i (help|assist)\b"),
    re.compile(r"\bi (cannot|can't|am unable to)\b"),
    re.compile(r"\bi'?m (happy|glad) to help\b"),
    re.compile(r"^#{1,6}\s", re.MULTILINE),  # markdown headers
    re.compile(r"\b(best regards|sincerely|kind regards)\b"),
    re.compile(r"\bthe user\b"),
    re.compile(r"\bact(ing)? as\b"),
    re.compile(r"\b(stay|staying|stick|in) (in )?character\b"),
    re.compile(r"\bmy (character|persona|role|goal is)\b"),
)


def role_entropy(roles: Iterable[str], universe_size: int) -> float:
    """Shannon entropy of a role distribution, normalized to ``[0, 1]`` against
    ``universe_size`` distinct roles (1.0 == perfectly uniform over the vocab)."""
    counts = Counter(r for r in roles)
    total = sum(counts.values())
    if total == 0 or universe_size <= 1:
        return 0.0
    h = -sum((c / total) * math.log(c / total) for c in counts.values())
    return h / math.log(universe_size)


def assistant_tell_rate(texts: Iterable[str]) -> float:
    """Fraction of texts that contain at least one assistant-tell phrase."""
    items = list(texts)
    if not items:
        return 0.0
    hits = sum(1 for t in items if any(p.search(t.lower()) for p in _ASSISTANT_TELLS))
    return hits / len(items)
