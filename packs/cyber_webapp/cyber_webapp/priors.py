"""Hand-authored default ``PackPrior`` for the cyber_webapp pack.

``default_prior()`` is the boot-path source of generation knowledge — the
prior ``WebappPack.make_builder(prior=None)`` falls back to before the
flywheel has produced any distilled BBG-driven prior. It carries the
same generic-stats shape ``openrange.core.distill`` emits (a
``PackPrior`` from ``openrange.core.pack``), so the builder has one
code path and never knows whether its prior was learned or authored.

The keys live in two places, by design:

  * ``PackPrior`` — generic graph statistics only. ``topology``,
    ``task_seeds``, ``difficulty``, ``coverage``. The builder
    INTERPRETS these into domain decisions; the prior never tells the
    builder what to do. This is what keeps the prior shape reusable
    across packs and identical to what ``distill()`` would produce.

  * ``_CYBER_GENERATION_CONFIG`` — pack-specific sampling knobs that
    have no place in the generic ``PackPrior``: service-kind weights,
    vulnerability catalog weights, chain depth caps. The pack's
    sampler (``sampling.py``, owned by another module) reads these
    when it lowers generic kind-frequencies into concrete service /
    vuln picks. These knobs would be a cyber-specific leak in
    ``PackPrior.topology`` — they belong inside the pack.

Each ``default_prior()`` call returns a fresh ``PackPrior`` so callers
can mutate it (e.g. a curriculum step re-weighting ``task_seeds``)
without affecting other consumers.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Any

from cyber_webapp.ontology_v2 import webapp_ontology
from openrange.core.pack import PackPrior, TaskSeed

# ---------------------------------------------------------------------------
# Pack-private generation config — NOT in PackPrior
# ---------------------------------------------------------------------------

# Cyber-specific sampler knobs the procedural builder reads when it lowers
# the generic ``PackPrior.topology`` frequencies into concrete service /
# vulnerability picks. These keys are deliberately ABSENT from the generic
# ``PackPrior`` so the prior shape stays domain-agnostic (and identical to
# what ``distill()`` emits). The webapp sampler in ``sampling.py`` imports
# this constant directly.
_CYBER_GENERATION_CONFIG: dict[str, Any] = {
    # service-count range — discrete uniform sample
    "service_count": {"min": 2, "max": 5},
    # service-kind weights — "web" is forced (one web service always); the
    # remaining services are drawn from this weighted set.
    "service_kinds": {
        "web": 0,
        "api": 3,
        "auth": 2,
        "db": 4,
    },
    # endpoints sampled per service — discrete uniform
    "endpoints_per_service": {"min": 1, "max": 3},
    # vulnerability-count range — discrete uniform
    "vuln_count": {"min": 1, "max": 3},
    # catalog ids (kept thin in v1 — broader catalog lives in
    # cyber_webapp.vulnerabilities)
    "vuln_kinds": {
        "sql_injection": 3,
        "ssrf": 2,
        "broken_authz": 2,
    },
    # account-count range — discrete uniform
    "account_count": {"min": 1, "max": 3},
    # how many vulns may form an `enables` chain. v1 caps at short chains;
    # MCTS curriculum (later) scales this intelligently.
    "vuln_chain_depth": {"min": 1, "max": 2},
}


# ---------------------------------------------------------------------------
# default_prior — the hand-authored PackPrior
# ---------------------------------------------------------------------------


def default_prior() -> PackPrior:
    """Return a fresh hand-authored ``PackPrior`` for the cyber_webapp pack.

    The values describe a "typical small webapp world" — roughly the shape
    a procedural sampler would produce at defaults: a couple of hosts, a
    handful of services exposing several endpoints, a small data store
    cluster, one secret, one to two vulnerabilities. These are the same
    GENERIC graph statistics ``distill()`` emits from a real BBG, so the
    builder code path is identical for hand-authored and distilled priors.

    Returns a new instance per call so callers may mutate it freely.
    """
    # Expected node-kind counts in a typical webapp world. All keys are
    # node kinds declared in ``webapp_ontology()`` — generic shape, no
    # cyber-only vocabulary leaks into the topology bag.
    node_kind_freq: dict[str, int] = {
        "host": 2,
        "service": 3,
        "endpoint": 5,
        "data_store": 2,
        "record": 4,
        "secret": 1,
        "vulnerability": 2,
        "account": 3,
        "credential": 3,
        "network": 1,
    }
    # Salient subset — what a builder should weight heavily. These are
    # the kinds that, in the agent's spatial memory, would be reached
    # by `record_thought` anchors (endpoints the agent reasoned about,
    # secrets it confirmed, vulns it exploited).
    salient_kind_freq: dict[str, int] = {
        "endpoint": 1,
        "secret": 1,
        "vulnerability": 1,
    }
    # Hand-authored expected dead-end ratio in a typical webapp world —
    # a fraction of paths look like progress and are not.
    dead_end_ratio: float = 0.2
    # Where to place hidden state. Keyed by node-kind; the builder
    # reads this to decide which kinds get Visibility.HIDDEN.
    hidden_signal: dict[str, int] = {
        "secret": 1,
        "vulnerability": 1,
    }

    topology: dict[str, Any] = {
        "node_kind_freq": node_kind_freq,
        "salient_kind_freq": salient_kind_freq,
        "dead_end_ratio": dead_end_ratio,
        "hidden_signal": hidden_signal,
    }

    # Two hand-authored task seeds, one per task family the pack ships.
    # ``family`` is set here because, unlike ``distill()`` (which never
    # tags), the hand-authored author KNOWS which family each seed feeds.
    task_seeds: list[TaskSeed] = [
        TaskSeed(
            theme="webapp.build.default",
            anchor_kinds=["service", "endpoint"],
            suggested_goal_kinds=["endpoint"],
            difficulty=0.4,
            evidence=1,
            family="webapp.build",
        ),
        TaskSeed(
            theme="webapp.pentest.default",
            anchor_kinds=["endpoint", "vulnerability"],
            suggested_goal_kinds=["secret"],
            difficulty=0.7,
            evidence=1,
            family="webapp.pentest",
        ),
    ]

    difficulty: dict[str, float] = {
        "webapp.build.default": 0.4,
        "webapp.pentest.default": 0.7,
    }

    # Explored-density per kind — what fraction of a kind a typical
    # agent would have stepped through by end-of-task. Endpoints and
    # services are heavily walked; secrets and vulns are mostly hidden
    # until discovered, so coverage is lower.
    coverage: dict[str, float] = {
        "host": 0.7,
        "service": 0.85,
        "endpoint": 0.9,
        "data_store": 0.6,
        "record": 0.55,
        "secret": 0.5,
        "vulnerability": 0.6,
        "account": 0.75,
        "credential": 0.7,
        "network": 0.5,
    }

    return PackPrior(
        source="cyber.webapp@v1 :: hand-authored",
        ontology=webapp_ontology(),
        topology=topology,
        task_seeds=task_seeds,
        difficulty=difficulty,
        coverage=coverage,
    )


# ---------------------------------------------------------------------------
# BACK-COMPAT SHIM — TEMPORARY (Phase 2-only)
# ---------------------------------------------------------------------------
#
# Old code paths (cyber_webapp/__init__.py's `generation_priors()` returning
# a Mapping for the legacy `CyberWebappPack`, plus the legacy
# sampling/mutation/builder/tests) still import a constant named ``PRIORS``
# from this module. They expect a flat ``Mapping[str, object]`` of
# cyber-specific knobs. Those consumers are being migrated in parallel by
# other agents; until they switch to ``default_prior()`` /
# ``_CYBER_GENERATION_CONFIG``, this alias keeps them compiling.
#
# Phase 2e (new ``__init__.py``) + Phase 4 (delete legacy modules) remove
# the last consumer. At that point this alias is unused and gets dropped.
PRIORS: MappingProxyType[str, object] = MappingProxyType(dict(_CYBER_GENERATION_CONFIG))
