"""Hand-authored default `PackPrior` for the cyber_webapp pack."""

from __future__ import annotations

from typing import Any

from openrange_pack_sdk import PackPrior, TaskSeed

from cyber_webapp.ontology import webapp_ontology

# Pack-private sampler knobs; deliberately not in `PackPrior` (which
# stays domain-agnostic). Imported by `cyber_webapp.sampling`.
_CYBER_GENERATION_CONFIG: dict[str, Any] = {
    "service_count": {"min": 2, "max": 5},
    # "web" is always forced as the first service; weight ignored.
    "service_kinds": {
        "web": 0,
        "api": 3,
        "auth": 2,
        "db": 4,
    },
    "endpoints_per_service": {"min": 1, "max": 3},
    "vuln_count": {"min": 1, "max": 3},
    "vuln_kinds": {
        "sql_injection": 3,
        "ssrf": 2,
        "broken_authz": 2,
    },
    "vuln_chain_depth": {"min": 1, "max": 2},
}


def default_prior() -> PackPrior:
    # fresh instance per call so callers may mutate without leaking into other consumers
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
    salient_kind_freq: dict[str, int] = {
        "endpoint": 1,
        "secret": 1,
        "vulnerability": 1,
    }
    dead_end_ratio: float = 0.2
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

    # ``family`` is set here because the author KNOWS which family each
    # seed feeds; producers without that knowledge leave it unset.
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
        source="cyber.webapp@v2 :: hand-authored",
        ontology=webapp_ontology(),
        topology=topology,
        task_seeds=task_seeds,
        difficulty=difficulty,
        coverage=coverage,
    )
