"""WebappBuilder — v1 hand-authored procedural builder for webapp worlds.

This v1 ships a deterministic, seeded generator that always emits a
small admittable webapp world: one repo, one auth service, one
authenticated endpoint, one datastore, one record, one HIDDEN flag
secret, and one HIDDEN weakness affecting the endpoint.

The shape is the load-bearing demo. A v2 will:
  - read the `PackPrior.topology` and scale up service / endpoint counts
  - read `PackPrior.task_seeds` and bias generation toward kinds the
    distilled BBG marked salient
  - support LLM-driven endpoint naming and weakness placement
  - implement `repair()` against a specific Issue list

For now: same world every time the seed is the same. Repair is a no-op
(returns the previous result, which guarantees the admission loop
gives up after the first failure — useful for catching bugs early).
"""

from __future__ import annotations

import random

from openrange import (
    Builder,
    BuildResult,
    Edge,
    Issue,
    Manifest,
    Node,
    PackPrior,
    Role,
    Visibility,
    WorldGraph,
)
from webapp.ontology import ONTOLOGY_ID


class WebappBuilder(Builder):
    """Deterministic v1 builder. Same seed → same world."""

    def __init__(self, prior: PackPrior | None) -> None:
        self._prior = prior
        # v1 ignores prior; v2 will read topology + task_seeds

    def build(self, manifest: Manifest) -> BuildResult:
        seed = int(manifest.get("seed", 0))
        rng = random.Random(seed)
        flag_value = self._mint_flag(rng)
        g = WorldGraph(ontology=ONTOLOGY_ID)

        # --- nodes ---
        g.add_node(
            Node(
                "repo.web",
                "repo",
                attrs={
                    "name": "primary-web",
                    "language": "python",
                },
            )
        )
        g.add_node(
            Node(
                "svc.auth",
                "service",
                attrs={
                    "name": "auth-service",
                    "kind": "auth",
                    "exposure": "public",
                },
                roles={Role.ACTOR},
            )
        )
        g.add_node(
            Node(
                "ep.login",
                "endpoint",
                attrs={
                    "path": "/login",
                    "method": "POST",
                    "auth_required": True,
                },
            )
        )
        g.add_node(
            Node(
                "store.users",
                "datastore",
                attrs={
                    "name": "users",
                    "engine": "sqlite",
                },
            )
        )
        g.add_node(
            Node(
                "rec.admin",
                "record",
                attrs={
                    "key": "users/admin",
                },
            )
        )
        g.add_node(
            Node(
                "sec.flag",
                "secret",
                attrs={
                    "name": "admin-flag",
                    "kind": "flag",
                    "value_ref": flag_value,
                },
                visibility=Visibility.HIDDEN,
            )
        )
        g.add_node(
            Node(
                "wk.sqli",
                "weakness",
                attrs={
                    "name": "login-sqli",
                    "kind": "sql_injection",
                },
                visibility=Visibility.HIDDEN,
            )
        )

        # --- edges ---
        g.add_edge(Edge("e.svc-repo", "owned_by", "svc.auth", "repo.web"))
        g.add_edge(Edge("e.svc-ep", "exposes", "svc.auth", "ep.login"))
        g.add_edge(
            Edge(
                "e.svc-store",
                "backed_by",
                "svc.auth",
                "store.users",
                attrs={"mode": "readwrite"},
            )
        )
        g.add_edge(Edge("e.store-rec", "contains", "store.users", "rec.admin"))
        g.add_edge(
            Edge(
                "e.rec-sec",
                "holds",
                "rec.admin",
                "sec.flag",
                attrs={"field": "secret_token"},
            )
        )
        g.add_edge(Edge("e.wk-ep", "affects", "wk.sqli", "ep.login"))

        # --- tasks come from the families, not from the builder ---
        # The builder produces the graph; the Pack stitches family-generated
        # tasks against it in admit(). v1 builder.build leaves tasks=[]
        # and admit's `_run_feasibility` reads pack.task_families() to
        # generate per-family tasks at call time. See WebappPack.
        # Actually: admit() does NOT call family.generate(); it only
        # checks the tasks that BuildResult carried. So the Builder is
        # the one that asks each family to contribute.
        from webapp.families import WebappBuild, WebappPentest

        tasks = []
        tasks.extend(WebappBuild().generate(g, manifest, self._prior))
        tasks.extend(WebappPentest().generate(g, manifest, self._prior))

        return BuildResult(
            graph=g,
            tasks=tasks,
            admission_meta={
                "builder": "webapp.v1",
                "seed": seed,
                "prior_source": (
                    self._prior.source if self._prior else "hand-authored:webapp.v1"
                ),
            },
        )

    def repair(
        self, prev: BuildResult, errors: list[Issue], infeasible: list[str]
    ) -> BuildResult:
        """v1 has no repair strategy — fail fast on the first attempt.

        If admission rejects a v1 world, the bug is in `build()`, not in
        the manifest, so resampling with the same seed would loop. v2's
        repair will resample with a perturbed seed and/or patch
        targeted issues.
        """
        del prev, errors, infeasible
        raise NotImplementedError(
            "webapp.v1 builder has no repair strategy; the v1 graph is "
            "hand-authored and admission should always pass. A v2 builder "
            "with procedural sampling will implement repair."
        )

    @staticmethod
    def _mint_flag(rng: random.Random) -> str:
        """Deterministic flag value derived from the rng."""
        return (
            "FLAG{" + "".join(rng.choice("0123456789abcdef") for _ in range(16)) + "}"
        )
