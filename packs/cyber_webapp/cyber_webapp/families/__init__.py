"""TaskFamilies for the cyber webapp pack.

A Pack ships a world-family (e.g. `webapp`); a TaskFamily owns one
*domain* of tasks against that world. Two families ship with this
pack:

  - `WebappBuild`  (id `webapp.build`)
        the agent implements / repairs a service to make a feature
        endpoint serve correctly. Entrypoint: a service node. Goal:
        an endpoint node. Success criterion: smoke test passes after
        the agent's edits.

  - `WebappPentest`  (id `webapp.pentest`)
        the agent discovers and exploits a vulnerability chain to
        recover a hidden flag-kind secret. Entrypoint: an exposed
        endpoint. Goal: the hidden secret. Success criterion: the
        agent's submitted flag matches the secret's value_ref.

Both families live against the SAME world graph in any given snapshot.
That cross-family-on-one-world property is the load-bearing
demonstration that "domain" lives on the TaskFamily, not on the Pack.
"""

from cyber_webapp.families.build import WebappBuild
from cyber_webapp.families.pentest import WebappPentest

__all__ = ["WebappBuild", "WebappPentest"]
