"""A consequence gate for ``auto_evolve`` (cyber webapp).

Rejects a "harden" add whose exploit actually leaks the flag — that's a second
solution (easier), not a hardening. Lives here, not in the pack or core,
because it composes pack pieces (reference exploit + verifier) with host
realization (``EpisodeService``).
"""

from __future__ import annotations

import urllib.error
import urllib.request
from pathlib import Path

from cyber_webapp.consequence import detect_leak
from cyber_webapp.reference_solver import exploit_and_benign
from openrange_pack_sdk import Mutation, Pack, Snapshot

from openrange.core.curriculum import EvolutionGate
from openrange.core.episode import EpisodeService


def _fetch(url: str) -> str:
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            return str(resp.read().decode())
    except urllib.error.HTTPError as exc:
        return str(exc.read().decode())
    except Exception:  # noqa: BLE001 — an unreachable surface just can't leak
        return ""


def consequence_gate(pack: Pack, workdir: str | Path) -> EvolutionGate:
    """Build a gate for ``auto_evolve(..., gate=…)``.

    Vets a ``harden`` add: realizes the candidate, runs the added vuln's
    reference exploit, and returns ``False`` if it leaks the flag (so the
    leaking add is skipped and a genuine non-leaking decoy is picked instead).
    Anything it can't assess (non-harden, no added vuln, no reference exploit)
    passes through unchanged.
    """
    root = Path(workdir)

    def gate(evolved: Snapshot, mutation: Mutation) -> bool:
        if mutation.direction != "harden":
            return True
        added_kinds = [
            str(n.attrs.get("kind"))
            for n in mutation.patch.nodes_added
            if n.kind == "vulnerability"
        ]
        if not added_kinds:
            return True
        task = next(
            (t for t in evolved.tasks if t.meta.get("family") == "webapp.pentest"),
            None,
        )
        if task is None:
            return True
        svc = EpisodeService(pack, root)
        try:
            handle = svc.start_episode(evolved, task.id)
            base = str(svc.surface(handle)["base_url"])
            for kind in added_kinds:
                try:
                    exploit_path, _benign = exploit_and_benign(evolved.graph, kind)
                except Exception:  # noqa: BLE001 — no reference exploit → can't leak-check
                    continue
                if detect_leak(evolved.graph, [_fetch(base + exploit_path)]).occurred:
                    return False  # a "harden" add that leaks the flag → reject
            return True
        finally:
            svc.close()

    return gate
