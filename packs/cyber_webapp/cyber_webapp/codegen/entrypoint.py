"""Filesystem conventions for the generated webapp runtime.

The generated ``app.py`` runs as a single Python subprocess managed by
``WebappRuntimeHandle``. This module owns the path conventions shared
between the codegen template (``--log`` argv, ``seed.json`` location)
and the runtime handle (request-log offset polling, ``result.json``
collection). Keeping these as module constants pins one source of
truth — the realizer and the generated app cannot disagree on a path.

Pre-refactor this module also returned an ``Entrypoint`` dataclass
consumed by the now-deleted ``HTTPBacking``; per-pack runtime is
``RuntimeHandle``-owned now, so the dataclass is gone but the
conventions it encoded survive here.
"""

from __future__ import annotations

# Names the runtime handle reads / writes inside the per-episode workspace.
# The codegen template embeds ``REQUEST_LOG_NAME`` as the ``--log`` argv,
# and ``SEED_FILE_NAME`` is the default sibling file the generated app
# loads then unlinks. ``RESULT_FILE_NAME`` is the agent-written terminal
# signal the handle polls in ``terminal()`` / reads in ``collect()``.
APP_FILE_NAME = "app.py"
SEED_FILE_NAME = "seed.json"
REQUEST_LOG_NAME = "requests.jsonl"
RESULT_FILE_NAME = "result.json"
TASK_FILE_NAME = "OPENRANGE_TASK.json"
