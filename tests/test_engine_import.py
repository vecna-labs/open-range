"""Importing the engine must pull in no torch/transformers/trl/httpx/examples.

Run in a fresh interpreter (not this polluted pytest process) so sys.modules is a
true reflection of what `import openrange_trl.engine` drags in — this is the guard
that catches an accidental shipped-code dependency on the non-wheel `examples`
package or a non-lazy heavy import.
"""

from __future__ import annotations

import subprocess
import sys


def test_importing_the_engine_pulls_in_no_heavy_or_example_deps() -> None:
    code = (
        "import openrange_trl.engine, sys; "
        "roots = {'torch', 'transformers', 'trl', 'httpx', 'examples'}; "
        "bad = sorted(m for m in sys.modules if m.split('.')[0] in roots); "
        "print(bad); "
        "assert not bad, bad"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stdout + result.stderr
