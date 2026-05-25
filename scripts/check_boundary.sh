#!/usr/bin/env bash
# OpenRange boundary checks — enforce the design rules in CI.
#
# Three checks:
#   1. The OpenRange core (world_ir.py + core/*.py) must not name any domain
#      concept. Domain vocabulary lives in packs.
#   2. OpenRange must not import any specific harness library (no
#      `from wayfinder`, no `from vecna`, etc.). The seam to agent-memory
#      runtimes is JSON, not Python imports.
#   3. (Future) The vendored copy of world_ir.py in vecna/wayfinder must
#      carry a vendor-source header naming the openrange commit it was
#      copied from. That check lives in the vecna repo, not here, but the
#      header convention is documented in DESIGN.md.
#
# Exit codes:
#   0  all checks passed
#   1  a domain-leak or forbidden-import was found
#
# Run locally: bash scripts/check_boundary.sh
# In CI: see .github/workflows/ci.yml — "Boundary checks" step.

set -euo pipefail

cd "$(dirname "$0")/.."

# ---------------------------------------------------------------------------
# 1. Domain-leak check
# ---------------------------------------------------------------------------
# Core code must be domain-free. These tokens are example domain words from
# packs that have existed or might exist; if any of them appears in core,
# the boundary has leaked.
#
# Add more tokens as new packs are written. The list is deliberately
# conservative — false positives are cheap to fix; missed leaks are not.

CORE_DIRS=(
  "src/openrange/world_ir.py"
  "src/openrange/core"
  "src/openrange/ontologies"
)

DOMAIN_TOKENS=(
  # cyber / webapp
  '\bhost\b'
  '\bhostname\b'
  '\bvuln\b'
  '\bvulnerability\b'
  '\bendpoint\b'
  '\bsqli\b'
  '\bsql_injection\b'
  '\bssrf\b'
  '\bflag\b'
  '\bexploit\b'
  '\bcredential\b'
  # other plausible domains
  '\btrading\b'
  '\bpendulum\b'
  '\bcluster\b'
  '\bnamespace\b'
)

# Allowed tokens — terms that look domain-shaped but are actually
# generic / cognitive / algorithmic / graph-theory vocabulary used in core.
#
# - thing, thought, traversed, etc.  : BBG ontology vocabulary (cognitive
#                                       primitives, not a domain).
# - cluster                          : union-find cluster in distill, NOT a
#                                       k8s cluster — purely algorithmic.
# - endpoint                         : OVERLOADED. In core, `endpoint(s)` is
#                                       a graph-theory term (edge endpoints:
#                                       `EdgeKind.endpoints: list[(src,dst)]`).
#                                       The cyber pack happens to use the
#                                       same word for HTTP endpoints, but
#                                       that's domain context not the core's.
ALLOWED_TOKENS_RE='(thing|thought|traversed|anchored_to|revises|part_of|cluster|endpoint)'

leaks_found=0
for token in "${DOMAIN_TOKENS[@]}"; do
  hits=$(grep -rIn --include='*.py' --include='*.md' -E "$token" "${CORE_DIRS[@]}" 2>/dev/null | \
         grep -vE "$ALLOWED_TOKENS_RE" || true)
  if [ -n "$hits" ]; then
    echo "DOMAIN-LEAK in core: pattern /$token/ found in"
    echo "$hits"
    echo
    leaks_found=$((leaks_found + 1))
  fi
done

if [ "$leaks_found" -gt 0 ]; then
  echo "FAIL: $leaks_found domain leak(s) detected in core."
  echo "      Core (world_ir.py, core/, ontologies/) must be domain-free."
  echo "      Domain vocabulary belongs in packs/ — see DESIGN.md §3.3."
  exit 1
fi

# ---------------------------------------------------------------------------
# 2. Forbidden-import check
# ---------------------------------------------------------------------------
# OpenRange must not import any specific harness library. The seam is JSON.

FORBIDDEN_PATTERNS=(
  '^from wayfinder'
  '^import wayfinder'
  '^from vecna'
  '^import vecna'
)

forbidden_found=0
for pat in "${FORBIDDEN_PATTERNS[@]}"; do
  hits=$(grep -rIn --include='*.py' -E "$pat" src/openrange/ tests/ 2>/dev/null || true)
  if [ -n "$hits" ]; then
    echo "FORBIDDEN IMPORT: pattern /$pat/ found in"
    echo "$hits"
    echo
    forbidden_found=$((forbidden_found + 1))
  fi
done

if [ "$forbidden_found" -gt 0 ]; then
  echo "FAIL: $forbidden_found forbidden import(s) detected."
  echo "      OpenRange does not import any harness library — the seam"
  echo "      is the JSON wire format declared in CONTRACTS.md."
  exit 1
fi

echo "Boundary checks: OK"
