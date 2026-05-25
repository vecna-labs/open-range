#!/usr/bin/env bash
#
# check_boundary.sh — enforce the core / pack split.
#
# Core code (the SDK, the world IR, the ontologies module, the dashboard
# minus its one documented domain leak) MUST NOT name any cyber-domain
# concept. Domain words live in packs; core stays domain-agnostic so a
# second non-cyber pack can be added without rewriting core.
#
# The list below is authoritative. Update it when a new domain-agnostic
# concern gets baked in.
#
# Strategy
# --------
# Whole-word, case-insensitive grep across the scanned paths. A line is
# allowed to mention a forbidden word ONLY if that line carries the
# marker `ALLOWED_DOMAIN_LEAK` — either as a code-line comment or as
# trailing text inside a docstring. Reviewers see the marker and the
# justification on the same line; new violations have to surface a real
# annotation to land.
#
# `src/openrange/dashboard/topology.py` is a KNOWN, documented domain
# leak (its module docstring spells out the cyber-pack coupling and a
# follow-up to move it onto a `Pack.topology_view()` hook). The whole
# file is excluded from the scan rather than annotated line-by-line.
# Phase 5 does not own the dashboard/pack refactor; the exclusion goes
# away with that follow-up.

set -euo pipefail

FORBIDDEN_WORDS=(
  host
  service
  endpoint
  vulnerability
  account
  secret
  credential
  webapp
  pentest
  cyber
  http
  sql_injection
  ssrf
  flag
  payload
)

CORE_PATHS=(
  src/openrange/core
  src/openrange/world_ir.py
  src/openrange/ontologies
  src/openrange/dashboard
)

# Files inside the CORE_PATHS list that are excluded wholesale. Keep
# this list short — every entry is a known leak that should be cleaned
# up upstream eventually.
EXCLUDED_FILES=(
  src/openrange/dashboard/topology.py
)

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Build one alternation regex: \b(host|service|...)\b
pattern="\\b($(IFS='|'; echo "${FORBIDDEN_WORDS[*]}"))\\b"

# Build a `find` exclusion expression for the wholesale-excluded files.
find_args=()
for path in "${CORE_PATHS[@]}"; do
  find_args+=("$path")
done
find_args+=(-type f -name '*.py')
for excluded in "${EXCLUDED_FILES[@]}"; do
  find_args+=(-not -path "$excluded")
done

violations=0
while IFS= read -r file; do
  # Per-file scan: print lines that match the forbidden pattern, drop
  # lines tagged with ALLOWED_DOMAIN_LEAK, prefix each survivor with
  # the file:line so the report is reviewable.
  while IFS= read -r match; do
    violations=$((violations + 1))
    echo "boundary: $file:$match" >&2
  done < <(
    grep -niE "$pattern" "$file" 2>/dev/null \
      | grep -v 'ALLOWED_DOMAIN_LEAK' || true
  )
done < <(find "${find_args[@]}" 2>/dev/null | sort)

if [ "$violations" -gt 0 ]; then
  echo "boundary: $violations domain-leak(s) found in core paths." >&2
  echo "boundary: tag intentional cases with '# ALLOWED_DOMAIN_LEAK:" \
    "<reason>' or refactor the offending file." >&2
  exit 1
fi

echo "boundary: clean."
