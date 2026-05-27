#!/usr/bin/env bash
#
# check_boundary.sh — enforce the core / pack split AND the
# proprietary-code firewall.
#
# Core code (the SDK, the world IR, the dashboard minus its one
# documented domain leak) MUST NOT name any cyber-domain concept.
# Domain words live in packs; core stays domain-agnostic so a second
# non-cyber pack can be added without rewriting core.
#
# Anywhere in the repo MUST NOT name proprietary external concepts
# (BBG, wayfinder). These belong to consumers of OpenRange, not to
# OpenRange itself. OpenRange is MIT-licensed open source; it does not
# ship code or designs sourced from proprietary projects.
#
# The lists below are authoritative. Update them when a new
# domain-agnostic concern gets baked in (FORBIDDEN_WORDS_CORE) or a
# new proprietary-leak vector is identified (FORBIDDEN_WORDS_REPO).
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

FORBIDDEN_WORDS_CORE=(
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

# Words that must NEVER appear ANYWHERE in the repo. These name
# proprietary external concepts (BBG / Wayfinder family and adjacent
# vocabulary) that OpenRange does not ship code or designs for. A hit
# here is a license / IP leak, not just a layering issue — fix it
# immediately and check with the project owner.
#
# Multi-word phrases use `[-_ ]` so hyphenated, underscored, and
# space-separated spellings all match (e.g. `spatial-memory`,
# `spatial_memory`, `spatial memory`).
FORBIDDEN_WORDS_REPO=(
  bbg
  wayfinder
  distill
  distillation
  distilled
  "spatial[-_ ]memory"
  "cognitive[-_ ]map"
  "agent[-_ ]memory"
  "epistemic[-_ ]graph"
  "big[-_ ]beautiful[-_ ]graph"
)

CORE_PATHS=(
  src/openrange/core
  src/openrange/dashboard
)

REPO_SCAN_PATHS=(
  src
  tests
  packs
  examples
  docs
  README.md
  ROADMAP.md
  CONTRACTS.md
  DESIGN.md
  scripts
  pyproject.toml
)

# Files inside the CORE_PATHS list that are excluded wholesale. Keep
# this list short — every entry is a known leak that should be cleaned
# up upstream eventually.
EXCLUDED_FILES=(
  src/openrange/dashboard/topology.py
)

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

violations=0

# --- Core-only scan: cyber-domain words must not leak into core. ---

core_pattern="\\b($(IFS='|'; echo "${FORBIDDEN_WORDS_CORE[*]}"))\\b"

core_find_args=()
for path in "${CORE_PATHS[@]}"; do
  core_find_args+=("$path")
done
core_find_args+=(-type f -name '*.py')
for excluded in "${EXCLUDED_FILES[@]}"; do
  core_find_args+=(-not -path "$excluded")
done

while IFS= read -r file; do
  while IFS= read -r match; do
    violations=$((violations + 1))
    echo "boundary: core-leak $file:$match" >&2
  done < <(
    grep -niE "$core_pattern" "$file" 2>/dev/null \
      | grep -v 'ALLOWED_DOMAIN_LEAK' || true
  )
done < <(find "${core_find_args[@]}" 2>/dev/null | sort)

# --- Repo-wide scan: proprietary words must not appear anywhere. ---

repo_pattern="\\b($(IFS='|'; echo "${FORBIDDEN_WORDS_REPO[*]}"))\\b"

# `--binary-files=without-match` skips binary blobs (the assets/ SVG has
# a base64 font that incidentally contains the substring "bBG"; we don't
# want to flag bytes inside encoded data, only source-level tokens).
# `-r` recurses; `-l` lists files; we then walk each match line.
while IFS= read -r file; do
  while IFS= read -r match; do
    violations=$((violations + 1))
    echo "boundary: proprietary-leak $file:$match" >&2
  done < <(
    grep -niE --binary-files=without-match "$repo_pattern" "$file" \
      2>/dev/null || true
  )
done < <(
  grep -rliE --binary-files=without-match --include='*' \
    "$repo_pattern" "${REPO_SCAN_PATHS[@]}" 2>/dev/null \
    | grep -v __pycache__ \
    | grep -v 'scripts/check_boundary.sh' \
    | sort -u
)

if [ "$violations" -gt 0 ]; then
  echo "boundary: $violations leak(s) found." >&2
  echo "boundary: core-leaks may be tagged with '# ALLOWED_DOMAIN_LEAK:" \
    "<reason>'." >&2
  echo "boundary: proprietary-leaks must be removed — no exemption." >&2
  exit 1
fi

echo "boundary: clean."
