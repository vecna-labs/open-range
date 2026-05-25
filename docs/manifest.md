# Manifest

The manifest is a free-form `Mapping[str, Any]` — the user's request layer.

## Cross-pack invariant

Core only reads one key: `pack.id` (the registered pack to admit against). The
shorthand `{"pack": "<id>"}` is accepted as equivalent to
`{"pack": {"id": "<id>"}}`. Every other key is the pack's contract; core never
branches on a manifest field. See [CONTRACTS.md](../CONTRACTS.md) for the full
shape declaration.

## Pack-specific keys

Each pack documents the keys it expects in its own source. For the built-in
`webapp` pack, the keys honored today are:

- `seed` (int) — deterministic sampling seed. Same seed + same prior →
  same world graph.
- `world` (mapping) — optional pre-baked topology hints honored by the
  dashboard. The pack does not read it during sampling.

Source of truth: `packs/cyber_webapp/cyber_webapp/builder.py` (see the
`WebappBuilder` docstring and `_seed_from_manifest`). Pack-specific manifest
docs are a known gap; the `webapp` pack does not yet ship a dedicated key
reference.
