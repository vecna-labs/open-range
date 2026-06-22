# Manifest

The manifest is a free-form `Mapping[str, Any]` — the user's request layer.

## Cross-pack invariant

Core only reads one key: `pack.id` (the registered pack to admit against). The
shorthand `{"pack": "<id>"}` is accepted as equivalent to
`{"pack": {"id": "<id>"}}`. Every other key is the pack's contract; core never
branches on a manifest field. See [CONTRACTS.md](../CONTRACTS.md) for the full
shape declaration.

## Pack-specific keys

Each pack documents the keys it expects in its own source. A few are handled
generically, above any single pack:

- `seed` (int) — deterministic sampling seed, extracted by the SDK base class
  `ProceduralBuilder.build`. Same seed + same prior → the same world graph.
- `runtime.backing` (string) — desired runtime substrate (`"process"`,
  `"container"`, `"simulator"`, `"hybrid"`). Read at episode start; an explicit
  `RunConfig.backing` overrides it, and an unknown token is a hard error.
- `world` (mapping) — optional pre-baked topology hints honored by the dashboard;
  not read during sampling.

Everything else is the pack's own contract. For the built-in `webapp` pack — its
auto↔specific control surface — see
[packs/cyber_webapp/MANIFEST.md](../packs/cyber_webapp/MANIFEST.md).
