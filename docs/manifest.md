# Manifest

The manifest is a free-form `Mapping[str, Any]` — the user's request layer.

## Cross-pack invariant

Core only reads one key: `pack.id` (the registered pack to admit against). The
shorthand `{"pack": "<id>"}` is accepted as equivalent to
`{"pack": {"id": "<id>"}}`. Every other key is the pack's contract; core never
branches on a manifest field. See [CONTRACTS.md](../CONTRACTS.md) for the full
shape declaration.

## Keys handled above the pack

A few keys are read generically, by the SDK base class or core, for any pack:

- `seed` (int) — deterministic sampling seed, extracted by the SDK base class
  `ProceduralBuilder.build`. Same seed + same prior → the same world graph.
- `runtime.backing` (string) — desired runtime substrate (`"process"`,
  `"container"`, `"simulator"`, `"hybrid"`). Read at episode start; an explicit
  `RunConfig.backing` overrides it, and an unknown token is a hard error.
- `world` (mapping) — optional pre-baked topology hints honored by the dashboard;
  not read during sampling.

## Pack-specific keys

Every other key is the pack's own contract. A pack defines the keys it accepts and
documents them alongside its own source — packs are independently packaged and may
live in their own repositories, so core neither defines nor branches on these keys.
See the pack you are admitting against for its key reference.
