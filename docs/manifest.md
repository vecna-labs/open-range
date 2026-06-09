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

- `seed` (int) — deterministic sampling seed. Same seed + same prior +
  same `scale` → same world graph.
- `scale` (mapping) — optional sampler count-range overrides, so a world
  can be scaled from the manifest without hand-building a `PackPrior`.
  Maps sampler count keys (`service_count`, `endpoints_per_service`,
  `vuln_count`, `account_count`) to `{"min": int, "max": int}`; keys left
  out keep their defaults. Example:
  `{"scale": {"service_count": {"min": 8, "max": 10}}}`.
- `runtime.backing` (string) — optional desired runtime substrate
  (`"process"`, `"container"`, `"simulator"`, `"hybrid"`). Read at
  episode start; an explicit `RunConfig.backing` overrides it, and the
  realizer raises if it doesn't support the choice (`webapp` wires only
  `"process"` today). An unknown token is a hard error.
- `world` (mapping) — optional pre-baked topology hints honored by the
  dashboard. The pack does not read it during sampling.

The seed itself is extracted by the SDK base class
`ProceduralBuilder.build` (via `manifest_int`), not the pack. World scale
folds into the prior in `WebappBuilder._effective_prior`; backing
resolution lives in `OpenRangeRun._resolve_backing`. Pack-specific
manifest docs are still thin; the `webapp` pack does not yet ship an
exhaustive key reference.
