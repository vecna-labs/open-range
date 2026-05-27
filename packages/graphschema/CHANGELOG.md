# Changelog

All notable changes documented here. Versions follow semver; breaking
changes to the wire format bump the ontology id (`@version`), not the
package version.

## [Unreleased]

## [0.1.0] — 2026-05-26

Initial release.

- `Node`, `Edge`, `WorldGraph`: typed property graph primitives.
- `Ontology`, `NodeKind`, `EdgeKind`, `AttrSpec`, `AttrType`: declarative schema.
- `Role`, `Visibility`: cross-cutting enums.
- `validate(graph, ontology, invariants=...)`: three-tier validator
  (structure → ontology conformance → caller invariants).
- `GraphPatch` + `apply_patch`: the only mutation primitive.
- `WorldGraph.content_hash()`: deterministic `sha256` over canonical JSON.
- `py.typed` marker — full type information ships with the package.
