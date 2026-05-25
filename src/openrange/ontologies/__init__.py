"""Built-in ontologies that ship with OpenRange.

These are declarative `Ontology` values (data, not runtime). Pack authors,
agent harnesses, and downstream consumers import them rather than redefining
the same shape independently — that prevents silent drift across the
distillation seam.

Today's catalog:

- `openrange.ontologies.bbg` — the `bbg@0.1.0` ontology used by any harness
  that maintains a long-horizon agent-memory graph of `thing`/`thought`
  nodes. OpenRange itself consumes this ontology in `distill()`.

Adding a new ontology here is light — it's just a module returning an
`Ontology` value. The bar for inclusion: the ontology should be useful to
multiple consumers (i.e. it would otherwise be duplicated), or sharp
enough as a cognitive primitive that pinning it once eliminates a class
of mistakes.
"""

from openrange.ontologies.bbg import BBG_ONTOLOGY_ID, bbg_ontology

__all__ = ["BBG_ONTOLOGY_ID", "bbg_ontology"]
