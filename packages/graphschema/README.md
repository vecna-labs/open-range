# graphschema

[![ci](https://github.com/vecna-labs/graphschema/actions/workflows/ci.yml/badge.svg)](https://github.com/vecna-labs/graphschema/actions/workflows/ci.yml)
[![python](https://img.shields.io/badge/python-3.14+-blue)](https://www.python.org)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Typed property graphs in Python. Declare an ontology in plain data;
get validation, diffs, and content hashing for free. Zero
dependencies, ~700 lines, fully typed.

```python
from graphschema import (
    AttrSpec, AttrType, Edge, EdgeKind, Node, NodeKind,
    Ontology, WorldGraph, validate,
)

rooms = Ontology(
    id="rooms@0.1.0",
    node_kinds={
        "room": NodeKind("room", attrs={
            "label": AttrSpec(AttrType.STRING, required=True),
        }),
    },
    edge_kinds={
        "leads_to": EdgeKind("leads_to", endpoints=[("room", "room")]),
    },
)

g = WorldGraph(ontology=rooms.id)
g.add_node(Node("a", "room", attrs={"label": "lobby"}))
g.add_node(Node("b", "room", attrs={"label": "vault"}))
g.add_edge(Edge("e1", "leads_to", "a", "b"))

assert not [i for i in validate(g, rooms) if i.severity == "error"]
print(g.content_hash())  # sha256:…  same content → same hash
```

## Install

```bash
uv add git+https://github.com/vecna-labs/graphschema.git
```

Requires Python ≥ 3.14.

## Develop

```bash
uv sync
uv run pytest
uv run ruff check
uv run mypy --strict src tests
```

## License

MIT.
