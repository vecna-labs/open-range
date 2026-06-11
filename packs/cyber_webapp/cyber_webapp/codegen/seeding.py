"""Project a webapp world graph into the seed payload the runtime loads at start."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from graphschema import Node, WorldGraph
from openrange_pack_sdk import PackError

_DEFAULT_TABLE = "records"
_DEFAULT_KEY_COLUMN = "key"
_DEFAULT_VALUE_COLUMN = "value"
# All possible leak_field values broken_authz might pick — every key
# under state["secrets"] must resolve to the flag so the in-memory leak
# path returns the secret regardless of which field name was sampled.
_BROKEN_AUTHZ_LEAK_FIELDS = ("value", "data", "secret", "content", "result", "flag")


def project_seed(graph: WorldGraph) -> Mapping[str, object]:
    """Project the runtime seed payload.

    The flag lives in exactly one place, fixed by the loot shape: a "db" loot
    keeps it in the in-memory records/secrets (read via a response-leak
    exploit); a "file" loot keeps it in the in-memory file map (read via a
    file-read exploit). It never lands on disk. Raises :class:`PackError` if
    the graph has no flag-kind secret.
    """
    flag = ""
    flag_secret_id = ""
    accounts: dict[str, dict[str, object]] = {}
    secrets: dict[str, str] = {}
    raw_records: dict[str, tuple[str, dict[str, str]]] = {}

    creds_by_account: dict[str, str] = {}
    for edge in graph.edges.values():
        if edge.kind == "has_credential":
            creds_by_account[edge.src] = edge.dst
    cred_by_id: dict[str, Node] = {
        n.id: n for n in graph.nodes.values() if n.kind == "credential"
    }

    for node in graph.nodes.values():
        if node.kind == "secret" and node.attrs.get("kind") == "flag":
            flag = str(node.attrs.get("value_ref", ""))
            flag_secret_id = node.id
        elif node.kind == "secret":
            secrets[str(node.attrs.get("kind", node.id))] = str(
                node.attrs.get("value_ref", ""),
            )
        elif node.kind == "account":
            cred_id = creds_by_account.get(node.id)
            password = ""
            if cred_id is not None:
                cred = cred_by_id.get(cred_id)
                if cred is not None:
                    password = str(cred.attrs.get("value_ref", ""))
            accounts[str(node.attrs.get("username", node.id))] = {
                "role": str(node.attrs.get("role", "user")),
                "password": password,
            }
        elif node.kind == "record":
            fields = node.attrs.get("fields", {})
            clean = (
                {str(k): str(v) for k, v in fields.items()}
                if isinstance(fields, Mapping)
                else {}
            )
            raw_records[node.id] = (str(node.attrs.get("key", node.id)), clean)

    if not flag:
        raise PackError("graph has no flag-kind secret; codegen needs one")

    store_kind_of_record = _store_kind_by_record(graph)
    flag_record_id = _record_holding(graph, flag_secret_id)
    loot_shape = store_kind_of_record.get(flag_record_id, "kv")

    db_records: dict[str, dict[str, str]] = {}
    files: dict[str, str] = {}
    for rec_id, (key, fields) in raw_records.items():
        if store_kind_of_record.get(rec_id) == "file":
            files[key] = fields.get("value", "")
        else:
            db_records[key] = fields

    schema = _derive_sql_schema(graph)
    if loot_shape == "file":
        # Flag lives in the file map only; db records / secrets carry decoys
        # so a stray response-leak vuln can't reach it.
        records_for_schema = _retarget_records(db_records, schema, flag="")
        secrets_out: dict[str, str] = dict(secrets)
        files_out = _populate_files(files)
    else:
        records_for_schema = _retarget_records(db_records, schema, flag)
        secrets_out = _populate_secrets_with_flag(secrets, flag)
        files_out = {}

    return MappingProxyType(
        {
            "flag": flag,
            "accounts": accounts,
            "secrets": secrets_out,
            "records": records_for_schema,
            "files": files_out,
            "schema": schema,
        },
    )


def _store_kind_by_record(graph: WorldGraph) -> dict[str, str]:
    contains: dict[str, str] = {
        e.dst: e.src for e in graph.edges.values() if e.kind == "contains"
    }
    store_kind: dict[str, str] = {
        n.id: str(n.attrs.get("kind", "")) for n in graph.by_kind("data_store")
    }
    return {rec: store_kind.get(store, "") for rec, store in contains.items()}


def _record_holding(graph: WorldGraph, secret_id: str) -> str:
    for edge in graph.edges.values():
        if edge.kind == "holds" and edge.dst == secret_id:
            return edge.src
    return ""


def _populate_files(files: Mapping[str, str]) -> dict[str, str]:
    return {k: str(v) for k, v in files.items()}


def _derive_sql_schema(graph: WorldGraph) -> Mapping[str, str]:
    # SQLi handler's query must match the schema — derive from vuln params,
    # fall back to defaults.
    for node in graph.nodes.values():
        if node.kind != "vulnerability":
            continue
        if str(node.attrs.get("kind", "")) != "sql_injection":
            continue
        params = node.attrs.get("params", {})
        if not isinstance(params, Mapping):
            continue
        table = str(params.get("table") or _DEFAULT_TABLE)
        value_col = str(params.get("leak_column") or _DEFAULT_VALUE_COLUMN)
        return MappingProxyType(
            {
                "table": _safe_ident(table, _DEFAULT_TABLE),
                "key_column": _DEFAULT_KEY_COLUMN,
                "value_column": _safe_ident(value_col, _DEFAULT_VALUE_COLUMN),
            },
        )
    return MappingProxyType(
        {
            "table": _DEFAULT_TABLE,
            "key_column": _DEFAULT_KEY_COLUMN,
            "value_column": _DEFAULT_VALUE_COLUMN,
        },
    )


_DECOY_ROWS: tuple[tuple[str, str], ...] = (
    ("schema_version", "3"),
    ("region", "us-east-1"),
    ("owner", "platform-ops"),
    ("retention_days", "30"),
    ("backup_enabled", "true"),
)


def _retarget_records(
    records: Mapping[str, Mapping[str, object]],
    schema: Mapping[str, str],
    flag: str,
) -> dict[str, dict[str, str]]:
    # Graph records always carry ``fields = {"value": <flag>}`` by
    # sampler convention; rename to the schema's actual value column so
    # the SQLi handler's ``SELECT key, <col> FROM ...`` resolves. Decoys
    # keep the table from being a single-row giveaway.
    value_column = schema["value_column"]
    out: dict[str, dict[str, str]] = {}
    for key, fields in records.items():
        row = {value_column: ""}
        for col, val in fields.items():
            target_col = value_column if col == "value" else col
            row[target_col] = str(val)
        if flag and not row.get(value_column):
            row[value_column] = flag
        out[key] = row
    if not out:
        out["admin_secret"] = {value_column: flag}
    for decoy_key, decoy_value in _DECOY_ROWS:
        out.setdefault(decoy_key, {value_column: decoy_value})
    return out


def _populate_secrets_with_flag(
    secrets: Mapping[str, str],
    flag: str,
) -> dict[str, str]:
    # Mirror the flag under every leak_field broken_authz might pick so
    # the in-memory leak path returns the secret regardless of the
    # sampled field name.
    populated = dict(secrets)
    for field in _BROKEN_AUTHZ_LEAK_FIELDS:
        populated.setdefault(field, flag)
    populated["flag"] = flag
    return populated


def _safe_ident(value: str, fallback: str) -> str:
    # SQL identifiers are interpolated unquoted into the rendered handler
    # (this is the bug being modeled); constrain at codegen time so a
    # sampled value can't break the rendered query.
    if not value:
        return fallback
    if not (value[0].isalpha() or value[0] == "_"):
        return fallback
    if not all(c.isalnum() or c == "_" for c in value):
        return fallback
    return value
