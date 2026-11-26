"""
eo_brain/agent/sql_exec.py - Read-only execution and catalog-driven schema docs
for the data agent.

``ReadOnlyExecutor`` connects with its own URL (AGENT_DATABASE_URL, the
least-privilege ``agent_reader`` role), never through the application's
read-write pool. Each query runs in a fresh read-only transaction with a
statement timeout, and the result is bounded:

  * at most ``max_rows`` rows, flagged ``truncated`` if there were more;
  * at most ``max_payload_chars`` of JSON, flagged ``truncated`` if cut;
  * only scalar column types are returned. Geometry, geography, arrays,
    records, json and binary values are replaced by a placeholder, so raw
    geometry never reaches the model's context.

``load_schema_doc`` builds the documentation the model sees from the
database catalog (view and column COMMENTs in eo_brain/db/agent_views.sql).
"""

from __future__ import annotations

import datetime as dt
import decimal
import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

DEFAULT_MAX_ROWS = 100
DEFAULT_MAX_PAYLOAD_CHARS = 16_000
DEFAULT_STATEMENT_TIMEOUT_MS = 3_000
MAX_TEXT_VALUE_CHARS = 300

# PostgreSQL built-in type OIDs whose values are returned as-is.
_SCALAR_TYPE_OIDS = {
    16: "bool", 20: "int8", 21: "int2", 23: "int4", 25: "text", 700: "float4",
    701: "float8", 1700: "numeric", 1042: "bpchar", 1043: "varchar", 19: "name",
    1082: "date", 1083: "time", 1114: "timestamp", 1184: "timestamptz",
    1186: "interval", 2950: "uuid",
}


@dataclass(frozen=True)
class ColumnDoc:
    name: str
    type: str
    comment: str = ""


@dataclass(frozen=True)
class ViewDoc:
    name: str
    comment: str
    columns: List[ColumnDoc]


@dataclass(frozen=True)
class SchemaDoc:
    schema: str
    views: List[ViewDoc]

    def relation_names(self) -> List[str]:
        return [v.name for v in self.views]

    def render(self) -> str:
        lines: List[str] = []
        for view in self.views:
            lines.append(f"{self.schema}.{view.name}: {view.comment}".rstrip(": "))
            for col in view.columns:
                note = f": {col.comment}" if col.comment else ""
                lines.append(f"  - {col.name} ({col.type}){note}")
            lines.append("")
        return "\n".join(lines).rstrip()


_SCHEMA_DOC_SQL = """
SELECT c.relname,
       coalesce(obj_description(c.oid, 'pg_class'), ''),
       a.attname,
       format_type(a.atttypid, a.atttypmod),
       coalesce(col_description(c.oid, a.attnum), '')
FROM pg_catalog.pg_class c
JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
JOIN pg_catalog.pg_attribute a ON a.attrelid = c.oid AND a.attnum > 0 AND NOT a.attisdropped
WHERE n.nspname = %s
  AND c.relkind IN ('v', 'm')
  AND has_table_privilege(c.oid, 'SELECT')
ORDER BY c.relname, a.attnum
"""


@dataclass
class QueryResult:
    columns: List[str]
    rows: List[List[Any]]
    row_count: int
    truncated: bool
    omitted_columns: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "columns": self.columns,
            "rows": self.rows,
            "row_count": self.row_count,
            "truncated": self.truncated,
            "omitted_columns": self.omitted_columns,
        }


class QueryError(Exception):
    """A database error, worded so it can be fed back to the model."""


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value
    if isinstance(value, decimal.Decimal):
        return float(value)
    if isinstance(value, (dt.datetime, dt.date, dt.time)):
        return value.isoformat()
    if isinstance(value, dt.timedelta):
        return f"{value.total_seconds():g} seconds"
    if isinstance(value, uuid.UUID):
        return str(value)
    text = str(value)
    return text if len(text) <= MAX_TEXT_VALUE_CHARS else text[:MAX_TEXT_VALUE_CHARS] + "..."


def shape_result(
    description: Sequence[Any],
    fetched: Sequence[Sequence[Any]],
    *,
    max_rows: int,
    max_payload_chars: int,
    type_names: Optional[Dict[int, str]] = None,
) -> QueryResult:
    """Apply the type allow-list and the row / payload caps. Pure function."""
    type_names = type_names or {}
    columns = [d[0] for d in description]
    type_codes = [d[1] for d in description]
    omitted = [
        f"{name} ({type_names.get(code, 'non-scalar')})"
        for name, code in zip(columns, type_codes)
        if code not in _SCALAR_TYPE_OIDS
    ]
    truncated = len(fetched) > max_rows
    rows: List[List[Any]] = []
    size = len(json.dumps(columns))
    for raw in fetched[:max_rows]:
        row = [
            _jsonable(v) if code in _SCALAR_TYPE_OIDS else f"<{type_names.get(code, 'value')} omitted>"
            for v, code in zip(raw, type_codes)
        ]
        size += len(json.dumps(row, default=str)) + 1
        if size > max_payload_chars:
            truncated = True
            break
        rows.append(row)
    return QueryResult(columns=columns, rows=rows, row_count=len(rows), truncated=truncated,
                       omitted_columns=omitted)


class ReadOnlyExecutor:
    """Runs gate-approved SQL as the least-privilege role, one transaction per call."""

    def __init__(
        self,
        dsn: str,
        *,
        schema: str = "agent",
        max_rows: int = DEFAULT_MAX_ROWS,
        max_payload_chars: int = DEFAULT_MAX_PAYLOAD_CHARS,
        statement_timeout_ms: int = DEFAULT_STATEMENT_TIMEOUT_MS,
        connect_timeout_s: int = 5,
    ):
        self.dsn = dsn
        self.schema = schema
        self.max_rows = max_rows
        self.max_payload_chars = max_payload_chars
        self.statement_timeout_ms = statement_timeout_ms
        self.connect_timeout_s = connect_timeout_s

    def _connect(self):
        import psycopg2

        conn = psycopg2.connect(
            self.dsn,
            connect_timeout=self.connect_timeout_s,
            application_name="data-agent",
            # Belt and braces: the role already sets both of these.
            options=f"-c statement_timeout={int(self.statement_timeout_ms)} "
                    f"-c default_transaction_read_only=on",
        )
        conn.set_session(readonly=True, autocommit=False)
        return conn

    def execute(self, sql: str) -> QueryResult:
        import psycopg2
        from psycopg2 import errors as pg_errors

        try:
            conn = self._connect()
        except psycopg2.Error as e:
            raise QueryError(f"database unavailable: {str(e).strip().splitlines()[0]}") from None
        try:
            with conn.cursor() as cur:
                cur.execute(sql)
                if cur.description is None:
                    raise QueryError("statement returned no rows")
                description = tuple(cur.description)  # captured before any other query
                fetched = cur.fetchmany(self.max_rows + 1)
            unknown = sorted({d[1] for d in description if d[1] not in _SCALAR_TYPE_OIDS})
            type_names: Dict[int, str] = {}
            if unknown:
                with conn.cursor() as lookup:
                    lookup.execute(
                        "SELECT oid::int, typname FROM pg_catalog.pg_type WHERE oid = ANY(%s)", (unknown,)
                    )
                    type_names = {oid: name for oid, name in lookup.fetchall()}
            return shape_result(
                description, fetched,
                max_rows=self.max_rows, max_payload_chars=self.max_payload_chars,
                type_names=type_names,
            )
        except pg_errors.QueryCanceled:
            raise QueryError(f"query cancelled after {self.statement_timeout_ms} ms (statement timeout)") from None
        except psycopg2.Error as e:
            raw = (e.pgerror or str(e) or type(e).__name__).strip()
            msg = raw.splitlines()[0].removeprefix("ERROR:").strip()
            code = f" {e.pgcode}" if e.pgcode else ""
            raise QueryError(f"database error{code}: {msg}") from None
        finally:
            conn.rollback()
            conn.close()

    def load_schema_doc(self) -> SchemaDoc:
        import psycopg2

        try:
            conn = self._connect()
        except psycopg2.Error as e:
            raise QueryError(f"database unavailable: {str(e).strip().splitlines()[0]}") from None
        try:
            with conn.cursor() as cur:
                cur.execute(_SCHEMA_DOC_SQL, (self.schema,))
                rows = cur.fetchall()
        finally:
            conn.rollback()
            conn.close()
        views: Dict[str, ViewDoc] = {}
        for relname, rel_comment, col, col_type, col_comment in rows:
            if relname not in views:
                views[relname] = ViewDoc(name=relname, comment=rel_comment, columns=[])
            views[relname].columns.append(ColumnDoc(name=col, type=col_type, comment=col_comment))
        if not views:
            raise QueryError(f"no readable views in schema '{self.schema}'")
        return SchemaDoc(schema=self.schema, views=list(views.values()))
