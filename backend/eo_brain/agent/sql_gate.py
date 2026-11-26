"""
eo_brain/agent/sql_gate.py - Deterministic gate between model-written SQL and PostGIS.

The data agent's model proposes SQL; nothing reaches the database unless
``SQLGate.check`` accepts it. The gate is deny-by-default: it parses the
text with sqlglot (postgres dialect) and accepts only

  * exactly one statement, a SELECT (optionally with WITH / UNION / EXCEPT /
    INTERSECT), built only from the syntax nodes listed in ``_ALLOWED_NODES``;
  * relations that are views of the ``agent`` schema named in the schema
    documentation, written schema-qualified (``agent.analyses``), or CTEs
    visible at the point of reference;
  * functions listed in ``ALLOWED_FUNCTIONS``, casts to ``_ALLOWED_CAST_TYPES``;
  * a literal ``LIMIT n`` on the outermost query with 1 <= n <= max_limit.

Before parsing, the raw text is rejected if it contains anything where
sqlglot's tokenizer and PostgreSQL's could disagree about statement or
string boundaries: semicolons (other than one trailing), comments,
backslashes, dollar quoting / parameters, Unicode-escape strings and
control characters. The model never needs them; rejecting them removes the
parser-differential class of bypasses instead of trying to reason about it.

The gate is the first of two independent layers. The second is the
database role the data agent connects with (see eo_brain/db/agent_role.sql):
it can only SELECT from the agent views, cannot create anything, and runs
in read-only transactions with a statement timeout. A gate bug alone must
not be enough to write or to read outside the views.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import FrozenSet, Iterable, Optional, Set

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError, TokenError

AGENT_SCHEMA = "agent"
DEFAULT_MAX_LIMIT = 200
MAX_OFFSET = 10_000
MAX_SQL_CHARS = 4000

# Every syntax node the gate accepts. Anything else is rejected by name.
_ALLOWED_NODES = (
    # statement structure
    exp.Select, exp.Union, exp.Except, exp.Intersect, exp.With, exp.CTE, exp.Subquery,
    exp.From, exp.Join, exp.Where, exp.Group, exp.Having, exp.Order, exp.Ordered,
    exp.Limit, exp.Offset, exp.Distinct, exp.Window, exp.WindowSpec, exp.Filter,
    # names and literals
    exp.Table, exp.TableAlias, exp.Alias, exp.Column, exp.Identifier, exp.Star,
    exp.Literal, exp.Boolean, exp.Null, exp.Var, exp.Interval, exp.Tuple, exp.Paren,
    exp.DataType, exp.DataTypeParam,
    # operators
    exp.And, exp.Or, exp.Not, exp.Neg,
    exp.EQ, exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE, exp.NullSafeEQ, exp.NullSafeNEQ,
    exp.Is, exp.In, exp.Between, exp.Like, exp.ILike, exp.Exists,
    exp.Add, exp.Sub, exp.Mul, exp.Div, exp.Mod,
    exp.Case, exp.If, exp.Cast,
)

# Function nodes whose "name" is syntax, not a callable.
_STRUCTURAL_FUNCS = (exp.And, exp.Or, exp.Exists, exp.Case, exp.If, exp.Cast)

# Lower-case function names as sqlglot reports them (Anonymous name or sql_name()).
ALLOWED_FUNCTIONS: FrozenSet[str] = frozenset({
    # aggregates
    "count", "sum", "avg", "min", "max", "stddev",
    # windows
    "row_number", "rank", "dense_rank", "lag", "lead",
    # scalar
    "round", "abs", "ceil", "floor", "coalesce", "nullif", "greatest", "least",
    # time
    "current_timestamp", "current_date", "timestamp_trunc", "extract", "age",
    # PostGIS predicates and measures (no output/serialisation functions)
    "st_intersects", "st_within", "st_contains", "st_covers", "st_coveredby",
    "st_overlaps", "st_touches", "st_disjoint", "st_dwithin", "st_equals",
    "st_area", "st_perimeter", "st_distance", "st_intersection", "st_union",
    "st_envelope", "st_centroid", "st_x", "st_y",
    "st_xmin", "st_ymin", "st_xmax", "st_ymax",
    "st_makeenvelope", "st_makepoint", "st_point", "st_setsrid",
})

_ALLOWED_CAST_TYPES = frozenset({
    exp.DataType.Type.SMALLINT, exp.DataType.Type.INT, exp.DataType.Type.BIGINT,
    exp.DataType.Type.DECIMAL, exp.DataType.Type.DOUBLE, exp.DataType.Type.FLOAT,
    exp.DataType.Type.BOOLEAN, exp.DataType.Type.DATE, exp.DataType.Type.TIMESTAMP,
    exp.DataType.Type.TIMESTAMPTZ, exp.DataType.Type.INTERVAL,
    exp.DataType.Type.GEOGRAPHY, exp.DataType.Type.GEOMETRY,
})

# Bare words PostgreSQL evaluates as functions (SQL-standard "niladic" functions).
_KEYWORD_FUNCTIONS = frozenset({
    "user", "current_user", "session_user", "current_role", "system_user",
    "current_catalog", "current_schema", "localtime", "localtimestamp",
})

# Friendlier reasons for constructs a model is likely to try.
_NODE_REASONS = {
    exp.Insert: "INSERT is a write operation",
    exp.Update: "UPDATE is a write operation",
    exp.Delete: "DELETE is a write operation",
    exp.Merge: "MERGE is a write operation",
    exp.Create: "CREATE is DDL",
    exp.Drop: "DROP is DDL",
    exp.Alter: "ALTER is DDL",
    exp.TruncateTable: "TRUNCATE is a write operation",
    exp.Copy: "COPY is not allowed",
    exp.Set: "SET is not allowed",
    exp.Command: "statement type not allowed",
    exp.Into: "SELECT ... INTO creates a table",
    exp.Lock: "row locking (FOR UPDATE / FOR SHARE) is not allowed",
    exp.Lateral: "LATERAL is not allowed",
    exp.Values: "VALUES lists are not allowed",
    exp.Fetch: "use LIMIT instead of FETCH",
    exp.Dot: "schema-qualified function calls are not allowed",
    exp.DPipe: "string concatenation is not allowed",
    exp.Returning: "RETURNING implies a write operation",
}

_RAW_RULES = [
    (re.compile(r";"), "multiple statements (';') are not allowed"),
    (re.compile(r"--|/\*|\*/"), "SQL comments are not allowed"),
    (re.compile(r"\\"), "backslashes are not allowed"),
    (re.compile(r"\$"), "dollar quoting and parameters are not allowed"),
    (re.compile(r"\b[uU]&|\buescape\b", re.IGNORECASE), "Unicode escape strings are not allowed"),
    (re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]"), "control characters are not allowed"),
]


@dataclass(frozen=True)
class GateDecision:
    ok: bool
    reason: str = ""
    sql: str = ""  # the accepted SQL (trailing ';' and whitespace removed)


class SQLRejected(ValueError):
    pass


def _func_name(node: exp.Func) -> str:
    if isinstance(node, exp.Anonymous):
        return str(node.name).lower()
    return node.sql_name().lower()


def _ident(identifier: Optional[exp.Expression]) -> Optional[str]:
    """Identifier as PostgreSQL resolves it: unquoted folds to lower case."""
    if identifier is None:
        return None
    if isinstance(identifier, exp.Identifier):
        return identifier.this if identifier.quoted else identifier.this.lower()
    return str(identifier.name).lower()


def _with_clause(node: exp.Expression) -> Optional[exp.With]:
    w = node.args.get("with_") or node.args.get("with")
    return w if isinstance(w, exp.With) else None


def _visible_ctes(table: exp.Table) -> Set[str]:
    """
    CTE names visible where ``table`` is referenced. A non-recursive CTE sees
    only the CTEs defined before it; the main query sees all of them.
    """
    visible: Set[str] = set()
    child: exp.Expression = table
    parent = table.parent
    while parent is not None:
        if isinstance(parent, exp.With):
            ctes = list(parent.expressions)
            idx = next((i for i, c in enumerate(ctes) if c is child), None)
            if idx is not None:
                visible.update(_ident(c.args["alias"].this) for c in ctes[:idx])
        else:
            w = _with_clause(parent)
            if w is not None and child is not w:
                visible.update(_ident(c.args["alias"].this) for c in w.expressions)
        child, parent = parent, parent.parent
    return visible


class SQLGate:
    def __init__(
        self,
        allowed_relations: Iterable[str],
        *,
        schema: str = AGENT_SCHEMA,
        max_limit: int = DEFAULT_MAX_LIMIT,
        allowed_functions: FrozenSet[str] = ALLOWED_FUNCTIONS,
    ):
        self.allowed_relations = frozenset(r.lower() for r in allowed_relations)
        if not self.allowed_relations:
            raise ValueError("SQLGate needs at least one allowed relation")
        self.schema = schema
        self.max_limit = max_limit
        self.allowed_functions = allowed_functions

    # -- public -------------------------------------------------------------

    def check(self, sql: object) -> GateDecision:
        try:
            return GateDecision(ok=True, sql=self._check(sql))
        except SQLRejected as e:
            return GateDecision(ok=False, reason=str(e))

    # -- implementation -----------------------------------------------------

    def _check(self, sql: object) -> str:
        if not isinstance(sql, str):
            raise SQLRejected("SQL must be a string")
        text = sql.strip()
        if text.endswith(";"):
            text = text[:-1].rstrip()
        if not text:
            raise SQLRejected("empty SQL")
        if len(text) > MAX_SQL_CHARS:
            raise SQLRejected(f"SQL longer than {MAX_SQL_CHARS} characters")
        for pattern, reason in _RAW_RULES:
            if pattern.search(text):
                raise SQLRejected(reason)

        try:
            statements = [s for s in sqlglot.parse(text, read="postgres") if s is not None]
        except (ParseError, TokenError) as e:
            raise SQLRejected(f"could not parse SQL: {str(e).splitlines()[0][:200]}") from None
        if len(statements) != 1:
            raise SQLRejected(f"expected exactly one statement, got {len(statements)}")
        root = statements[0]

        if not isinstance(root, (exp.Select, exp.Union, exp.Except, exp.Intersect)):
            reason = _NODE_REASONS.get(type(root), f"{type(root).__name__} statements are not allowed")
            raise SQLRejected(f"only SELECT queries are allowed ({reason})")

        for node in root.walk():
            self._check_node(node)
        self._check_limit(root)
        return text

    def _check_node(self, node: exp.Expression) -> None:
        for cls, reason in _NODE_REASONS.items():
            if isinstance(node, cls):
                raise SQLRejected(reason)

        if isinstance(node, exp.Func) and not isinstance(node, _STRUCTURAL_FUNCS):
            name = _func_name(node)
            if name not in self.allowed_functions:
                raise SQLRejected(f"function not allowed: {name}")
        elif not isinstance(node, _ALLOWED_NODES):
            raise SQLRejected(f"construct not allowed: {type(node).__name__}")

        if isinstance(node, exp.With) and node.args.get("recursive"):
            raise SQLRejected("WITH RECURSIVE is not allowed")
        if isinstance(node, exp.Cast):
            target = node.to.this if isinstance(node.to, exp.DataType) else None
            if target not in _ALLOWED_CAST_TYPES:
                raise SQLRejected(f"cast to {node.to.sql('postgres')} is not allowed")
        if isinstance(node, exp.Column):
            self._check_column(node)
        if isinstance(node, exp.Table):
            self._check_table(node)
        if isinstance(node, (exp.Limit, exp.Offset)):
            self._literal_int(node.expression, type(node).__name__.upper())

    def _check_column(self, col: exp.Column) -> None:
        if col.args.get("db") is not None or col.args.get("catalog") is not None:
            raise SQLRejected(f"column reference {col.sql('postgres')} has too many qualifiers")
        if not col.table and isinstance(col.this, exp.Identifier) and not col.this.quoted:
            if col.this.this.lower() in _KEYWORD_FUNCTIONS:
                raise SQLRejected(f"function not allowed: {col.this.this.lower()}")

    def _check_table(self, table: exp.Table) -> None:
        if not isinstance(table.this, exp.Identifier):
            raise SQLRejected("table functions are not allowed in FROM")
        if table.args.get("catalog") is not None:
            raise SQLRejected(f"relation {table.sql('postgres')} is not allowed")
        schema = _ident(table.args.get("db"))
        name = _ident(table.this)
        if schema is not None:
            if schema != self.schema or name not in self.allowed_relations:
                raise SQLRejected(
                    f"relation {schema}.{name} is not allowed; available: "
                    + ", ".join(f"{self.schema}.{r}" for r in sorted(self.allowed_relations))
                )
            return
        if name in _visible_ctes(table):
            return
        if name in self.allowed_relations:
            raise SQLRejected(f"write the relation schema-qualified: {self.schema}.{name}")
        raise SQLRejected(f"relation {name} is not allowed")

    def _literal_int(self, node: Optional[exp.Expression], what: str) -> int:
        if not (isinstance(node, exp.Literal) and not node.is_string and str(node.this).isdigit()):
            raise SQLRejected(f"{what} must be a literal integer")
        value = int(node.this)
        if what == "OFFSET" and value > MAX_OFFSET:
            raise SQLRejected(f"OFFSET above {MAX_OFFSET} is not allowed")
        return value

    def _check_limit(self, root: exp.Expression) -> None:
        limit = root.args.get("limit")
        if not isinstance(limit, exp.Limit):
            raise SQLRejected(f"the outermost query needs LIMIT n with n <= {self.max_limit}")
        value = self._literal_int(limit.expression, "LIMIT")
        if not 1 <= value <= self.max_limit:
            raise SQLRejected(f"LIMIT must be between 1 and {self.max_limit}, got {value}")
