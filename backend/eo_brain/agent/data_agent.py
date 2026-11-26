"""
eo_brain/agent/data_agent.py - Text-to-SQL sub-agent over the curated PostGIS views.

Isolation model:
  * its own system prompt, built from the catalog-generated schema doc;
  * its own context: only that prompt and the question (plus an optional
    area hint computed by code) - never the orchestrator's conversation;
  * its own database role and connection (see sql_exec.py, agent_role.sql);
  * every SQL string passes the deterministic SQLGate before execution.

Through the provider seam it must answer with exactly one structured
action per turn: ``run_sql(sql)`` or ``refuse(reason)``. Gate rejections and
database errors are fed back for a fixed, small number of attempts; after
that it fails honestly with the last error.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from eo_brain.agent.llm import (
    LLMProvider,
    Message,
    ProviderError,
    ToolResult,
    ToolResults,
    ToolSpec,
    UserMessage,
)
from eo_brain.agent.sql_exec import QueryError, SchemaDoc
from eo_brain.agent.sql_gate import ALLOWED_FUNCTIONS, SQLGate

logger = logging.getLogger(__name__)

MAX_SQL_ATTEMPTS = 3          # one try + two corrections, fixed on purpose
MAX_QUESTION_CHARS = 1000

DATA_TOOL_SPECS: List[ToolSpec] = [
    ToolSpec(
        name="run_sql",
        description=(
            "Run ONE read-only PostgreSQL SELECT over the documented views and get the "
            "rows back, or an error to fix."
        ),
        parameters={
            "type": "object",
            "properties": {"sql": {"type": "string", "description": "A single SELECT statement."}},
            "required": ["sql"],
        },
    ),
    ToolSpec(
        name="refuse",
        description=(
            "Decline: the question cannot be answered from the documented views, "
            "or it asks to change data or to read anything else."
        ),
        parameters={
            "type": "object",
            "properties": {"reason": {"type": "string", "description": "Short reason, in the user's language."}},
            "required": ["reason"],
        },
    ),
]

DATA_SYSTEM_PROMPT = """\
You translate one question about stored satellite analyses into one PostgreSQL query.
Answer ONLY by calling a tool: run_sql(sql) or refuse(reason). One call per turn.

RULES
1. Use only the views documented below, always schema-qualified (e.g. agent.analyses).
2. One SELECT statement (WITH is allowed). It must end with LIMIT n, 1 <= n <= {max_limit}.
3. No semicolons, comments, string concatenation, casts to text, or functions outside this list:
   {functions}
4. Geometry columns (footprint, geometry) may only appear inside spatial functions, e.g.
   ST_Intersects(a.footprint, b.footprint) or ST_Area(footprint::geography). They are never returned.
5. Prefer the ready-made unit columns (area_ha, total_area_ha, area_km2...) and labels
   (severity_label: watch / suspect / alert). "Latest" means the highest created_at.
6. Return only the columns needed to answer, with readable aliases.
7. If the question cannot be answered from these views, or asks to modify data or to read
   anything else (other schemas, system catalogs, settings, files), call refuse.
8. If run_sql returns an error, fix the query and call run_sql again.

VIEWS
{schema_doc}
"""


@dataclass
class SQLAttempt:
    sql: str
    error: Optional[str] = None
    stage: str = ""  # "gate" | "database" | "" (success)


@dataclass
class DataAgentResult:
    status: str                       # "ok" | "refused" | "failed"
    question: str
    sql: Optional[str] = None         # the SQL that was executed (status ok)
    columns: List[str] = field(default_factory=list)
    rows: List[List[Any]] = field(default_factory=list)
    row_count: int = 0
    truncated: bool = False
    omitted_columns: List[str] = field(default_factory=list)
    reason: Optional[str] = None      # refusal reason
    error: Optional[str] = None       # failure reason
    attempts: List[SQLAttempt] = field(default_factory=list)
    model: Optional[str] = None

    @property
    def gate_rejections(self) -> int:
        return sum(1 for a in self.attempts if a.stage == "gate")

    def to_tool_response(self) -> Dict[str, Any]:
        """What the orchestrator model sees."""
        if self.status == "ok":
            out: Dict[str, Any] = {
                "status": "ok", "sql": self.sql, "columns": self.columns, "rows": self.rows,
                "row_count": self.row_count, "truncated": self.truncated,
            }
            if self.omitted_columns:
                out["omitted_columns"] = self.omitted_columns
            return out
        if self.status == "refused":
            return {"status": "refused", "reason": self.reason}
        return {"status": "failed", "error": self.error}

    def to_api_payload(self) -> Dict[str, Any]:
        """What the chat API returns to the client (executed SQL + rows)."""
        return {
            "sql": self.sql, "columns": self.columns, "rows": self.rows,
            "row_count": self.row_count, "truncated": self.truncated,
        }


class DataAgent:
    def __init__(
        self,
        provider: LLMProvider,
        *,
        models: Sequence[str],
        executor: Any,
        schema_doc: Optional[SchemaDoc] = None,
        gate: Optional[SQLGate] = None,
        max_attempts: int = MAX_SQL_ATTEMPTS,
    ):
        if not models:
            raise ValueError("At least one model id is required")
        self.provider = provider
        self.models = list(models)
        self.executor = executor
        self._schema_doc = schema_doc
        self._gate = gate
        self.max_attempts = max_attempts

    # Lazily read the catalog once, through the agent's own connection.
    def _ensure_schema(self) -> None:
        if self._schema_doc is None:
            self._schema_doc = self.executor.load_schema_doc()
        if self._gate is None:
            self._gate = SQLGate(
                self._schema_doc.relation_names(),
                schema=self._schema_doc.schema,
                max_limit=getattr(self.executor, "max_rows", 100),
            )

    def system_prompt(self) -> str:
        self._ensure_schema()
        return DATA_SYSTEM_PROMPT.format(
            max_limit=self._gate.max_limit,
            functions=", ".join(sorted(ALLOWED_FUNCTIONS)),
            schema_doc=self._schema_doc.render(),
        )

    def answer(self, question: str, *, area_hint: Optional[str] = None) -> DataAgentResult:
        if not isinstance(question, str) or not question.strip():
            return DataAgentResult(status="failed", question=str(question), error="empty question")
        question = question.strip()[:MAX_QUESTION_CHARS]
        try:
            prompt = self.system_prompt()
        except QueryError as e:
            return DataAgentResult(status="failed", question=question, error=str(e))

        text = question if not area_hint else f"{question}\n\n(Context from the app: {area_hint})"
        messages: List[Message] = [UserMessage(text=text)]
        attempts: List[SQLAttempt] = []
        model_idx = 0

        while len(attempts) < self.max_attempts:
            model = self.models[model_idx]
            try:
                turn = self.provider.generate(
                    model=model, system_prompt=prompt, tools=DATA_TOOL_SPECS, messages=messages,
                )
            except ProviderError as e:
                logger.warning(f"data agent: {model} failed: {e}")
                model_idx += 1
                if model_idx >= len(self.models):
                    return DataAgentResult(status="failed", question=question, attempts=attempts,
                                           error=f"model unavailable: {e}")
                continue

            if turn is None or not turn.tool_calls:
                attempts.append(SQLAttempt(sql="", error="no tool call", stage="protocol"))
                if turn is not None and turn.text:
                    messages.append(turn)
                    messages.append(UserMessage(text="Answer by calling run_sql or refuse."))
                continue

            call, extra = turn.tool_calls[0], turn.tool_calls[1:]
            responses: List[ToolResult] = []

            if call.name == "refuse":
                return DataAgentResult(status="refused", question=question, attempts=attempts,
                                       reason=str(call.args.get("reason") or "refused"), model=model)

            if call.name != "run_sql":
                attempt = SQLAttempt(sql="", error=f"unknown tool {call.name}", stage="protocol")
            else:
                attempt, result = self._try_sql(call.args.get("sql"))
                if result is not None:
                    attempts.append(attempt)
                    return DataAgentResult(
                        status="ok", question=question, sql=attempt.sql, columns=result.columns,
                        rows=result.rows, row_count=result.row_count, truncated=result.truncated,
                        omitted_columns=result.omitted_columns, attempts=attempts, model=model,
                    )
            attempts.append(attempt)
            responses.append(ToolResult(name=call.name, response={"error": attempt.error}))
            responses.extend(
                ToolResult(name=c.name, response={"error": "only one tool call per turn is processed"})
                for c in extra
            )
            messages.append(turn)
            messages.append(ToolResults(results=responses))

        last = attempts[-1].error if attempts else "no attempt"
        return DataAgentResult(
            status="failed", question=question, attempts=attempts,
            sql=None, error=f"gave up after {len(attempts)} attempts; last error: {last}",
        )

    def _try_sql(self, sql: Any):
        decision = self._gate.check(sql)
        if not decision.ok:
            logger.info(f"data agent: gate rejected SQL: {decision.reason}")
            return SQLAttempt(sql=str(sql or ""), error=f"rejected by SQL gate: {decision.reason}", stage="gate"), None
        try:
            result = self.executor.execute(decision.sql)
        except QueryError as e:
            return SQLAttempt(sql=decision.sql, error=str(e), stage="database"), None
        return SQLAttempt(sql=decision.sql), result
