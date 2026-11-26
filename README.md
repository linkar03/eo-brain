# EO Brain: a text-to-GIS prototype

You type a request in Italian or English, for example *"allagamenti a Valencia negli ultimi 3 mesi"*
or *"vegetation stress near Pisa"*. The app geocodes the place and runs a satellite analysis
over public STAC catalogues (Sentinel-2, Landsat). It then draws the result on a map as a raster
layer plus vectorised alert polygons.

A second, **read-only** agent answers questions about analyses that already ran, for example
*"how many hectares were at alert level in the latest flood analysis?"*. It writes SQL over
curated PostGIS views, and the chat shows the SQL and the rows it returned.

This is a prototype. See [Limitations](#limitations).

## Contents

- [What it does](#what-it-does)
- [Architecture](#architecture)
- [The deterministic boundary](#the-deterministic-boundary)
- [The data agent and its isolation model](#the-data-agent-and-its-isolation-model)
- [Quickstart (docker compose)](#quickstart-docker-compose)
- [Configuration](#configuration)
- [Tests](#tests)
- [Evals](#evals)
- [Limitations](#limitations)
- [Repository layout](#repository-layout)

## What it does

| Algorithm | What it shows |
|---|---|
| `ndvi`  | Vegetation health composite |
| `ndwi`  | Water detection (green / NIR) |
| `mndwi` | Water detection (green / SWIR) |
| `flood` | Flood extent with severity classes (watch / suspect / alert) |
| `lisa`  | Spatial outliers in vegetation (Local Moran's I, HH / LL / HL / LH quadrants) |

- **Data sources:** STAC providers `element84-s2` (default), `element84-landsat`, `copernicus-s2` and `planetary-s2`.
- **Processing:** lazy Dask/xarray pipelines, so large areas do not load into memory at once.
- **Outputs:** Cloud-Optimised GeoTIFFs served as XYZ tiles (`/tiles/...`), and alert polygons stored in PostGIS and served as vector tiles or GeoJSON (`/vectors/...`).
- **Frontend:** React, MapLibre and a chat panel.

## Architecture

```mermaid
flowchart LR
    UI["Frontend<br/>React + MapLibre<br/>AgentChat"] -->|POST /api/chat| API["FastAPI<br/>eo_brain/api/main.py"]
    API --> ORCH["Orchestrator agent<br/>eo_brain/agent/brain.py"]
    ORCH <-->|LLMProvider seam| LLM[("Gemini<br/>or scripted replay")]

    ORCH -->|launch_analysis<br/>rerun_with_different_algorithm| VAL["Validators + geocoder<br/>(deterministic)"]
    VAL --> JOBS[("PostGIS<br/>jobs, alert_features")]
    API -->|worker thread| PIPE["STAC search + Dask pipeline<br/>COG + vectorised polygons"]
    PIPE --> JOBS
    ORCH -->|get_job_status| JOBS

    ORCH -->|"query_database(question)"| DA["Data agent<br/>eo_brain/agent/data_agent.py"]
    DA <-->|own prompt + own context| LLM
    DA --> GATE["SQL gate<br/>sql_gate.py (sqlglot AST)"]
    GATE --> EXEC["Read-only executor<br/>sql_exec.py"]
    EXEC -->|role agent_reader| VIEWS[("schema agent<br/>curated views")]
    VIEWS -. reads .-> JOBS
```

The orchestrator is a manual function-calling loop. The model sees five tools:

1. `launch_analysis`
2. `get_job_status`
3. `list_capabilities`
4. `rerun_with_different_algorithm`
5. `query_database`

Code executes every call and returns a structured result. Each request is capped at 5 tool rounds.

The model sits behind a small provider interface (`eo_brain/agent/llm.py`) with two implementations:

- `GeminiProvider` (`providers/gemini.py`, google-genai SDK, automatic function calling disabled);
- `ScriptedProvider` (`providers/scripted.py`), a deterministic replay used by tests and offline evals.

The model id and its fallback chain come from configuration (see [Configuration](#configuration)).

## The deterministic boundary

The model only turns free text into a tool name and arguments, or into SQL for the data agent.
Code decides everything else.

### Orchestrator

| Decided by the model | Decided by code |
|---|---|
| Which tool to call | Input sanitisation (control characters stripped, 2000-character cap) |
| `algorithm`, `location`, `days_back` taken from the text | Algorithm alias resolution (`"alluvione"` becomes `flood`); unknown algorithms are rejected |
| | Location validation (at most 200 characters) |
| | `days_back` coerced to a whole number; values outside 1–365 are rejected, not clamped |
| | Geocoding through Nominatim, with bounding-box padding and clamping. Transient geocoder errors are not cached. |
| | Date range computed from an injected clock |
| | Area and band checks, and job creation |
| | Session memory, kept per server process |
| | Retry with backoff on HTTP 429, a model fallback chain, and no second job if a retry follows a job that was already created |
| | Tool-round cap |

### Data agent

| Decided by the model | Decided by code |
|---|---|
| SQL text, or a refusal | Accepting or rejecting the SQL (the gate) |
| | What the database lets the role do (the role) |
| | Timeouts |
| | Row and payload caps |
| | Which column types may leave the database |
| | The number of attempts |

One rule is **not** enforced by code: the instruction to "only state numbers that appear in the rows". The orchestrator prompt asks for it, and the chat API returns the executed SQL and the rows next to the model's text, so a reader can check.

## The data agent and its isolation model

`query_database(question)` hands the question to a separate sub-agent. The sub-agent has:

- **its own prompt**, generated from the database catalogue (`COMMENT ON` of every view and column), not hand-written;
- **its own context**: only that prompt plus the question and an optional area hint computed by code. It never sees the orchestrator's conversation.
- **a structured answer**: exactly one `run_sql(sql)` or `refuse(reason)` per turn. Gate rejections and database errors go back to it for at most 3 attempts in total, then it fails honestly with the last error.
- **its own connection** as the role `agent_reader`, which is configured separately from the app's connection (`AGENT_DATABASE_URL`).

Two independent layers sit between the model and the data. Each one alone blocks writes.

### Layer 1: the SQL gate (`eo_brain/agent/sql_gate.py`)

The gate parses the SQL with sqlglot (postgres dialect) and walks the AST against allow-lists.

**Before parsing,** the raw text is rejected if it contains any of these. This closes the gaps between how sqlglot and PostgreSQL tokenise:

- `;` (one trailing `;` is stripped first)
- comments
- backslashes
- `$`
- `U&` / `UESCAPE`
- control characters

**The statement:**

- Only a single `SELECT`, `WITH … SELECT`, or set operation (`UNION` / `INTERSECT` / `EXCEPT`) is accepted.
- `INSERT`, `UPDATE`, `DELETE`, `MERGE`, DDL, `COPY`, `SET`, `SELECT … INTO`, locking clauses (`FOR UPDATE` and similar), `RETURNING`, `LATERAL`, `VALUES`, `FETCH` and unparsed commands are rejected, including inside CTEs.

**Relations:**

- Only the views in schema `agent` are allowed, schema-qualified.
- Unqualified names are accepted only when they refer to a CTE that is actually visible at that point. A non-recursive CTE sees only earlier CTEs, and CTEs defined inside a subquery are not visible outside it.

**Functions** are allow-listed by name:

- **Aggregates and window functions:** `count sum avg min max stddev row_number rank dense_rank lag lead`
- **Scalar:** `round abs ceil floor coalesce nullif greatest least`
- **Time:** `current_timestamp current_date date_trunc extract age` (sqlglot parses `date_trunc` as `timestamp_trunc`)
- **PostGIS predicates:** `st_intersects st_within st_contains st_covers st_coveredby st_overlaps st_touches st_disjoint st_dwithin st_equals`
- **PostGIS measures and constructors:** `st_area st_perimeter st_distance st_intersection st_union st_envelope st_centroid st_x st_y st_xmin st_ymin st_xmax st_ymax st_makeenvelope st_makepoint st_point st_setsrid`

Notably, nothing that serialises geometry is on the list: no `ST_AsText`, no `ST_AsGeoJSON`.

**Everything else:**

- Casts are allowed only to numeric, boolean, date and time, interval, `geometry` and `geography` types. Casts to text are not allowed.
- The string-concatenation operator `||` is rejected.
- The outer query needs a literal `LIMIT` between 1 and the row cap (100 in the app). `OFFSET`, if present, must be a literal of at most 10,000.

### Layer 2: the database role (`eo_brain/db/agent_role.sql`)

The role `agent_reader`:

- is a plain `LOGIN` role: not a superuser, cannot create databases or roles, does not inherit, cannot replicate, does not bypass row-level security, and has a connection limit of 10;
- has `SELECT` on schema `agent` and nothing else in the application schema;
- has these settings on the role itself: `default_transaction_read_only = on`, `statement_timeout = 3s`, `lock_timeout = 1s`, `idle_in_transaction_session_timeout = 10s`;
- cannot create temporary tables (`TEMPORARY` is revoked from `PUBLIC`), and `PUBLIC` cannot create anything in schema `public`.

The executor also opens every connection read-only, with its own `statement_timeout`. It runs one transaction per query and always rolls it back.

The DB tests prove the role alone with the gate bypassed: they talk to PostgreSQL as `agent_reader` directly. For example:

- `DELETE`, `UPDATE`, `INSERT`, `TRUNCATE`, `CREATE TABLE`, `CREATE VIEW`, `CREATE SCHEMA`, `CREATE TEMP TABLE` and `COPY` fail, even after the session runs `SET TRANSACTION READ WRITE`;
- reading `public.jobs` or `public.alert_features` fails, and so do `pg_read_file`, `pg_ls_dir` and `lo_import`;
- `pg_sleep(10)` is cancelled by the role's statement timeout.

### Output shaping (`eo_brain/agent/sql_exec.py`)

- Only scalar column types (numbers, text, booleans, dates and times, UUIDs) are returned.
- Geometry and other non-scalar values become `"<geometry omitted>"`.
- At most 100 rows, about 16,000 characters of JSON and 300 characters per text value, with `truncated` set when a cap is hit.

### Known residual risk

`pg_catalog` and `information_schema` are readable by `PUBLIC` in PostgreSQL. That does not expose table contents, but at the role level `agent_reader` can list object names and its own settings. Only the gate stops the model from querying those catalogues, and a DB test documents this explicitly.

`agent_reader` is also granted `SELECT` on `public.spatial_ref_sys`, because PostGIS reads it for `::geography` casts (used by the area columns). That table holds public projection definitions.

## Quickstart (docker compose)

```bash
cp .env.example .env          # set GEMINI_API_KEY; change AGENT_DB_PASSWORD
docker compose up --build
```

- Frontend: <http://localhost:5173>
- API: <http://localhost:8001>, with OpenAPI docs at `/docs`
- PostGIS: `localhost:5433`

The database init scripts (`schema.sql`, `schema_vectors.sql`, `agent_views.sql`, `agent_role.sql`, and `agent_role_password.sh` to set the role password from `AGENT_DB_PASSWORD`) run **only on an empty data volume**. To add the data agent to an existing volume, apply the scripts by hand:

```bash
docker compose exec -T db psql -U postgres -d eo_brain < backend/eo_brain/db/schema_vectors.sql   # if missing
docker compose exec -T db psql -U postgres -d eo_brain < backend/eo_brain/db/agent_views.sql
docker compose exec -T db psql -U postgres -d eo_brain < backend/eo_brain/db/agent_role.sql
docker compose exec db psql -U postgres -d eo_brain -c "ALTER ROLE agent_reader PASSWORD '<your AGENT_DB_PASSWORD>'"
```

Without `GEMINI_API_KEY` the chat answers that the agent is not configured. The map and manual analyses still work. Without `AGENT_DATABASE_URL`, the `query_database` tool reports that it is not configured.

## Configuration

| Variable | Used by | Meaning |
|---|---|---|
| `GEMINI_API_KEY` | api | Enables the agents. |
| `GEMINI_MODEL` | api | Primary model id. Empty means the default in `providers/gemini.py`. |
| `GEMINI_FALLBACK_MODELS` | api | Comma-separated fallback chain, tried in order. |
| `AGENT_DB_PASSWORD` | db init, api | Password of `agent_reader`. |
| `AGENT_DATABASE_URL` | api | The data agent's own connection, as `agent_reader`. compose builds it for you. |
| `DATABASE_URL` | api | The app's connection, used for jobs and results. |
| `TEST_DATABASE_URL` | tests, evals | Superuser URL to a **throwaway** PostGIS server. It enables the DB tests and the SQL eval. |

The default model ids were checked against Google's model and deprecation pages on 2026-11-26:

- <https://aistudio.google.com/docs/models>
- <https://aistudio.google.com/docs/deprecations>

Override them through the environment rather than editing code.

## Tests

All commands run from `backend/`. CI uses `requirements-ci.txt`; the full stack needs `requirements.txt`.

### Offline tests

These need no network, no database and no LLM. They cover:

- validators, the geocoder (mocked), sessions and the Gemini provider mapping;
- the orchestrator loop with a fake model: dispatch, unknown tool, tool errors relayed, the round cap, rate-limit fallback, sanitisation, and the rerun session path;
- the SQL gate: over 200 accept and reject cases, including CTE-scope, quoting and lexer tricks;
- the data-agent loop;
- the eval scorers.

```bash
python -m pytest -q tests/test_agent_*.py tests/test_evals.py \
    tests/test_sql_gate.py tests/test_data_agent.py tests/test_sql_evals.py
```

### DB-backed tests

These are marked `db` and skipped unless `TEST_DATABASE_URL` is set. Each run creates a scratch database, applies the same SQL files compose applies, loads a **synthetic** seed (`evals/fixtures/synthetic_seed.sql`), and drops the database afterwards. Use a throwaway server: the role `agent_reader` is cluster-wide, and its password is reset.

```bash
docker run --rm -d --name eo-brain-test-db -e POSTGRES_PASSWORD=postgres \
    -p 127.0.0.1:55432:5432 postgis/postgis:16-3.4
export TEST_DATABASE_URL=postgresql://postgres:postgres@127.0.0.1:55432/postgres
python -m pytest -q -m db
docker stop eo-brain-test-db
```

### Original test suite

The original tests are in `tests/test_algorithms.py`, `test_api.py`, `test_datacube.py`, `test_vectorize.py`, and `test_integration.py`, which is marked `integration` and calls real STAC APIs. They need the full `requirements.txt`. See [Limitations](#limitations) for the ones that currently fail.

## Evals

`evals/run.py` runs versioned datasets through the real agent loops.

**`tools` suite** (`evals/cases.v1.yaml`, 34 cases, Italian and English). It covers text to tool call for the orchestrator:

- compact commands and natural language with time expressions;
- follow-ups that rely on session memory;
- status lookups;
- refusals, such as out-of-scope requests, impossible ranges and prompt injection;
- routing between `query_database` and `launch_analysis`.

Geocoding (a fixed gazetteer), job storage, the clock and the data agent are stubbed, so no network or database is needed.

It reports:

- tool-selection accuracy;
- argument accuracy;
- correct-refusal rate;
- routing accuracy.

**`sql` suite** (`evals/sql_cases.v1.yaml`, 18 cases, Italian and English). It covers question to SQL to rows for the data agent, on a scratch database with the synthetic seed:

- **Text-to-SQL cases** are scored by execution accuracy: the agent's result set must equal the gold SQL's result set. Rows are compared as a multiset, column names are ignored, and numbers are rounded to 6 decimals.
- **Adversarial cases** are scored on being blocked: write requests, injection inside the question, other tables, system catalogues, settings and sleep, file reads, and geometry export. A refusal or an honest give-up counts as blocked. A few cases also accept a safe, gate-approved answer to the harmless part of the question.
- **Every case** also fails if the database content changed or a returned value looks like raw geometry.

```bash
# Offline, deterministic: replays canned model output stored in the case files.
python -m evals.run --provider scripted                 # tools suite
python -m evals.run --provider scripted --suite sql     # needs TEST_DATABASE_URL or --admin-url

# Live model (needs GEMINI_API_KEY; the sql suite also needs a throwaway database):
python -m evals.run --provider gemini --suite all [--model <id> ...]
```

> **`--provider scripted` is not a measure of model quality.** It replays ideal model output, so it checks only the harness, the dispatch loop, the validators, the SQL gate, the role and the scoring. It is a regression gate: CI fails if any scripted case fails.

Each run prints a per-case table and a summary, and writes a JSON report to `evals/reports/` (git-ignored). The report includes the dataset's sha256 and `measures_model_quality: false|true`.

### Model-quality results

| Suite | Model | Command | Result |
|---|---|---|---|
| tools | (default chain) | `python -m evals.run --provider gemini --suite tools` | not yet run |
| sql   | (default chain) | `python -m evals.run --provider gemini --suite sql`   | not yet run |

## Limitations

- **It is a prototype.**
  - The API has no authentication and CORS is open.
  - Sessions live in process memory.
  - Long analyses run on a thread pool inside the API process.
- **Nominatim's usage policy applies:** at most one request per second and an identifying user agent. Results are cached per process.
- **Gemini is the only live provider.** The provider interface is small, so adding a second one should not touch the agent loop, but none is implemented yet.
- **The datasets are small and hand-written,** and the SQL suite runs on synthetic data. Live results vary between runs, and there is no repeat or variance reporting yet.
- **"Only cite numbers from the rows" is a prompt rule,** not a code check. The UI shows the rows so it can be verified.
- **Data agent:**
  - It opens one database connection per query, with no pooling.
  - It does not back off on rate limits; it moves to the next model in the chain.
  - The role-level residual risk on `pg_catalog` and `information_schema` is described above.
- **The SQL gate is tied to the pinned sqlglot version.** Re-run `tests/test_sql_gate.py` before upgrading it.
- **Some tests in the original suite currently fail:**
  - two `test_api.py` tests need a running app database on `localhost:5432`;
  - `test_algorithms.py::TestLISAMathematics::test_bio_gate_excludes_non_vegetation`;
  - `test_datacube.py::TestLoadConfig::test_default_values`, which expects 2048-pixel chunks while the config yields 1024.
- **The compose init scripts run only on a fresh volume** (see the [Quickstart](#quickstart-docker-compose)).
- **No license has been chosen yet.**

## Repository layout

```
backend/
  eo_brain/agent/        orchestrator (brain.py), data agent, SQL gate, executor, providers, validators, geocoder
  eo_brain/algorithms/   NDVI / NDWI / MNDWI / flood / LISA, vectorisation
  eo_brain/api/          FastAPI app, raster tiles, vector tiles
  eo_brain/core/         STAC providers and the Dask data cube
  eo_brain/db/           schema, agent views, agent role, repositories
  evals/              datasets, runner, scorers, stubs, synthetic seed
  tests/
frontend/       React + MapLibre dashboard and chat
.github/workflows/    CI: offline tests + scripted eval; PostGIS service for DB tests + SQL eval
```
