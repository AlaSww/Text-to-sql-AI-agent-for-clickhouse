# DOCUMENTATION — Text-to-SQL AI Agent for ClickHouse

> A FastAPI-based service that converts natural-language questions into executable ClickHouse SQL queries, using LlamaIndex for schema-aware retrieval and LLMs (Groq or vLLM) for SQL generation.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Architecture](#architecture)
3. [Directory Structure](#directory-structure)
4. [End-to-End Data Flow](#end-to-end-data-flow)
5. [Configuration & Environment Variables](#configuration--environment-variables)
6. [Schema Extraction](#schema-extraction)
7. [Vector Index & Retrieval](#vector-index--retrieval)
8. [LLM Abstraction](#llm-abstraction)
9. [SQL Generation](#sql-generation)
10. [SQL Safety Validation](#sql-safety-validation)
11. [Query Execution](#query-execution)
12. [API Routes](#api-routes)
13. [Prompt Templates](#prompt-templates)
14. [Error Handling & Self-Correction](#error-handling--self-correction)
15. [Logging Strategy](#logging-strategy)
16. [Running Locally](#running-locally)
17. [Deployment](#deployment)
18. [Testing](#testing)
19. [Extending the System](#extending-the-system)

---

## System Overview

This project is a **Text-to-SQL AI agent** designed specifically for **ClickHouse** databases running a star-schema data warehouse (database `gold`). A user sends a natural-language question via a REST API; the system:

1. **Extracts** the ClickHouse schema at startup.
2. **Embeds** table metadata into a LlamaIndex `VectorStoreIndex` using a local HuggingFace embedding model (`BAAI/bge-small-en-v1.5`).
3. **Retrieves** the most relevant tables for each query via semantic similarity search.
4. **Generates** ClickHouse-compatible SQL using an LLM (Groq or self-hosted vLLM).
5. **Validates** the generated SQL against a safety policy (read-only).
6. **Executes** the query against ClickHouse and returns results.
7. **Self-corrects** on execution failure by feeding the error back to the LLM.

---

## Architecture

```
User Request
    │
    ▼
┌─────────────────┐
│   FastAPI App   │  (app/main.py)
└────────┬────────┘
         │
    ┌────┴──────────────────────────────┐
    ▼                                   ▼
┌──────────────┐              ┌─────────────────────┐
│ SchemaRetriever│            │    SQLGenerator       │
│ (LlamaIndex)  │──context──▶│  (Groq / vLLM LLM)    │
└──────────────┘              └──────────┬──────────┘
                                         │
                                    ┌────▼────┐
                                    │SQLSafety│
                                    └────┬────┘
                                         │
                                   ┌─────▼────────────┐
                                   │ClickHouseExecutor│
                                   └──────────────────┘
```

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **LlamaIndex VectorStoreIndex** over raw schema | Semantic retrieval avoids feeding the entire schema to the LLM, reducing token cost and improving accuracy |
| **Local HuggingFace embeddings** | No external API key needed for embeddings; `BAAI/bge-small-en-v1.5` runs on CPU |
| **Pluggable LLM backend** | Supports both Groq (cloud) and vLLM (self-hosted) via a common `LLMClient` interface |
| **Safety layer before execution** | Prevents destructive operations (DROP, DELETE, etc.) from reaching the database |
| **Self-correction on error** | If execution fails, the error message is fed back to the LLM for a retry |

---

## Directory Structure

```
Text-to-sql-AI-agent-for-clickhouse/
├── README.md                          # Project name (minimal)
├── requirements.txt                   # Python dependencies
├── app/
│   ├── __init__.py
│   ├── main.py                        # FastAPI app; wires all components together
│   ├── config.py                      # Settings class (env-based configuration)
│   ├── cache.py                       # (referenced but not present on disk)
│   ├── llm/
│   │   ├── base.py                    # Abstract LLMClient interface
│   │   ├── groq_client.py             # Groq API implementation
│   │   └── vllm_client.py             # Self-hosted vLLM implementation
│   ├── sql/
│   │   ├── generator.py               # SQLGenerator + prompt templates + clean_sql()
│   │   ├── safety.py                  # SQLSafety — read-only enforcement
│   │   ├── executor.py                # ClickHouseExecutor — runs queries
│   │   └── metrics.py                 # METRIC_MAP — business metric → column mapping
│   └── retrieval/
│       ├── schema_extractor.py        # TableSchema, Column, SchemaExtractor
│       ├── schema_index.py            # SchemaIndexBuilder — builds LlamaIndex
│       ├── retriever.py               # SchemaRetriever — semantic table lookup
│       ├── schema_loader.py           # SchemaLoader — Qdrant-based indexing (alternate path)
│       └── qdrant_store.py            # QdrantStore — Qdrant vector DB wrapper (alternate path)
└── index_cache/                       # Cache directory for schema/index (optional)
    ├── schema_cache.json
    ├── patterns/
    └── schema/
```

---

## End-to-End Data Flow

Here is the complete journey of a user request:

### Step 0 — Startup (runs once when FastAPI starts)

1. `SchemaExtractor` connects to ClickHouse and reads `system.tables` and `system.columns`.
2. It builds `TableSchema` objects for every table, inferring foreign keys from `*_key` column naming conventions.
3. `SchemaIndexBuilder` converts each `TableSchema` into a LlamaIndex `Document` with rich metadata.
4. Documents are split via `SentenceSplitter` (chunk_size=512, overlap=64) and indexed into a `VectorStoreIndex` using `BAAI/bge-small-en-v1.5` embeddings.
5. A `SchemaRetriever` is initialized with `similarity_top_k=6`.

### Step 1 — Request Arrives

A `POST /chat` request arrives with JSON:
```json
{
  "session_id": "user-123",
  "message": "What were total sales last month?"
}
```

### Step 2 — Schema Retrieval

The `SchemaRetriever.retrieve_with_context()` method:
- Embeds the user query using the same HuggingFace model.
- Finds the top-6 most similar table documents via cosine similarity.
- Expands results to include dimension tables that join to retrieved fact tables.
- Returns a tuple: `(list_of_table_names, schema_context_string)`.

If no tables match, the API returns an error immediately.

### Step 3 — SQL Generation

`SQLGenerator.generate()` builds a prompt containing:
- The retrieved schema context (table DDL snippets).
- Star schema notes (list of all fact and dimension tables).
- Strict rules (SELECT only, no markdown, use `*_key` joins).
- The user's natural-language query.

The LLM (Groq or vLLM) returns raw SQL, which is cleaned by `clean_sql()` (strips markdown code fences, backticks, and `SQL:` prefixes).

### Step 4 — Safety Check

`SQLSafety.validate()`:
- Confirms the query starts with `SELECT` or `WITH`.
- Rejects any query containing forbidden keywords: `DROP`, `TRUNCATE`, `DELETE`, `INSERT`, `UPDATE`, `ALTER`, `DETACH`, `ATTACH`, `OPTIMIZE`, `SYSTEM`.
- Detects multi-query injection via semicolons.

### Step 5 — Execution

`ClickHouseExecutor.run()` executes the cleaned SQL against ClickHouse and returns `result_rows`.

### Step 6 — Self-Correction (on failure)

If execution throws an exception, `SQLGenerator.regenerate_with_error()` is called with the original query, schema context, failed SQL, and the error message. The LLM attempts to fix the query. The corrected SQL goes through safety validation again before re-execution.

### Step 7 — Response

```json
{
  "type": "answer",
  "sql": "SELECT sum(line_sales_amount) FROM gold.fact_sales ...",
  "result": [[125000.00]]
}
```

---

## Configuration & Environment Variables

All configuration lives in `app/config.py` as the `Settings` class. Values are read from environment variables with sensible defaults.

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | `""` | API key for Groq. **Required** when using Groq (default mode). |
| `GROQ_MODEL` | `meta-llama/llama-4-scout-17b-16e-instruct` | Groq model identifier. |
| `USE_VLLM` | `"0"` | Set to `"1"` to use a self-hosted vLLM server instead of Groq. |
| `VLLM_URL` | `http://localhost:8000` | Base URL of the vLLM server. |
| `VLLM_MODEL` | `local-model` | Model name to request from vLLM. |
| `CLICKHOUSE_HOST` | `localhost` | ClickHouse server hostname. |
| `CLICKHOUSE_PORT` | `8123` | ClickHouse HTTP port (not native TCP port 9000). |
| `CLICKHOUSE_USER` | `default` | ClickHouse username. |
| `CLICKHOUSE_PASSWORD` | `""` | ClickHouse password. |
| `CLICKHOUSE_DB` | `gold` | Target database name. |

### Usage

```bash
# With Groq (default)
export GROQ_API_KEY="gsk_your-key-here"
export CLICKHOUSE_HOST="clickhouse.example.com"

# With self-hosted vLLM
export USE_VLLM=1
export VLLM_URL="http://localhost:8000"
export VLLM_MODEL="meta-llama/Llama-3-8B-Instruct"
```

---

## Schema Extraction

**File:** `app/retrieval/schema_extractor.py`

### Core Data Structures

#### `Column` (dataclass)

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Column name |
| `dtype` | `str` | ClickHouse data type (e.g. `Nullable(Float64)`) |
| `is_nullable` | `bool` | Auto-detected from `Nullable(...)` type wrapper |

#### `TableSchema` (dataclass)

| Field | Type | Description |
|-------|------|-------------|
| `database` | `str` | Database name (e.g. `gold`) |
| `name` | `str` | Table name |
| `columns` | `List[Column]` | All columns in the table |
| `primary_key` | `List[str]` | Primary key columns (from ClickHouse `system.tables`) |
| `foreign_keys` | `List[Dict]` | Inferred FK relationships |
| `description` | `str` | Optional description |

**Key methods:**
- `full_name` — returns `database.name` (e.g. `gold.fact_sales`).
- `ddl_snippet()` — generates a simplified `CREATE TABLE` DDL snippet.
- `relationships_text()` — returns human-readable join instructions.

### `SchemaExtractor`

Connects to ClickHouse and queries system tables:

| Method | Query | Purpose |
|--------|-------|---------|
| `get_tables()` | `system.tables` | Lists all tables in the database |
| `get_columns(table)` | `system.columns` | Gets column names, types, and PK status |
| `get_primary_key(table)` | `system.tables` | Extracts the primary key definition |
| `infer_foreign_keys(table, all_tables, all_cols)` | — | Heuristic FK inference |
| `extract_all()` | — | Orchestrates full schema extraction |

#### Foreign Key Inference Logic

ClickHouse does not enforce foreign keys natively, so the system infers relationships from the star-schema naming convention:

1. Any column ending in `_key` is a candidate FK column.
2. The prefix before `_key` (e.g. `customer` from `customer_key`) is matched against dimension table names containing `dim_`.
3. Example: `fact_sales.customer_key` → `dim_customer.customer_key`.

---

## Vector Index & Retrieval

**Files:** `app/retrieval/schema_index.py`, `app/retrieval/retriever.py`

### Embedding Model

```python
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
```

- **Model:** `BAAI/bge-small-en-v1.5` (~133 MB)
- **Dimensions:** 384
- **Distance:** Cosine similarity (default for LlamaIndex)
- **Hardware:** Runs on CPU; no GPU required

### `SchemaIndexBuilder`

Converts `TableSchema` objects into LlamaIndex documents and builds a `VectorStoreIndex`.

#### Document Construction (`_build_table_document`)

Each table becomes a rich text document:

```
Table: gold.fact_sales
Columns:
  - date_key (Nullable(UInt32))
  - product_key (Nullable(UInt32))
  - customer_key (Nullable(UInt32))
  - line_sales_amount (Nullable(Float64))
  ...

Primary/Order key: date_key, product_key, customer_key

Relationships (joins):
  - gold.fact_sales.customer_key references gold.dim_customer.customer_key
  - gold.fact_sales.product_key references gold.dim_product.product_key
```

**Metadata attached to each document:**

| Key | Example | Purpose |
|-----|---------|---------|
| `table_name` | `gold.fact_sales` | Full qualified name |
| `database` | `gold` | Database filter |
| `table_short` | `fact_sales` | Short name |
| `column_names` | `[date_key, product_key, ...]` | Column list |
| `has_fk` | `true` | Whether FK relationships exist |
| `is_fact` | `true` | Whether table starts with `fact_` |
| `is_dim` | `false` | Whether table starts with `dim_` |

**Embedding exclusion:** `database`, `table_short`, and `column_names` are excluded from embeddings (too structured for semantic matching) but remain available for filtering.

### `SchemaRetriever`

Wraps LlamaIndex's retrieval API:

| Method | Description |
|--------|-------------|
| `retrieve_tables(query)` | Returns list of relevant table names |
| `retrieve_with_context(query)` | Returns `(table_list, schema_text)` tuple |
| `_expand_with_dimensions(fact_tables)` | Adds dimension tables that join to retrieved fact tables |

The dimension expansion ensures that if a query retrieves `fact_sales`, the related `dim_customer`, `dim_product`, etc. are also included in the context.

### Alternate Vector Store (Qdrant)

**Files:** `app/retrieval/qdrant_store.py`, `app/retrieval/schema_loader.py`

These files provide an alternate indexing path using **Qdrant** as the vector store and **OpenAI embeddings**. They are **not used by the current main pipeline** but are available for integration:

- `QdrantStore` — manages a Qdrant collection with cosine similarity.
- `SchemaLoader` — embeds schema text and upserts into Qdrant.

To use this path, you would need to modify `main.py` to instantiate `QdrantStore` and `SchemaLoader` instead of the in-memory LlamaIndex.

---

## LLM Abstraction

**Files:** `app/llm/base.py`, `app/llm/groq_client.py`, `app/llm/vllm_client.py`

### `LLMClient` (Abstract Base Class)

```python
class LLMClient(ABC):
    @abstractmethod
    def generate_text(self, messages: List[str]) -> str:
        pass
```

A single-method interface that all LLM implementations must implement. Takes a list of message strings and returns the generated text.

### `GroqClient`

Calls the Groq API (OpenAI-compatible endpoint):

- **Endpoint:** `https://api.groq.com/openai/v1/chat/completions`
- **Temperature:** 0 (deterministic output)
- **Timeout:** 60 seconds
- **Auth:** Bearer token via `GROQ_API_KEY`

### `VLLMClient`

Calls a self-hosted vLLM server (also OpenAI-compatible):

- **Endpoint:** `{VLLM_URL}/v1/chat/completions`
- **Temperature:** 0
- **Timeout:** 60 seconds
- **Auth:** None (assumes local network)

Both implementations squash the message list into a single `user` role message joined by `\n\n`, since the current prompt design uses a single system prompt.

---

## SQL Generation

**File:** `app/sql/generator.py`

### `clean_sql(text)`

Post-processing function that sanitizes LLM output:

1. Extracts content from markdown code fences (` ```sql ... ``` `).
2. Removes all backticks.
3. Strips `SQL:` prefix if present.
4. Trims whitespace and removes trailing semicolon.

### `SQLGenerator`

| Method | Description |
|--------|-------------|
| `generate(query, schema_context)` | Initial SQL generation |
| `regenerate_with_error(query, schema_context, failed_sql, error)` | Self-correction on execution failure |
| `regenerate_with_unknown_columns(query, schema_context, failed_sql, unknown_cols)` | Self-correction when column names don't exist |

Both regeneration methods include the original failed SQL and a description of what went wrong, giving the LLM concrete debugging information.

---

## SQL Safety Validation

**File:** `app/sql/safety.py`

### `SQLSafety`

Enforces a **read-only** policy through multiple checks:

| Check | Mechanism | Example Rejected |
|-------|-----------|------------------|
| Empty query | String emptiness after cleaning | `""` |
| First keyword | Regex finds first SQL keyword; must be `SELECT` or `WITH` | `DROP TABLE ...` |
| Forbidden keywords | Word-boundary regex scan | `SELECT ...; DROP TABLE ...` |
| Multi-query injection | Semicolon detection in stripped SQL | `SELECT ...; DELETE FROM ...` |

**Forbidden keywords:** `drop`, `truncate`, `delete`, `insert`, `update`, `alter`, `detach`, `attach`, `optimize`, `system`.

Note: The safety check runs `clean_sql()` internally, ensuring it validates the same SQL that will be executed.

---

## Query Execution

**File:** `app/sql/executor.py`

### `ClickHouseExecutor`

Uses `clickhouse-connect` (the official ClickHouse Python client) to connect and execute:

```python
self.client = get_client(
    host=host, port=port, username=username,
    password=password, database=database,
)
```

The `run(sql)` method:
1. Cleans the SQL (same pipeline as safety).
2. Executes via `self.client.query(sql)`.
3. Returns `result_rows` — a list of tuples, one per row.

---

## Metrics Map

**File:** `app/sql/metrics.py`

```python
METRIC_MAP = {
    "sales": "line_sales_amount",
    "total_sales": "line_sales_amount",
    "revenue": "line_sales_amount",
    "turnover": "line_sales_amount",
    "net_sales": "line_net_amount",
    "profit": "gross_margin",
    "margin": "gross_margin",
    "quantity": "qty_sold",
    "units": "qty_sold",
}
```

This map defines business-level metric terms → actual column names. It is **defined but not currently imported** in the active code path. It exists as a reference for future enhancement — e.g., a pre-processing step that translates metric keywords before schema retrieval or SQL generation.

---

## API Routes

**File:** `app/main.py`

### `POST /chat`

Main endpoint. Accepts JSON:

```json
{
  "session_id": "user-123",
  "message": "What were total sales last month?"
}
```

Returns:

**Success:**
```json
{
  "type": "answer",
  "sql": "SELECT sum(line_sales_amount) FROM gold.fact_sales ...",
  "result": [[125000.00]]
}
```

**Error:**
```json
{
  "type": "error",
  "message": "No relevant tables found for the query.",
  "sql": ""
}
```

### `POST /chat/quick`

Alias for `/chat` without session management. Identical behavior.

### `GET /schema/tables`

Lists all known table names from the built index:

```json
{
  "tables": ["gold.fact_sales", "gold.dim_product", ...]
}
```

### `GET /schema/table/{table_name}`

Returns DDL for a specific table:

```json
{
  "ddl": "CREATE TABLE gold.fact_sales (\n  date_key Nullable(UInt32) NULL,\n  ..."
}
```

Returns error if table not found.

---

## Prompt Templates

### Initial Generation Prompt

```
You are an expert ClickHouse SQL generator.

DATABASE SCHEMA:
{schema_context}

STAR SCHEMA NOTES:
- Fact tables: gold.fact_sales, gold.fact_sales_header, gold.fact_invoice_lines, ...
- Dimension tables: gold.dim_date, gold.dim_product, gold.dim_customer, ...
- Join fact tables to dimension tables using *_key columns.
- Use gold.dim_date for time-based filtering via date_key joins.
- Always use the exact column names shown in the schema. Never invent column names.

USER QUERY:
{user_query}

STRICT RULES:
1. Output ONLY the SQL query. No explanation, no markdown.
2. Start with SELECT or WITH.
3. Use correct JOINs between fact and dimension tables via *_key columns.
4. If you cannot fulfill the request, output exactly: ERROR: CANNOT_BUILD_QUERY
```

### Regeneration Prompt (on error)

```
You are an expert ClickHouse SQL generator.

The following SQL query failed:

FAILED SQL:
{failed_sql}

ERROR:
{error}

DATABASE SCHEMA:
{schema_context}

USER QUERY:
{user_query}

STRICT RULES:
1. Output ONLY the corrected SQL query. No explanation, no markdown.
2. Start with SELECT or WITH.
3. Fix the error described above.
4. Use correct JOINs between fact and dimension tables via *_key columns.
```

### Regeneration Prompt (unknown columns)

```
You are an expert ClickHouse SQL generator.

The following SQL used columns that do not exist in the schema:
{unknown_cols}

FAILED SQL:
{failed_sql}

DATABASE SCHEMA:
{schema_context}

USER QUERY:
{user_query}

STRICT RULES:
1. Output ONLY the corrected SQL query. No explanation, no markdown.
2. Start with SELECT or WITH.
3. Use ONLY the column names shown in the schema above.
```

---

## Error Handling & Self-Correction

The system employs a **defense-in-depth** approach to error handling:

### Layer 1 — Retrieval Failure

If `_retriever.retrieve_with_context()` returns no tables:
```json
{"type": "error", "message": "No relevant tables found for the query.", "sql": ""}
```

### Layer 2 — LLM Generation Failure

If the LLM returns `ERROR: CANNOT_BUILD_QUERY` or empty output:
```json
{"type": "error", "message": "LLM could not generate a valid query. Response: ...", "sql": ""}
```

### Layer 3 — Safety Violation

If `SQLSafety.validate()` detects forbidden operations:
```json
{"type": "error", "message": "Forbidden keyword detected: drop", "sql": "drop table ..."}
```

### Layer 4 — Execution Failure + Self-Correction

If `executor.run(sql)` throws:

1. **First attempt:** The error message is captured.
2. **Retry:** `regenerate_with_error()` is called with the failed SQL and error.
3. **Re-validation:** The corrected SQL goes through `SQLSafety` again.
4. **Re-execution:** If it succeeds, the answer is returned.
5. **Final failure:** If retry also fails, the error from the second attempt is returned.

```json
{"type": "error", "message": "<second error>", "sql": "<original sql>"}
```

### Layer 5 — Schema Detail Error

If a requested table doesn't exist in the index:
```json
{"error": "Table {table_name} not found"}
```

---

## Logging Strategy

The project currently has **no explicit logging configuration**. Errors are returned as HTTP responses but are not logged to files or external monitoring systems.

### Recommended Enhancements

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)
```

Key places to add logging:
- Schema extraction at startup (table count, FK count).
- Retrieval results (tables matched, similarity scores).
- Generated SQL (for debugging).
- Safety violations (audit trail).
- Execution errors and self-correction attempts.
- Request latency per endpoint.

---

## Running Locally

### Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.11+ | Tested on 3.13 |
| ClickHouse | Any version | Must have the `gold` database populated |
| Groq API key | — | Or a vLLM server |

### Step-by-Step

```bash
# 1. Clone the repository
git clone <repo-url>
cd Text-to-sql-AI-agent-for-clickhouse

# 2. Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set environment variables
export GROQ_API_KEY="gsk_your-key"
export CLICKHOUSE_HOST="localhost"
export CLICKHOUSE_PORT="8123"
export CLICKHOUSE_USER="default"
export CLICKHOUSE_PASSWORD=""
export CLICKHOUSE_DB="gold"

# 5. Start the FastAPI server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# 6. Test the endpoint
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "test", "message": "Show all products"}'
```

### Using vLLM Instead of Groq

```bash
# Start vLLM server (separate process)
vllm serve meta-llama/Llama-3-8B-Instruct --host 0.0.0.0 --port 8000

# In another terminal
export USE_VLLM=1
export VLLM_URL="http://localhost:8000"
export VLLM_MODEL="meta-llama/Llama-3-8B-Instruct"

uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Quick Test Commands

```bash
# List available tables
curl http://localhost:8000/schema/tables

# Get DDL for a table
curl http://localhost:8000/schema/table/gold.fact_sales

# Send a query
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "1", "message": "What is the total revenue by product last quarter?"}'
```

---

## Deployment

### Docker (Recommended)

No `Dockerfile` is provided in the repository, but here is a production-ready configuration:

**Dockerfile:**

```dockerfile
FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

**docker-compose.yml:**

```yaml
version: "3.9"
services:
  text-to-sql:
    build: .
    ports:
      - "8000:8000"
    environment:
      GROQ_API_KEY: ${GROQ_API_KEY}
      CLICKHOUSE_HOST: clickhouse
      CLICKHOUSE_PORT: "8123"
      CLICKHOUSE_USER: ${CLICKHOUSE_USER}
      CLICKHOUSE_PASSWORD: ${CLICKHOUSE_PASSWORD}
      CLICKHOUSE_DB: gold
    depends_on:
      - clickhouse

  clickhouse:
    image: clickhouse/clickhouse-server:latest
    ports:
      - "8123:8123"
      - "9000:9000"
    volumes:
      - clickhouse_data:/var/lib/clickhouse

volumes:
  clickhouse_data:
```

### Cloud Deployment

| Platform | Notes |
|----------|-------|
| **AWS ECS/Fargate** | Containerize; set env vars in task definition; place behind ALB |
| **Google Cloud Run** | Deploy container; set env vars; ensure VPC access to ClickHouse |
| **Azure Container Apps** | Similar to Cloud Run; set secrets via Key Vault |
| **Kubernetes** | Deploy as a Deployment + Service; use ConfigMap for non-secret env vars, Secret for API keys |

### Scaling Considerations

- The LlamaIndex vector store is built **in-memory** at startup. For multi-worker deployments, each worker will independently build the index on startup. This is acceptable for small schema but could be optimized by caching the index to disk (e.g., using `index.storage_context`).
- The HuggingFace embedding model is downloaded on first use. In containerized environments, pre-warm the model during image build.
- Consider adding a request queue or rate limiter if the LLM API has throughput limits.

---

## Testing

The repository does not include a test suite. Here is a recommended testing strategy:

### Unit Tests (pytest)

```bash
pip install pytest httpx
```

**Test `SQLSafety`:**

```python
# tests/test_safety.py
from app.sql.safety import SQLSafety

def test_allows_select():
    SQLSafety().validate("SELECT * FROM gold.fact_sales")

def test_rejects_drop():
    import pytest
    with pytest.raises(ValueError, match="Forbidden"):
        SQLSafety().validate("DROP TABLE gold.fact_sales")

def test_rejects_insert():
    with pytest.raises(ValueError, match="Forbidden"):
        SQLSafety().validate("INSERT INTO gold.fact_sales VALUES (1)")

def test_rejects_multi_query():
    with pytest.raises(ValueError, match="Multiple queries"):
        SQLSafety().validate("SELECT 1; DROP TABLE x")
```

**Test `clean_sql`:**

```python
# tests/test_clean_sql.py
from app.sql.generator import clean_sql

def test_strips_markdown():
    raw = '```sql\nSELECT * FROM t\n```'
    assert clean_sql(raw) == "SELECT * FROM t"

def test_strips_sql_prefix():
    assert clean_sql("SQL: SELECT 1") == "SELECT 1"

def test_removes_backticks():
    assert clean_sql("SELECT `id` FROM `t`") == "SELECT id FROM t"
```

**Test `SchemaRetriever` (mocked):**

```python
# tests/test_retriever.py
from unittest.mock import MagicMock
from app.retrieval.retriever import SchemaRetriever

def test_retrieve_tables():
    mock_index = MagicMock()
    mock_retriever = MagicMock()
    mock_index.as_retriever.return_value = mock_retriever

    mock_node = MagicMock()
    mock_node.metadata = {"table_name": "gold.fact_sales"}
    mock_retriever.retrieve.return_value = [mock_node]

    retriever = SchemaRetriever(mock_index)
    tables = retriever.retrieve_tables("sales")
    assert "gold.fact_sales" in tables
```

### Integration Tests

```bash
# With a live ClickHouse instance
pytest tests/integration/ --clickhouse-host=localhost --clickhouse-port=8123
```

### Running Tests

```bash
python -m pytest tests/ -v
```

---

## Extending the System

### 1. Add a New LLM Provider

Create `app/llm/your_provider.py`:

```python
from app.llm.base import LLMClient

class YourClient(LLMClient):
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model

    def generate_text(self, messages: list[str]) -> str:
        # Call your LLM API
        return response_text
```

Update `main.py`'s `get_llm()` function to instantiate it.

### 2. Switch to Qdrant Vector Store

Currently, `qdrant_store.py` and `schema_loader.py` exist but are unused. To activate:

```python
from app.retrieval.qdrant_store import QdrantStore
from app.retrieval.schema_loader import SchemaLoader
from llama_index.embeddings.openai import OpenAIEmbedding

qdrant = QdrantStore(host="localhost", port=6333)
qdrant.ensure_collection(vector_size=1536)  # OpenAI ada-002
loader = SchemaLoader(qdrant, OpenAIEmbedding())
loader.index_schema(schema_texts)
```

### 3. Add Prompt Caching

Cache generated SQL per (query, schema_context) hash to avoid redundant LLM calls:

```python
import hashlib
from app.cache import get_cache, set_cache

query_hash = hashlib.sha256(f"{user_query}:{schema_context}".encode()).hexdigest()
cached = get_cache(query_hash)
if cached:
    sql = cached
else:
    sql = sql_gen.generate(user_query, schema_context)
    set_cache(query_hash, sql)
```

### 4. Add Conversation History

The current system is stateless — `session_id` is received but not used. To add history:

```python
class ConversationStore:
    def __init__(self):
        self._store: Dict[str, List[dict]] = {}

    def add(self, session_id: str, role: str, content: str):
        self._store.setdefault(session_id, []).append({"role": role, "content": content})

    def get(self, session_id: str) -> List[dict]:
        return self._store.get(session_id, [])
```

Include history in the LLM prompt for multi-turn conversations.

### 5. Use the Metrics Map

Integrate `METRIC_MAP` into the retrieval or generation pipeline to improve metric accuracy:

```python
from app.sql.metrics import METRIC_MAP

def enrich_query_with_metrics(user_query: str) -> str:
    for metric, column in METRIC_MAP.items():
        if metric.lower() in user_query.lower():
            # Hint to the retriever that this column is relevant
            pass
    return user_query
```

### 6. Add Response Formatting

Instead of returning raw rows, format results as natural language:

```python
def format_result(sql: str, result: list) -> str:
    if not result:
        return "No data found."
    # Use an LLM to generate a natural language summary
    return summary
```

### 7. Add Structured Output

Return typed results instead of raw `result_rows`:

```python
class ChatResponse(BaseModel):
    type: str  # "answer" | "error"
    sql: str
    result: list[list]
    columns: list[str]  # Add column names for tabular rendering
    row_count: int
```

---

## Star Schema Reference

The system is designed around a star schema in the `gold` ClickHouse database:

### Fact Tables

| Table | Description |
|-------|-------------|
| `gold.fact_sales` | Individual sales line items |
| `gold.fact_sales_header` | Sales header/transaction info |
| `gold.fact_invoice_lines` | Invoice line details |
| `gold.fact_consumption` | Consumption records |
| `gold.fact_financial_exchange` | Currency exchange rates |
| `gold.fact_goods_receipts` | Goods received notes |
| `gold.fact_inventory_count` | Inventory counts |
| `gold.fact_inventory_snapshot` | Inventory snapshots |
| `gold.fact_payment_schedule` | Payment schedule entries |
| `gold.fact_product_history` | Product history data |
| `gold.fact_purchase_order_lines` | Purchase order line items |
| `gold.fact_service_lines` | Service line items |
| `gold.fact_service_request` | Service requests |
| `gold.fact_stock_movement` | Stock movements |

### Dimension Tables

| Table | Description |
|-------|-------------|
| `gold.dim_date` | Date dimension (for time-based joins) |
| `gold.dim_product` | Product information |
| `gold.dim_customer` | Customer information |
| `gold.dim_currency` | Currency codes and rates |
| `gold.dim_depot` | Depot/warehouse information |
| `gold.dim_employee` | Employee information |
| `gold.dim_invoice` | Invoice information |
| `gold.dim_user` | User information |
| `gold.dim_provider` | Provider/vendor information |
| `gold.dim_transporter` | Transporter information |
| `gold.dim_package` | Package information |
| `gold.dim_article` | Article information |
| `gold.dim_manager` | Manager information |
| `dim_responsable` | Responsable information |
| `dim_service` | Service information |
| `dim_purchase_order` | Purchase order information |
| `gold.dim_delivery_note` | Delivery note information |

---

## Limitations & Known Gaps

1. **No persistent index caching:** The vector index is rebuilt from ClickHouse on every server restart. The `index_cache/` directory exists but is not wired into the pipeline.
2. **No session management:** `session_id` is accepted but never used — each request is stateless.
3. **No explicit logging:** Errors are returned in HTTP responses but not persisted for debugging.
4. **No test suite:** The project has no `tests/` directory or pytest configuration.
5. **No Docker setup:** No `Dockerfile` or `docker-compose.yml` is provided.
6. **`cache.py` is missing:** Referenced in the file listing but not present on disk.
7. **`metrics.py` is unused:** The `METRIC_MAP` is defined but never imported.
8. **Qdrant path is dead code:** `qdrant_store.py` and `schema_loader.py` are not used in the active pipeline.
9. **Single-query LLM format:** Messages are squashed into a single user message, losing the system/user distinction.
10. **No rate limiting:** The API has no built-in request throttling.
