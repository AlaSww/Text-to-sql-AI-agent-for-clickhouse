"""FastAPI Text-to-SQL agent powered by LlamaIndex for schema-aware retrieval."""
from __future__ import annotations
from fastapi import FastAPI
from pydantic import BaseModel
from app.config import Settings
from app.llm.groq_client import GroqClient
from app.llm.vllm_client import VLLMClient
from app.sql.generator import SQLGenerator
from app.sql.safety import SQLSafety
from app.sql.executor import ClickHouseExecutor
from app.retrieval.schema_extractor import SchemaExtractor
from app.retrieval.schema_index import SchemaIndexBuilder
from app.retrieval.retriever import SchemaRetriever

app = FastAPI()

# --- One-time setup: extract schema and build LlamaIndex ---
_schema_extractor = SchemaExtractor(
    host=Settings.CLICKHOUSE_HOST,
    port=Settings.CLICKHOUSE_PORT,
    username=Settings.CLICKHOUSE_USER,
    password=Settings.CLICKHOUSE_PASSWORD,
    database=Settings.CLICKHOUSE_DB,
)

_schemas = _schema_extractor.extract_all()
_index_builder = SchemaIndexBuilder()
_index_builder.load_schemas(_schemas)
_index_builder.build_index()

_retriever = SchemaRetriever(_index_builder.index, similarity_top_k=6)


def get_llm():
    if Settings.USE_VLLM:
        return VLLMClient(Settings.VLLM_URL, Settings.VLLM_MODEL)
    return GroqClient(Settings.GROQ_API_KEY, Settings.GROQ_MODEL)


class ChatRequest(BaseModel):
    session_id: str
    message: str


@app.post("/chat")
def chat(req: ChatRequest):
    llm = get_llm()

    sql_gen = SQLGenerator(llm)
    safety = SQLSafety()
    executor = ClickHouseExecutor(
        Settings.CLICKHOUSE_HOST,
        Settings.CLICKHOUSE_PORT,
        Settings.CLICKHOUSE_USER,
        Settings.CLICKHOUSE_PASSWORD,
        Settings.CLICKHOUSE_DB,
    )

    user_query = req.message.strip()

    # 1. Retrieve relevant schema from LlamaIndex
    relevant_tables, schema_context = _retriever.retrieve_with_context(user_query)

    if not relevant_tables:
        return {"type": "error", "message": "No relevant tables found for the query.", "sql": ""}

    # 2. Generate SQL
    try:
        sql = sql_gen.generate(user_query, schema_context)
    except ValueError as e:
        return {"type": "error", "message": str(e), "sql": ""}

    # 3. Safety check
    try:
        safety.validate(sql)
    except ValueError as e:
        return {"type": "error", "message": str(e), "sql": sql}

    # 4. Execute with retry
    try:
        result = executor.run(sql)
        return {"type": "answer", "sql": sql, "result": result}
    except Exception as e:
        # Retry once with error context
        try:
            corrected_sql = sql_gen.regenerate_with_error(
                user_query, schema_context, sql, str(e)
            )
            safety.validate(corrected_sql)
            result = executor.run(corrected_sql)
            return {"type": "answer", "sql": corrected_sql, "result": result}
        except Exception as e2:
            return {"type": "error", "message": str(e2), "sql": sql}


@app.post("/chat/quick")
def chat_quick(req: ChatRequest):
    """Simplified endpoint without session management."""
    return chat(req)


@app.get("/schema/tables")
def list_tables():
    """Return all known table names from the schema."""
    return {"tables": list(_index_builder.table_map.keys())}


@app.get("/schema/table/{table_name}")
def table_detail(table_name: str):
    """Return DDL for a specific table."""
    if table_name not in _index_builder.table_map:
        return {"error": f"Table {table_name} not found"}
    return {"ddl": _index_builder.table_map[table_name].ddl_snippet()}
