"""FastAPI Text-to-SQL agent powered by LlamaIndex for schema-aware retrieval."""
from __future__ import annotations
import logging
import time
import os
import json
from datetime import datetime
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from app.config import Settings
from app.llm.groq_client import GroqClient
from app.llm.vllm_client import VLLMClient
from app.sql.generator import SQLGenerator, classify_visualization
from app.sql.safety import SQLSafety
from app.sql.executor import ClickHouseExecutor
from app.retrieval.schema_extractor import SchemaExtractor
from app.retrieval.schema_index import SchemaIndexBuilder
from app.retrieval.retriever import SchemaRetriever
from app.cache import get_cache, set_cache, clear_cache
from app.session import get_conversation_store

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
MAX_SQL_GENERATION_RETRIES = 3  # Maximum retry attempts for SQL generation on error


def log_request(
    session_id: str,
    user_query: str,
    retrieved_tables: list,
    sql: str,
    execution_time_ms: float,
    row_count: int,
    viz_hint: str = "",
    error: str = ""
) -> None:
    """Log request details in structured format.
    
    Args:
        session_id: Session identifier.
        user_query: Original user query.
        retrieved_tables: Tables retrieved by the retriever.
        sql: Generated SQL.
        execution_time_ms: Execution time in milliseconds.
        row_count: Number of result rows.
        viz_hint: Visualization hint.
        error: Error message if any.
    """
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "session_id": session_id,
        "user_query": user_query,
        "retrieved_tables": retrieved_tables,
        "sql": sql,
        "execution_time_ms": execution_time_ms,
        "row_count": row_count,
        "viz_hint": viz_hint,
        "error": error,
    }
    logger.info(json.dumps(log_entry))


app = FastAPI()

# --- One-time setup: extract schema and build LlamaIndex ---
_SCHEMA_EXTRACTOR = SchemaExtractor(
    host=Settings.CLICKHOUSE_HOST,
    port=Settings.CLICKHOUSE_PORT,
    username=Settings.CLICKHOUSE_USER,
    password=Settings.CLICKHOUSE_PASSWORD,
    database=Settings.CLICKHOUSE_DB,
)

_SCHEMAS = _SCHEMA_EXTRACTOR.extract_all()
_INDEX_BUILDER = SchemaIndexBuilder()
_INDEX_BUILDER.load_schemas(_SCHEMAS)

# Try to load from cache; if not, build new index
if not _INDEX_BUILDER.load_index_from_cache():
    logger.info("Index not found in cache. Building fresh index...")
    _INDEX_BUILDER.build_index()
    _INDEX_BUILDER.save_index_to_cache()
    logger.info("Index built and cached successfully.")
else:
    logger.info("Index loaded from cache.")

_RETRIEVER = SchemaRetriever(_INDEX_BUILDER.index, similarity_top_k=6)
_CONVERSATION_STORE = get_conversation_store()


def get_llm():
    """Instantiate the appropriate LLM client based on settings."""
    if Settings.USE_VLLM:
        return VLLMClient(Settings.VLLM_URL, Settings.VLLM_MODEL)
    return GroqClient(Settings.GROQ_API_KEY, Settings.GROQ_MODEL)


class ChatRequest(BaseModel):
    """Request model for /chat endpoint."""
    session_id: str
    message: str


class ChatResponse(BaseModel):
    """Response model for /chat endpoint with structured data."""
    type: str  # "answer" or "error"
    sql: str
    columns: list[str]
    result: list[list]
    row_count: int
    viz_hint: str
    error_message: str = ""


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """Main chat endpoint: process natural language query and return SQL results.
    
    Args:
        req: ChatRequest with session_id and message.
    
    Returns:
        ChatResponse with SQL, results, visualization hint, and metadata.
    """
    start_time = time.time()
    session_id = req.session_id
    user_query = req.message.strip()
    retrieved_tables = []
    sql = ""
    columns = []
    rows = []
    viz_hint = ""
    error_message = ""

    try:
        llm = get_llm()
        sql_gen = SQLGenerator(llm, cache=None)  # Can pass cache object here if needed
        safety = SQLSafety()
        executor = ClickHouseExecutor(
            Settings.CLICKHOUSE_HOST,
            Settings.CLICKHOUSE_PORT,
            Settings.CLICKHOUSE_USER,
            Settings.CLICKHOUSE_PASSWORD,
            Settings.CLICKHOUSE_DB,
        )

        # 1. Retrieve relevant schema from LlamaIndex
        relevant_tables, schema_context = _RETRIEVER.retrieve_with_context(user_query)
        retrieved_tables = list(relevant_tables)

        if not retrieved_tables:
            error_message = "No relevant tables found for the query."
            execution_time = (time.time() - start_time) * 1000
            log_request(session_id, user_query, retrieved_tables, "", execution_time, 0, "", error_message)
            return ChatResponse(
                type="error",
                sql="",
                columns=[],
                result=[],
                row_count=0,
                viz_hint="",
                error_message=error_message
            )

        # 2. Get last 6 messages from conversation history
        history = _CONVERSATION_STORE.get_last_n_messages(session_id, n=6)
        
        # Add history context to SQL generator if needed
        # (Currently not integrated into prompt, but available for future use)

        # 3. Generate SQL
        try:
            sql = sql_gen.generate(user_query, schema_context)
        except ValueError as e:
            error_message = str(e)
            execution_time = (time.time() - start_time) * 1000
            log_request(session_id, user_query, retrieved_tables, "", execution_time, 0, "", error_message)
            return ChatResponse(
                type="error",
                sql="",
                columns=[],
                result=[],
                row_count=0,
                viz_hint="",
                error_message=error_message
            )

        # 4. Safety check
        try:
            safety.validate(sql)
        except ValueError as e:
            error_message = str(e)
            execution_time = (time.time() - start_time) * 1000
            log_request(session_id, user_query, retrieved_tables, sql, execution_time, 0, "", error_message)
            return ChatResponse(
                type="error",
                sql=sql,
                columns=[],
                result=[],
                row_count=0,
                viz_hint="",
                error_message=error_message
            )

        # 5. Execute with retry loop
        last_error = None
        
        for attempt in range(1, MAX_SQL_GENERATION_RETRIES + 1):
            try:
                columns, rows = executor.run(sql)
                viz_hint = classify_visualization(sql, columns, rows)
                
                # Store in conversation history
                _CONVERSATION_STORE.add_message(session_id, "user", user_query)
                if attempt == 1:
                    result_summary = f"SQL: {sql}. Returned {len(rows)} rows with columns: {', '.join(columns)}"
                else:
                    result_summary = f"SQL (corrected, attempt {attempt}): {sql}. Returned {len(rows)} rows with columns: {', '.join(columns)}"
                _CONVERSATION_STORE.add_message(session_id, "assistant", result_summary)
                
                execution_time = (time.time() - start_time) * 1000
                log_request(session_id, user_query, retrieved_tables, sql, execution_time, len(rows), viz_hint)
                
                return ChatResponse(
                    type="answer",
                    sql=sql,
                    columns=columns,
                    result=rows,
                    row_count=len(rows),
                    viz_hint=viz_hint,
                    error_message=""
                )
            except Exception as e:
                last_error = str(e)
                
                # If this is the last attempt, give up
                if attempt >= MAX_SQL_GENERATION_RETRIES:
                    break
                
                # Retry with error context
                try:
                    logger.info(f"SQL generation attempt {attempt} failed: {last_error}. Retrying with error context...")
                    sql = sql_gen.regenerate_with_error(
                        user_query,
                        schema_context,
                        sql,
                        last_error
                    )
                    safety.validate(sql)
                    logger.info(f"Regenerated SQL for attempt {attempt + 1}: {sql[:100]}...")
                except Exception as regen_error:
                    last_error = str(regen_error)
                    logger.error(f"Failed to regenerate SQL: {last_error}")
                    break
        
        # All retries exhausted
        error_message = f"Failed to generate valid SQL after {MAX_SQL_GENERATION_RETRIES} attempts. Last error: {last_error}"
        execution_time = (time.time() - start_time) * 1000
        log_request(session_id, user_query, retrieved_tables, sql, execution_time, 0, "", error_message)
        return ChatResponse(
            type="error",
            sql=sql,
            columns=[],
            result=[],
            row_count=0,
            viz_hint="",
            error_message=error_message
        )

    except Exception as e:
        error_message = f"Unexpected error: {str(e)}"
        execution_time = (time.time() - start_time) * 1000
        log_request(session_id, user_query, retrieved_tables, sql, execution_time, 0, "", error_message)
        return ChatResponse(
            type="error",
            sql=sql,
            columns=[],
            result=[],
            row_count=0,
            viz_hint="",
            error_message=error_message
        )


@app.post("/chat/quick", response_model=ChatResponse)
def chat_quick(req: ChatRequest):
    """Simplified endpoint without session management. Alias for /chat."""
    return chat(req)


@app.post("/admin/refresh-index")
def refresh_index(x_admin_key: str = Header(None)):
    """Refresh the schema index by deleting cache and rebuilding.
    
    Protected by X-Admin-Key header check against ADMIN_KEY environment variable.
    
    Args:
        x_admin_key: Admin key from header.
    
    Returns:
        Success message or error.
    
    Raises:
        HTTPException: If admin key is invalid.
    """
    admin_key = os.getenv("ADMIN_KEY", "")
    if not admin_key or x_admin_key != admin_key:
        raise HTTPException(status_code=403, detail="Invalid admin key")
    
    try:
        logger.info("Clearing index cache and rebuilding...")
        _INDEX_BUILDER.clear_cache()
        _INDEX_BUILDER.build_index()
        _INDEX_BUILDER.save_index_to_cache()
        logger.info("Index rebuild complete.")
        return {"message": "Index refreshed successfully"}
    except Exception as e:
        logger.error(f"Error refreshing index: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error refreshing index: {str(e)}")


@app.get("/schema/tables")
def list_tables():
    """Return all known table names from the schema."""
    return {"tables": list(_INDEX_BUILDER.table_map.keys())}


@app.get("/schema/table/{table_name}")
def table_detail(table_name: str):
    """Return DDL for a specific table."""
    if table_name not in _INDEX_BUILDER.table_map:
        return {"error": f"Table {table_name} not found"}
    return {"ddl": _INDEX_BUILDER.table_map[table_name].ddl_snippet()}

