"""Generate ClickHouse SQL from user query and retrieved schema context."""
from __future__ import annotations
import re
from app.llm.base import LLMClient


def clean_sql(text: str) -> str:
    """Extract raw SQL from LLM output, stripping markdown and prefixes."""
    # Extract content from markdown code blocks
    match = re.search(r"```(?:sql)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        text = match.group(1)

    # Remove any remaining backticks
    text = text.replace("`", "")

    # Remove leading "SQL:" prefix the LLM may echo
    text = re.sub(r"^\s*sql\s*:", "", text, flags=re.IGNORECASE)

    return text.strip().rstrip(";")


class SQLGenerator:
    """Generate ClickHouse SQL using an LLM and retrieved schema."""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def generate(self, user_query: str, schema_context: str) -> str:
        prompt = f"""You are an expert ClickHouse SQL generator.

DATABASE SCHEMA:
{schema_context}

STAR SCHEMA NOTES:
- Fact tables: gold.fact_sales, gold.fact_sales_header, gold.fact_invoice_lines,
  gold.fact_consumption, gold.fact_financial_exchange, gold.fact_goods_receipts,
  gold.fact_inventory_count, gold.fact_inventory_snapshot, gold.fact_payment_schedule,
  gold.fact_product_history, gold.fact_purchase_order_lines, gold.fact_service_lines,
  gold.fact_service_request, gold.fact_stock_movement
- Dimension tables: gold.dim_date, gold.dim_product, gold.dim_customer,
  gold.dim_currency, gold.dim_depot, gold.dim_employee, gold.dim_invoice,
  gold.dim_user, gold.dim_provider, gold.dim_transporter, gold.dim_package,
  gold.dim_article, gold.dim_manager, dim_responsable, dim_service, dim_purchase_order,
  gold.dim_delivery_note
- Join fact tables to dimension tables using *_key columns (e.g. date_key, product_key, customer_key).
- Use gold.dim_date for time-based filtering via date_key joins.
- Always use the exact column names shown in the schema above. Never invent column names.

USER QUERY:
{user_query}

STRICT RULES:
1. Output ONLY the SQL query. No explanation, no markdown.
2. Start with SELECT or WITH.
3. Use correct JOINs between fact and dimension tables via *_key columns.
4. If you cannot fulfill the request, output exactly: ERROR: CANNOT_BUILD_QUERY
"""
        raw = self.llm.generate_text([prompt])
        sql = clean_sql(raw)

        if not sql or "ERROR" in sql.upper():
            raise ValueError(f"LLM could not generate a valid query. Response: {raw.strip()[:200]}")

        return sql

    def regenerate_with_error(self, user_query: str, schema_context: str, failed_sql: str, error: str) -> str:
        """Regenerate SQL after a previous attempt failed."""
        prompt = f"""You are an expert ClickHouse SQL generator.

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
"""
        raw = self.llm.generate_text([prompt])
        sql = clean_sql(raw)

        if not sql or "ERROR" in sql.upper():
            raise ValueError(f"LLM could not generate a corrected query. Response: {raw.strip()[:200]}")

        return sql

    def regenerate_with_unknown_columns(self, user_query: str, schema_context: str, failed_sql: str, unknown_cols: list[str]) -> str:
        """Regenerate SQL after unknown columns were detected."""
        prompt = f"""You are an expert ClickHouse SQL generator.

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
"""
        raw = self.llm.generate_text([prompt])
        sql = clean_sql(raw)

        if not sql or "ERROR" in sql.upper():
            raise ValueError(f"LLM could not generate a corrected query. Response: {raw.strip()[:200]}")

        return sql
