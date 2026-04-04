"""Generate ClickHouse SQL from user query and retrieved schema context."""
from __future__ import annotations
import re
import hashlib
from typing import List
from app.llm.base import LLMClient
from app.sql.metrics import METRIC_MAP


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


def _build_metric_hints(user_query: str) -> str:
    """Build metric hints from METRIC_MAP for terms found in user query."""
    hints = []
    query_lower = user_query.lower()
    for term, column in METRIC_MAP.items():
        if term.lower() in query_lower:
            hints.append(f"Hint: for '{term}', use column '{column}'")
    return "\n".join(hints)


def classify_visualization(sql: str, columns: List[str], rows: List[List]) -> str:
    """Classify the visualization type based on SQL and result.
    
    Returns one of: 'kpi', 'bar_chart', 'line_chart', 'pie_chart', 'table'.
    """
    # Single cell result -> KPI
    if len(rows) == 1 and len(columns) == 1:
        return "kpi"
    
    # Check if SQL contains GROUP BY
    sql_upper = sql.upper()
    if "GROUP BY" not in sql_upper:
        # No GROUP BY, multiple columns -> table
        return "table"
    
    # GROUP BY exists - check the grouping column
    # If it contains date/month/year keywords -> line_chart
    if any(keyword in sql_upper for keyword in ["TODATE", "TOMONTH", "TOYEAR", "DATE_TRUNC", "FORMATDATETIME", "DATE", "MONTH", "YEAR"]):
        return "line_chart"
    
    # GROUP BY on non-date: check distinct group count
    if len(rows) <= 10:
        return "bar_chart"
    else:
        return "table"


class SQLGenerator:
    """Generate ClickHouse SQL using an LLM and retrieved schema."""

    def __init__(self, llm: LLMClient, cache=None):
        """Initialize SQL generator.
        
        Args:
            llm: LLMClient instance (Groq or vLLM).
            cache: Optional cache object with get/set methods.
        """
        self.llm = llm
        self.cache = cache

    def generate(self, user_query: str, schema_context: str) -> str:
        """Generate initial SQL from user query and schema context.
        
        Args:
            user_query: Natural language question.
            schema_context: Formatted schema DDL and relationships.
        
        Returns:
            Clean SQL string.
        """
        # Check cache
        if self.cache:
            cache_key = hashlib.sha256(
                f"{user_query}|{schema_context}".encode()
            ).hexdigest()
            cached = self.cache.get_cache(cache_key)
            if cached:
                return cached

        # Build metric hints
        metric_hints = _build_metric_hints(user_query)
        metric_section = ""
        if metric_hints:
            metric_section = f"\n\nMETRIC HINTS:\n{metric_hints}"

        system_message = """You are an expert ClickHouse SQL generator for a star-schema data warehouse.

Your role is to:
1. Understand natural language business questions
2. Convert them into efficient, correct ClickHouse SQL
3. Always follow ClickHouse best practices and the rules below

STRICT RULES:
- Output ONLY the SQL query. No explanation, no markdown, no code block markers.
- Start with SELECT or WITH.
- Use ClickHouse date functions CORRECTLY:
  * toDate(string_or_timestamp) - convert to date, requires 1 argument
  * toDate(now()) or toDate(today()) - for current date
  * today() - simpler way to get current date
  * toStartOfDay(date_or_timestamp) - start of day, requires 1 argument
  * toStartOfDay(now()) - for current day start
  * toYear(date) - extract year, requires 1 argument
  * toMonth(date) - extract month, requires 1 argument
  * formatDateTime(timestamp, 'format') - format timestamp, requires 2 arguments
  * CRITICAL: ALL date functions require at least 1 argument - NEVER call toDate(), toStartOfDay(), etc. with no arguments
- Never use ANSI date functions like YEAR(), MONTH(), DATE_FORMAT(), DATEDIFF()
- Always join dim_date using date_key for any time-based filtering
- Only join on *_key integer columns — never join on string fields
- CRITICAL TABLE MAPPING AND TERMINOLOGY - ARTICLE vs PRODUCT:
  * ARTICLE = CATEGORY (brand/type) - 29 categories like 'HIKVISION', 'SAMSUNG' in dim_article
  * PRODUCT = INDIVIDUAL ITEM - 3302+ SKUs in dim_product that belong to an article
  * fact_invoice_lines has product_key (individual products) - NOT article_key
  * To reach ARTICLE/CATEGORY level: fact_invoice_lines.product_key → dim_product.product_key → GROUP BY dim_product.article_id or dim_product.article_name
  * dim_product.article_id joins to dim_article.article_id (dim_article.nom = category name)
  * When user says "article", they mean CATEGORY - group products by article_id or filter by article_name from dim_product
  * NEVER use article_key directly in fact_invoice_lines - it does not exist there
- For dim_product, always add: WHERE dim_product.is_current = 1 when filtering active products
- Use countIf(), sumIf(), avgIf() instead of subqueries where possible
- For COUNT aggregates: use COUNT(*) or COUNT(primary_key), NOT COUNT(column_names_you_invent)
- Never use SELECT * — always name columns explicitly
- For currency amounts, always note which currency_key is being used if filtering by currency
- Use LIMIT 1000 by default unless the user asks for all data
- Always use table aliases for readability (e.g., fs for fact_sales)
- Column naming: always alias aggregates clearly (e.g., sum(line_sales_amount) AS total_revenue)
- CRITICAL FOR CTEs: In WITH clauses, explicitly alias ALL selected columns (e.g., SELECT col AS col_name, not just SELECT col)
- CRITICAL FOR CTEs: In the outer query, reference columns by their aliased names from the CTE, not original table names
- CRITICAL FOR WHERE FILTERS: Only use column names that are EXPLICITLY listed in the schema below
- CRITICAL FOR NAME FILTERING: When filtering by ANY name column (product_name, customer_name, depot_name, nom, firstname, lastname, email, description, adresse, etc.), ALWAYS use LOWER() wrapped around the column:
  * WRONG: WHERE product_name LIKE '%iphone%' (searches are case-sensitive)
  * WRONG: WHERE product_name = 'iphone' (exact match, fails with case/extra chars)
  * CORRECT: WHERE lower(product_name) LIKE '%iphone%' (case-insensitive substring match)
  * Mandatory pattern: WHERE lower(column_name) LIKE '%search_term%' — the lower() function is REQUIRED for ALL name searches
  * NEVER reference a name column directly in WHERE without wrapping it in lower()
  * Examples of CORRECT usage:
    - WHERE lower(dim_product.product_name) LIKE '%iphone%'
    - WHERE lower(dim_customer.customer_name) LIKE '%acme%'
    - WHERE lower(dim_depot.nom) LIKE '%warehouse%'
    - WHERE lower(dim_employee.firstname) LIKE '%john%'
    - WHERE lower(dim_employee.email) LIKE '%@example%'
  * Examples:
    - WHERE lower(dim_product.product_name) LIKE '%iphone%'
    - WHERE lower(dim_customer.customer_name) LIKE '%acme%'
    - WHERE lower(dim_depot.nom) LIKE '%warehouse%'
- CRITICAL: Do NOT invent column names for COUNT - use COUNT(*) or a known primary key, never COUNT(unknown_column)
- If you're unsure about a column name in the schema, remove the WHERE clause rather than guess
- Only use columns you can see in the provided schema - do not invent or assume column names
- Use INNER JOIN by default, never LEFT JOIN unless explicitly requested
- If you cannot fulfill the request, output exactly: ERROR: CANNOT_BUILD_QUERY"""

        user_message = f"""DATABASE SCHEMA:
{schema_context}{metric_section}

STAR SCHEMA NOTES:
- Fact tables: gold.fact_sales, gold.fact_sales_header, gold.fact_invoice_lines,
  gold.fact_consumption, gold.fact_financial_exchange, gold.fact_goods_receipts,
  gold.fact_inventory_count, gold.fact_inventory_snapshot, gold.fact_payment_schedule,
  gold.fact_purchase_order_lines, gold.fact_service_lines, gold.fact_service_request,
  gold.fact_stock_movement
- Dimension tables: gold.dim_date, gold.dim_product, gold.dim_customer,
  gold.dim_currency, gold.dim_depot, gold.dim_employee, gold.dim_invoice,
  gold.dim_user, gold.dim_provider, gold.dim_transporter, gold.dim_package,
  gold.dim_article, gold.dim_manager, gold.dim_responsable, gold.dim_service,
  gold.dim_purchase_order, gold.dim_delivery_note
- Join fact tables to dimension tables using *_key columns (e.g. date_key, product_key, customer_key).
- Use gold.dim_date for time-based filtering via date_key joins.
- Always use the exact column names shown in the schema. Never invent column names.

CRITICAL REMINDERS:
- For COUNT aggregates: use COUNT(*) — do NOT make up column names like "sales_fact_key"
- Before referencing ANY dimension column (like dim_depot.nom): verify it exists in the schema above
- If a column like "nom" doesn't appear in the schema, try common alternatives: name, depot_name, description
- Never use columns that aren't explicitly shown in the provided schema

USER QUERY:
{user_query}"""

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]

        raw = self.llm.generate_text(messages)
        sql = clean_sql(raw)

        if not sql or "ERROR" in sql.upper():
            raise ValueError(f"LLM could not generate a valid query. Response: {raw.strip()[:200]}")

        # Cache the result
        if self.cache:
            self.cache.set_cache(cache_key, sql)

        return sql

    def regenerate_with_error(self, user_query: str, schema_context: str, failed_sql: str, error: str) -> str:
        """Regenerate SQL after a previous attempt failed.
        
        Args:
            user_query: Original natural language question.
            schema_context: Formatted schema DDL and relationships.
            failed_sql: The SQL that failed.
            error: Error message from execution.
        
        Returns:
            Clean corrected SQL string.
        """
        metric_hints = _build_metric_hints(user_query)
        metric_section = ""
        if metric_hints:
            metric_section = f"\n\nMETRIC HINTS:\n{metric_hints}"

        # Detect specific error types and provide targeted guidance
        error_guidance = ""
        if "UNKNOWN_IDENTIFIER" in error:
            # Check if it's a common hallucinated column
            hallucinated_cols = {
                "article_key": "fact_invoice_lines does NOT have article_key. ARTICLE=CATEGORY (29 total from dim_article). To get article-level revenue: JOIN fact_invoice_lines.product_key → dim_product.product_key, then GROUP BY dim_product.article_id or dim_product.article_name. Or join dim_product to dim_article via article_id.",
                "sales_fact_key": "Use COUNT(*) or COUNT(primary_key) instead. This column may not exist in all tables.",
                "depot_id": "This is a dim_depot column. Use depot_key for joins, or dim_depot.depot_id for display. Verify the table you're referencing has this column.",
            }
            
            col_hint = ""
            for bad_col, suggestion in hallucinated_cols.items():
                if bad_col in error:
                    col_hint = f"\n\nCOMMON HALLUCINATION FIX: {suggestion}"
                    break
            
            error_guidance = f"""
SPECIFIC ERROR - Unknown Identifier:
You're referencing a column name that doesn't exist.
Common causes:
1. In a CTE (WITH clause), you selected a column but didn't explicitly alias it
2. In the outer query, you're using the original table prefix instead of the CTE column alias
3. You used a column name that doesn't exist in that table (hallucination) {col_hint}

CRITICAL TABLE-SPECIFIC MAPPINGS TO REMEMBER:
- fact_invoice_lines has product_key, NOT article_key
- When column not found, check the schema provided — only use columns listed there
- Use the exact column names shown in the schema

FIX: Always use "AS column_name" when selecting in CTEs and reference by those exact aliases in outer queries.
Example:
  WITH my_cte AS (
    SELECT table.col1 AS col1, sum(table.col2) AS total FROM table GROUP BY table.col1
  )
  SELECT col1, total FROM my_cte  -- Use CTE aliases, not original table column names
"""
        elif "EmptyDataError" in error or ("Empty result" in error and re.search(r"LIKE|=\s*['\"]", failed_sql)):
            # Empty result with name filtering - likely exact match vs substring match issue
            error_guidance = """
SPECIFIC ERROR - Empty result (likely name filtering issue):
You searched by name but used the WRONG pattern. You either used exact match or didn't use lower().

CRITICAL FIX FOR NAME FILTERING - MANDATORY PATTERN:
For EVERY name-based search, you MUST:
1. Wrap the column name in lower()
2. Use LIKE '%search_term%' for substring matching (not = for exact match)

- ❌ WRONG: WHERE product_name = 'iphone'
- ❌ WRONG: WHERE product_name LIKE '%iphone%' (case-sensitive, misses IPHONE, IPhone, etc.)
- ✅ CORRECT: WHERE lower(product_name) LIKE '%iphone%'

THE MANDATORY PATTERN FOR ALL NAME SEARCHES:
WHERE lower(column_name) LIKE '%search_term%'

YOU MUST ALWAYS:
1. Use lower(column_name) — wrap the column in lower()
2. Use LIKE '%term%' — NOT = 'term'
3. Make the search term lowercase in the LIKE clause

EXAMPLES (ALL CORRECT - follow this pattern):
- WHERE lower(dim_product.product_name) LIKE '%iphone%'
- WHERE lower(dim_customer.customer_name) LIKE '%acme%'  
- WHERE lower(dim_depot.nom) LIKE '%warehouse%'
- WHERE lower(dim_employee.firstname) LIKE '%john%'
- WHERE lower(dim_employee.email) LIKE '%example%'

Applies to ALL text/name columns: product_name, customer_name, nom, firstname, lastname, email, description, adresse, etc.

FIX: Replace your WHERE clause with the mandatory lower() LIKE '%term%' pattern.
"""
        elif "SYNTAX_ERROR" in error or "Syntax error" in error:
            error_guidance = """
SPECIFIC ERROR - Syntax Error:
Check that:
1. All JOINs have proper ON clauses with table aliases
2. All column references use proper table aliases
3. CTEs are properly defined and referenced
4. GROUP BY columns match the SELECT list (unless using ORDER BY or aggregate functions)
"""
        elif "NOT_FOUND" in error or "doesn't exist" in error.lower() or "UNKNOWN_TABLE_FUNCTION" in error:
            error_guidance = """
SPECIFIC ERROR - Table, Column, or Function Not Found:
This usually means:
1. You used a column name that doesn't exist, OR
2. You used COUNT(column_name) instead of COUNT(*), OR
3. You used a dimension column that isn't in the schema (like dim_depot.nom)

CRITICAL FIXES:
1. Check that all table names use the gold. prefix (gold.fact_sales, not just fact_sales)
2. Verify column names exactly match the schema - this is CRITICAL!  
3. Ensure all joins use the correct *_key columns
4. Only use columns that are EXPLICITLY shown in the provided schema - do not guess at column names
5. For COUNT: use COUNT(*) or COUNT(primary_key), NEVER COUNT(unknown_column)
6. For dimension columns: if unsure a column exists, try alternatives (name, depot_name, description) OR just omit it
7. Remove any WHERE filters that use column names not in the schema

COMMON FIXES:
- Replace COUNT(fs.sales_fact_key) with COUNT(*)
- Replace COUNT(unknown_col) with COUNT(*)
- Remove "WHERE dim_depot.nom = ..." and try "WHERE dim_depot.name = ..." or drop the filter
- Remove dimension column filters you're uncertain about

FIX: Remove any WHERE filters with questionable column names and try the query without filters first.
"""
        elif "NUMBER_OF_ARGUMENTS_DOESNT_MATCH" in error or "incorrect number of arguments" in error.lower():
            error_guidance = """
SPECIFIC ERROR - Function called with wrong number of arguments:
This happens when ClickHouse functions are called with missing or extra arguments.

CRITICAL ClickHouse DATE FUNCTION FIXES:
- ❌ WRONG: toDate() — NO arguments
- ✅ CORRECT: toDate(now()) or toDate(today())
- ✅ CORRECT: toDate(column_name) when column contains strings/timestamps

- ❌ WRONG: toStartOfDay() — NO arguments  
- ✅ CORRECT: toStartOfDay(now())
- ✅ CORRECT: toStartOfDay(today())

- ❌ WRONG: toYear() or toMonth() with no argument
- ✅ CORRECT: toYear(date_column) or toYear(now())

FOR CURRENT DATE/TIME IN ClickHouse:
- today() — Returns current date (simplest)
- now() — Returns current timestamp
- toDate(now()) — Convert timestamp to date
- toStartOfDay(now()) — Start of current day
- formatDateTime(now(), '%Y-%m-%d') — Format current time

COMMON ERROR PATTERNS TO FIX:
WHERE date_key = toStartOfDay(toDate())  ← WRONG (missing arguments)
→ Fix: WHERE date_key >= toStartOfDay(today())

WHERE created_at = toDate()  ← WRONG (missing argument)
→ Fix: WHERE created_at >= today()

FOR AGGREGATES:
- sum(amount) — 1 argument
- count(*) — 0 arguments  
- countIf(condition) — 1 argument
- sumIf(amount, condition) — 2 arguments

RULE: Every ClickHouse function that isn't a simple aggregate must have at least 1 argument specified."""
        elif "Empty result from aggregation" in error:
            if "with JOIN" in error:
                error_guidance = """
SPECIFIC ERROR - Empty result from JOIN with aggregation:
This likely means the JOIN keys don't match. Common causes:
1. Foreign key column has NULL values (nothing matches the dimension table)
2. The join condition references wrong columns
3. No records in the dimension table match the fact table keys

CRITICAL FIXES:
1. Verify the join key is NOT NULL in the fact table
2. Use LEFT JOIN first to see which keys match and which don't
3. Check that both tables have the same values for the join key
4. Example: If joining on depot_key, verify fact_sales.depot_key matches dim_depot.depot_key
5. Try removing the JOIN first to confirm the fact table has data
6. Use countDistinct(join_key) to see how many unique values match

DEBUGGING STEPS:
- First: SELECT COUNT(*) FROM fact_table (verify data exists)
- Second: SELECT DISTINCT join_key FROM fact_table (see what values are there)
- Third: SELECT DISTINCT join_key FROM dim_table (see what matches)
- Fourth: Try LEFT JOIN and look for rows where dimension key is NULL (no match)
"""
            else:
                error_guidance = """
SPECIFIC ERROR - Empty result from aggregation:
The aggregation query returns 0 rows. Likely causes:
1. WHERE filters are too restrictive
2. JOIN keys don't match
3. Aggregation columns don't exist

FIXES:
1. Remove WHERE filters one by one to find the culprit
2. Try running without filters first
3. For JOINs, verify keys match (see JOIN debugging steps above)
4. Use COUNT(*) instead of COUNT(specific_column)
"""

        system_message = f"""You are an expert ClickHouse SQL generator. Your job is to fix broken SQL.

{error_guidance}

CRITICAL RULES FOR FIXES:
- Output ONLY the corrected SQL query. No explanation, no markdown.
- Start with SELECT or WITH.
- Use ClickHouse date functions: toYear(), toMonth(), toStartOfMonth(), toDate()
- Always join dim_date using date_key for any time-based filtering
- Only join on *_key integer columns — never join on string fields
- For dim_product, always add: WHERE dim_product.is_current = 1
- Use countIf(), sumIf(), avgIf() instead of subqueries where possible
- Never use SELECT * — always name columns explicitly
- Use LIMIT 1000 by default unless user asks for all data
- Always use table aliases for readability
- Column naming: always alias aggregates clearly (e.g., sum(line_sales_amount) AS total_revenue)
- CRITICAL: In WITH clauses, explicitly alias ALL columns: SELECT col AS col_name
- CRITICAL: In outer queries, reference CTE columns by their aliases, NOT original table prefixes
- CRITICAL: Only use column names from the schema - if a column name doesn't appear in the schema, remove it
- If getting unknown column errors: try removing WHERE clauses that use questionable column names
- If the error is about missing columns: try building a simpler query without filters first"""

        user_message = f"""The following SQL query failed:

FAILED SQL:
{failed_sql}

ERROR:
{error}

DATABASE SCHEMA:
{schema_context}{metric_section}

USER QUERY:
{user_query}

Please fix the error in the SQL. Pay special attention to:
1. All columns in WITH clauses must have explicit aliases
2. Outer queries must reference CTE columns by their alias names only
3. Table names must use gold. prefix
4. All column references must use table aliases
5. Only use column names that appear in the schema - do not guess at column names
6. If there's COUNT(something): replace with COUNT(*) immediately
7. If there's a dimension column you're uncertain about: try alternatives (name, depot_name, description) OR remove it
8. If a column doesn't exist in schema: DO NOT use it - only select columns explicitly shown
9. Remove any WHERE clauses with questionable column names
10. MANDATORY FOR EVERY NAME SEARCH: Use WHERE lower(column_name) LIKE '%search_term%' — NEVER use equals (=) or name without lower()
11. Try building a simple query first without WHERE filters to verify joins work

Return only the corrected SQL."""

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]

        raw = self.llm.generate_text(messages)
        sql = clean_sql(raw)

        if not sql or "ERROR" in sql.upper():
            raise ValueError(f"LLM could not generate a corrected query. Response: {raw.strip()[:200]}")

        return sql

    def regenerate_with_unknown_columns(self, user_query: str, schema_context: str, failed_sql: str, unknown_cols: List[str]) -> str:
        """Regenerate SQL after unknown columns were detected.
        
        Args:
            user_query: Original natural language question.
            schema_context: Formatted schema DDL and relationships.
            failed_sql: The SQL with invalid column names.
            unknown_cols: List of column names that don't exist.
        
        Returns:
            Clean corrected SQL string.
        """
        system_message = """You are an expert ClickHouse SQL generator. Your job is to fix SQL with invalid column names.

CRITICAL RULES:
- Output ONLY the corrected SQL query. No explanation, no markdown.
- Start with SELECT or WITH.
- Use ONLY the column names shown in the schema.
- Never invent column names.
- In WITH clauses, explicitly alias ALL columns: SELECT col AS col_name
- In outer queries, reference CTE columns by their aliases only"""

        user_message = f"""The following SQL used columns that do not exist in the schema:
{', '.join(unknown_cols)}

FAILED SQL:
{failed_sql}

DATABASE SCHEMA:
{schema_context}

USER QUERY:
{user_query}

Please fix by using the correct column names from the schema. Return only the corrected SQL."""

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]

        raw = self.llm.generate_text(messages)
        sql = clean_sql(raw)

        if not sql or "ERROR" in sql.upper():
            raise ValueError(f"LLM could not generate a corrected query. Response: {raw.strip()[:200]}")

        return sql
