"""Execute SQL queries against ClickHouse and fetch results."""
from __future__ import annotations
import re
from typing import Tuple, List
from clickhouse_connect import get_client


def _clean_sql(sql: str) -> str:
    """Extract content from markdown code blocks and clean SQL."""
    # Extract content from markdown code blocks: ```sql\n...\n```
    match = re.search(r"```(?:sql)?\s*(.*?)\s*```", sql, flags=re.DOTALL | re.IGNORECASE)
    if match:
        sql = match.group(1)

    # Remove any remaining backticks
    sql = sql.replace("`", "")

    # Remove leading "SQL:" or "sql:" prefix
    sql = re.sub(r"^\s*sql\s*:", "", sql, flags=re.IGNORECASE)

    return sql.strip().rstrip(";")


class ClickHouseExecutor:
    """Execute ClickHouse queries and return structured results."""

    def __init__(self, host: str, port: int, username: str, password: str, database: str):
        """Initialize ClickHouse executor.
        
        Args:
            host: ClickHouse server hostname.
            port: ClickHouse HTTP port.
            username: ClickHouse username.
            password: ClickHouse password.
            database: Target database name.
        """
        self.client = get_client(
            host=host,
            port=port,
            username=username,
            password=password,
            database=database,
        )

    def run(self, sql: str) -> Tuple[List[str], List[List]]:
        """Execute SQL and return columns and rows.
        
        Args:
            sql: SQL query to execute.
        
        Returns:
            Tuple of (column_names, rows) where rows is list of lists.
            
        Raises:
            Exception: If query execution fails.
        """
        sql = _clean_sql(sql)
        
        # Pre-execution validation: catch common function argument issues
        validation_error = self._validate_clickhouse_syntax(sql)
        if validation_error:
            raise ValueError(validation_error)
        
        # Execute query
        result = self.client.query(sql)
        
        # Extract column names - click house-connect should provide these
        column_names = []
        try:
            # First try the column_names attribute
            if hasattr(result, 'column_names') and result.column_names:
                column_names = list(result.column_names)
            
            # If that fails, try to infer from the first row
            if not column_names and result.result_rows:
                sample_row = result.result_rows[0]
                column_names = [f"col_{i}" for i in range(len(sample_row))]
            
            # If still empty but no rows, generate column names from query if possible
            if not column_names:
                # Try to extract column names from SELECT clause as fallback
                column_names = self._extract_columns_from_sql(sql)
        except Exception:
            # If all else fails, use generic names
            if result.result_rows:
                column_names = [f"col_{i}" for i in range(len(result.result_rows[0]))]
        
        # Convert result tuples to lists
        rows = [list(row) for row in result.result_rows]
        
        # Check for empty results with aggregates - common sign of invalid column references or data issues
        if len(rows) == 0 and self._has_aggregates(sql):
            # Check if it's a join query
            if "JOIN" in sql.upper():
                raise ValueError(
                    "Empty result from aggregation with JOIN - potential causes:\n"
                    "1. Join keys don't match (e.g., foreign key column is NULL or missing values)\n"
                    "2. COUNT/SUM used with non-existent columns\n"
                    "3. Dimension table has no matching records\n"
                    "4. Data quality issue: check that join keys are populated in source table\n"
                    "Try: 1) Use LEFT JOIN to see which keys don't match, 2) Verify foreign keys have non-NULL values, 3) Check sample data"
                )
            else:
                raise ValueError(
                    "Empty result from aggregation query - likely cause: COUNT/SUM/AVG used with non-existent columns. "
                    "Common issues: COUNT(unknown_column) should be COUNT(*), or dimension column doesn't exist in schema. "
                    "Try: 1) Replace COUNT(col) with COUNT(*), 2) Remove questionable WHERE filters, 3) Verify all column names in schema."
                )
        
        return column_names, rows

    def _extract_columns_from_sql(self, sql: str) -> List[str]:
        """Try to extract column names from the final SELECT clause of a query.
        
        This is a best-effort heuristic for cases where metadata is unavailable.
        
        Args:
            sql: SQL query string.
        
        Returns:
            List of extracted column names, or empty list if extraction fails.
        """
        try:
            # Handle CTEs and subqueries - extract from the final SELECT
            sql_upper = sql.upper()
            
            # Find the last/main SELECT statement
            # Look for pattern: FROM ... SELECT or end of query ending with SELECT
            select_positions = [m.start() for m in re.finditer(r'\bSELECT\b', sql, re.IGNORECASE)]
            
            if not select_positions:
                return []
            
            # Get the last SELECT (the main one, not in a subquery/CTE)
            main_select_pos = select_positions[-1]
            select_section = sql[main_select_pos:]
            
            # Find FROM or UNION after this SELECT
            from_match = re.search(r'\bFROM\b', select_section, re.IGNORECASE)
            order_match = re.search(r'\bORDER\s+BY\b|\bUNION\b|\bLIMIT\b', select_section, re.IGNORECASE)
            
            if from_match:
                cols_end = from_match.start()
            elif order_match:
                cols_end = order_match.start()
            else:
                cols_end = len(select_section)
            
            cols_text = select_section[6:cols_end].strip()  # Skip "SELECT"
            
            if not cols_text:
                return []
            
            # Split by comma and extract column names/aliases
            columns = []
            current_col = ""
            paren_depth = 0
            
            for char in cols_text:
                if char == '(':
                    paren_depth += 1
                    current_col += char
                elif char == ')':
                    paren_depth -= 1
                    current_col += char
                elif char == ',' and paren_depth == 0:
                    if current_col.strip():
                        col_name = self._extract_col_name(current_col.strip())
                        if col_name:
                            columns.append(col_name)
                    current_col = ""
                else:
                    current_col += char
            
            # Don't forget the last column
            if current_col.strip():
                col_name = self._extract_col_name(current_col.strip())
                if col_name:
                    columns.append(col_name)
            
            return columns
        except Exception:
            return []

    def _extract_col_name(self, col_expr: str) -> str:
        """Extract the actual column name/alias from an expression.
        
        Args:
            col_expr: A single column expression like "sum(amount) AS total" or "table.name"
        
        Returns:
            The column name (alias if present, else the column identifier).
        """
        col_expr = col_expr.strip()
        
        # Handle "expr AS name" - extract the alias (highest priority)
        as_match = re.search(r'\bAS\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*$', col_expr, re.IGNORECASE)
        if as_match:
            return as_match.group(1)
        
        # Handle "expr AS 'name'" - quoted alias
        quoted_match = re.search(r'\bAS\s+["\']([^"\']+)["\']\s*$', col_expr, re.IGNORECASE)
        if quoted_match:
            return quoted_match.group(1)
        
        # If it's a simple identifier (no function call, no arithmetic)
        if not re.search(r'[\(\)+-/*]', col_expr):
            # It's like: "table.column" or just "column"
            parts = col_expr.split('.')
            return parts[-1]  # Get the last part after dot if qualified
        
        # For function calls without alias, use a generic name based on function
        if re.match(r'^count\s*\(', col_expr, re.IGNORECASE):
            return "count"
        if re.match(r'^sum\s*\(', col_expr, re.IGNORECASE):
            return "sum"
        if re.match(r'^avg\s*\(', col_expr, re.IGNORECASE):
            return "avg"
        if re.match(r'^max\s*\(', col_expr, re.IGNORECASE):
            return "max"
        if re.match(r'^min\s*\(', col_expr, re.IGNORECASE):
            return "min"
        
        # Fallback for unaggregated function without alias
        return "col"

    def run_raw(self, sql: str) -> List[List]:
        """Execute SQL and return raw rows (legacy interface).
        
        Args:
            sql: SQL query to execute.
        
        Returns:
            List of result rows (as tuples).
        """
        sql = _clean_sql(sql)
        return self.client.query(sql).result_rows

    def _has_aggregates(self, sql: str) -> bool:
        """Check if SQL contains aggregate functions (COUNT, SUM, AVG, MAX, MIN).
        
        Args:
            sql: SQL query string.
        
        Returns:
            True if aggregates detected, False otherwise.
        """
        aggregates = r'COUNT\s*\(|SUM\s*\(|AVG\s*\(|MAX\s*\(|MIN\s*\(|countIf\s*\(|sumIf\s*\(|avgIf\s*\('
        return bool(re.search(aggregates, sql, re.IGNORECASE))

    def _validate_clickhouse_syntax(self, sql: str) -> str:
        """Check for common ClickHouse syntax errors before execution.
        
        Args:
            sql: SQL query string.
        
        Returns:
            Error message if issues found, empty string if OK.
        """
        # Check for date functions called with no arguments
        # Pattern: toDate() or toStartOfDay() with empty parentheses (possibly with whitespace)
        empty_arg_patterns = [
            (r'toDate\s*\(\s*\)', 'toDate() called with no arguments - use toDate(now()) or toDate(today())'),
            (r'toStartOfDay\s*\(\s*\)', 'toStartOfDay() called with no arguments - use toStartOfDay(now()) or toStartOfDay(today())'),
            (r'toStartOfMonth\s*\(\s*\)', 'toStartOfMonth() called with no arguments - use toStartOfMonth(now())'),
            (r'toYear\s*\(\s*\)', 'toYear() called with no arguments - use toYear(date_column) or toYear(now())'),
            (r'toMonth\s*\(\s*\)', 'toMonth() called with no arguments - use toMonth(date_column) or toMonth(now())'),
        ]
        
        for pattern, error_msg in empty_arg_patterns:
            if re.search(pattern, sql, re.IGNORECASE):
                return f"ClickHouse Syntax Error: {error_msg}\n\nFix: Add argument like now(), today(), or a column name to the function call.\nExample: WHERE date_key >= toStartOfDay(now())"
        
        return ""