import re


def _clean_sql(sql: str) -> str:
    # Extract content from markdown code blocks: ```sql\n...\n```
    match = re.search(r"```(?:sql)?\s*(.*?)\s*```", sql, flags=re.DOTALL | re.IGNORECASE)
    if match:
        sql = match.group(1)

    # Remove any remaining backticks
    sql = sql.replace("`", "")

    # Remove leading "SQL:" or "sql:" prefix
    sql = re.sub(r"^\s*sql\s*:", "", sql, flags=re.IGNORECASE)

    return sql.strip().rstrip(";")


class SQLSafety:
    FORBIDDEN = [
        "drop", "truncate", "delete", "insert", "update",
        "alter", "detach", "attach", "optimize", "system"
    ]

    def validate(self, sql: str) -> None:
        # 1. Clean the SQL first (remove markdown, prefixes, etc.)
        sql = _clean_sql(sql)

        # 2. Check if the string is empty
        if not sql or not sql.strip():
            raise ValueError("The generated SQL is empty. The LLM may have failed to produce a query.")

        s = sql.strip().lower()

        # 3. Find the first SQL keyword properly (select/with)
        # Look for SELECT or WITH as the first meaningful SQL keyword
        match = re.search(r"\b(select|with|drop|truncate|delete|insert|update|alter|create)\b", s)
        if not match:
            raise ValueError(f"Only SELECT/WITH queries are allowed. No valid SQL keyword found.")

        first_keyword = match.group(1)
        if first_keyword not in ("select", "with"):
            raise ValueError(f"Only SELECT/WITH queries are allowed. Found: '{first_keyword}'")

        # 4. Check for forbidden keywords using word boundaries
        for word in self.FORBIDDEN:
            if re.search(rf"\b{word}\b", s):
                raise ValueError(f"Forbidden keyword detected: {word}")

        # 5. Check for multiple queries (semicolon injection)
        # Strip trailing semicolons, then check if any remain
        stripped = sql.strip().rstrip(";")
        if ";" in stripped:
            raise ValueError("Multiple queries detected.")