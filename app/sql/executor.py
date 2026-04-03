from clickhouse_connect import get_client
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


class ClickHouseExecutor:
    def __init__(self, host: str, port: int, username: str, password: str, database: str):
        self.client = get_client(
            host=host,
            port=port,
            username=username,
            password=password,
            database=database,
        )

    def run(self, sql: str):
        sql = _clean_sql(sql)
        return self.client.query(sql).result_rows