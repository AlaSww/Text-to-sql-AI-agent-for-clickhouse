"""Extract schema from ClickHouse: tables, columns, and inferred foreign keys."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from clickhouse_connect import get_client


@dataclass
class Column:
    name: str
    dtype: str
    is_nullable: bool = False


@dataclass
class TableSchema:
    database: str
    name: str
    columns: List[Column] = field(default_factory=list)
    primary_key: List[str] = field(default_factory=list)
    foreign_keys: List[Dict[str, str]] = field(default_factory=list)  # {"column": ..., "ref_table": ..., "ref_column": ...}
    description: str = ""

    @property
    def full_name(self) -> str:
        return f"{self.database}.{self.name}"

    def ddl_snippet(self) -> str:
        lines = [f"CREATE TABLE {self.full_name} ("]
        for c in self.columns:
            null = "NULL" if c.is_nullable else "NOT NULL"
            lines.append(f"  {c.name} {c.dtype} {null},")
        if self.primary_key:
            lines.append(f"  PRIMARY KEY ({', '.join(self.primary_key)}),")
        lines.append(")")
        return "\n".join(lines)

    def relationships_text(self) -> str:
        if not self.foreign_keys:
            return ""
        parts = []
        for fk in self.foreign_keys:
            parts.append(f"{self.full_name}.{fk['column']} -> {fk['ref_table']}.{fk['ref_column']}")
        return "Relationships: " + "; ".join(parts)


class SchemaExtractor:
    """Extract schema from a live ClickHouse instance."""

    def __init__(self, host: str, port: int, username: str, password: str, database: str):
        self.database = database
        self.client = get_client(
            host=host,
            port=port,
            username=username,
            password=password,
            database=database,
        )

    def get_tables(self) -> List[str]:
        result = self.client.query(
            f"SELECT name FROM system.tables WHERE database = '{self.database}' ORDER BY name"
        )
        return [row[0] for row in result.result_rows]

    def get_columns(self, table_name: str) -> List[Column]:
        result = self.client.query(
            f"SELECT name, type, is_in_primary_key FROM system.columns "
            f"WHERE database = '{self.database}' AND table = '{table_name}' "
            f"ORDER BY position"
        )
        cols = []
        for row in result.result_rows:
            name, dtype, in_pk = row[0], row[1], bool(row[2])
            is_nullable = dtype.startswith("Nullable(")
            cols.append(Column(name=name, dtype=dtype, is_nullable=is_nullable))
        return cols

    def get_primary_key(self, table_name: str) -> List[str]:
        result = self.client.query(
            f"SELECT primary_key FROM system.tables "
            f"WHERE database = '{self.database}' AND name = '{table_name}'"
        )
        if not result.result_rows:
            return []
        pk_str = result.result_rows[0][0]
        if pk_str:
            return [c.strip() for c in pk_str.split(",")]
        return []

    def infer_foreign_keys(self, table_name: str, all_tables: List[str], all_columns: Dict[str, List[Column]]) -> List[Dict[str, str]]:
        """Infer FK relationships from *_key column naming convention (star schema)."""
        fks = []
        cols = all_columns.get(table_name, [])
        col_names = {c.name for c in cols}

        for col in cols:
            if col.name.endswith("_key") and col.name not in (f"{table_name.rsplit('_', 1)[-1]}_key",):
                # Try to find the ref table: e.g. customer_key -> dim_customer
                base = col.name.replace("_key", "")
                for other_table in all_tables:
                    if other_table == table_name:
                        continue
                    other_short = other_table.split(".")[-1]
                    # Match: customer -> dim_customer
                    if base in other_short and "dim_" in other_short:
                        # Find the _key column in the ref table
                        other_cols = all_columns.get(other_table, [])
                        ref_col = f"{base}_key"
                        if any(c.name == ref_col for c in other_cols):
                            fks.append({
                                "column": col.name,
                                "ref_table": other_table,
                                "ref_column": ref_col,
                            })
                        break

        return fks

    def extract_all(self) -> List[TableSchema]:
        """Extract full schema with inferred relationships."""
        table_names = self.get_tables()
        all_columns: Dict[str, List[Column]] = {}
        primary_keys: Dict[str, List[str]] = {}

        for t in table_names:
            all_columns[t] = self.get_columns(t)
            primary_keys[t] = self.get_primary_key(t)

        schemas = []
        for t in table_names:
            fks = self.infer_foreign_keys(t, table_names, all_columns)
            schema = TableSchema(
                database=self.database,
                name=t,
                columns=all_columns[t],
                primary_key=primary_keys.get(t, []),
                foreign_keys=fks,
            )
            schemas.append(schema)

        return schemas
