"""Build a LlamaIndex VectorStoreIndex from ClickHouse schema."""
from __future__ import annotations
from typing import List, Dict
from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from app.retrieval.schema_extractor import TableSchema

# Use a local HuggingFace embedding model (no API key needed)
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")


def _build_table_document(schema: TableSchema) -> Document:
    """Create a rich text document for a single table."""
    col_lines = "\n".join(
        f"  - {c.name} ({c.dtype})" for c in schema.columns
    )

    relationships = ""
    if schema.foreign_keys:
        rel_lines = []
        for fk in schema.foreign_keys:
            rel_lines.append(
                f"  - {schema.name}.{fk['column']} references "
                f"{fk['ref_table']}.{fk['ref_column']}"
            )
        relationships = "\nRelationships (joins):\n" + "\n".join(rel_lines)

    pk_text = ""
    if schema.primary_key:
        pk_text = f"\nPrimary/Order key: {', '.join(schema.primary_key)}"

    table_text = (
        f"Table: {schema.full_name}\n"
        f"Columns:\n{col_lines}"
        f"{pk_text}"
        f"{relationships}"
    )

    return Document(
        text=table_text,
        metadata={
            "table_name": schema.full_name,
            "database": schema.database,
            "table_short": schema.name,
            "column_names": [c.name for c in schema.columns],
            "has_fk": len(schema.foreign_keys) > 0,
            "is_fact": schema.name.startswith("fact_"),
            "is_dim": schema.name.startswith("dim_"),
        },
        excluded_llm_metadata_keys=[],
        excluded_embed_metadata_keys=["database", "table_short", "column_names"],
    )


class SchemaIndexBuilder:
    """Build and manage a LlamaIndex over the database schema."""

    def __init__(self):
        self._schemas: List[TableSchema] = []
        self._index: VectorStoreIndex | None = None
        self._table_map: Dict[str, TableSchema] = {}

    def load_schemas(self, schemas: List[TableSchema]) -> "SchemaIndexBuilder":
        """Load extracted table schemas into the builder."""
        self._schemas = schemas
        self._table_map = {s.full_name: s for s in schemas}
        return self

    def build_index(self) -> VectorStoreIndex:
        """Build a vector index over all table documents."""
        documents = [_build_table_document(s) for s in self._schemas]
        parser = SentenceSplitter(chunk_size=512, chunk_overlap=64)
        self._index = VectorStoreIndex.from_documents(
            documents,
            transformations=[parser],
        )
        return self._index

    @property
    def index(self) -> VectorStoreIndex:
        if self._index is None:
            raise RuntimeError("Index not built yet. Call build_index() first.")
        return self._index

    @property
    def table_map(self) -> Dict[str, TableSchema]:
        return self._table_map

    def get_full_ddl(self, table_names: List[str]) -> str:
        """Get complete DDL snippets for the given tables."""
        blocks = []
        for t in table_names:
            if t in self._table_map:
                blocks.append(self._table_map[t].ddl_snippet())
        return "\n\n".join(blocks)

    def get_relationships_text(self, table_names: List[str]) -> str:
        """Get join instructions for the given tables."""
        parts = []
        for t in table_names:
            if t in self._table_map:
                rel = self._table_map[t].relationships_text()
                if rel:
                    parts.append(rel)
        return "\n".join(parts)
