"""Retrieve relevant tables/columns from LlamaIndex for a user query."""
from __future__ import annotations
from typing import List, Tuple, Set
from llama_index.core import VectorStoreIndex


class SchemaRetriever:
    """Use LlamaIndex to find tables relevant to a natural language query."""

    def __init__(self, index: VectorStoreIndex, similarity_top_k: int = 5):
        self._retriever = index.as_retriever(similarity_top_k=similarity_top_k)
        self._index = index

    def retrieve_tables(self, query: str) -> List[str]:
        """Return list of table full names relevant to the query."""
        nodes = self._retriever.retrieve(query)
        tables = []
        seen = set()
        for node in nodes:
            tname = node.metadata.get("table_name")
            if tname and tname not in seen:
                seen.add(tname)
                tables.append(tname)
        return tables

    def retrieve_with_context(self, query: str) -> Tuple[List[str], str]:
        """Retrieve tables and build a schema context string for SQL generation."""
        tables = self.retrieve_tables(query)

        # If we found fact tables, also include the dimension tables they join to
        expanded = self._expand_with_dimensions(tables)

        context_parts = []
        for node in self._retriever.retrieve(query):
            tname = node.metadata.get("table_name")
            if tname in expanded:
                context_parts.append(node.get_content())

        # If retrieval returned nothing, fall back to all tables
        if not context_parts:
            all_nodes = self._index.docstore.docs
            for doc_id, doc in all_nodes.items():
                tname = doc.metadata.get("table_name")
                if tname in expanded:
                    context_parts.append(doc.text)

        return expanded, "\n\n".join(context_parts)

    def _expand_with_dimensions(self, fact_tables: List[str]) -> Set[str]:
        """Add dimension tables that join to the retrieved fact tables."""
        result = set(fact_tables)
        for node in self._retriever.retrieve(" ".join(fact_tables)):
            tname = node.metadata.get("table_name")
            if tname:
                result.add(tname)
        return result
