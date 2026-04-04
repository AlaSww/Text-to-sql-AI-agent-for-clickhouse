"""Retrieve relevant tables/columns from LlamaIndex for a user query."""
from __future__ import annotations
from typing import List, Tuple, Set
from llama_index.core import VectorStoreIndex


class SchemaRetriever:
    """Use LlamaIndex to find tables relevant to a natural language query."""

    def __init__(self, index: VectorStoreIndex, similarity_top_k: int = 6):
        """Initialize schema retriever.
        
        Args:
            index: LlamaIndex VectorStoreIndex instance.
            similarity_top_k: Default number of similar items to retrieve.
        """
        self._index = index
        self._default_top_k = similarity_top_k

    def _get_adaptive_top_k(self, query: str) -> int:
        """Compute adaptive similarity_top_k based on query keywords.
        
        Returns 10 if query contains comparison/breakdown keywords, else 6.
        
        Args:
            query: User query string.
        
        Returns:
            Adaptive similarity_top_k value.
        """
        keywords = ["compare", "breakdown", "by customer", "by depot", "by product", "vs", "versus", "each", "per"]
        query_lower = query.lower()
        for keyword in keywords:
            if keyword in query_lower:
                return 10
        return self._default_top_k

    def retrieve_tables(self, query: str) -> List[str]:
        """Return list of table full names relevant to the query.
        
        Args:
            query: User query string.
        
        Returns:
            List of relevant table full names.
        """
        top_k = self._get_adaptive_top_k(query)
        retriever = self._index.as_retriever(similarity_top_k=top_k)
        nodes = retriever.retrieve(query)
        tables = []
        seen = set()
        for node in nodes:
            tname = node.metadata.get("table_name")
            if tname and tname not in seen:
                seen.add(tname)
                tables.append(tname)
        return tables

    def retrieve_with_context(self, query: str) -> Tuple[List[str], str]:
        """Retrieve tables and build a schema context string for SQL generation.
        
        Args:
            query: User query string.
        
        Returns:
            Tuple of (list of table names, context string).
        """
        tables = self.retrieve_tables(query)

        # If we found fact tables, also include the dimension tables they join to
        expanded = self._expand_with_dimensions(query, tables)

        top_k = self._get_adaptive_top_k(query)
        retriever = self._index.as_retriever(similarity_top_k=top_k)
        context_parts = []
        for node in retriever.retrieve(query):
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

    def _expand_with_dimensions(self, query: str, fact_tables: List[str]) -> Set[str]:
        """Add dimension tables that join to the retrieved fact tables.
        
        Args:
            query: User query string.
            fact_tables: List of initially retrieved fact table names.
        
        Returns:
            Set of expanded table names including dimensions.
        """
        result = set(fact_tables)
        top_k = self._get_adaptive_top_k(query)
        retriever = self._index.as_retriever(similarity_top_k=top_k)
        for node in retriever.retrieve(" ".join(fact_tables)):
            tname = node.metadata.get("table_name")
            if tname:
                result.add(tname)
        return result
