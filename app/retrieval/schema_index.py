"""Build a LlamaIndex VectorStoreIndex from ClickHouse schema."""
from __future__ import annotations
import os
from typing import List, Dict
from llama_index.core import Document, VectorStoreIndex, Settings, StorageContext, load_index_from_storage
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from app.retrieval.schema_extractor import TableSchema

# Use a local HuggingFace embedding model (no API key needed)
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# Cache directory path
INDEX_CACHE_DIR = "./index_cache"


# Business descriptions for each table
BUSINESS_DESCRIPTIONS = {
    "fact_sales": "Individual sales line items from POS and invoices. PRIMARY KEY: sales_fact_key. For COUNT use: COUNT(*) or COUNT(sales_fact_key). Columns: sales_fact_key, document_key, depot_key, product_key, customer_key, date_key, line_net_amount, gross_margin, qty_sold. Use line_net_amount for net revenue, gross_margin for profit, qty_sold for volume.",
    "fact_sales_header": "Sales transaction totals. Use total_sale for transaction value, reglement for amount paid.",
    "fact_invoice_lines": "Invoice line details with product/pricing info. KEY COLUMNS: invoice_line_fact_key (PK), invoice_key, product_key (INDIVIDUAL PRODUCT, not article), customer_key, depot_key, date_key. CRITICAL: fact_invoice_lines has product_key which links to INDIVIDUAL products. To get ARTICLE/CATEGORY level: JOIN product_key → dim_product.product_key → use dim_product.article_id or dim_product.article_name. NEVER use article_key directly here. Amounts: montant_total (line total), prix_vente (selling price), prix_achat (purchase price), prix_transport (transport cost). Use montant_total for revenue.",
    "fact_goods_receipts": "Goods received from suppliers. Use qty_received vs qty_purchased for receiving variance.",
    "fact_inventory_snapshot": "Daily inventory levels. Use quantity_on_hand, stock_value_at_cost, low_stock_flag.",
    "fact_payment_schedule": "Payment installments. Use statut_payement_flag=1 for paid tranches.",
    "fact_consumption": "Internal stock consumption by employees.",
    "fact_financial_exchange": "Cash flow in/out by currency.",
    "fact_stock_movement": "Stock transfers between depots.",
    "fact_service_request": "Customer service/repair requests with pricing.",
    "fact_service_lines": "Individual service line items.",
    "fact_purchase_order_lines": "Purchase order lines sent to suppliers.",
    "fact_inventory_count": "Physical inventory count results vs system quantity.",
    "dim_date": "Date dimension. Always join here for time filtering using date_key.",
    "dim_product": "Individual product catalog (3302+ items, SCD Type 2). CRITICAL FIELDS: product_key (join key), product_name (individual product), article_id (link to article/category), article_name (CATEGORY/ARTICLE name). Use is_current=1 for active products. BRIDGE: To group by article/category, GROUP BY article_id or article_name.",
    "dim_article": "Article/Category dimension (29 categories total). ARTICLE = CATEGORY not individual product. Columns: article_key, article_id, nom (article/category name), description, unite (unit). Examples: 'HIKVISION' is an article/brand/category. Use for filtering/grouping by article type. Join to dim_product via article_id.",
    "dim_customer": "Customer master. Use nom_societe_client for company name.",
    "dim_depot": "Warehouse/depot locations. Columns: depot_key, name (warehouse name), location, region. Use 'name' for warehouse name, NOT 'nom'. Join with fact_sales.depot_key.",
}


def _build_table_document(schema: TableSchema) -> Document:
    """Create a rich text document for a single table with business context."""
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

    # Add business description if available
    short_name = schema.name
    business_desc = BUSINESS_DESCRIPTIONS.get(short_name, "")
    business_section = ""
    if business_desc:
        business_section = f"\n\nBusiness Context:\n{business_desc}"

    table_text = (
        f"Table: {schema.full_name}\n"
        f"Columns:\n{col_lines}"
        f"{pk_text}"
        f"{relationships}"
        f"{business_section}"
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
    """Build and manage a LlamaIndex over the database schema with persistence."""

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

    def save_index_to_cache(self) -> None:
        """Save the built index to disk for persistence."""
        if self._index is None:
            raise RuntimeError("Index not built yet. Call build_index() first.")
        
        os.makedirs(INDEX_CACHE_DIR, exist_ok=True)
        self._index.storage_context.persist(persist_dir=INDEX_CACHE_DIR)

    def load_index_from_cache(self) -> bool:
        """Try to load index from cache.
        
        Returns:
            True if cache was loaded successfully, False otherwise.
        """
        if not os.path.exists(os.path.join(INDEX_CACHE_DIR, "docstore.json")):
            return False
        
        try:
            storage_context = StorageContext.from_persist_dir(persist_dir=INDEX_CACHE_DIR)
            self._index = load_index_from_storage(storage_context)
            return True
        except Exception:
            return False

    def clear_cache(self) -> None:
        """Delete the cache directory to force a rebuild."""
        import shutil
        if os.path.exists(INDEX_CACHE_DIR):
            shutil.rmtree(INDEX_CACHE_DIR)
        self._index = None

    @property
    def index(self) -> VectorStoreIndex:
        if self._index is None:
            raise RuntimeError("Index not built yet. Call build_index() or load_index_from_cache() first.")
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
