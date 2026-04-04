#!/usr/bin/env python3
"""
Initialize Qdrant collections for Text-to-SQL AI agent.

Collections:
1. gold_schema: Semantic embeddings of all ClickHouse tables
2. gold_query_cache: Cache of successful NL→SQL pairs (starts empty)

Usage:
    python scripts/init_collections.py                    # Init both collections
    python scripts/init_collections.py --force            # Recreate collections
    python scripts/init_collections.py --collection gold_schema  # Init one collection
    python scripts/init_collections.py --force --collection gold_query_cache
"""

import os
import sys
import argparse
import json
from typing import Dict, List, Tuple
from datetime import datetime

# Third-party imports
print("Loading dependencies...", flush=True)
try:
    import clickhouse_connect
    print("  ✓ clickhouse_connect", flush=True)
except ImportError as e:
    print(f"ERROR: Missing required dependency. {e}")
    print("Install with: pip install qdrant-client sentence-transformers clickhouse-connect")
    sys.exit(1)

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
    print("  ✓ qdrant_client", flush=True)
except ImportError as e:
    print(f"ERROR: Missing required dependency. {e}")
    print("Install with: pip install qdrant-client sentence-transformers clickhouse-connect")
    sys.exit(1)

try:
    print("  Loading sentence_transformers (this may take a moment)...", flush=True)
    from sentence_transformers import SentenceTransformer
    print("  ✓ sentence_transformers", flush=True)
except ImportError as e:
    print(f"ERROR: Missing required dependency. {e}")
    print("Install with: pip install qdrant-client sentence-transformers clickhouse-connect")
    sys.exit(1)


# Hardcoded business descriptions
BUSINESS_DESCRIPTIONS = {
    # Fact tables
    "gold.fact_sales": "Individual sales line items from POS and invoices. Use line_net_amount for net revenue after discount and tax, gross_margin for profit, qty_sold for volume sold, unit_price for listed price, selling_price for actual price.",
    "gold.fact_sales_header": "Sales transaction totals per session. Use total_sale for full transaction value, reglement for amount already paid, frais for extra charges, paye_flag=1 for fully paid transactions.",
    "gold.fact_invoice_lines": "Invoice line details linking purchases and sales. Contains prix_achat (purchase price), prix_vente (selling price), qty_invoiced, remise and taxe per line.",
    "gold.fact_goods_receipts": "Goods received from suppliers via delivery notes. Use qty_received vs qty_purchased for receiving variance, unpaid_amount for outstanding supplier balance, transport_cost for freight cost.",
    "gold.fact_inventory_snapshot": "Daily snapshot of stock levels per product per depot. Use quantity_on_hand for current stock, stock_value_at_cost for inventory cost, stock_value_at_sale for inventory at retail value, low_stock_flag=1 for items below minimum.",
    "gold.fact_inventory_count": "Physical inventory count results. Use quantite_reelle for counted quantity, difference for variance between system and physical count.",
    "gold.fact_payment_schedule": "Payment installment schedule for invoices and purchase orders. Use montant_tranche for installment amount, statut_payement_flag=1 for paid installments, closure_validated_amount for validated closure amount.",
    "gold.fact_consumption": "Internal stock consumption by employees. Use quantity_consumed for units used, total_cost for consumption value, unit_cost for per-unit cost.",
    "gold.fact_financial_exchange": "Cash flow records by currency. Use flow_in_amount for incoming cash, flow_out_amount for outgoing cash, montant for net amount.",
    "gold.fact_stock_movement": "Stock transfer records between depots. Use qty_transferred for units sent, qty_received for units confirmed received, estimated_unit_cost for valuation.",
    "gold.fact_service_request": "Customer service or repair requests. Use prix_total for total billed, avance for advance payment, paid_flag=1 for settled requests, total_cout_service for service cost, total_produits for parts cost.",
    "gold.fact_service_lines": "Individual service line items within a service request. Use cout_service for cost, prix_vente for billed price, remise and taxe per line.",
    "gold.fact_purchase_order_lines": "Purchase order lines sent to suppliers. Use ordered_qty for quantity ordered, montant_total for line total, ordered_net_amount for net after discount and tax.",
    "gold.fact_product_history": "Historical log of product price and quantity changes. Use ecart_valeur for value variance, prix_unitaire_applique for the price that was applied.",
    
    # Dimension tables
    "gold.dim_date": "Date dimension for all time-based filtering and grouping. Always join using date_key. Use day_num, month_num, year_num, quarter_num for period grouping, is_weekend=1 for weekend days.",
    "gold.dim_product": "Product catalog with SCD Type 2 history. Always filter with is_current=1 to get active products. Use prix_unitaire for sale price, prix_gros for wholesale price, gain_unitaire for unit margin, taxe for tax rate, levelstock for minimum stock level.",
    "gold.dim_customer": "Customer master data. Use nom_societe_client for company name, type_client to distinguish between individual and corporate clients, solde_client for outstanding balance, matricule_fiscale for tax ID.",
    "gold.dim_depot": "Warehouse and depot locations. Use nom for depot name, capitale for depot capital value.",
    "gold.dim_employee": "Employee master data. Use type_employer and tache_employer for role filtering.",
    "gold.dim_currency": "Currency reference with exchange rates. Use taux_change for the exchange rate, symbole for currency symbol.",
    "gold.dim_invoice": "Invoice header information. Use dtype to distinguish invoice types, paye=1 for paid invoices, ref_type and id_ref for linking to source documents.",
    "gold.dim_provider": "Supplier/vendor master data. Use nom_societe for company name, matricule_fiscale for tax ID.",
    "gold.dim_transporter": "Transporter/carrier master data. Use immatriculev for vehicle registration.",
    "gold.dim_user": "System user master data. Use role to filter by user role.",
    "gold.dim_manager": "Manager master data. Use attribut_manager for manager type.",
    "gold.dim_responsable": "Depot responsible person data. Links to dim_depot via responsable_id.",
    "gold.dim_package": "Package/paquet configuration. Use qty_paquet_unite for units per package, prix_unitaire for package price, levelstock for stock level.",
    "gold.dim_article": "Base article/item master. Use nom for article name, unite for unit of measure, is_showing=1 for visible articles.",
    "gold.dim_service": "Service catalog. Use cout_service for service cost, taxe for tax rate, etat_service=1 for active services, out_service=1 for discontinued services.",
    "gold.dim_purchase_order": "Purchase order headers. Use statut_commande for order status, montant_total for order total.",
    "gold.dim_delivery_note": "Delivery note headers from suppliers. Use converted_to_facture=1 for notes already invoiced, paye=1 for paid notes, frais_transport for freight charges.",
}

COLLECTION_CONFIG = {
    "gold_schema": {
        "vector_size": 384,
        "distance": Distance.COSINE,
    },
    "gold_query_cache": {
        "vector_size": 384,
        "distance": Distance.COSINE,
    },
}


def get_env_config() -> Dict[str, str]:
    """Read configuration from environment variables with defaults."""
    return {
        "clickhouse_host": os.getenv("CLICKHOUSE_HOST", "localhost"),
        "clickhouse_port": int(os.getenv("CLICKHOUSE_PORT", 8123)),
        "clickhouse_user": os.getenv("CLICKHOUSE_USER", "default"),
        "clickhouse_password": os.getenv("CLICKHOUSE_PASSWORD", ""),
        "clickhouse_db": os.getenv("CLICKHOUSE_DB", "gold"),
        "qdrant_host": os.getenv("QDRANT_HOST", "localhost"),
        "qdrant_port": int(os.getenv("QDRANT_PORT", 6333)),
    }


def connect_clickhouse(config: Dict[str, str]):
    """Create ClickHouse connection."""
    try:
        print("  Connecting to ClickHouse...", flush=True)
        client = clickhouse_connect.get_client(
            host=config["clickhouse_host"],
            port=config["clickhouse_port"],
            username=config["clickhouse_user"],
            password=config["clickhouse_password"],
            database=config["clickhouse_db"],
        )
        print("  ✓ ClickHouse connected", flush=True)
        return client
    except Exception as e:
        print(f"ERROR: Failed to connect to ClickHouse: {e}", flush=True)
        sys.exit(1)


def connect_qdrant(config: Dict[str, str]):
    """Create Qdrant connection."""
    try:
        print("  Connecting to Qdrant...", flush=True)
        client = QdrantClient(
            host=config["qdrant_host"],
            port=config["qdrant_port"],
        )
        print("  ✓ Qdrant connected", flush=True)
        return client
    except Exception as e:
        print(f"ERROR: Failed to connect to Qdrant: {e}", flush=True)
        sys.exit(1)


def get_embedding_model():
    """Load sentence transformer model for embeddings."""
    try:
        print("    Loading embedding model BAAI/bge-small-en-v1.5...", flush=True)
        model = SentenceTransformer("BAAI/bge-small-en-v1.5")
        print("    ✓ Embedding model loaded", flush=True)
        return model
    except Exception as e:
        print(f"ERROR: Failed to load embedding model: {e}", flush=True)
        sys.exit(1)


def get_tables_from_clickhouse(ch_client, database: str = "gold") -> Dict[str, List[str]]:
    """
    Query ClickHouse system.columns to get all tables and their columns.
    
    Returns:
        Dict mapping table_name -> list of column names
    """
    try:
        print("    Querying ClickHouse for tables and columns...", flush=True)
        result = ch_client.query_df(
            f"""
            SELECT table, name 
            FROM system.columns 
            WHERE database = '{database}'
            ORDER BY table, position
            """
        )
        
        tables = {}
        for _, row in result.iterrows():
            table = row["table"]
            name = row["name"]
            if table not in tables:
                tables[table] = []
            tables[table].append(name)
        
        print(f"    Found {len(tables)} tables", flush=True)
        return tables
    except Exception as e:
        print(f"ERROR: Failed to query ClickHouse system tables: {e}", flush=True)
        sys.exit(1)


def build_embedding_text(table_name: str, description: str, columns: List[str]) -> str:
    """Build text to embed: table_name. description. Columns: col1, col2, ..."""
    columns_str = ", ".join(columns)
    return f"{table_name}. {description}. Columns: {columns_str}"


def init_gold_schema_collection(
    ch_client,
    qdrant_client,
    embedding_model,
    force: bool = False,
) -> Tuple[int, str]:
    """
    Initialize gold_schema collection with table embeddings.
    
    Returns:
        (point_count, status_message)
    """
    collection_name = "gold_schema"
    
    print(f"Creating collection: {collection_name}...")
    
    # Check if collection exists
    try:
        collections = [c.name for c in qdrant_client.get_collections().collections]
        if collection_name in collections:
            if not force:
                print(f"  ⚠ Collection {collection_name} already exists. Use --force to recreate.")
                return 0, "SKIPPED"
            else:
                print(f"  Deleting existing collection {collection_name}...")
                qdrant_client.delete_collection(collection_name)
    except Exception as e:
        print(f"  WARNING: Could not check existing collections: {e}")
    
    # Create collection
    try:
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=COLLECTION_CONFIG[collection_name]["vector_size"],
                distance=COLLECTION_CONFIG[collection_name]["distance"],
            ),
        )
    except Exception as e:
        print(f"ERROR: Failed to create collection {collection_name}: {e}")
        sys.exit(1)
    
    # Get tables from ClickHouse
    print("  Querying ClickHouse for tables and columns...")
    tables = get_tables_from_clickhouse(ch_client)
    
    # Filter to only tables we have descriptions for
    tables_to_index = {}
    for table_name in sorted(tables.keys()):
        full_name = f"gold.{table_name}"
        if full_name in BUSINESS_DESCRIPTIONS:
            tables_to_index[table_name] = tables[table_name]
        else:
            print(f"  WARNING: No business description for {full_name}, skipping")
    
    if not tables_to_index:
        print("ERROR: No tables found to index!")
        sys.exit(1)
    
    print(f"  Embedding {len(tables_to_index)} table documents...")
    
    # Build embeddings
    texts_to_embed = []
    table_metadata = []
    
    for table_name in sorted(tables_to_index.keys()):
        full_name = f"gold.{table_name}"
        description = BUSINESS_DESCRIPTIONS[full_name]
        columns = tables_to_index[table_name]
        
        embedding_text = build_embedding_text(full_name, description, columns)
        texts_to_embed.append(embedding_text)
        
        table_metadata.append({
            "table_name": full_name,
            "description": description,
            "columns": columns,
            "type": "fact" if "fact_" in table_name else "dim",
        })
    
    # Generate embeddings
    try:
        embeddings = embedding_model.encode(texts_to_embed, convert_to_numpy=True)
    except Exception as e:
        print(f"ERROR: Failed to generate embeddings: {e}")
        sys.exit(1)
    
    # Build points for Qdrant
    points = []
    for idx, (metadata, embedding) in enumerate(zip(table_metadata, embeddings)):
        point = PointStruct(
            id=idx,
            vector=embedding.tolist(),
            payload={
                "table_name": metadata["table_name"],
                "description": metadata["description"],
                "columns": metadata["columns"],
                "type": metadata["type"],
                "column_count": len(metadata["columns"]),
            },
        )
        points.append(point)
    
    # Upsert to Qdrant
    print("  Upserting points to Qdrant...")
    try:
        qdrant_client.upsert(
            collection_name=collection_name,
            points=points,
        )
    except Exception as e:
        print(f"ERROR: Failed to upsert points: {e}")
        sys.exit(1)
    
    point_count = len(points)
    print(f"  Collection {collection_name} ready. {point_count} points indexed.")
    
    return point_count, "OK"


def init_gold_query_cache_collection(
    qdrant_client,
    force: bool = False,
) -> Tuple[int, str]:
    """
    Initialize gold_query_cache collection (starts empty).
    
    Returns:
        (point_count, status_message)
    """
    collection_name = "gold_query_cache"
    
    print(f"Creating collection: {collection_name}...")
    
    # Check if collection exists
    try:
        collections = [c.name for c in qdrant_client.get_collections().collections]
        if collection_name in collections:
            if not force:
                print(f"  ⚠ Collection {collection_name} already exists. Use --force to recreate.")
                return 0, "SKIPPED"
            else:
                print(f"  Deleting existing collection {collection_name}...")
                qdrant_client.delete_collection(collection_name)
    except Exception as e:
        print(f"  WARNING: Could not check existing collections: {e}")
    
    # Create collection
    try:
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=COLLECTION_CONFIG[collection_name]["vector_size"],
                distance=COLLECTION_CONFIG[collection_name]["distance"],
            ),
        )
    except Exception as e:
        print(f"ERROR: Failed to create collection {collection_name}: {e}")
        sys.exit(1)
    
    print(f"  Collection {collection_name} created (empty, ready for runtime inserts).")
    
    return 0, "OK (runtime)"


def print_summary(results: Dict[str, Tuple[int, str]]):
    """Print summary table."""
    print("\n" + "=" * 60)
    print(f"{'Collection':<20} | {'Points':<7} | Status")
    print("-" * 60)
    for collection_name in sorted(results.keys()):
        point_count, status = results[collection_name]
        print(f"{collection_name:<20} | {point_count!s:<7} | {status}")
    print("=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Initialize Qdrant collections for Text-to-SQL AI agent"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recreate collections even if they already exist",
    )
    parser.add_argument(
        "--collection",
        choices=["gold_schema", "gold_query_cache"],
        help="Initialize only a specific collection (default: all)",
    )
    
    args = parser.parse_args()
    
    # Determine which collections to initialize
    collections_to_init = [args.collection] if args.collection else ["gold_schema", "gold_query_cache"]
    
    print(f"Initializing Qdrant collections at {datetime.now().isoformat()}", flush=True)
    print("-" * 60, flush=True)
    
    # Load config and connect
    print("Loading configuration...", flush=True)
    config = get_env_config()
    print(f"ClickHouse: {config['clickhouse_host']}:{config['clickhouse_port']}/{config['clickhouse_db']}", flush=True)
    print(f"Qdrant: {config['qdrant_host']}:{config['qdrant_port']}", flush=True)
    print("Establishing connections...", flush=True)
    
    ch_client = connect_clickhouse(config)
    qdrant_client = connect_qdrant(config)
    
    if "gold_schema" in collections_to_init:
        embedding_model = get_embedding_model()
    else:
        embedding_model = None
    
    print()
    
    results = {}
    
    # Initialize collections
    if "gold_schema" in collections_to_init:
        try:
            print(flush=True)
            point_count, status = init_gold_schema_collection(
                ch_client, qdrant_client, embedding_model, force=args.force
            )
            results["gold_schema"] = (point_count, status)
        except Exception as e:
            print(f"ERROR initializing gold_schema: {e}", flush=True)
            sys.exit(1)
    
    if "gold_query_cache" in collections_to_init:
        try:
            print(flush=True)
            point_count, status = init_gold_query_cache_collection(
                qdrant_client, force=args.force
            )
            results["gold_query_cache"] = (point_count, status)
        except Exception as e:
            print(f"ERROR initializing gold_query_cache: {e}", flush=True)
            sys.exit(1)
    
    # Print summary
    print_summary(results)
    print(f"Completed at {datetime.now().isoformat()}", flush=True)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
