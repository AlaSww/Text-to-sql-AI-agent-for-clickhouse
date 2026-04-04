#!/usr/bin/env python3
"""Quick test of script imports and Qdrant connectivity."""
import sys
import time

print("Testing imports...", flush=True)
try:
    import clickhouse_connect
    print("  ✓ clickhouse_connect imported", flush=True)
except Exception as e:
    print(f"  ✗ Failed to import clickhouse_connect: {e}", flush=True)
    sys.exit(1)

try:
    from qdrant_client import QdrantClient
    print("  ✓ qdrant_client imported", flush=True)
except Exception as e:
    print(f"  ✗ Failed to import qdrant_client: {e}", flush=True)
    sys.exit(1)

try:
    from sentence_transformers import SentenceTransformer
    print("  ✓ sentence_transformers imported", flush=True)
except Exception as e:
    print(f"  ✗ Failed to import sentence_transformers: {e}", flush=True)
    sys.exit(1)

print("\nTesting Qdrant connectivity...", flush=True)
try:
    client = QdrantClient(host="localhost", port=6333)
    print("  Connecting to Qdrant...", flush=True)
    collections = client.get_collections()
    print(f"  ✓ Connected to Qdrant. Collections: {len(collections.collections)}", flush=True)
    for c in collections.collections:
        print(f"    - {c.name}", flush=True)
except Exception as e:
    print(f"  ✗ Failed to connect to Qdrant: {e}", flush=True)
    print("  → Is Qdrant running? Try: docker run -p 6333:6333 qdrant/qdrant", flush=True)
    sys.exit(1)

print("\nTesting ClickHouse connectivity...", flush=True)
try:
    ch = clickhouse_connect.get_client(host="localhost", port=8123)
    print("  Connecting to ClickHouse...", flush=True)
    result = ch.query("SELECT 1")
    print(f"  ✓ Connected to ClickHouse", flush=True)
except Exception as e:
    print(f"  ✗ Failed to connect to ClickHouse: {e}", flush=True)
    sys.exit(1)

print("\n✅ All systems ready!")
