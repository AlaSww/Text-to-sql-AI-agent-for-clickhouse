from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance


class QdrantStore:
    def __init__(self, host="localhost", port=6333, collection="clickhouse_schema"):
        self.client = QdrantClient(host=host, port=port)
        self.collection = collection

    def ensure_collection(self, vector_size: int):
        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection in collections:
            return

        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )