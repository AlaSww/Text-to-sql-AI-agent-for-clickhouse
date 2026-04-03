from llama_index.embeddings.openai import OpenAIEmbedding
from qdrant_client.models import PointStruct
from app.retrieval.qdrant_store import QdrantStore


class SchemaLoader:
    def __init__(self, qdrant: QdrantStore, embedding_model: OpenAIEmbedding):
        self.qdrant = qdrant
        self.embed = embedding_model

    def index_schema(self, schema_texts: list[str]):
        vectors = [self.embed.get_text_embedding(t) for t in schema_texts]

        points = []
        for i, (text, vec) in enumerate(zip(schema_texts, vectors)):
            points.append(PointStruct(id=i, vector=vec, payload={"text": text}))

        self.qdrant.client.upsert(
            collection_name=self.qdrant.collection,
            points=points
        )