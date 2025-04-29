import uuid
from qdrant_client import QdrantClient, models as rest
from typing import List, Dict, Any, Optional
import config
from interfaces.vector_store import VectorStoreBase
from interfaces.llm import LLMBase

class QdrantVectorStore(VectorStoreBase):
    def __init__(self, uri: str, llm_client: Optional[LLMBase] = None):
        self.uri = uri
        self.llm_client = llm_client
        # Initialize client immediately
        self._client = self._get_client()

    def _get_client(self) -> QdrantClient:
        """Initialize and return the Qdrant client."""
        return QdrantClient(
            url=self.uri,
            timeout=60.0
        )

    def create_point_struct(self, point_id: str, vector: List[float], payload: Dict[str, Any]) -> rest.PointStruct:
        """Create a PointStruct with the given parameters."""
        if not self._client:
            self._client = self._get_client()
        return rest.PointStruct(id=point_id, vector=vector, payload=payload)

    def close_client(self):
        if self._client:
            # QdrantClient doesn't have an explicit close method in this version,
            # but we can reset the client to None to allow re-initialization if needed.
            self._client = None

    def ensure_collection_exists(self, collection_name: str, vector_size: int, distance_metric: str):
        client = self._get_client()
        try:
            client.get_collection(collection_name=collection_name)
            print(f"Collection '{collection_name}' already exists.")
        except Exception: # Catching generic Exception for simplicity, could be more specific
            print(f"Collection '{collection_name}' not found. Creating...")
            client.recreate_collection(
                collection_name=collection_name,
                vectors_config=rest.VectorParams(size=vector_size, distance=rest.Distance[distance_metric.upper()]),
            )
            print(f"Collection '{collection_name}' created.")

    def delete_collection(self, collection_name: str):
        client = self._get_client()
        try:
            client.delete_collection(collection_name=collection_name, timeout=60)
            print(f"Collection '{collection_name}' deleted.")
        except Exception: # Catching generic Exception for simplicity
            print(f"Collection '{collection_name}' not found. Skipping deletion.")

    def upsert_points(self, collection_name: str, points: List[rest.PointStruct]):
        if not points: return
        client = self._get_client()
        client.upsert(
            collection_name=collection_name,
            wait=True,
            points=points,
        )

    def search_points(self, collection_name: str, vector: List[float], query_filter: Optional[rest.Filter] = None, limit: int = 10):
        client = self._get_client()
        search_result = client.search(
            collection_name=collection_name,
            query_vector=vector,
            query_filter=query_filter,
            limit=limit,
            with_payload=True
        )
        return search_result

    def scroll_points(self, collection_name: str, scroll_filter: Optional[rest.Filter] = None, limit: int = 100):
        client = self._get_client()
        all_points = []
        offset = None
        while True:
            results, next_offset = client.scroll(
                collection_name=collection_name,
                scroll_filter=scroll_filter,
                limit=limit,
                offset=offset,
                with_payload=True,
                with_vectors=False
            )
            if not results: break
            all_points.extend(results)
            if next_offset is None: break
            offset = next_offset
        return all_points

    def generate_uuidv5(self, text: str) -> str:
        # Use a fixed namespace UUID for generating consistent UUIDs from text
        namespace_uuid = uuid.UUID('f9a7f9b0-0b9b-4b9c-8c9c-8b9b0b9b0b9b')
        return str(uuid.uuid5(namespace_uuid, text))

    # Implementation of VectorStoreBase methods
    def search_context(self, question: str, source_type: str, k: Optional[int] = None) -> List[Dict[str, Any]]:
        if not self.llm_client:
            raise ValueError("LLM client is required for search_context")
        
        # Get embedding for the question
        embedding = self.llm_client.get_embedding(question)
        
        # Create filter for source type
        query_filter = rest.Filter(
            must=[rest.FieldCondition(key="metadata.source_type", match=rest.MatchValue(value=source_type))]
        )
        
        # Get k from config if not provided
        if k is None:
            k = getattr(config, f"VECTOR_STORE_K_{source_type.upper()}", 5)
        
        # Search points
        search_results = self.search_points(
            collection_name=config.VECTOR_DB_COLLECTION_NAME,
            vector=embedding,
            query_filter=query_filter,
            limit=k
        )
        
        # Format results
        return [
            {
                "id": hit.id,
                "score": hit.score,
                "payload": hit.payload or {}
            }
            for hit in search_results
        ]

    def add_entry(self, text: str, metadata: Dict[str, Any], embedding: Optional[List[float]] = None):
        if embedding is None:
            if not self.llm_client:
                raise ValueError("LLM client is required when embedding is not provided")
            embedding = self.llm_client.get_embedding(text)
        
        point = rest.PointStruct(
            id=self.generate_uuidv5(text),
            vector=embedding,
            payload={
                "text": text,
                "metadata": metadata
            }
        )
        
        self.upsert_points(config.VECTOR_DB_COLLECTION_NAME, [point])

    def add_batch(self, texts: List[str], metadatas: List[Dict[str, Any]], embeddings: Optional[List[List[float]]] = None):
        if embeddings is None:
            if not self.llm_client:
                raise ValueError("LLM client is required when embeddings are not provided")
            embeddings = self.llm_client.get_embeddings_batch(texts)
        
        points = [
            rest.PointStruct(
                id=self.generate_uuidv5(text),
                vector=embedding,
                payload={
                    "text": text,
                    "metadata": metadata
                }
            )
            for text, metadata, embedding in zip(texts, metadatas, embeddings)
        ]
        
        self.upsert_points(config.VECTOR_DB_COLLECTION_NAME, points)

    def ensure_store_exists(self):
        self.ensure_collection_exists(
            collection_name=config.VECTOR_DB_COLLECTION_NAME,
            vector_size=config.VECTOR_DB_VECTOR_SIZE,
            distance_metric=config.VECTOR_DB_DISTANCE_METRIC
        )

    def empty_store(self):
        self.delete_collection(config.VECTOR_DB_COLLECTION_NAME)
        self.ensure_store_exists()

    def show_store_content_by_type(self, source_type: str, limit: int = 100) -> List[Dict[str, Any]]:
        scroll_filter = rest.Filter(must=[rest.FieldCondition(key="metadata.source_type", match=rest.MatchValue(value=source_type))])
        records = self.scroll_points(
            collection_name=config.VECTOR_DB_COLLECTION_NAME,
            scroll_filter=scroll_filter,
            limit=limit
        )
        return [{"id": rec.id, "payload": rec.payload or {}} for rec in records]