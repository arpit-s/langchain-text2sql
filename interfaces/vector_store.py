from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class VectorStoreBase(ABC):
    @abstractmethod
    def search_context(self, question: str, source_type: str, k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Searches the vector store for relevant context.
        """
        pass

    @abstractmethod
    def add_entry(self, text: str, metadata: Dict[str, Any], embedding: Optional[List[float]] = None):
        """
        Adds a single entry to the vector store.
        """
        pass

    @abstractmethod
    def add_batch(self, texts: List[str], metadatas: List[Dict[str, Any]], embeddings: Optional[List[List[float]]] = None):
        """
        Adds a batch of entries to the vector store.
        """
        pass

    @abstractmethod
    def ensure_store_exists(self):
        """
        Ensures the vector store exists.
        """
        pass

    @abstractmethod
    def empty_store(self):
        """
        Empties the vector store.
        """
        pass

    @abstractmethod
    def show_store_content_by_type(self, source_type: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Shows content in the vector store filtered by source type.
        """
        pass

    @abstractmethod
    def close_client(self):
        """
        Closes the vector store client.
        """
        pass