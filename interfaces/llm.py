import json
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class LLMBase(ABC):
    @abstractmethod
    def generate_sql(self, question: str, chat_history: List[Dict[str, Any]], formatted_context: str) -> Dict[str, Any]:
        """
        Generates a SQL query based on the question, chat history, and context.
        """
        pass

    @abstractmethod
    def summarize(self, question: str, results: List[Dict[str, Any]]) -> str:
        """
        Summarizes the results of a SQL query based on the question.
        """
        pass

    @abstractmethod
    def comment_ddl(self, ddl_string: str, sql_dialect: str, sample_data: Optional[List[Dict[str, Any]]] = None) -> Optional[str]:
        """
        Adds comments to a DDL string.
        """
        pass

    @abstractmethod
    def get_embedding(self, text: str) -> List[float]:
        """
        Generates an embedding for a given text.
        """
        pass

    @abstractmethod
    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generates embeddings for a batch of texts.
        """
        pass