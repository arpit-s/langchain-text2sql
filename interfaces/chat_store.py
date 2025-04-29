from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from langchain_core.messages import BaseMessage

class ChatStoreBase(ABC):
    @abstractmethod
    def create_session(self, user_id: str) -> str:
        """Create a new session for a user and return the session_id."""
        pass

    @abstractmethod
    def validate_session(self, session_id: str, user_id: str) -> bool:
        """Validate if a session belongs to a user."""
        pass

    @abstractmethod
    def list_sessions(self, user_id: str) -> List[Dict]:
        """List all sessions for a given user."""
        pass

    @abstractmethod
    def add_message(self, session_id: str, message: BaseMessage, user_id: str) -> Optional[str]:
        """Add a message to the chat history for a given session."""
        pass

    @abstractmethod
    def get_message_by_id(self, message_id_str: str, user_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a message by its ID, ensuring it belongs to the user."""
        pass

    @abstractmethod
    def update_message_history_content(self, message_id_str: str, new_history_content_json_str: str, user_id: str) -> bool:
        """Update the content of a message in the chat history."""
        pass

    @abstractmethod
    def get_messages(self, session_id: str, user_id: str) -> List[BaseMessage]:
        """Retrieve all messages for a given session."""
        pass

    @abstractmethod
    def delete_session(self, session_id: str, user_id: str) -> bool:
        """Delete all messages for a given session."""
        pass