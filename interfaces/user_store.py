from abc import ABC, abstractmethod
from typing import Optional, List, Dict
from datetime import datetime

class UserStoreBase(ABC):
    @abstractmethod
    def create_user(self) -> str:
        """Create a new user and return the user_id."""
        pass

    @abstractmethod
    def get_user(self, user_id: str) -> Optional[Dict]:
        """Get user details by user_id."""
        pass

    @abstractmethod
    def list_users(self) -> List[Dict]:
        """List all users."""
        pass

    @abstractmethod
    def delete_user(self, user_id: str) -> bool:
        """Delete a user by user_id."""
        pass 