import uuid
from datetime import datetime, timezone
import pymongo
from typing import Optional, List, Dict

from interfaces.user_store import UserStoreBase
import config

class MongoDBUserStore(UserStoreBase):
    def __init__(self, uri: str, db_name: str):
        self.uri = uri
        self.db_name = db_name
        self._client = None
        self._db = None
        self._collection = None

    def _get_collection(self):
        if self._client is None:
            try:
                self._client = pymongo.MongoClient(self.uri, serverSelectionTimeoutMS=5000)
                self._client.admin.command('ismaster')
                self._db = self._client[self.db_name]
                self._collection = self._db['users']
                print("MongoDB user store connection successful.")
            except Exception as e:
                print(f"MongoDB user store connection failed: {e}")
                self._client = None
                self._db = None
                self._collection = None
        return self._collection

    def close_client(self):
        if self._client:
            self._client.close()
            self._client = None
            self._db = None
            self._collection = None

    def create_user(self) -> str:
        """Create a new user and return the user_id."""
        collection = self._get_collection()
        if collection is None:
            print("Warning: MongoDB not connected. Cannot create user.")
            return None
        
        user_id = str(uuid.uuid4())
        collection.insert_one({
            "user_id": user_id,
            "created_at": datetime.now(timezone.utc)
        })
        return user_id

    def get_user(self, user_id: str) -> Optional[Dict]:
        """Get user details by user_id."""
        collection = self._get_collection()
        if collection is None:
            return None
        
        try:
            return collection.find_one({"user_id": user_id})
        except Exception as e:
            print(f"Error retrieving user {user_id}: {e}")
            return None

    def list_users(self) -> List[Dict]:
        """List all users."""
        collection = self._get_collection()
        if collection is None:
            return []
        
        try:
            return list(collection.find({}, {"_id": 0}))
        except Exception as e:
            print(f"Error listing users: {e}")
            return []

    def delete_user(self, user_id: str) -> bool:
        """Delete a user by user_id."""
        collection = self._get_collection()
        if collection is None:
            return False
        
        try:
            result = collection.delete_one({"user_id": user_id})
            return result.deleted_count > 0
        except Exception as e:
            print(f"Error deleting user {user_id}: {e}")
            return False 