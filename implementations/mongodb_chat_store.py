import json
from datetime import datetime, timezone
from bson import ObjectId
from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory
from langchain_core.messages import BaseMessage, message_to_dict, messages_from_dict, HumanMessage, AIMessage
from typing import List, Dict, Any, Optional
import pymongo
import uuid

import config
from interfaces.chat_store import ChatStoreBase

class MongoDBChatStore(ChatStoreBase):
    """
    MongoDB implementation of chat store.
    
    This store manages:
    1. Chat sessions (create, list, delete)
    2. Session-user mapping (ownership)
    3. Chat messages (add, get, update)
    
    The session-user mapping is stored in the 'session_user_map' collection
    to maintain session ownership and enable efficient querying of sessions by user.
    """
    def __init__(self, uri: str, db_name: str, collection_name: str):
        self.uri = uri
        self.db_name = db_name
        self.collection_name = collection_name
        self._mongo_client = None
        self._db = None
        self._messages_collection = None
        self._session_user_map_collection = None

    def _get_collection(self):
        if self._mongo_client is None:
            try:
                self._mongo_client = pymongo.MongoClient(self.uri, serverSelectionTimeoutMS=5000)
                self._mongo_client.admin.command('ismaster')
                self._db = self._mongo_client[self.db_name]
                self._messages_collection = self._db[self.collection_name]
                self._session_user_map_collection = self._db['session_user_map']
                print("MongoDB connection successful.")
            except Exception as e:
                print(f"MongoDB connection failed: {e}")
                self._mongo_client = None
                self._db = None
                self._messages_collection = None
                self._session_user_map_collection = None
        return self._messages_collection

    def close_client(self):
        if self._mongo_client:
            self._mongo_client.close()
            self._mongo_client = None
            self._db = None
            self._messages_collection = None
            self._session_user_map_collection = None

    def create_session(self, user_id: str) -> str:
        """
        Create a new session for a user and return the session_id.
        
        This method:
        1. Generates a new session ID
        2. Creates a session-user mapping in the session_user_map collection
        3. Returns the session ID for future use
        """
        collection = self._get_collection()
        if collection is None:
            print("Warning: MongoDB not connected. Cannot create session.")
            return None
        
        session_id = str(uuid.uuid4())
        self._session_user_map_collection.insert_one({
            "session_id": session_id,
            "user_id": user_id,
            "created_at": datetime.now(timezone.utc)
        })
        return session_id

    def validate_session(self, session_id: str, user_id: str) -> bool:
        """
        Validate if a session belongs to a user.
        
        This method checks the session_user_map collection to verify
        that the session is owned by the specified user.
        """
        if self._session_user_map_collection is None:
            return False
        return self._session_user_map_collection.find_one({
            "session_id": session_id,
            "user_id": user_id
        }) is not None

    def list_sessions(self, user_id: str) -> List[Dict]:
        """
        List all sessions for a given user.
        
        This method queries the session_user_map collection to find
        all sessions owned by the specified user.
        """
        if self._session_user_map_collection is None:
            return []
        
        try:
            sessions = list(self._session_user_map_collection.find(
                {"user_id": user_id},
                {"_id": 0, "session_id": 1, "created_at": 1}
            ))
            return sessions
        except Exception as e:
            print(f"Error listing sessions for user {user_id}: {e}")
            return []

    def add_message(self, session_id: str, message=None, user_id: str = None, content: str = None, is_user: bool = None) -> Optional[str]:
        """
        Add a message to the chat history for a given session.
        
        This method:
        1. Validates session ownership if user_id is provided
        2. Stores the message in the messages collection
        3. Returns the message ID
        
        Args:
            session_id: The session ID to add the message to
            message: The message as a BaseMessage object (positional or named)
            user_id: The user ID for session validation (optional)
            content: The message content as a string (if not using BaseMessage)
            is_user: Whether the message is from the user (required if using content string)
        """
        if self._messages_collection is None:
            print("Warning: MongoDB not connected. Cannot add message.")
            return None
        
        # Validate session ownership if user_id is provided
        if user_id and not self.validate_session(session_id, user_id):
            print(f"Warning: Session {session_id} does not belong to user {user_id}")
            return None
        
        try:
            # Prepare message document
            message_doc = {
                "session_id": session_id,
                "timestamp": datetime.now(timezone.utc)
            }
            
            # Handle different message input types
            if message is not None:
                # Handle BaseMessage object
                message_doc.update({
                    "type": "human" if isinstance(message, HumanMessage) else "assistant",
                    "content": message.content
                })
            elif content is not None and is_user is not None:
                # Handle direct content and type
                message_doc.update({
                    "type": "human" if is_user else "assistant",
                    "content": content
                })
            else:
                print("Warning: Invalid message format. Must provide either message object or content+is_user.")
                return None
            
            # Insert message and return ID
            result = self._messages_collection.insert_one(message_doc)
            return str(result.inserted_id)
            
        except Exception as e:
            print(f"Error adding message: {e}")
            return None

    def get_message_by_id(self, message_id_str: str, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a message by its ID, ensuring it belongs to the user.
        
        This method:
        1. Finds the message by ID
        2. Validates that the message's session belongs to the user
        3. Returns the message if valid
        """
        collection = self._get_collection()
        if collection is None or not message_id_str:
            return None
        
        try:
            message = collection.find_one({"_id": ObjectId(message_id_str)})
            if message and self.validate_session(message.get("session_id"), user_id):
                return message
            return None
        except Exception as e:
            print(f"Error retrieving message by ID {message_id_str}: {e}")
            return None

    def update_message_history_content(self, message_id_str: str, new_history_content_json_str: str, user_id: str) -> bool:
        """
        Update the content of a message in the chat history.
        
        This method:
        1. Finds the message by ID
        2. Validates that the message's session belongs to the user
        3. Updates the message content if valid
        """
        collection = self._get_collection()
        if collection is None or not message_id_str:
            print("Warning: MongoDB not connected or no ID provided. Cannot update message.")
            return False
        
        try:
            message = collection.find_one({"_id": ObjectId(message_id_str)})
            if not message or not self.validate_session(message.get("session_id"), user_id):
                return False

            result = collection.update_one(
                {"_id": ObjectId(message_id_str)},
                {"$set": {"history": new_history_content_json_str}}
            )
            return result.modified_count > 0
        except Exception as e:
            print(f"Error updating message by ID {message_id_str}: {e}")
            return False

    def get_messages(self, session_id: str, user_id: str) -> List[BaseMessage]:
        """
        Get all messages for a given session.
        
        Args:
            session_id: The session ID to get messages for
            user_id: The user ID for session validation
            
        Returns:
            List of BaseMessage objects
        """
        if not self._mongo_client:
            raise Exception("MongoDB not connected")
            
        # Validate session ownership
        if not self.validate_session(session_id, user_id):
            raise Exception(f"Session {session_id} does not belong to user {user_id}")
            
        try:
            # Get messages from MongoDB
            messages = list(self._messages_collection.find(
                {"session_id": session_id},
                sort=[("timestamp", pymongo.ASCENDING)]
            ))
            
            # Convert stored messages to BaseMessage objects
            result = []
            for msg in messages:
                if msg["type"] == "human":
                    result.append(HumanMessage(content=msg["content"]))
                else:
                    result.append(AIMessage(content=msg["content"]))
                    
            return result
            
        except Exception as e:
            print(f"Error getting messages: {e}")
            return []

    def delete_session(self, session_id: str, user_id: str) -> bool:
        """
        Delete all messages for a given session.
        
        This method:
        1. Validates session ownership
        2. Deletes all messages for the session
        3. Deletes the session-user mapping
        """
        collection = self._get_collection()
        if collection is None:
            print("Warning: ngoDB not connected. Cannot delete session.")
            return False
        
        if not self.validate_session(session_id, user_id):
            print(f"Warning: Session {session_id} does not belong to user {user_id}")
            return False

        try:
            # Delete messages
            messages_result = collection.delete_many({"session_id": session_id})
            # Delete session mapping
            session_result = self._session_user_map_collection.delete_one({
                "session_id": session_id,
                "user_id": user_id
            })
            return messages_result.deleted_count > 0 or session_result.deleted_count > 0
        except Exception as e:
            print(f"Error deleting session {session_id}: {e}")
            return False

    def delete_messages(self, session_id: str) -> bool:
        """
        Delete all messages for a given session without deleting the session itself.
        
        This method:
        1. Deletes all messages associated with the session
        2. Returns True if any messages were deleted, False otherwise
        """
        collection = self._get_collection()
        if collection is None:
            print("Warning: MongoDB not connected. Cannot delete messages.")
            return False

        try:
            result = collection.delete_many({"session_id": session_id})
            return result.deleted_count > 0
        except Exception as e:
            print(f"Error deleting messages for session {session_id}: {e}")
            return False