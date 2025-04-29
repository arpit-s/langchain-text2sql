from typing import Optional, List, Dict
from datetime import datetime

from interfaces.user_store import UserStoreBase
from interfaces.chat_store import ChatStoreBase

class UserService:
    def __init__(self, user_store: UserStoreBase, chat_store: ChatStoreBase):
        self.user_store = user_store
        self.chat_store = chat_store

    def delete_user(self, user_id: str) -> bool:
        sessions = self.chat_store.list_sessions(user_id)        
        for session in sessions:
            self.chat_store.delete_session(session["session_id"], user_id)
        return self.user_store.delete_user(user_id)
