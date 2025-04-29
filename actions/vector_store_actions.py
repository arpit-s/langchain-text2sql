import os
import json
from typing import List, Dict, Any, Optional

import config as config
from services.training_service import TrainingService
from interfaces.vector_store import VectorStoreBase # Need VectorStoreBase for show/empty actions

class VectorStoreActions:
    def __init__(self, training_service: TrainingService, vector_store: VectorStoreBase):
        self.training_service = training_service
        self.vector_store = vector_store

    def train_from_file(self, file_path: str, data_type: str) -> int:
        """Trains the vector store from a given file using the training service."""
        print(f"--- Training from file: {file_path} (Type: {data_type}) ---")
        added_count = self.training_service.train_from_file(file_path, data_type)
        print(f"--- Finished training from file: {file_path} ---")
        return added_count

    def train_from_sql_schema(self, dataset_id: str) -> int:
        """Trains the vector store from the SQL schema using the training service."""
        print("--- Training from SQL Schema ---")
        added_count = self.training_service.train_from_sql_schema(dataset_id)
        print("--- Finished training from SQL Schema ---")
        return added_count

    def run_empty_action(self):
        """Empties the vector store using the vector store implementation."""
        print(f"--- Emptying Vector Store (Collection: {config.VECTOR_DB_COLLECTION_NAME}) ---")
        self.vector_store.empty_store()
        print("--- Vector Store Emptied ---")

    def run_show_action(self):
        """Shows vector store contents using the vector store implementation."""
        print(f"--- Showing Vector Store Contents (Collection: {config.VECTOR_DB_COLLECTION_NAME}) ---")
        limit_per_type = 100 # This should probably be a config parameter
        all_results = {}

        for source_type in ["ddl", "documentation", "query"]: # This should probably be a config parameter
            print(f"\nFetching type: {source_type}...")
            points_data = self.vector_store.show_store_content_by_type(source_type, limit=limit_per_type)
            print(f"  Found {len(points_data)} entries for {source_type} (showing up to {limit_per_type}).")
            all_results[source_type] = points_data

        print("\n--- Vector Store Contents (JSON Output) ---")
        print(json.dumps(all_results, indent=2, default=str))
        print("--- Finished Showing Vector Store ---")

    # The train_from_chat_history action from the original file is related to the chat confirmation flow
    # and might need to be adapted or moved depending on how that flow is refactored.
    # For now, we will omit it as the plan focuses on the core training workflow.