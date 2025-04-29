import os
import json
from dotenv import load_dotenv
from typing import List, Dict, Any, Tuple

import config as config
from interfaces.llm import LLMBase
from interfaces.vector_store import VectorStoreBase
from interfaces.sql_client import SQLClientBase

class TrainingService:
    def __init__(self, llm_client: LLMBase, vector_store: VectorStoreBase, sql_client: SQLClientBase):
        self.llm_client = llm_client
        self.vector_store = vector_store
        self.sql_client = sql_client

    def train_from_file(self, file_path: str, file_type: str) -> int:
        """Trains the vector store from a given file."""
        added_count = 0
        try:
            with open(file_path, 'r') as f:
                if file_type == "query":
                    data = json.load(f)
                    texts = [item["question"] for item in data]
                    metadatas = [{"source_type": file_type, "sql": item["sql"]} for item in data]
                elif file_type == "documentation":
                    # Read entire file as one document
                    texts = [f.read().strip()]
                    metadatas = [{"source_type": file_type}]
                else:
                    print(f"Warning: Unsupported file type for training: {file_type}")
                    return 0

            if texts:
                embeddings = self.llm_client.get_embeddings_batch(texts)
                points_to_add = []
                for text, metadata, embedding in zip(texts, metadatas, embeddings):
                    # For DDL entries, normalize the text to ensure consistent UUIDs
                    if metadata.get('source_type') == 'ddl':
                        # Extract table name from metadata if available
                        table_name = metadata.get('table_name')
                        if table_name:
                            # Use table name as part of the UUID generation
                            normalized_text = f"ddl_{table_name}"
                        else:
                            # Fallback: normalize the text by removing extra whitespace and comments
                            normalized_text = ' '.join(text.split())
                    else:
                        normalized_text = text

                    point_id = self.vector_store.generate_uuidv5(normalized_text)
                    payload = {"text": text, "metadata": metadata}
                    point = self.vector_store.create_point_struct(point_id, embedding, payload)
                    points_to_add.append(point)

                # Use the upsert_points method from the vector store implementation
                self.vector_store.upsert_points(
                    collection_name=config.VECTOR_DB_COLLECTION_NAME,
                    points=points_to_add
                )
                added_count = len(points_to_add)

        except FileNotFoundError:
            print(f"Error: Training file not found at {file_path}")
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {file_path}")
        except Exception as e:
            print(f"An error occurred during training from file {file_path}: {e}")

        return added_count

    def train_from_sql_schema(self, dataset_id: str) -> int:
        """Trains the vector store from the SQL schema."""
        added_count = 0
        if not dataset_id:
            print("Dataset ID not provided. Skipping schema training.")
            return 0

        try:
            tables = self.sql_client.list_tables(dataset_id)
            if not tables:
                print(f"No tables found in dataset {dataset_id}. Skipping schema training.")
                return 0

            ddl_texts = []
            metadatas = []
            for table_id in tables:
                ddl = self.sql_client.get_formatted_ddl(dataset_id, table_id)
                if ddl:
                    # Optionally add comments to DDL using LLM
                    sample_data, _ = self.sql_client.get_sample_data(dataset_id, table_id, config.SQL_CLIENT_COMMENT_SAMPLE_SIZE)
                    commented_ddl = self.llm_client.comment_ddl(ddl, config.SQL_DIALECT, sample_data)
                    final_ddl = commented_ddl if commented_ddl else ddl

                    ddl_texts.append(final_ddl)
                    metadatas.append({"source_type": "ddl", "table_name": table_id})

            if ddl_texts:
                embeddings = self.llm_client.get_embeddings_batch(ddl_texts)
                points_to_add = []
                for text, metadata, embedding in zip(ddl_texts, metadatas, embeddings):
                    # For DDL entries, normalize the text to ensure consistent UUIDs
                    if metadata.get('source_type') == 'ddl':
                        # Extract table name from metadata if available
                        table_name = metadata.get('table_name')
                        if table_name:
                            # Use table name as part of the UUID generation
                            normalized_text = f"ddl_{table_name}"
                        else:
                            # Fallback: normalize the text by removing extra whitespace and comments
                            normalized_text = ' '.join(text.split())
                    else:
                        normalized_text = text

                    point_id = self.vector_store.generate_uuidv5(normalized_text)
                    payload = {"text": text, "metadata": metadata}
                    point = self.vector_store.create_point_struct(point_id, embedding, payload)
                    points_to_add.append(point)

                # Use the upsert_points method from the vector store implementation
                self.vector_store.upsert_points(
                    collection_name=config.VECTOR_DB_COLLECTION_NAME,
                    points=points_to_add
                )
                added_count = len(points_to_add)

        except Exception as e:
            print(f"An error occurred during schema training: {e}")

        return added_count

    def populate_vector_database(self, dataset_id: str):
        """Populate the vector database with all available training data."""
        print("\n=== Populating Vector Database ===")

        # Ensure vector store exists
        self.vector_store.ensure_store_exists()

        # Train from SQL schema
        if dataset_id:
            print("\nTraining from SQL schema...")
            added_count = self.train_from_sql_schema(dataset_id)
            print(f"Successfully embedded {added_count} documents from SQL schema.")

        # Train from training set files
        training_dir = config.TRAINING_SET_DIR
        if os.path.exists(training_dir):
            print("\nTraining from training set files...")
            # Map file types to their actual file names
            file_type_mapping = {
                "query": "queries.json",
                "documentation": "docs.txt"
            }

            for file_type, file_name in file_type_mapping.items():
                file_path = os.path.join(training_dir, file_name)
                if os.path.exists(file_path):
                    print(f"Training from {file_path}...")
                    added_count = self.train_from_file(file_path, file_type)
                    print(f"Successfully embedded {added_count} documents from {file_path}.")

        print("\n=== Vector Database Population Complete ===")
        # The run_show_action call will be moved to the actions file

        print("\nClosing connections...")
        self.vector_store.close_client()
        self.sql_client.close_client()
        # The chat store client is not used in training, so no need to close it here.
        print("Connections closed.")