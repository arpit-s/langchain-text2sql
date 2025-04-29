import os
import json
from dotenv import load_dotenv
from typing import Optional, List, Dict, Tuple
from langchain_core.messages import HumanMessage, AIMessage

import config
from implementations.gemini_llm import GeminiLLM
from implementations.qdrant_vector_store import QdrantVectorStore
from implementations.mongodb_chat_store import MongoDBChatStore
from implementations.mongodb_user_store import MongoDBUserStore
from implementations.bigquery_sql_client import BigQuerySqlClient
from services.chat_service import ChatService
from services.user_service import UserService
from services.training_service import TrainingService
from actions.vector_store_actions import VectorStoreActions

def get_choice(prompt: str, options: List[str]) -> str:
    """Get user choice from given options."""
    while True:
        choice = input(f"{prompt} ({'/'.join(options)}): ").lower()
        if choice in options:
            return choice
        print(f"Invalid choice. Please enter one of: {', '.join(options)}")

def display_messages(messages: List[Dict]):
    """Display messages in user: message \n ai: message format."""
    for msg in messages:
        if isinstance(msg, HumanMessage):
            print(f"User: {msg.content}")
        elif isinstance(msg, AIMessage):
            try:
                content = json.loads(msg.content)
                print(f"AI: {content.get('answer', '')}")
                if content.get('sql'):
                    print(f"Generated SQL: {content['sql']}")
            except json.JSONDecodeError:
                print(f"AI: {msg.content}")

def initialize_dependencies() -> Tuple[MongoDBUserStore, MongoDBChatStore, UserService, ChatService, VectorStoreActions]:
    """Initialize all required stores and services."""
    try:
        user_store = MongoDBUserStore(uri=config.CHAT_STORE_URI, db_name=config.CHAT_STORE_DB_NAME)
        chat_store = MongoDBChatStore(uri=config.CHAT_STORE_URI, db_name=config.CHAT_STORE_DB_NAME, collection_name=config.CHAT_STORE_COLLECTION_NAME)
        user_service = UserService(user_store=user_store, chat_store=chat_store)

        llm_client = GeminiLLM(
            api_key=config.API_KEY,
            llm_model=config.LLM_MODEL,
            embedding_model_name=config.EMBEDDING_MODEL_NAME
        )

        vector_store = QdrantVectorStore(
            uri=config.VECTOR_DB_URI,
            llm_client=llm_client
        )

        sql_client = BigQuerySqlClient(
            project_id=config.SQL_CLIENT_PROJECT_ID,
            service_account_path=config.SQL_CLIENT_SERVICE_ACCOUNT_PATH
        )

        chat_service = ChatService(
            llm_client=llm_client,
            vector_store=vector_store,
            chat_store=chat_store,
            sql_client=sql_client
        )

        vector_store_actions = VectorStoreActions(
            training_service=TrainingService(
                llm_client=llm_client,
                vector_store=vector_store,
                sql_client=sql_client
            ),
            vector_store=vector_store
        )

        return user_store, chat_store, user_service, chat_service, vector_store_actions
    except Exception as e:
        print(f"Error initializing dependencies: {e}")
        raise

def handle_entity_management(store, entity_type: str, create_func, list_func, delete_func=None) -> str:
    """Generic function to handle user/session management."""
    choice = get_choice(f"Do you want to create a new {entity_type}, resume an existing one, or delete one?", ['new', 'resume', 'delete'])
    
    if choice == 'new':
        entity_id = create_func()
        print(f"Created new {entity_type} with ID: {entity_id}")
        return entity_id
    elif choice == 'delete':
        if not delete_func:
            print(f"Delete operation not supported for {entity_type}")
            return handle_entity_management(store, entity_type, create_func, list_func, delete_func)
            
        entities = list_func()
        if not entities:
            print(f"No existing {entity_type}s found to delete.")
            return handle_entity_management(store, entity_type, create_func, list_func, delete_func)

        print(f"\nExisting {entity_type}s:")
        for i, entity in enumerate(entities, 1):
            print(f"{i}. {entity_type.title()} ID: {entity[f'{entity_type}_id']}")
        
        while True:
            try:
                choice = int(input(f"\nSelect {entity_type} number to delete (or 0 to cancel): "))
                if choice == 0:
                    return handle_entity_management(store, entity_type, create_func, list_func, delete_func)
                if 1 <= choice <= len(entities):
                    entity_id = entities[choice-1][f'{entity_type}_id']
                    if delete_func(entity_id):
                        print(f"Successfully deleted {entity_type} {entity_id}")
                    else:
                        print(f"Failed to delete {entity_type} {entity_id}")
                    return handle_entity_management(store, entity_type, create_func, list_func, delete_func)
                print("Invalid choice. Please try again.")
            except ValueError:
                print("Please enter a valid number.")
    else:  # resume
        entities = list_func()
        if not entities:
            print(f"No existing {entity_type}s found. Creating new {entity_type}...")
            entity_id = create_func()
            print(f"Created new {entity_type} with ID: {entity_id}")
            return entity_id

        print(f"\nExisting {entity_type}s:")
        for i, entity in enumerate(entities, 1):
            print(f"{i}. {entity_type.title()} ID: {entity[f'{entity_type}_id']}")
        
        while True:
            try:
                choice = int(input(f"\nSelect {entity_type} number: "))
                if 1 <= choice <= len(entities):
                    return entities[choice-1][f'{entity_type}_id']
                print("Invalid choice. Please try again.")
            except ValueError:
                print("Please enter a valid number.")

def handle_sql_correction(chat_service, session_id, question, user_id, initial_sql, initial_error):
    """Handle SQL correction loop."""
    while True:
        print("\n--- SQL Requires Correction ---")
        print(f"SQL:\n{initial_sql}")
        print(f"Error: {initial_error}")
        corrected_sql = input("Enter corrected SQL query (or type 'cancel'): ")
        if corrected_sql.lower() == 'cancel':
            return None
        
        response = chat_service.finalize_response(session_id, question, corrected_sql, user_id)
        if response.get("status") == "success":
            return response
        
        initial_sql = response.get("sql", corrected_sql)
        initial_error = response.get("error_message", "Unknown error.")

def handle_chat_interaction(chat_service, chat_store, session_id, user_id):
    """Main chat interaction loop."""
    while True:
        question = input("\nEnter your question (or 'quit' to exit): ")
        if question.lower() == 'quit':
            break

        # Generate SQL
        gen_response = chat_service.generate_sql_only(session_id, question, user_id)
        if gen_response.get("status") != "success":
            print("Initial SQL generation failed validation after retries.")
            final_response = handle_sql_correction(
                chat_service, session_id, question, user_id,
                gen_response["sql"], gen_response.get("error_message", "Unknown error.")
            )
            if not final_response:
                print("SQL correction cancelled.")
                continue
        else:
            # Get user approval
            sql = gen_response["sql"]
            print("\nGenerated SQL:")
            print(sql)
            
            if get_choice("Is the generated SQL query OK?", ['yes', 'no']) == 'no':
                sql = input("Enter your updated SQL query: ")
            
            final_response = chat_service.finalize_response(
                session_id, question, sql, user_id,
                follow_ups=gen_response.get("follow_up", [])  # Pass follow-ups through
            )
            if final_response.get("status") != "success":
                final_response = handle_sql_correction(
                    chat_service, session_id, question, user_id,
                    final_response["sql"], final_response.get("error_message", "Unknown error.")
                )
                if not final_response:
                    print("SQL correction cancelled.")
                    continue

        # Display results
        print("\nAI Response:")
        if final_response.get('results'):
            print("\nQuery Results:")
            print(final_response['results'])
        if final_response.get('summary'):
            print("\nSummary:")
            print(final_response['summary'])
        if final_response.get('follow_up'):
            print("\nSuggested follow-up questions:")
            for i, q in enumerate(final_response['follow_up'], 1):
                print(f"{i}. {q}")

        # Add to vector DB if enabled
        executed_sql = final_response.get("sql")
        if config.ADD_QUESTION_QUERY_TO_VECTOR_DB and question and executed_sql and not executed_sql.startswith("-- Error"):
            print("Adding question and executed SQL to vector database...")
            chat_service.add_query_to_vector_db(question, executed_sql)

def cleanup_connections(vector_store: QdrantVectorStore, chat_store: MongoDBChatStore, sql_client: BigQuerySqlClient) -> None:
    """Clean up all connections."""
    print("\nClosing connections...")
    vector_store.close_client()
    chat_store.close_client()
    sql_client.close_client()
    print("Connections closed.")

def handle_vector_db_training(chat_service: ChatService) -> None:
    """Handle vector database training options."""
    print("\n=== Vector Database Training ===")
    print("1. Train from SQL schema")
    print("2. Train from training files")
    print("3. Train from both")
    print("4. Back to main menu")
    
    try:
        choice = int(input("\nSelect training option (1-4): "))
        if choice == 4:
            return
        
        if choice not in [1, 2, 3]:
            print("Invalid choice. Please try again.")
            return
            
        # Initialize training service
        training_service = TrainingService(
            llm_client=chat_service.llm_client,
            vector_store=chat_service.vector_store,
            sql_client=chat_service.sql_client
        )
        
        # Train from schema if selected
        if choice in [1, 3]:
            dataset_id = input("Enter dataset ID for schema training: ")
            added_count = training_service.train_from_sql_schema(dataset_id)
            print(f"Successfully embedded {added_count} documents from SQL schema.")
        
        # Train from files if selected
        if choice in [2, 3]:
            training_dir = config.TRAINING_SET_DIR
            if os.path.exists(training_dir):
                for file_type, file_name in {"query": "queries.json", "documentation": "docs.txt"}.items():
                    file_path = os.path.join(training_dir, file_name)
                    if os.path.exists(file_path):
                        added_count = training_service.train_from_file(file_path, file_type)
                        print(f"Successfully embedded {added_count} documents from {file_path}")
            else:
                print(f"Training directory not found at {training_dir}")
        
        print("\nTraining completed successfully!")
        
    except ValueError:
        print("Please enter a valid number.")
    except Exception as e:
        print(f"An error occurred during training: {e}")

def main():
    print("Setting up dependencies...")
    try:
        user_store, chat_store, user_service, chat_service, vector_store_actions = initialize_dependencies()
        print("Dependencies setup complete.")

        while True:
            print("\nMain Menu:")
            print("1. Chat with SQL Assistant")
            print("2. Train Vector Database")
            print("3. Empty Vector Database")
            print("4. Show Vector Database Contents")
            print("5. Exit")
            
            try:
                choice = int(input("\nSelect option (1-5): "))
                if choice == 5:
                    break
                elif choice == 4:
                    vector_store_actions.run_show_action()
                elif choice == 3:
                    vector_store_actions.run_empty_action()
                elif choice == 2:
                    handle_vector_db_training(chat_service)
                elif choice == 1:
                    # User and session management
                    user_id = handle_entity_management(
                        user_store, "user", 
                        user_store.create_user,
                        user_store.list_users,
                        user_store.delete_user
                    )
                    
                    session_id = handle_entity_management(
                        chat_store, "session",
                        lambda: chat_store.create_session(user_id),
                        lambda: chat_store.list_sessions(user_id),
                        chat_store.delete_session
                    )

                    # Display existing messages if resuming session
                    messages = chat_store.get_messages(session_id, user_id)
                    if messages:
                        print("\nPrevious conversation:")
                        display_messages(messages)

                    # Main chat loop
                    handle_chat_interaction(chat_service, chat_store, session_id, user_id)
                else:
                    print("Invalid choice. Please try again.")
                    
            except ValueError:
                print("Please enter a valid number.")
            except Exception as e:
                print(f"An error occurred: {e}")

        # Cleanup
        cleanup_connections(chat_service.vector_store, chat_store, chat_service.sql_client)

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()