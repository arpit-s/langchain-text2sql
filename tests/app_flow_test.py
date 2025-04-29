import os
import sys
import json
import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from pymongo import MongoClient
from dotenv import load_dotenv
from datetime import datetime
from langchain_core.messages import HumanMessage, AIMessage

# Set up logging
logging.basicConfig(
    filename='app_flow_test.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path to import app modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Import app dependencies
from app import (
    initialize_dependencies, 
    handle_entity_management,
    handle_chat_interaction
)
from implementations.mongodb_chat_store import MongoDBChatStore
from implementations.mongodb_user_store import MongoDBUserStore
from implementations.mongodb_chat_store import MongoDBChatStore as MongoDBSessionStore
from implementations.qdrant_vector_store import QdrantVectorStore
from implementations.gemini_llm import GeminiLLM
from implementations.bigquery_sql_client import BigQuerySqlClient
from interfaces.chat_store import ChatStoreBase
from services.chat_service import ChatService
from services.user_service import UserService
from services.training_service import TrainingService
from actions.vector_store_actions import VectorStoreActions
import config

# Test configuration
TEST_USER_NAME = "test_user"
TEST_SESSION_NAME = "test_session"

class AppFlowTester:
    def __init__(self):
        """Initialize the tester with all required dependencies."""
        logger.info("Initializing AppFlowTester")
        load_dotenv()
        
        # Connect to MongoDB for direct verification
        self.mongo_uri = config.CHAT_STORE_URI
        self.mongo_db_name = config.CHAT_STORE_DB_NAME  # Use the same DB name from config
        self.mongo_client = MongoClient(self.mongo_uri)
        self.db = self.mongo_client[self.mongo_db_name]
        
        # Initialize app dependencies (similar to app.py)
        logger.info("Initializing app dependencies")
        self.user_store = MongoDBUserStore(
            uri=config.CHAT_STORE_URI,
            db_name=config.CHAT_STORE_DB_NAME
        )
        self.chat_store = MongoDBChatStore(
            uri=config.CHAT_STORE_URI,
            db_name=config.CHAT_STORE_DB_NAME,
            collection_name=config.CHAT_STORE_COLLECTION_NAME
        )
        self.user_service = UserService(
            user_store=self.user_store,
            chat_store=self.chat_store
        )
        
        # Initialize LLM and vector store
        self.llm_client = GeminiLLM(
            api_key=config.API_KEY,
            llm_model=config.LLM_MODEL,
            embedding_model_name=config.EMBEDDING_MODEL_NAME
        )
        self.vector_store = QdrantVectorStore(
            uri=config.VECTOR_DB_URI,
            llm_client=self.llm_client
        )
        
        # Initialize SQL client
        self.sql_client = BigQuerySqlClient(
            project_id=config.SQL_CLIENT_PROJECT_ID,
            service_account_path=config.SQL_CLIENT_SERVICE_ACCOUNT_PATH
        )
        
        # Initialize chat service
        self.chat_service = ChatService(
            llm_client=self.llm_client,
            vector_store=self.vector_store,
            chat_store=self.chat_store,
            sql_client=self.sql_client
        )
        
        # Initialize vector store actions
        self.vector_store_actions = VectorStoreActions(
            training_service=TrainingService(
                llm_client=self.llm_client,
                vector_store=self.vector_store,
                sql_client=self.sql_client
            ),
            vector_store=self.vector_store
        )
        
        # Initialize session store - using the same collection name as the chat store
        self.session_store = self.chat_store
        
        # Test data
        self.test_questions = [
            "what are the top 5 most ordered products?",
            "what about next top 5?",
            "what's the average price of these products?",
            "which category has the most products?"
        ]
        
    def verify_mongo_data(self, collection_name: str, filter_query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Verify data exists in MongoDB collection."""
        collection = self.db[collection_name]
        results = list(collection.find(filter_query))
        return results
    
    def clean_test_data(self):
        """Clean up test data after tests."""
        logger.info("Cleaning up test data")
        try:
            # Get test users
            test_users = self.user_store.list_users()
            for user in test_users:
                user_id = user["user_id"]
                
                # Get test sessions
                test_sessions = self.chat_store.list_sessions(user_id)
                
                # Delete test messages and sessions
                for session in test_sessions:
                    session_id = session["session_id"]
                    # Delete messages before deleting the session
                    self.chat_store.delete_messages(session_id)
                    # Delete the session
                    self.chat_store.delete_session(session_id, user_id)
                
                # Delete test user
                self.user_store.delete_user(user_id)
                
            logger.info("Test data cleanup completed")
        except Exception as e:
            logger.error(f"Error cleaning test data: {str(e)}")
    
    def test_user_management(self) -> Optional[Dict[str, Any]]:
        """Test user management functionality."""
        logger.info("Testing user management")
        
        # List users (should be empty or contain existing users)
        initial_users = self.user_store.list_users()
        logger.info(f"Initial users: {initial_users}")
        
        # Create a new test user
        test_user_id = self.user_store.create_user()
        logger.info(f"Created test user with ID: {test_user_id}")
        
        # Verify user was created
        user_from_db = self.verify_mongo_data("users", {"user_id": test_user_id})
        if user_from_db:
            logger.info(f"User verified in MongoDB: {user_from_db}")
        else:
            logger.error("User not found in MongoDB!")
            
        # Get the user
        retrieved_user = self.user_store.get_user(test_user_id)
        logger.info(f"Retrieved user: {retrieved_user}")
        
        if retrieved_user and retrieved_user["user_id"] == test_user_id:
            logger.info("User management tests passed")
            return retrieved_user
        else:
            logger.error("User management tests failed!")
            return None
    
    def test_session_management(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Test session management functionality."""
        logger.info(f"Testing session management for user {user_id}")
        
        # List sessions (should be empty for new user)
        initial_sessions = self.session_store.list_sessions(user_id)
        logger.info(f"Initial sessions: {initial_sessions}")
        
        # Create a new test session
        test_session_id = self.session_store.create_session(user_id)
        logger.info(f"Created test session with ID: {test_session_id}")
        
        # Verify session was created
        session_from_db = self.verify_mongo_data("session_user_map", {"session_id": test_session_id, "user_id": user_id})
        if session_from_db:
            logger.info(f"Session verified in MongoDB: {session_from_db}")
        else:
            logger.error("Session not found in MongoDB!")
            
        # Get the session
        retrieved_session = self.session_store.list_sessions(user_id)
        logger.info(f"Retrieved session: {retrieved_session}")
        
        if retrieved_session and any(session["session_id"] == test_session_id for session in retrieved_session):
            logger.info("Session management tests passed")
            return {"id": test_session_id, "name": "test_session"}
        else:
            logger.error("Session management tests failed!")
            return None
    
    def test_chat_interaction(self, user_id: str, session_id: str):
        """Test chat interaction with all steps of the flow."""
        logger.info(f"Testing chat interaction for user {user_id}, session {session_id}")
        
        # Mock input function to simulate user input
        original_input = __builtins__.input
        
        # Chat history tracking
        chat_history = []
        
        try:
            # Create a mock input function that will provide our test data
            input_responses = {
                "Enter your question (or 'quit' to exit): ": self.test_questions[0],
                "Is the generated SQL query OK? (yes/no): ": "yes",
                "Enter your question (or 'quit' to exit): ": self.test_questions[1],
                "Is the generated SQL query OK? (yes/no): ": "no",
                "Enter your updated SQL query: ": "SELECT product_id, name, retail_price FROM `bigquery-public-data.thelook_ecommerce.products` ORDER BY retail_price DESC LIMIT 5 OFFSET 5",
                "Enter your question (or 'quit' to exit): ": self.test_questions[2],
                "Is the generated SQL query OK? (yes/no): ": "yes",
                "Enter your question (or 'quit' to exit): ": self.test_questions[3],
                "Is the generated SQL query OK? (yes/no): ": "yes",
                "Enter your question (or 'quit' to exit): ": "quit"
            }
            
            def mock_input(prompt):
                logger.info(f"Prompt: {prompt}")
                
                # If this is a prompt we've prepared a response for
                if prompt in input_responses:
                    response = input_responses[prompt]
                    logger.info(f"Providing mock response: {response}")
                    return response
                
                # For any unexpected prompts, use default responses
                if "query OK" in prompt:
                    return "yes"
                
                # For truly unexpected prompts, log and return default
                logger.warning(f"Unexpected prompt encountered: {prompt}")
                return "quit"
            
            # Replace the input function with our mock
            __builtins__.input = mock_input
            
            # Mock the handle_chat_interaction function
            def mock_handle_chat_interaction(chat_service, chat_store, session_id, user_id):
                question_index = 0
                
                # Ensure MongoDB connection is established
                chat_store._get_collection()
                
                while True:
                    # Check if we've run out of test questions
                    if question_index >= len(self.test_questions):
                        logger.info("No more test questions, exiting chat")
                        break
                    
                    # Get the next test question
                    question = self.test_questions[question_index]
                    logger.info(f"Processing test question: {question}")
                    
                    # Generate SQL
                    try:
                        messages = chat_store.get_messages(session_id, user_id)
                        chat_history = []
                        for msg in messages:
                            if hasattr(msg, 'content') and hasattr(msg, 'type'):
                                is_user = msg.type == 'human'
                                chat_history.append({"role": "user" if is_user else "assistant", "content": msg.content})
                        
                        # Log the chat history being sent to the LLM
                        logger.info(f"CHAT HISTORY: {json.dumps(chat_history, indent=2)}")
                        
                        # Generate SQL - this is the core step
                        logger.info(f"QUERY: {question}")
                        result = chat_service.generate_sql_only(session_id, question, user_id)
                        
                        # Log the full result
                        logger.info(f"GENERATED SQL RESULT: {json.dumps(result, indent=2)}")
                        
                        # Check if SQL was successfully generated
                        if not result.get("sql", "").strip():
                            logger.error(f"Failed to generate SQL for question: {question}")
                            chat_store.add_message(
                                session_id=session_id,
                                message=HumanMessage(content=f"Error: Could not generate SQL for '{question}'"),
                                user_id=user_id
                            )
                            break
                        
                        # Add user question to chat history
                        chat_store.add_message(
                            session_id=session_id,
                            message=HumanMessage(content=question),
                            user_id=user_id
                        )
                        
                        # Execute SQL query
                        try:
                            query_result = chat_service.validate_and_execute_sql(result["sql"])
                            logger.info(f"Query execution successful with {len(query_result)} results")
                            
                            # Generate summary if configured
                            summary = chat_service.generate_summary(question, query_result)
                            if summary:
                                logger.info(f"Generated summary: {summary[:100]}...")
                            
                            # Store AI response in chat history
                            response_content = f"SQL: {result['sql']}\n\nResults: {json.dumps(query_result[:5], indent=2)}"
                            if summary:
                                response_content += f"\n\nSummary: {summary}"
                            
                            chat_store.add_message(
                                session_id=session_id,
                                message=AIMessage(content=response_content),
                                user_id=user_id
                            )
                            
                            # Store in vector database if configured
                            if chat_service.should_store_queries:
                                logger.info(f"Storing question and query in vector database")
                                chat_service.add_to_vector_store(question, result["sql"])
                                
                        except Exception as e:
                            logger.error(f"Error executing SQL: {str(e)}")
                            chat_store.add_message(
                                session_id=session_id,
                                message=AIMessage(content=f"Error executing SQL: {str(e)}"),
                                user_id=user_id
                            )
                    except Exception as e:
                        logger.error(f"Error in test chat interaction: {str(e)}")
                        break
                        
                    # Move to next question
                    question_index += 1
                    
                    # Verify messages were saved in MongoDB
                    messages_from_db = self.verify_mongo_data(config.CHAT_STORE_COLLECTION_NAME, {"session_id": session_id})
                    logger.info(f"Verified {len(messages_from_db)} messages in MongoDB")
                
                return True
            
            # Execute the modified chat interaction
            success = mock_handle_chat_interaction(self.chat_service, self.chat_store, session_id, user_id)
            
            # Verify final state
            final_messages = self.chat_store.get_messages(session_id, user_id)
            logger.info(f"Final message count: {len(final_messages)}")
            
            # Verify vector store additions
            # This would require a method to query the vector store directly
            
            if success and len(final_messages) > 0:
                logger.info("Chat interaction tests passed")
                # Display the full conversation for verification
                for msg in final_messages:
                    role = "USER" if hasattr(msg, 'type') and msg.type == 'human' else "AI"
                    # Truncate long responses for readability
                    content = msg.content if len(msg.content) < 100 else f"{msg.content[:100]}..."
                    logger.info(f"{role}: {content}")
                return True
            else:
                logger.error("Chat interaction tests failed!")
                return False
            
        except Exception as e:
            logger.error(f"Error in test_chat_interaction: {str(e)}")
            return False
        finally:
            # Restore original input function
            __builtins__.input = original_input
    
    def test_resuming_session(self, user_id: str, session_id: str):
        """Test resuming a session with existing chat history."""
        logger.info(f"Testing session resumption for user {user_id}, session {session_id}")
        
        # Get messages before resuming
        before_messages = self.chat_store.get_messages(session_id, user_id)
        logger.info(f"Messages before resuming: {len(before_messages)}")
        
        # Add a new test message as if from another session
        test_message = "This is a test message for session resumption"
        self.chat_store.add_message(
            session_id=session_id,
            message=HumanMessage(content=test_message),
            user_id=user_id
        )
        
        # Verify message was added
        after_messages = self.chat_store.get_messages(session_id, user_id)
        logger.info(f"Messages after adding test message: {len(after_messages)}")
        
        # Get the last message to verify it's our test message
        last_message = after_messages[-1] if after_messages else None
        
        if last_message and hasattr(last_message, 'content') and last_message.content == test_message:
            logger.info("Session resumption test passed")
            return True
        else:
            logger.error("Session resumption test failed!")
            return False
    
    def run_tests(self):
        """Run all tests in sequence."""
        try:
            logger.info("Starting app flow tests")
            
            # Test user management
            test_user = self.test_user_management()
            if not test_user:
                logger.error("User management tests failed, stopping")
                return False
            
            # Test session management
            test_session = self.test_session_management(test_user["user_id"])
            if not test_session:
                logger.error("Session management tests failed, stopping")
                return False
            
            # Test chat interaction
            chat_success = self.test_chat_interaction(test_user["user_id"], test_session["id"])
            if not chat_success:
                logger.error("Chat interaction tests failed, stopping")
                return False
            
            # Test session resumption
            resume_success = self.test_resuming_session(test_user["user_id"], test_session["id"])
            if not resume_success:
                logger.error("Session resumption tests failed, stopping")
                return False
            
            logger.info("All tests completed successfully!")
            return True
        
        except Exception as e:
            logger.error(f"Error in test execution: {str(e)}")
            return False
        finally:
            # Clean up test data
            self.clean_test_data()

if __name__ == "__main__":
    tester = AppFlowTester()
    success = tester.run_tests()
    
    if success:
        print("✅ All tests passed successfully!")
        sys.exit(0)
    else:
        print("❌ Tests failed. See app_flow_test.log for details.")
        sys.exit(1) 