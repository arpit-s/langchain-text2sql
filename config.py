import os
from dotenv import load_dotenv

load_dotenv(override=True)

UUID_NAMESPACE = os.getenv('UUID_NAMESPACE')
SQL_DIALECT = os.getenv('SQL_DIALECT', 'BigQuery')
TRAINING_SET_DIR = os.getenv('TRAINING_SET_DIR', './training_set')

API_KEY = os.getenv('API_KEY')
LLM_MODEL = os.getenv('LLM_MODEL', 'gemini-1.5-flash-latest')
EMBEDDING_MODEL_NAME = os.getenv('EMBEDDING_MODEL_NAME', 'models/text-embedding-004')

VECTOR_DB_HOST = os.getenv("VECTOR_DB_HOST", "localhost")
VECTOR_DB_PORT = int(os.getenv("VECTOR_DB_PORT", 6333))
VECTOR_DB_API_KEY = os.getenv("VECTOR_DB_API_KEY")
VECTOR_DB_COLLECTION_NAME = os.getenv("VECTOR_DB_COLLECTION_NAME", "text2sql_vector_store")
VECTOR_DB_VECTOR_SIZE = int(os.getenv('VECTOR_DB_VECTOR_SIZE', 768))
VECTOR_DB_DISTANCE_METRIC = os.getenv('VECTOR_DB_DISTANCE_METRIC', 'COSINE').upper()

SQL_CLIENT_PROJECT_ID = os.getenv('SQL_CLIENT_PROJECT_ID')
SQL_CLIENT_SERVICE_ACCOUNT_PATH = os.getenv('SQL_CLIENT_SERVICE_ACCOUNT_PATH')

CHAT_STORE_URI = os.getenv('CHAT_STORE_URI', 'mongodb://localhost:27017/')
CHAT_STORE_DB_NAME = os.getenv('CHAT_STORE_DB_NAME', 'text2sql_chat_history')
CHAT_STORE_COLLECTION_NAME = os.getenv('CHAT_STORE_COLLECTION_NAME', 'message_store') # Default for Langchain class

SUMMARIZE_RESULTS = os.getenv('SUMMARIZE_RESULTS', 'true').lower() == 'true'
ADD_QUESTION_QUERY_TO_VECTOR_DB = os.getenv('ADD_QUESTION_QUERY_TO_VECTOR_DB', 'true').lower() == 'true'

# Maximum character limits for content included in prompts to avoid exceeding context windows
MAX_SUMMARY_RESULTS_CHARS = int(os.getenv('MAX_SUMMARY_RESULTS_CHARS', 2000))
MAX_DDL_SAMPLE_CHARS = int(os.getenv('MAX_DDL_SAMPLE_CHARS', 2000))

SQL_CLIENT_COMMENT_SAMPLE_SIZE = int(os.getenv('SQL_CLIENT_COMMENT_SAMPLE_SIZE', 5))

# Vector Store K Values
VECTOR_STORE_K_DDL = int(os.getenv("VECTOR_STORE_K_DDL", "3"))
VECTOR_STORE_K_DOC = int(os.getenv("VECTOR_STORE_K_DOC", "5"))
VECTOR_STORE_K_QUERY = int(os.getenv("VECTOR_STORE_K_QUERY", "5"))
VECTOR_STORE_K_OTHER = int(os.getenv("VECTOR_STORE_K_OTHER", "3"))

# SQL Validation Retries
SQL_VALIDATION_RETRIES = int(os.getenv('SQL_VALIDATION_RETRIES', 2))

if not API_KEY: raise ValueError("API_KEY not found in .env")
if not UUID_NAMESPACE: raise ValueError("UUID_NAMESPACE not found in .env")

VECTOR_DB_URL = f"http://{VECTOR_DB_HOST}:{VECTOR_DB_PORT}"
VECTOR_DB_URI = VECTOR_DB_URL  # For compatibility with older code