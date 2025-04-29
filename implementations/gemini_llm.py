import json
import re
import logging
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from typing import List, Dict, Any, Optional, Tuple
import config
from interfaces.llm import LLMBase
from core.models import SQLOutput

# Set up logging
logging.basicConfig(
    filename='gemini_llm.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define the base system prompt template with placeholders for context and schema
# Note: No f-string embedding of schema here!
SQL_SYSTEM_PROMPT_BASE = f"""You are an expert {config.SQL_DIALECT} SQL generator. Generate a valid and efficient query based on the user's question, provided context (DDL, documentation, similar queries), and chat history.

Relevant Context:
---
{{context}}
---

IMPORTANT DATE HANDLING RULES:
1. ALWAYS use DATE() to convert TIMESTAMP to DATE: DATE(timestamp_column)
2. For date arithmetic, use DATE_SUB(DATE(timestamp_column), INTERVAL X MONTH/QUARTER/YEAR)
3. NEVER use TIMESTAMP_SUB with MONTH/QUARTER/YEAR - it will fail!
4. For date comparisons, ensure both sides are of the same type (either both DATE or both TIMESTAMP)

Example of CORRECT date handling:
```sql
SELECT FORMAT_TIMESTAMP('%Y-%m', created_at) as month, COUNT(order_id) as order_count
FROM `bigquery-public-data.thelook_ecommerce.orders`
WHERE DATE(created_at) >= DATE_SUB(CURRENT_DATE(), INTERVAL 6 MONTH)
GROUP BY month ORDER BY month
```

Example of INCORRECT date handling (will fail):
```sql
SELECT FORMAT_TIMESTAMP('%Y-%m', created_at) as month, COUNT(order_id) as order_count
FROM `bigquery-public-data.thelook_ecommerce.orders`
WHERE created_at >= DATE_SUB(CURRENT_DATE(), INTERVAL 6 MONTH)  -- WRONG: comparing TIMESTAMP with DATE
GROUP BY month ORDER by month
```

IMPORTANT OUTPUT FORMAT RULES:
1. Your response MUST be a valid JSON object matching the provided schema.
2. The SQL query should be a string value in the JSON, not a code block.
3. Do NOT include ```sql ... ``` markers in the JSON output.
4. The JSON object should have exactly these fields:
   - "sql": The SQL query as a plain string
   - "follow_up": A list of 2-3 relevant follow-up questions
   - "answer": The answer to the question based on the SQL results
Prioritize using schema elements mentioned in the question or context.
Output ONLY a JSON object matching the following Pydantic schema: {{schema}}. Do not include explanations."""

SUMMARY_SYSTEM_PROMPT = """Summarize the key insights from the provided query results data based on the user's question: {question}.
Results (sample):
{results_str}
Provide a concise, bulleted summary."""

COMMENT_DDL_SYSTEM_PROMPT = """You are an expert SQL commentator. Given the following DDL ({sql_dialect}), add a concise, descriptive comment ('-- comment') on the same line after EACH column definition ONLY.
Infer the column's purpose from its name, type, and context. {sample_data_instructions}
Maintain the original DDL structure precisely.
If sample data is provided, ALWAYS include examples from it in your comments (e.g., '-- User ID (e.g., 12345)').
If no sample data, skip the 'e.g.' part (e.g., '-- User ID').
Output ONLY the modified DDL. Do NOT include explanations or ```sql fences."""

class GeminiLLM(LLMBase):
    def __init__(self, api_key: str, llm_model: str, embedding_model_name: str):
        self.api_key = api_key
        self.llm_model = llm_model
        self.embedding_model_name = embedding_model_name
        self._llm = None
        self._embeddings = None

    def _get_llm(self):
        if self._llm is None:
            self._llm = ChatGoogleGenerativeAI(model=self.llm_model, temperature=0.1, google_api_key=self.api_key)
        return self._llm

    def _get_embeddings(self):
        if self._embeddings is None:
            self._embeddings = GoogleGenerativeAIEmbeddings(model=self.embedding_model_name, google_api_key=self.api_key)
        return self._embeddings

    def get_embedding(self, text: str) -> List[float]:
        return self._get_embeddings().embed_query(text)

    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        return self._get_embeddings().embed_documents(texts)

    def generate_sql(self, question: str, chat_history: List[Dict[str, Any]], formatted_context: str) -> Dict[str, Any]:
        parser = JsonOutputParser(pydantic_object=SQLOutput)

        # Log original chat history
        logger.debug("\nOriginal Chat History:")
        for msg in chat_history:
            logger.debug(f"Type: {type(msg)}, Content: {msg.content if hasattr(msg, 'content') else msg}")

        # Filter out SystemMessages from chat_history
        filtered_chat_history = [
            msg for msg in chat_history
            if isinstance(msg, (HumanMessage, AIMessage)) # Keep only conversation turns
        ]

        # Log filtered chat history
        logger.debug("\nFiltered Chat History:")
        for msg in filtered_chat_history:
            logger.debug(f"Type: {type(msg)}, Content: {msg.content}")

        # Get the schema as a string to pass *as a variable* to the template
        schema_json = json.dumps(SQLOutput.model_json_schema(), indent=2) # Use indent for readability in prompt

        # Define the prompt template structure using the base string with placeholders
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(SQL_SYSTEM_PROMPT_BASE), # The main system prompt
            MessagesPlaceholder(variable_name="chat_history"), # Placeholder for the chat history list
            HumanMessagePromptTemplate.from_template("{question}") # The current question
        ])
        
        # Log prompt structure
        logger.debug("\nPrompt Structure:")
        logger.debug(f"Input Variables: {prompt.input_variables}")
        logger.debug(f"Messages: {prompt.messages}")

        chain = prompt | self._get_llm() | parser

        # Build the input dictionary. Provide the schema JSON string under the 'schema' key
        input_data = {
            "context": formatted_context,
            "chat_history": filtered_chat_history,
            "question": question,
            "schema": schema_json # This matches the {schema} placeholder in SQL_SYSTEM_PROMPT_BASE
        }

        # Log input data
        logger.debug("\nInput Data:")
        logger.debug(f"Context length: {len(formatted_context)}")
        logger.debug(f"Chat history length: {len(filtered_chat_history)}")
        logger.debug(f"Question: {question}")
        logger.debug(f"Schema length: {len(schema_json)}")

        # Format the prompt to see what actually gets sent to the LLM
        formatted_prompt = prompt.format_messages(**input_data)
        logger.debug("\nFormatted Prompt (Actual content sent to LLM):")
        for msg in formatted_prompt:
            logger.debug(f"Message Type: {type(msg).__name__}")
            logger.debug(f"Content: {msg.content}")
            logger.debug("---")

        # The JsonOutputParser will parse the output JSON directly into a dictionary
        try:
            output_dict = chain.invoke(input_data)
            logger.debug(f"Output: {output_dict}")

            # Convert the output to SQLOutput object
            sql_output = SQLOutput(
                sql=output_dict.get('sql', ''),
                follow_up=output_dict.get('follow_up', []),
                answer=output_dict.get('answer', '')
            )
            return sql_output.model_dump() # Return as dict to match interface
        except Exception as e:
            logger.error(f"Error: {str(e)}")
            # Return a default SQLOutput object with an error message
            return SQLOutput(
                sql="-- Error: Failed to generate SQL",
                follow_up=[],
                answer=f"I encountered an error: {str(e)}"
            ).model_dump() # Return as dict

    def summarize(self, question: str, results: List[Dict[str, Any]]) -> str:
        if not config.SUMMARIZE_RESULTS or not results:
            return ""
        # Limit results string length to avoid exceeding model context window
        max_chars = getattr(config, 'MAX_SUMMARY_RESULTS_CHARS', 2000) # Use getattr for robustness
        results_str = json.dumps(results[:10], indent=2, default=str)[:max_chars]

        prompt = ChatPromptTemplate.from_template(SUMMARY_SYSTEM_PROMPT)
        chain = prompt | self._get_llm() | StrOutputParser()
        summary = chain.invoke({"question": question, "results_str": results_str})
        return summary.strip()

    def comment_ddl(self, ddl_string: str, sql_dialect: str, sample_data: Optional[List[Dict[str, Any]]] = None) -> Optional[str]:
        if not ddl_string: return None
        # Basic check if it looks like DDL
        if not ("CREATE TABLE" in ddl_string.upper() or "CREATE VIEW" in ddl_string.upper()):
            return None

        sample_str = ""
        if sample_data:
            # Limit sample data size if necessary
            max_chars = getattr(config, 'MAX_DDL_SAMPLE_CHARS', 2000) # Use getattr for robustness
            sample_str = f"\n\nSample rows (JSON):\n```json\n{json.dumps(sample_data, indent=2, default=str)[:max_chars]}\n```"

        sys_prompt_content = COMMENT_DDL_SYSTEM_PROMPT.format(
            sql_dialect=sql_dialect,
            sample_data_instructions="Use the sample data to inform comments." if sample_str else ""
        )
        # This prompt structure is correct (System followed by Human)
        prompt_template = ChatPromptTemplate.from_messages([
            SystemMessage(content=sys_prompt_content),
            HumanMessagePromptTemplate.from_template("Input DDL:\n```sql\n{ddl_string}\n```\n{sample_data_context}\nCommented DDL:")
        ])
        commenting_chain = prompt_template | self._get_llm() | StrOutputParser()

        # Use .strip("` \n") to remove potential code fences and surrounding whitespace
        commented_ddl = commenting_chain.invoke({"ddl_string": ddl_string, "sample_data_context": sample_str}).strip("` \n")

        # Check if comments were likely added and it's still valid-ish SQL DDL structure
        if "--" in commented_ddl and "(" in commented_ddl and "CREATE " in commented_ddl.upper():
             return commented_ddl
        else:
             print(f"Warning: LLM did not seem to add comments to DDL successfully based on heuristics. Output beginning: {commented_ddl[:200]}...")
             return None # Return None if it doesn't look commented