import json
from datetime import datetime, timezone
from tabulate import tabulate
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from typing import List, Dict, Any, Optional, Tuple

import config
from interfaces.llm import LLMBase
from interfaces.vector_store import VectorStoreBase
from interfaces.chat_store import ChatStoreBase
from interfaces.sql_client import SQLClientBase
from core.models import SQLOutput 
from langchain_core.output_parsers import StrOutputParser

class ChatService:
    def __init__(self, llm_client: LLMBase, vector_store: VectorStoreBase, chat_store: ChatStoreBase, sql_client: SQLClientBase):
        self.llm_client = llm_client
        self.vector_store = vector_store
        self.chat_store = chat_store
        self.sql_client = sql_client
        self.sql_parser = StrOutputParser()
        self.should_store_queries = config.ADD_QUESTION_QUERY_TO_VECTOR_DB

    def _format_context_for_llm(self, context_docs_by_type: Dict[str, List[Dict[str, Any]]]) -> str:
        parts = ["-- Relevant Context --"]
        has_context = any(bool(docs) for docs in context_docs_by_type.values())
        if not has_context: return "-- No context retrieved --"

        for source_type, docs in context_docs_by_type.items():
            if docs:
                parts.append(f"\n## Context from {source_type.upper()} ##")
                for doc_hit in docs:
                     payload = doc_hit.get('payload', {})
                     text = payload.get('text', '')
                     meta = payload.get('metadata', {})

                     if source_type == 'query':
                         parts.append(f"- Similar Q: {text}\n  Similar SQL: ```sql\n{meta.get('sql', '')}\n```")
                     elif source_type == 'ddl':
                         parts.append(f"- Schema (DDL): ```sql\n{text}\n```")
                     else:
                         parts.append(f"- {source_type.capitalize()}: {text[:500]}{'...' if len(text)>500 else ''}")

        parts.append("-- End Context --")
        full_context = "\n".join(parts)
        max_len = getattr(config, "MAX_CONTEXT_LENGTH", 15000)
        return full_context[:max_len] + ("\n... [Context Truncated]" if len(full_context) > max_len else "")

    def _validate_sql(self, sql: str) -> Tuple[bool, Optional[str]]:
        cleaned_sql = sql.removeprefix("```sql").removesuffix("```").strip()
        sql = cleaned_sql
        try:
            self.sql_parser.parse(sql)
        except Exception as e:
            return False, f"SQL parsing error: {e}"
        is_valid, error = self.sql_client.dry_run_query(sql)
        if not is_valid:
            return False, f"SQL validation error: {error}"
        return True, None

    def generate_sql_only(self, session_id: str, question: str, user_id: str) -> Dict[str, Any]:
        """
        Phase 1: Generate SQL only.
        Tries to auto-correct invalid SQL by appending error messages and retrying.
        """
        # Store user message
        human_message_obj = HumanMessage(content=question)
        self.chat_store.add_message(session_id, human_message_obj, user_id)
        chat_history = self.chat_store.get_messages(session_id, user_id)

        # Get context from vector store
        context_types = ["ddl", "documentation", "query"]
        context_docs_by_type = {}
        for c_type in context_types:
            context_docs_by_type[c_type] = self.vector_store.search_context(question, c_type)
        formatted_context = self._format_context_for_llm(context_docs_by_type)
        
        # Log the formatted context
        print(f"\n--- CONTEXT FOR PROMPT ---\n{formatted_context}\n--- END CONTEXT ---\n")

        retries = getattr(config, "SQL_VALIDATION_RETRIES", 2)
        
        # First attempt
        sql_output_dict = self.llm_client.generate_sql(question, chat_history, formatted_context)
        generated_sql = sql_output_dict.get('sql', '')
        is_valid, error_msg = self._validate_sql(generated_sql)
        if is_valid:
            print(f"\n--- FOLLOW-UP SUGGESTIONS ---\n{sql_output_dict.get('follow_up', [])}\n--- END FOLLOW-UPS ---\n")
            return {
                "status": "success",
                "sql": generated_sql,
                "question": question,
                "follow_up": sql_output_dict.get('follow_up', [])
            }

        # Retry loop only if there was a validation error
        for attempt in range(retries):
            # Compose a correction prompt for the LLM
            correction_prompt = (
                f"{question}\n\n"
                f"# Previous SQL attempt:\n"
                f"```sql\n{generated_sql}\n```\n"
                f"# Validation error:\n"
                f"{error_msg}\n"
                f"# Please fix the SQL query based on the above error."
            )
            sql_output_dict = self.llm_client.generate_sql(
                correction_prompt, chat_history, formatted_context
            )
            generated_sql = sql_output_dict.get('sql', '')
            is_valid, error_msg = self._validate_sql(generated_sql)
            if is_valid:
                return {
                    "status": "success",
                    "sql": generated_sql,
                    "question": question,
                    "follow_up": sql_output_dict.get('follow_up', [])
                }

        # If still invalid after retries, return the last error and SQL
        return {
            "status": "validation_error",
            "sql": generated_sql,
            "error_message": error_msg,
            "question": question
        }

    def finalize_response(self, session_id: str, question: str, sql_query: str, user_id: str, follow_ups: List[str] = None) -> Dict[str, Any]:
        """
        Phase 2: Execute SQL, summarize results, and store AI message.
        Only called after user approves or updates the SQL.
        """
        # Validate SQL before execution
        is_valid, error = self._validate_sql(sql_query)
        if not is_valid:
            ai_response_dict = {
                "status": "validation_error",
                "sql": sql_query,
                "error_message": error
            }
            ai_message_obj = AIMessage(content=json.dumps(ai_response_dict))
            self.chat_store.add_message(session_id, ai_message_obj, user_id)
            return ai_response_dict

        # Execute SQL
        query_results, query_error = (None, None)
        if sql_query and not sql_query.startswith("-- Error"):
            query_results, query_error = self.sql_client.execute_query(sql_query)
        elif sql_query and sql_query.startswith("-- Error"):
            query_error = sql_query

        if query_error:
            error_response = {
                "status": "execution_error",
                "sql": sql_query,
                "error_message": query_error
            }
            ai_message_obj = AIMessage(content=json.dumps(error_response))
            self.chat_store.add_message(session_id, ai_message_obj, user_id)
            return error_response

        # Success path
        ai_response_dict = {
            "status": "success",
            "sql": sql_query or "No SQL generated."
        }

        if query_results is not None:
            results_table = tabulate(query_results, headers="keys", tablefmt="grid", maxcolwidths=[None, 50])
            if len(results_table) > 2000:
                results_table = tabulate(query_results[:10], headers="keys", tablefmt="grid", maxcolwidths=[None, 50]) + "\n... (Results truncated)"
            ai_response_dict["results"] = results_table

            if config.SUMMARIZE_RESULTS:
                summary = self.llm_client.summarize(question, query_results)
                if summary:
                    ai_response_dict["summary"] = summary

            # Use the follow-ups passed from generate_sql_only
            if follow_ups:
                ai_response_dict["follow_up"] = follow_ups

        ai_message_obj = AIMessage(content=json.dumps(ai_response_dict))
        self.chat_store.add_message(session_id, ai_message_obj, user_id)

        return ai_response_dict

    def add_query_to_vector_db(self, question: str, sql: str):
        if not config.ADD_QUESTION_QUERY_TO_VECTOR_DB:
            print("Skipping adding question/query to vector DB (config disabled).")
            return
        if not question or not sql or sql.startswith("-- Error"):
            print("Skipping adding question/query to vector DB (invalid input).")
            return
        try:
            embedding = self.llm_client.get_embedding(question)
            metadata = {"source_type": "query","sql": sql}
            self.vector_store.add_entry(text=question, metadata=metadata, embedding=embedding)
            print(f"Successfully added question/query to vector database.\nQuestion: {question}\nSQL: {sql}")
        except Exception as e:
            print(f"Error adding question/query to vector DB: {e}")

    def validate_and_execute_sql(self, sql: str) -> List[Dict[str, Any]]:
        """
        Validate and execute SQL query. Used in tests.
        This combines _validate_sql and execute_query in one function.
        """
        # First validate
        is_valid, error = self._validate_sql(sql)
        if not is_valid:
            raise ValueError(f"SQL validation error: {error}")
        
        # Then execute
        query_results, query_error = self.sql_client.execute_query(sql)
        if query_error:
            raise ValueError(f"SQL execution error: {query_error}")
            
        return query_results

    def generate_summary(self, question: str, query_results: List[Dict[str, Any]]) -> Optional[str]:
        """
        Generate a summary of SQL query results. Used in tests.
        This is a wrapper around the LLM summarize method.
        """
        if not config.SUMMARIZE_RESULTS:
            return None
            
        return self.llm_client.summarize(question, query_results)
        
    def add_to_vector_store(self, question: str, sql: str):
        """
        Add a question and its corresponding SQL to the vector store.
        Used in tests, wraps add_query_to_vector_db.
        """
        return self.add_query_to_vector_db(question, sql)