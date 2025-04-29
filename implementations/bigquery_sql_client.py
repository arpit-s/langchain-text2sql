import re
import os
from google.cloud import bigquery
from google.oauth2 import service_account
from typing import List, Dict, Any, Tuple, Optional
from langchain_core.output_parsers import StrOutputParser

import config
from interfaces.sql_client import SQLClientBase

class BigQuerySqlClient(SQLClientBase):
    def __init__(self, project_id: str, service_account_path: str):
        self.project_id = project_id
        self.service_account_path = service_account_path
        self._client = None
        self.sql_parser = StrOutputParser()
        # Initialize client immediately
        self._get_client()

    def _get_client(self):
        if self._client is None and config.SQL_DIALECT == 'BigQuery' and self.project_id and self.service_account_path:
            try:
                bq_creds = service_account.Credentials.from_service_account_file(
                    self.service_account_path
                )
                self._client = bigquery.Client(
                    project=self.project_id,
                    credentials=bq_creds
                )
                print("BigQuery client initialized.")
            except Exception as e:
                print(f"BigQuery client initialization failed: {e}")
                self._client = None
        return self._client

    def close_client(self):
        # BigQuery client doesn't require explicit close in this library version
        pass

    def dry_run_query(self, query: str) -> Tuple[bool, Optional[str]]:
        query = query.strip().removeprefix("```sql").removesuffix("```").strip()
        if not query:
            return False, "Generated SQL query is empty."
        
        try:
            self.sql_parser.parse(query)
        except Exception as e:
            return False, f"SQL parsing error: {str(e)}"

        client = self._get_client()
        if not client:
            return False, "SQL client not initialized."

        try:
            job_config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
            query_job = client.query(query, job_config=job_config)
            return True, None
        except Exception as e:
            return False, f"BigQuery validation error: {str(e)}"

    def execute_query(self, query: str) -> Tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
        client = self._get_client()
        if not client:
            return None, "SQL client not initialized."

        clean_query = query.strip().removeprefix("```sql").removesuffix("```").strip()
        if not clean_query:
            return None, "Generated SQL query is empty."

        try:
            query_job = client.query(clean_query)
            results = [dict(row) for row in query_job.result()]
            return results, None
        except Exception as e:
            print(f"Error executing query: {e}")
            return None, str(e)

    def list_tables(self, dataset_id: str) -> List[str]:
        client = self._get_client()
        if not client: return []

        try:
            tables_iter = client.list_tables(dataset_id)
            return [table.table_id for table in tables_iter]
        except Exception as e:
            print(f"Error listing tables for dataset {dataset_id}: {e}")
            return []

    def get_table(self, dataset_id: str, table_id: str) -> Any:
        client = self._get_client()
        if not client: return None

        try:
            table_ref = f"{dataset_id}.{table_id}"
            return client.get_table(table_ref)
        except Exception as e:
            print(f"Error getting table {table_id} in dataset {dataset_id}: {e}")
            return None

    def get_formatted_ddl(self, dataset_id: str, table_id: str) -> Optional[str]:
        table = self.get_table(dataset_id, table_id)
        if not table: return None

        ddl_parts = [f"CREATE TABLE `{table.project}.{table.dataset_id}.{table.table_id}` ("]
        ddl_parts.extend([f"  `{field.name}` {field.field_type}," for field in table.schema])
        if ddl_parts[-1].endswith(','): ddl_parts[-1] = ddl_parts[-1][:-1]
        ddl_parts.append(");")
        return "\n".join(ddl_parts)

    def get_sample_data(self, dataset_id: str, table_id: str, sample_size: int) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        client = self._get_client()
        if not client: return [], "SQL client not initialized."

        full_table_id = f"`{dataset_id}.{table_id}`"
        sample_query = f"SELECT * FROM {full_table_id} LIMIT {sample_size}"
        sample_data, error = self.execute_query(sample_query)
        if error:
            print(f"    Warning: Could not fetch sample data for {table_id}: {error}")
            return [], error
        return sample_data, None