from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional

class SQLClientBase(ABC):
    @abstractmethod
    def dry_run_query(self, query: str) -> Tuple[bool, Optional[str]]:
        """
        Perform a dry run of the query to validate it without executing it.
        
        Args:
            query: The SQL query to validate
            
        Returns:
            Tuple[bool, Optional[str]]: (is_valid, error_message)
            - is_valid: True if query is valid, False otherwise
            - error_message: Error message if query is invalid, None if valid
        """
        pass

    @abstractmethod
    def execute_query(self, query: str) -> Tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
        """
        Executes a SQL query and returns the results and any error.
        """
        pass

    @abstractmethod
    def list_tables(self, dataset_id: str) -> List[str]:
        """
        Lists tables in a given dataset.
        """
        pass

    @abstractmethod
    def get_table(self, dataset_id: str, table_id: str) -> Any: # Return type might be specific to client library
        """
        Retrieves information about a specific table.
        """
        pass

    @abstractmethod
    def get_formatted_ddl(self, dataset_id: str, table_id: str) -> Optional[str]:
        """
        Retrieves the formatted DDL for a table.
        """
        pass

    @abstractmethod
    def get_sample_data(self, dataset_id: str, table_id: str, sample_size: int) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """
        Retrieves sample data from a table.
        """
        pass

    @abstractmethod
    def close_client(self):
        """
        Closes the SQL client connection.
        """
        pass