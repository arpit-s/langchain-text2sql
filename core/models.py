from pydantic import BaseModel, Field
from typing import List

import config

class SQLOutput(BaseModel):
    sql: str = Field(description=f"The generated {config.SQL_DIALECT} SQL query.")
    follow_up: list = Field(description="Suggested follow-up questions.", default=[])
    answer: str = Field(description="The answer to the question based on the SQL results.", default="")