from sqlalchemy import create_engine, text
import pandas as pd
from typing import Optional


class PostgresLoader:
    def __init__(self, conn_str: str):
        self.engine = create_engine(conn_str)

    def load_dataframe(
        self, df: pd.DataFrame, table: str, if_exists: str = "append"
    ):
        """
        Loads DataFrame rows into target PostgreSQL table.
        """
        if df.empty:
            return
        df.to_sql(table, self.engine, if_exists=if_exists, index=False)
