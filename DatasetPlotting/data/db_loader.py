# --- FILE: db_loader.py ---
#!/usr/bin/env python3
"""
DB / CSV loading utilities.
"""
from typing import Optional, Dict
import pandas as pd
import psycopg2

def load_from_db(sql: str, db_config: Dict, limit: Optional[int] = None) -> pd.DataFrame:
    if limit:
        sql = sql.strip().rstrip(';') + f"\nLIMIT {limit};"
    conn = psycopg2.connect(**db_config)
    try:
        df = pd.read_sql(sql, conn)
    finally:
        conn.close()
    return df

def load_from_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)