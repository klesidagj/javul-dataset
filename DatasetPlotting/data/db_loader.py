# --- FILE: db_loader.py ---
#!/usr/bin/env python3
"""
DB / CSV loading utilities.
"""
from sqlalchemy import create_engine
import pandas as pd

def make_pg_uri(cfg: dict) -> str:
    return (
        f"postgresql+psycopg2://{cfg['user']}:{cfg['password']}"
        f"@{cfg['host']}:{cfg['port']}/{cfg['dbname']}"
    )

def load_from_db(sql: str, db_cfg: dict) -> pd.DataFrame:
    uri = make_pg_uri(db_cfg)
    engine = create_engine(uri)
    return pd.read_sql(sql, engine)

def load_from_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)