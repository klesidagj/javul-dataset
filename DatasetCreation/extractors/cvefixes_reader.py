import sqlite3
import pandas as pd
from pathlib import Path
from typing import Any
from utils.cwe_utils import normalize_cwe_token, enforce_cwe_rules


def load_sql_query(sql_path: str) -> str:
    return Path(sql_path).read_text(encoding="utf-8")


def read_java_snippets(sqlite_path: str, sql_path: str) -> pd.DataFrame:
    """
    Loads raw Java snippets + vulnerability info + CWE IDs into a DataFrame.
    """
    query = load_sql_query(sql_path)

    with sqlite3.connect(sqlite_path) as conn:
        df = pd.read_sql_query(query, conn)

    df["cwe_id"] = df["cwe_id"].apply(normalize_cwe_token)
    df["is_vulnerable"] = df["is_vulnerable"].apply(_to_bool)
    df = enforce_cwe_rules(df)

    return df


def _to_bool(v: Any) -> bool:
    if isinstance(v, (int, float)):
        return bool(int(v))
    return str(v).strip().lower() in ("1", "true", "yes")
