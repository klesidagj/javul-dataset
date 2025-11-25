# DatasetCreation/readers/cvefixes_reader.py
import sqlite3
import uuid
from pathlib import Path
from typing import Any
import pandas as pd

from DatasetCreation.config.log import get_logger
from DatasetCreation.utils.cwe_utils import normalize_cwe_token, enforce_cwe_rules
from DatasetCreation.config.config import CVEFIXES_DB_PATH, CVEFIXES_QUERY_SQL, CVE_SOURCE_NAME

logger = get_logger("cvefixes_reader")


def load_sql_query(sql_path: str) -> str:
    return Path(sql_path).read_text(encoding="utf-8")


def _to_bool(v: Any) -> bool:
    if isinstance(v, (int, float)):
        return bool(int(v))
    return str(v).strip().lower() in ("1", "true", "yes")


def read_java_snippets(sqlite_path: str = CVEFIXES_DB_PATH, sql_path: str = CVEFIXES_QUERY_SQL) -> pd.DataFrame:
    """
    Loads raw Java snippets + vulnerability info + CWE IDs into a DataFrame.
    Uses the pre-existing CVEfixes SQL query file (left unchanged).
    """
    logger.info("Reading CVEFixes data from sqlite: %s using query: %s", sqlite_path, sql_path)
    query = load_sql_query(sql_path)

    with sqlite3.connect(sqlite_path) as conn:
        df = pd.read_sql_query(query, conn)
    # Set dataset source
    df["id"] = [str(uuid.uuid4()) for _ in range(len(df))]
    df["source"] = CVE_SOURCE_NAME

    # Normalize
    if "cwe_id" in df.columns:
        df["cwe_id"] = df["cwe_id"].apply(normalize_cwe_token)
    if "is_vulnerable" in df.columns:
        df["is_vulnerable"] = df["is_vulnerable"].apply(_to_bool)

    df = enforce_cwe_rules(df)
    logger.info("Loaded %d CVEFixes records", len(df))
    return df