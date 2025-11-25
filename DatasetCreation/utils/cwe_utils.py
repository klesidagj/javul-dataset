# DatasetCreation/utils/cwe_utils.py
import re
import pandas as pd
from typing import Optional

def normalize_cwe_token(token: Optional[str]) -> Optional[str]:
    """Normalize something like 15 or 'CWE15' to 'CWE-15', None -> None."""
    if token is None:
        return None
    m = re.search(r"(\d+)", str(token))
    return f"CWE-{int(m.group(1))}" if m else None


def extract_cwe_from_classname(class_name: str) -> Optional[str]:
    """Juliet-specific: e.g., CWE15_X -> CWE-15"""
    m = re.match(r"CWE(\d+)", class_name) if class_name else None
    return f"CWE-{m.group(1)}" if m else None


def enforce_cwe_rules(df: pd.DataFrame) -> pd.DataFrame:
    """
    If not vulnerable -> cwe_id becomes None.
    Accepts DataFrame with 'is_vulnerable' boolean column.
    """
    df = df.copy()
    if "is_vulnerable" in df.columns and "cwe_id" in df.columns:
        df.loc[~df["is_vulnerable"], "cwe_id"] = None
    return df