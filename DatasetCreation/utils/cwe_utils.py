import re
import pandas as pd


def normalize_cwe_token(token: str):
    if not token:
        return None
    m = re.search(r"(\d+)", str(token))
    return f"CWE-{int(m.group(1))}" if m else None


def enforce_cwe_rules(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enforces: if not vulnerable => cwe_id = None
    """
    df = df.copy()
    df.loc[~df["is_vulnerable"], "cwe_id"] = None
    return df
