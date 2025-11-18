# cve_extract.py
import sqlite3
import pandas as pd
import uuid
from typing import List, Dict, Any
from db_utils import enforce_cwe_rules

def _normalize_cwe_token(token: str):
    import re
    if not token:
        return None
    m = re.search(r'(\d+)', str(token))
    return f"CWE-{int(m.group(1))}" if m else None

def extract_java_snippets_from_sqlite(sqlite_path: str) -> List[Dict[str, Any]]:
    """
    Extract Java method/code snippets from CVEfixes SQLite file.
    Returns list of dicts ready for insertion (but not yet enforced).
    """
    conn = sqlite3.connect(sqlite_path)
    query = """
    SELECT 
        m.code           AS raw_code,
        m.before_change  AS is_vulnerable,
        fi.cve_id        AS source,
        cc.cwe_id
    FROM method_change m
    JOIN file_change f      ON m.file_change_id   = f.file_change_id
    JOIN fixes fi           ON f.hash             = fi.hash
    JOIN cwe_classification cc ON fi.cve_id        = cc.cve_id
    WHERE f.programming_language = 'Java'
      AND m.code IS NOT NULL
      AND cc.cwe_id IS NOT NULL
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    records = []
    for _, row in df.iterrows():
        raw = row["raw_code"]
        is_vuln_raw = row["is_vulnerable"]
        # Try to coerce to boolean
        if isinstance(is_vuln_raw, (int, float)):
            is_vuln = bool(int(is_vuln_raw))
        else:
            is_vuln = str(is_vuln_raw).strip().lower() in ("1", "true", "yes")
        cwe = _normalize_cwe_token(row.get("cwe_id"))
        records.append({
            "id": str(uuid.uuid4()),
            "raw_code": raw,
            "ast_graph": None,
            "cfg_graph": None,
            "dfg_graph": None,
            "css_vector": None,
            "cwe_id": cwe,
            "is_vulnerable": is_vuln,
            "source": f"CVEfixes:{row.get('source')}"
        })

    # enforce rules (cwe null when not vulnerable, format normalization)
    records = enforce_cwe_rules(records)
    return records