# db_utils.py
import os
import logging
from typing import Iterable, Dict, Any, Optional, List
import psycopg2
from psycopg2.extras import execute_values, Json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("etl")

# Prefer env vars for credentials
PG_CREDENTIALS = {
    "dbname": os.getenv("PG_DATABASE", "postgres"),
    "user": os.getenv("PG_USER", "klesi"),
    "password": os.getenv("PG_PASSWORD", ""),
    "host": os.getenv("PG_HOST", "localhost"),
    "port": os.getenv("PG_PORT", "5432"),
}

INSERT_SQL = """
INSERT INTO public.code_snippets (
    id, raw_code, ast_graph, cfg_graph, dfg_graph, css_vector,
    cwe_id, is_vulnerable, source
) VALUES %s
ON CONFLICT (id) DO NOTHING
"""

def _prepare_row_tuple(row: Dict[str, Any]):
    """
    Prepare tuple in deterministic column order expected by INSERT_SQL.
    ast/cfg/dfg are converted via Json(...) for JSONB columns if present.
    css_vector should be a Python list (or None).
    """
    return (
        row.get("id"),
        row.get("raw_code"),
        Json(row.get("ast_graph")) if row.get("ast_graph") is not None else None,
        Json(row.get("cfg_graph")) if row.get("cfg_graph") is not None else None,
        Json(row.get("dfg_graph")) if row.get("dfg_graph") is not None else None,
        row.get("css_vector"),
        row.get("cwe_id"),
        row.get("is_vulnerable"),
        row.get("source"),
    )

def insert_rows_batch(rows: Iterable[Dict[str, Any]],
                      pg_credentials: Optional[dict] = None,
                      chunk_size: int = 1000):
    """
    Insert rows (iterable of dicts) into Postgres in chunks.
    """
    creds = pg_credentials or PG_CREDENTIALS
    all_rows = list(rows)
    if not all_rows:
        logger.info("No rows to insert.")
        return

    conn = psycopg2.connect(**creds)
    try:
        with conn.cursor() as cur:
            # Template uses explicit cast for jsonb and relies on psycopg2 for arrays
            template = "(%s, %s, %s::jsonb, %s::jsonb, %s::jsonb, %s::double precision[], %s, %s, %s)"
            total = len(all_rows)
            for i in range(0, total, chunk_size):
                chunk = all_rows[i:i+chunk_size]
                values = [_prepare_row_tuple(r) for r in chunk]
                execute_values(cur, INSERT_SQL, values, template=template)
                logger.info("Inserted rows %d..%d", i, i + len(chunk) - 1)
        conn.commit()
        logger.info("Successfully inserted %d rows.", len(all_rows))
    except Exception:
        conn.rollback()
        logger.exception("Failed insert; rolled back.")
        raise
    finally:
        conn.close()

# -------------------------
# Validation helpers
# -------------------------
def normalize_cwe(cwe_raw: Optional[str]) -> Optional[str]:
    """Return normalized 'CWE-<num>' or None."""
    if not cwe_raw:
        return None
    import re
    m = re.search(r'(\d+)', str(cwe_raw))
    return f"CWE-{int(m.group(1))}" if m else None

def enforce_cwe_rules(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Enforce:
      - is_vulnerable is boolean
      - if not vulnerable -> cwe_id becomes None
      - cwe_id normalized to 'CWE-<num>' when present
    Returns new list (modifies copies).
    """
    out = []
    for r in records:
        rec = dict(r)  # shallow copy
        # normalize bool
        v = rec.get("is_vulnerable")
        if isinstance(v, str):
            rec["is_vulnerable"] = str(v).strip().lower() in ("1", "true", "yes")
        else:
            rec["is_vulnerable"] = bool(v)
        # normalize cwe
        if rec["is_vulnerable"]:
            rec["cwe_id"] = normalize_cwe(rec.get("cwe_id"))
        else:
            rec["cwe_id"] = None
        # ensure placeholders are explicit None
        for k in ("ast_graph", "cfg_graph", "dfg_graph", "css_vector"):
            if k not in rec:
                rec[k] = None
        out.append(rec)
    return out

def summarize_by_source(records: List[Dict[str, Any]]):
    """
    Print a summary per source and warn about missing CWE for vulnerable entries.
    """
    from collections import defaultdict
    per = defaultdict(list)
    for r in records:
        per[r.get("source", "UNKNOWN")].append(r)
    for src, items in per.items():
        total = len(items)
        vuln = sum(1 for x in items if x.get("is_vulnerable"))
        non_vuln = total - vuln
        missing_cwe = sum(1 for x in items if x.get("is_vulnerable") and not x.get("cwe_id"))
        logger.info("Source=%s: total=%d vulnerable=%d non_vulnerable=%d missing_cwe_for_vuln=%d",
                    src, total, vuln, non_vuln, missing_cwe)
        if missing_cwe:
            logger.warning("Source=%s has %d vulnerable entries missing cwe_id. These will be inserted with cwe_id=NULL.",
                           src, missing_cwe)