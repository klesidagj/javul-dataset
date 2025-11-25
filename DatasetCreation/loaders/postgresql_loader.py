# DatasetCreation/loader/db_loader.py
from pathlib import Path
import psycopg2
from psycopg2.extras import execute_values, Json
from typing import Iterable, Dict, Any, Optional, List

from DatasetCreation.config.log import get_logger
from DatasetCreation.utils.cwe_utils import normalize_cwe_token

logger = get_logger("db_loader")


def load_sql_template(sql_path: str) -> str:
    return Path(sql_path).read_text(encoding="utf-8")


def _prepare_row_tuple(row: Dict[str, Any]):
    """
    Prepare tuple in deterministic column order expected by psql_loader.sql.
    Converts graphs to Json for JSONB columns and leaves arrays as-is.
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


def _normalize_and_prepare(records: Iterable[Dict[str, Any]]) -> List[tuple]:
    """
    Normalize records to DB expectations, return list of tuples ready for execute_values.
    """
    out = []
    for r in records:
        rec = dict(r)  # shallow copy
        # normalize is_vulnerable
        v = rec.get("is_vulnerable")
        if isinstance(v, str):
            rec["is_vulnerable"] = v.strip().lower() in ("1", "true", "yes")
        else:
            rec["is_vulnerable"] = bool(v)

        # normalize cwe
        rec["cwe_id"] = normalize_cwe_token(rec.get("cwe_id")) if rec.get("cwe_id") else None

        # ensure graph/vector keys exist
        rec.setdefault("ast_graph", None)
        rec.setdefault("cfg_graph", None)
        rec.setdefault("dfg_graph", None)
        rec.setdefault("css_vector", None)
        if rec["is_vulnerable"] and rec["cwe_id"] is None:
            logger.warning(f"Skipping snippet {rec['id']} because missing CWE for vulnerable code.")
        else:
            out.append(_prepare_row_tuple(rec))
    return out


def insert_rows_batch(records: Iterable[Dict[str, Any]],
                      conn_string: str,
                      sql_path: str,
                      chunk_size: int = 1000):
    """
    Insert records into postgres using the SQL template at sql_path.
    The SQL template must contain a '{table_name}' placeholder for format substitution.
    """
    sql_text = load_sql_template(sql_path)
    # The psql template should include the complete INSERT statement with VALUES %s
    insert_sql = sql_text  # keep as-is, assume user included {table_name} already filled in file if needed

    tuples = _normalize_and_prepare(records)
    if not tuples:
        logger.info("No records to insert.")
        return

    conn = psycopg2.connect(conn_string)
    try:
        with conn.cursor() as cur:
            # template for execute_values: use explicit casts for jsonb/array as needed by your psql_loader.sql
            template = "(%s, %s, %s::jsonb, %s::jsonb, %s::jsonb, %s::double precision[], %s, %s, %s)"
            total = len(tuples)
            for i in range(0, total, chunk_size):
                chunk = tuples[i:i + chunk_size]
                execute_values(cur, insert_sql, chunk, template=template)
                logger.info("Inserted rows %d..%d", i, i + len(chunk) - 1)
        conn.commit()
        logger.info("Successfully inserted %d rows.", total)
    except Exception:
        conn.rollback()
        logger.exception("Failed insert; rolled back.")
        raise
    finally:
        conn.close()