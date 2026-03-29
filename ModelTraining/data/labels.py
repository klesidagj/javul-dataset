import logging
from .db import get_conn
from .config import DBConfig, REQUIRED_GRAPHS, TABLE_NAME

logger = logging.getLogger(__name__)

def binary_vulnerability_labels(cfg: DBConfig) -> dict:
    sql = f"""
    SELECT is_vulnerable, COUNT(*)
    FROM {TABLE_NAME}
    WHERE {" AND ".join(REQUIRED_GRAPHS)}
    GROUP BY is_vulnerable;
    """
    with get_conn(cfg) as conn:
        cur = conn.cursor()
        cur.execute(sql)
        counts = dict(cur.fetchall())

    classes = ["NOT_VULNERABLE", "VULNERABLE"]
    if None in counts:
        classes.append("NA")

    mapping = {label: i for i, label in enumerate(classes)}
    logger.info("Binary labels mapping: %s", mapping)
    logger.info("Binary class counts: %s", counts)
    return mapping

def topk_cwe_labels(cfg: DBConfig, k: int = 3) -> dict:
    sql = f"""
    SELECT cwe_id, COUNT(*) AS cnt
    FROM {TABLE_NAME}
    WHERE cwe_id IS NOT NULL
      AND {" AND ".join(REQUIRED_GRAPHS)}
    GROUP BY cwe_id
    ORDER BY cnt DESC
    LIMIT %s;
    """
    with get_conn(cfg) as conn:
        cur = conn.cursor()
        cur.execute(sql, (k,))
        rows = cur.fetchall()

    mapping = {cwe: idx for idx, (cwe, _) in enumerate(rows)}
    logger.info("Top-%d CWE classes selected: %s", k, mapping)
    return mapping