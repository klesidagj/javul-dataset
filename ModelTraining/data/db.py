import psycopg2
import logging
from contextlib import contextmanager
from .config import DBConfig, REQUIRED_GRAPHS, TABLE_NAME

logger = logging.getLogger(__name__)

@contextmanager
def get_conn(cfg: DBConfig):
    conn = psycopg2.connect(**cfg.__dict__)
    try:
        yield conn
    finally:
        conn.close()

def ensure_indices(cfg: DBConfig) -> None:
    sql = f"""
    CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_complete_graphs
    ON {TABLE_NAME}(id)
    WHERE {" AND ".join(REQUIRED_GRAPHS)};
    """
    with get_conn(cfg) as conn:
        cur = conn.cursor()
        cur.execute(sql)
        conn.commit()
    logger.info("Database indices verified")

def dataset_stats(cfg: DBConfig) -> tuple[int, int]:
    sql = f"""
    SELECT
        COUNT(*) AS total,
        COUNT(*) FILTER (WHERE {" AND ".join(REQUIRED_GRAPHS)}) AS complete
    FROM {TABLE_NAME};
    """
    with get_conn(cfg) as conn:
        cur = conn.cursor()
        cur.execute(sql)
        total, complete = cur.fetchone()
    logger.info("Dataset stats | total=%d complete=%d", total, complete)
    return complete, total

def fetch_column(cfg: DBConfig, column: str) -> list:
    sql = f"""
    SELECT {column}
    FROM {TABLE_NAME}
    WHERE {column} IS NOT NULL;
    """
    logger.info("Fetching column '%s'", column)
    with get_conn(cfg) as conn:
        cur = conn.cursor()
        cur.execute(sql)
        rows = [r[0] for r in cur.fetchall()]
    logger.info("Fetched %d rows for '%s'", len(rows), column)
    return rows

def fetch_distinct_node_types(cfg: DBConfig, graph_col: str) -> set[str]:
    sql = f"""
    SELECT DISTINCT COALESCE(n->>'type', n->>'label')
    FROM {TABLE_NAME}
    CROSS JOIN LATERAL jsonb_array_elements({graph_col}->'nodes') AS n
    WHERE {graph_col} IS NOT NULL
      AND COALESCE(n->>'type', n->>'label') IS NOT NULL;
    """
    logger.info("Discovering node types for %s", graph_col)
    with get_conn(cfg) as conn:
        cur = conn.cursor()
        cur.execute(sql)
        types = {r[0] for r in cur.fetchall()}
    logger.info("%s | discovered %d node types", graph_col, len(types))
    return types