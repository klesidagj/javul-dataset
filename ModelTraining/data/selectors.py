# ModelTraining/data/selectors.py
import logging
import torch
from typing import Dict
from ModelTraining.data.config import REQUIRED_GRAPHS, TABLE_NAME

logger = logging.getLogger(__name__)


class BaseSelector:
    """Defines dataset selection policy and label extraction."""

    name: str = "base"

    def where_clause(self) -> str:
        raise NotImplementedError

    def label_from_row(self, row):
        raise NotImplementedError

    def log_stats(self, conn) -> None:
        pass


class BinaryVulnerabilitySelector(BaseSelector):
    name = "binary_vulnerability"

    def where_clause(self) -> str:
        return " AND ".join(REQUIRED_GRAPHS)

    def label_from_row(self, row):
        v = row["is_vulnerable"]
        label = 1 if v is True else 0
        return torch.tensor(label, dtype=torch.long)

    def log_stats(self, conn) -> None:
        sql = f"""
        SELECT
            COUNT(*) FILTER (WHERE is_vulnerable IS TRUE),
            COUNT(*) FILTER (WHERE is_vulnerable IS FALSE),
            COUNT(*) FILTER (WHERE is_vulnerable IS NULL)
        FROM {TABLE_NAME}
        WHERE {" AND ".join(REQUIRED_GRAPHS)};
        """
        with conn.cursor() as cur:
            cur.execute(sql)
            t, f, n = cur.fetchone()
        logger.info(
            "[BinarySelector] vulnerable=%d not_vulnerable=%d null=%d",
            t, f, n
        )


class TopCWEMulticlassSelector(BaseSelector):
    name = "topk_cwe_multiclass"

    def __init__(self, cwe2id: Dict[str, int]):
        self.cwe2id = cwe2id

    def where_clause(self) -> str:
        keys = ",".join(f"'{k}'" for k in self.cwe2id)
        return f"cwe_id IN ({keys}) AND " + " AND ".join(REQUIRED_GRAPHS)

    def label_from_row(self, row):
        cwe = row["cwe_id"]
        if cwe not in self.cwe2id:
            raise KeyError(f"Unexpected CWE '{cwe}'")
        return torch.tensor(self.cwe2id[cwe], dtype=torch.long)

    def log_stats(self, conn) -> None:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT
                    cwe_id,
                    COUNT(*)
                FROM {TABLE_NAME}
                WHERE cwe_id = ANY(%s)
                GROUP BY cwe_id
                """,
                (list(self.cwe2id.keys()),)
            )
            for cwe, cnt in cur.fetchall():
                logger.info("[CWESelector] %s=%d", cwe, cnt)