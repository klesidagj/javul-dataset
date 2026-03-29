# ModelTraining/datasets/base.py
import json
import logging
import psycopg2
import psycopg2.extras
import torch
import numpy as np
from torch.utils.data import Dataset
from ModelTraining.data.config import TABLE_NAME
from ModelTraining.transforms.ast import ASTGraphTransform
from ModelTraining.transforms.cfg import CFGGraphTransform
from ModelTraining.transforms.dfg import DFGGraphTransform

logger = logging.getLogger(__name__)


class CodeSnippetDataset(Dataset):
    """Shared mechanics for all code-snippet datasets."""

    # Connects to the database
    def __init__(
        self,
        db_params,
        selector,
        ast_type2id,
        cfg_type2id,
        dfg_type2id,
        max_nodes,
        cache_size=1024,
        verbose=False,
    ):
        self.selector = selector
        self.conn = psycopg2.connect(**db_params.__dict__)
        self.verbose = verbose

        #Graph transforms initialized
        self.ast_t = ASTGraphTransform(ast_type2id, max_nodes, verbose)
        self.cfg_t = CFGGraphTransform(cfg_type2id, max_nodes, verbose)
        self.dfg_t = DFGGraphTransform(dfg_type2id, max_nodes, verbose)

        self.cache = {}
        self.cache_order = []
        self.cache_size = cache_size

        self._load_sample_ids()
        self.selector.log_stats(self.conn)

        logger.info(
            "Dataset initialized | selector=%s | samples=%d | cache=%d",
            selector.name, len(self.sample_ids), cache_size
        )

    def _load_sample_ids(self):
        sql = f"""
        SELECT id
        FROM {TABLE_NAME}
        WHERE {self.selector.where_clause()}
        ORDER BY id;
        """
        with self.conn.cursor() as cur:
            cur.execute(sql)
            self.sample_ids = [r[0] for r in cur.fetchall()]

    def __len__(self):
        return len(self.sample_ids)

    def _cache_get(self, idx):
        if idx in self.cache:
            self.cache_order.remove(idx)
            self.cache_order.append(idx)
            return self.cache[idx]
        return None

    def _cache_put(self, idx, item):
        if len(self.cache) >= self.cache_size:
            old = self.cache_order.pop(0)
            del self.cache[old]
        self.cache[idx] = item
        self.cache_order.append(idx)

    def __getitem__(self, idx):
        cached = self._cache_get(idx)
        if cached is not None:
            return cached

        sid = self.sample_ids[idx]
        with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute(
                f"""
                SELECT ast_graph, cfg_graph, dfg_graph, css_vector,
                       is_vulnerable, cwe_id
                FROM {TABLE_NAME}
                WHERE id = %s
                  AND {self.selector.where_clause()};
                """,
                (sid,),
            )
            row = cur.fetchone()

        ast = self.ast_t(row["ast_graph"])
        cfg = self.cfg_t(row["cfg_graph"])
        dfg = self.dfg_t(row["dfg_graph"])
        css = self._parse_css(row["css_vector"])
        label = self.selector.label_from_row(row)

        item = (ast, cfg, dfg, css, label)
        self._cache_put(idx, item)
        return item

    def _parse_css(self, css):
        D = int(globals().get("D_MODEL", 768))
        if css is None:
            return torch.zeros(1, D)

        if isinstance(css, str):
            s = css.strip()
            if s.startswith("{"):
                s = "[" + s[1:-1] + "]"
            css = json.loads(s)

        arr = []
        for v in css:
            try:
                fv = float(v)
                arr.append(0.0 if np.isnan(fv) or np.isinf(fv) else fv)
            except Exception:
                arr.append(0.0)

        arr = (arr[:D] if len(arr) > D else arr + [0.0] * (D - len(arr)))
        return torch.tensor(arr, dtype=torch.float).unsqueeze(0)

    def __del__(self):
        try:
            self.conn.close()
        except Exception:
            pass