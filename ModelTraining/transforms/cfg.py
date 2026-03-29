import json
import torch
import logging
from collections import Counter

logger = logging.getLogger(__name__)

class CFGGraphTransform:
    def __init__(self, type2id, max_nodes, verbose=False):
        self.type2id = type2id
        self.max_nodes = max_nodes
        self.pad = type2id["PAD"]
        self.unk = type2id["UNK"]
        self.verbose = verbose

    def __call__(self, graph_json):
        graph = json.loads(graph_json) if isinstance(graph_json, str) else graph_json
        nodes = graph.get("nodes", [])
        edges = graph.get("edges", [])

        x = torch.full((self.max_nodes,), self.pad, dtype=torch.long)
        mask = torch.zeros(self.max_nodes, dtype=torch.bool)

        mapped = [
            self.type2id.get(n.get("type") or n.get("label"), self.unk)
            for n in nodes[: self.max_nodes]
        ]

        n = len(mapped)
        if n > 0:
            x[:n] = torch.tensor(mapped)
            mask[:n] = True

        pairs = []
        for e in edges:
            try:
                src = int(e.get("from"))
                dst = int(e.get("to"))
            except (TypeError, ValueError):
                continue  # malformed edge

            if 0 <= src < self.max_nodes and 0 <= dst < self.max_nodes:
                pairs.append((src, dst))

        edge_index = (
            torch.tensor(pairs, dtype=torch.long).t()
            if pairs
            else torch.zeros((2, 0), dtype=torch.long)
        )

        if self.verbose:
            logger.info("[CFG] nodes=%d uniq=%d", n, len(set(mapped)))
            logger.info("[CFG] dist=%s", Counter(mapped))

        return x, edge_index, mask