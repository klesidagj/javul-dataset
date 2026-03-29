import json
import torch
import logging

logger = logging.getLogger(__name__)

class ASTGraphTransform:
    def __init__(self, type2id, max_nodes, verbose=False):
        self.type2id = type2id
        self.max_nodes = max_nodes
        self.pad = type2id["PAD"]
        self.unk = type2id["UNK"]
        self.verbose = verbose

    def __call__(self, graph_json):
        graph = json.loads(graph_json) if isinstance(graph_json, str) else graph_json

        x = torch.full((self.max_nodes,), self.pad, dtype=torch.long)
        mask = torch.zeros(self.max_nodes, dtype=torch.bool)

        types, edge_pairs = [], []

        if isinstance(graph, dict) and "nodes" in graph:
            for i, node in enumerate(graph["nodes"][: self.max_nodes]):
                t = (
                    node.get("type")
                    or node.get("label")
                    or node.get("!")
                    or "Unknown"
                )
                types.append(self.type2id.get(t, self.unk))
        else:
            def walk(n):
                if not isinstance(n, dict):
                    return
                t = n.get("!")
                if t:
                    types.append(self.type2id.get(t, self.unk))
                for v in n.values():
                    if isinstance(v, dict):
                        walk(v)
                    elif isinstance(v, list):
                        for i in v:
                            walk(i)
            walk(graph)

        n = min(len(types), self.max_nodes)
        if n == 0:
            x[0] = self.unk
            mask[0] = True
            n = 1
        else:
            x[:n] = torch.tensor(types[:n])
            mask[:n] = True

        edge_index = (
            torch.tensor(edge_pairs, dtype=torch.long).t()
            if edge_pairs
            else torch.zeros((2, 0), dtype=torch.long)
        )

        if self.verbose:
            logger.info("[AST] nodes=%d unique=%d", n, len(set(types[:n])))

        return x, edge_index, mask