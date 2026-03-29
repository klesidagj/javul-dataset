import json
import logging
from typing import Dict, Set

logger = logging.getLogger(__name__)

PAD, UNK = "PAD", "UNK"

def build_vocab(types: Set[str]) -> Dict[str, int]:
    vocab = {PAD: 0, UNK: 1}
    for idx, t in enumerate(sorted(types), start=2):
        vocab[t] = idx
    logger.info("Built vocab | size=%d", len(vocab))
    return vocab

# def extract_ast_types(ast_jsons: list) -> Set[str]:
#     logger.info("Extracting AST node types from %d samples", len(ast_jsons))
#     types = set()
#
#     def walk(node):
#         if not isinstance(node, dict):
#             return
#         t = node.get("!")
#         if t:
#             types.add(t)
#         for v in node.values():
#             if isinstance(v, dict):
#                 walk(v)
#             elif isinstance(v, list):
#                 for i in v:
#                     walk(i)
#
#     for js in ast_jsons:
#         walk(js if isinstance(js, dict) else json.loads(js))
#
#     logger.info("AST node types discovered: %d", len(types))
#     return types