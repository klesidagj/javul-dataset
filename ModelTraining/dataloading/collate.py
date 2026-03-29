# ModelTraining/dataloading/collate.py
import logging
import torch
from typing import List, Tuple

logger = logging.getLogger(__name__)

#This function handles one graph view at a time (AST or CFG or DFG).
def collate_graphs(
    batch: List[Tuple[torch.Tensor, list, torch.Tensor]],
    view: str,
):
    """
    Collate a single graph view (AST / CFG / DFG).
    Stacks nodes and masks
    Edge lists are kept as-is (list of tensors).
    """
    try:
        xs, edge_lists, masks = zip(*batch)

        x_batch = torch.stack(xs, dim=0)
        mask_batch = torch.stack(masks, dim=0)

        if x_batch.size(0) != mask_batch.size(0):
            logger.warning(
                "[collate_graphs:%s] Batch size mismatch x=%s mask=%s",
                view, tuple(x_batch.shape), tuple(mask_batch.shape)
            )

        logger.debug(
            "[collate_graphs:%s] x=%s mask=%s edges=%d",
            view, tuple(x_batch.shape), tuple(mask_batch.shape), len(edge_lists)
        )

        return x_batch, list(edge_lists), mask_batch

    except Exception:
        logger.exception("[collate_graphs:%s] Failed", view)
        raise


def optimized_collate(batch):
    """
    Produces:
      (ast_x, ast_edges, ast_mask),
      (cfg_x, cfg_edges, cfg_mask),
      (dfg_x, dfg_edges, dfg_mask),
      css_batch,
      labels
    """
    if not batch:
        logger.error("optimized_collate: empty batch")
        raise ValueError("Empty batch received")

    try:
        ast_batch = collate_graphs([b[0] for b in batch], "ast")
        cfg_batch = collate_graphs([b[1] for b in batch], "cfg")
        dfg_batch = collate_graphs([b[2] for b in batch], "dfg")

        css_batch = torch.stack([b[3] for b in batch], dim=0)
        labels = torch.stack([b[4] for b in batch], dim=0)

        logger.info(
            "[Collate] B=%d | AST=%s CFG=%s DFG=%s CSS=%s Labels=%s",
            len(batch),
            tuple(ast_batch[0].shape),
            tuple(cfg_batch[0].shape),
            tuple(dfg_batch[0].shape),
            tuple(css_batch.shape),
            tuple(labels.shape),
        )

        logger.debug(
            "[Collate] label_dist=%s css_nan=%s",
            torch.unique(labels, return_counts=True),
            torch.isnan(css_batch).any().item(),
        )

        return ast_batch, cfg_batch, dfg_batch, css_batch, labels

    except Exception:
        logger.exception("optimized_collate failed")
        raise