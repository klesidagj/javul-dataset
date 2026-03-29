# ModelTraining/models/multiview.py
import logging
import torch
import torch.nn as nn

from .attention import OptimizedQuadSelfAttention
from .heads import OptimizedClassificationHead

logger = logging.getLogger(__name__)


class OptimizedMultiViewCWEModel(nn.Module):
    """
    End-to-end model:
    Embedding → Quad Self-Attention Fusion → Classification
    """

    def __init__(
        self,
        ast_vocab,
        cfg_vocab,
        dfg_vocab,
        d_model,
        n_heads,
        num_classes,
        dropout=0.1,
        debug=False,
    ):
        super().__init__()
        self.debug = debug

        self.embed_ast = nn.Embedding(len(ast_vocab), d_model, padding_idx=0)
        self.embed_cfg = nn.Embedding(len(cfg_vocab), d_model, padding_idx=0)
        self.embed_dfg = nn.Embedding(len(dfg_vocab), d_model, padding_idx=0)

        self.fusion = OptimizedQuadSelfAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            debug=debug,
        )

        self.classifier = OptimizedClassificationHead(
            d_model=d_model,
            num_classes=num_classes,
            dropout=dropout,
            debug=debug,
        )

        self._init_weights()

        logger.info(
            "Initialized MultiViewModel | d_model=%d heads=%d classes=%d",
            d_model, n_heads, num_classes,
        )

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.padding_idx is not None:
                    with torch.no_grad():
                        m.weight[m.padding_idx].fill_(0.0)

    @staticmethod
    def _sanitize(x: torch.Tensor) -> torch.Tensor:
        return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    def forward(self, ast_x, mask_ast, cfg_x, mask_cfg, dfg_x, mask_dfg, css_x):
        css_x = self._sanitize(css_x)
        if css_x.dim() == 2:
            css_x = css_x.unsqueeze(1)

        ast_emb = self.embed_ast(ast_x)
        cfg_emb = self.embed_cfg(cfg_x)
        dfg_emb = self.embed_dfg(dfg_x)

        # 🔑 ENSURE BATCH DIM
        if ast_emb.dim() == 2:
            ast_emb = ast_emb.unsqueeze(0)
            cfg_emb = cfg_emb.unsqueeze(0)
            dfg_emb = dfg_emb.unsqueeze(0)

        if mask_ast is not None and mask_ast.dim() == 1:
            mask_ast = mask_ast.unsqueeze(0)
        if mask_cfg is not None and mask_cfg.dim() == 1:
            mask_cfg = mask_cfg.unsqueeze(0)
        if mask_dfg is not None and mask_dfg.dim() == 1:
            mask_dfg = mask_dfg.unsqueeze(0)

        fused = self.fusion(
            ast_emb, cfg_emb, dfg_emb, css_x,
            mask_ast=mask_ast,
            mask_cfg=mask_cfg,
            mask_dfg=mask_dfg,
        )

        logits = self.classifier(fused)

        if self.debug:
            assert not torch.isnan(logits).any(), "NaN detected in logits"
            assert not torch.isinf(logits).any(), "Inf detected in logits"

        return logits