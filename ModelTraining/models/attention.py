# ModelTraining/models/attention.py
import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class OptimizedQuadSelfAttention(nn.Module):
    """
    Fuses four different views of a code snippet:
    AST, CFG, DFG, and CSS.
    """

    def __init__(self, d_model, n_heads, dropout=0.1, debug=False):
        super().__init__()
        self.debug = debug

        self.attn_ast = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.attn_cfg = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.attn_dfg = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.attn_css = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

        self.norm_ast = nn.LayerNorm(d_model)
        self.norm_cfg = nn.LayerNorm(d_model)
        self.norm_dfg = nn.LayerNorm(d_model)
        self.norm_css = nn.LayerNorm(d_model)

        self.fuse = nn.Sequential(
            nn.Linear(4 * d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
        )

    # -------------------- utilities --------------------

    @staticmethod
    def _sanitize(x: torch.Tensor) -> torch.Tensor:
        return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    @staticmethod
    def _masked_mean(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        if mask is None:
            return x.mean(dim=1)
        m = mask.to(x.dtype).unsqueeze(-1)
        denom = m.sum(dim=1).clamp(min=1.0)
        return (x * m).sum(dim=1) / denom

    @staticmethod
    def _ensure_nonempty_mask(mask: torch.Tensor | None, device):
        if mask is None:
            return None
        m = mask.clone()
        if m.ndim == 1:
            m = m.unsqueeze(0)
        bad = (m.sum(dim=1) == 0)
        if bad.any():
            m[bad, 0] = True
        return m.to(device)

    # -------------------- forward --------------------

    def forward(self, ast, cfg, dfg, css, mask_ast=None, mask_cfg=None, mask_dfg=None):
        ast = self._sanitize(ast)
        cfg = self._sanitize(cfg)
        dfg = self._sanitize(dfg)
        css = self._sanitize(css)

        mask_ast = self._ensure_nonempty_mask(mask_ast, ast.device)
        mask_cfg = self._ensure_nonempty_mask(mask_cfg, cfg.device)
        mask_dfg = self._ensure_nonempty_mask(mask_dfg, dfg.device)

        ast_att, _ = self.attn_ast(
            ast, ast, ast,
            key_padding_mask=(~mask_ast.bool()) if mask_ast is not None else None
        )
        ast_out = self.norm_ast(ast + ast_att)

        cfg_att, _ = self.attn_cfg(
            cfg, cfg, cfg,
            key_padding_mask=(~mask_cfg.bool()) if mask_cfg is not None else None
        )
        cfg_out = self.norm_cfg(cfg + cfg_att)

        dfg_att, _ = self.attn_dfg(
            dfg, dfg, dfg,
            key_padding_mask=(~mask_dfg.bool()) if mask_dfg is not None else None
        )
        dfg_out = self.norm_dfg(dfg + dfg_att)

        css_att, _ = self.attn_css(css, css, css)
        css_out = self.norm_css(css + css_att)

        ast_vec = self._masked_mean(ast_out, mask_ast)
        cfg_vec = self._masked_mean(cfg_out, mask_cfg)
        dfg_vec = self._masked_mean(dfg_out, mask_dfg)

        css_mask = torch.ones(css_out.shape[:2], device=css_out.device, dtype=torch.bool)
        css_vec = self._masked_mean(css_out, css_mask)

        fused = torch.cat([ast_vec, cfg_vec, dfg_vec, css_vec], dim=-1)
        fused = self._sanitize(self.fuse(fused))

        if self.debug:
            logger.info(
                "[QuadAttention] fused=%s NaN=%s",
                tuple(fused.shape),
                torch.isnan(fused).any().item(),
            )

        return fused