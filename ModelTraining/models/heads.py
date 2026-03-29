# ModelTraining/models/heads.py
import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class OptimizedClassificationHead(nn.Module):
    def __init__(self, d_model, num_classes, dropout=0.2, debug=False):
        super().__init__()
        self.debug = debug

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

    @staticmethod
    def _sanitize(x: torch.Tensor) -> torch.Tensor:
        return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    def forward(self, x):
        logits = self._sanitize(self.classifier(self._sanitize(x)))

        if self.debug:
            logger.info(
                "[Classifier] logits=%s NaN=%s Inf=%s",
                tuple(logits.shape),
                torch.isnan(logits).any().item(),
                torch.isinf(logits).any().item(),
            )

        return logits