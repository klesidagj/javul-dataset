# train/losses.py
import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce)
        loss = self.alpha * (1 - pt) ** self.gamma * ce

        if torch.isnan(loss).any() or torch.isinf(loss).any():
            logger.error("FocalLoss produced NaN/Inf")

        return loss.mean() if self.reduction == 'mean' else loss.sum()


def build_criterion(task, class_weights, label_smoothing=0.0):
    if task.loss_type == "focal":
        logger.info("Using FocalLoss")
        return FocalLoss(alpha=1.0, gamma=2.0)

    logger.info("Using CrossEntropyLoss")
    return nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=label_smoothing,
    )