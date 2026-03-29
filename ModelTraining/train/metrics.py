# train/metrics.py
import torch
import logging
from collections import Counter

logger = logging.getLogger(__name__)

class TrainingMetrics:
    def __init__(self):
        self.reset()

    def reset(self):
        self.losses = []
        self.accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        self.learning_rates = []

    def update(self, loss, acc, val_loss=None, val_acc=None, lr=None):
        self.losses.append(loss)
        self.accuracies.append(acc)
        if val_loss is not None:
            self.val_losses.append(val_loss)
        if val_acc is not None:
            self.val_accuracies.append(val_acc)
        if lr is not None:
            self.learning_rates.append(lr)

def get_class_weights(dataset) -> torch.Tensor:
    """
    Compute inverse-frequency class weights from a dataset or Subset.

    Assumes dataset[i] returns (..., label) where label is int or 0-dim tensor.
    """

    labels = []
    for i in range(len(dataset)):
        label = dataset[i][-1]
        labels.append(int(label))

    counts = Counter(labels)
    num_classes = max(counts.keys()) + 1
    total = sum(counts.values())

    weights = torch.zeros(num_classes, dtype=torch.float)

    for cls in range(num_classes):
        if cls in counts:
            weights[cls] = total / (num_classes * counts[cls])
        else:
            # Defensive: unseen class in train split
            weights[cls] = 0.0
            logger.warning("Class %d missing in training split", cls)
    if num_classes == 2:
        logger.info(
            "Binary acc | neg=%.2f%% pos=%.2f%%",)
    logger.info("Class distribution: %s", dict(counts))
    logger.info("Class weights: %s", weights.tolist())

    return weights