# train/loops.py
from typing import Tuple
from torch.utils.data import Subset
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

def log_binary_class_accuracy(preds: torch.Tensor, labels: torch.Tensor):
    """
    Logs per-class accuracy for binary classification.
    Assumes:
      - labels ∈ {0,1}
      - preds  ∈ {0,1}
    """

    preds = preds.view(-1)
    labels = labels.view(-1)

    neg_mask = labels == 0
    pos_mask = labels == 1

    neg_total = neg_mask.sum().item()
    pos_total = pos_mask.sum().item()

    neg_correct = ((preds == labels) & neg_mask).sum().item()
    pos_correct = ((preds == labels) & pos_mask).sum().item()

    neg_acc = 100.0 * neg_correct / max(1, neg_total)
    pos_acc = 100.0 * pos_correct / max(1, pos_total)

    logger.info(
        "Binary acc | neg=%.2f%% (%d/%d) pos=%.2f%% (%d/%d)",
        neg_acc, neg_correct, neg_total,
        pos_acc, pos_correct, pos_total,
    )

def forward_and_predict(model, batch, task, device):
    ast_b, cfg_b, dfg_b, css_b, labels = batch
    ast_x, _, ast_m = ast_b
    cfg_x, _, cfg_m = cfg_b
    dfg_x, _, dfg_m = dfg_b

    ast_x, ast_m = ast_x.to(device), ast_m.to(device)
    cfg_x, cfg_m = cfg_x.to(device), cfg_m.to(device)
    dfg_x, dfg_m = dfg_x.to(device), dfg_m.to(device)
    css_b = css_b.to(device)
    labels = labels.to(device)

    # 🔑 ENSURE LABEL IS BATCHED
    if labels.dim() == 0:
        labels = labels.unsqueeze(0)

    logits = model(ast_x, ast_m, cfg_x, cfg_m, dfg_x, dfg_m, css_b)
    preds = logits.argmax(dim=-1)

    return logits, logits, preds, labels


def train_epoch(model, loader, optimizer, criterion, task, device, grad_clip=None):
    model.train()
    total_loss, total_correct, total_samples = 0, 0, 0

    for batch_idx, batch in enumerate(tqdm(loader, desc="Training")):
        optimizer.zero_grad(set_to_none=True)

        logits, loss_input, preds, labels = forward_and_predict(
            model, batch, task, device
        )

        loss = criterion(logits, labels)

        if torch.isnan(loss):
            logger.error(f"NaN loss at batch {batch_idx}")
            continue

        loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

    return (
        total_loss / max(1, len(loader)),
        100.0 * total_correct / max(1, total_samples),
    )

def validate_epoch(model, loader, criterion, task, device):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc="Validation")):
            logits, loss_input, preds, labels = forward_and_predict(
                model, batch, task, device
            )

            loss = criterion(logits, labels)

            if torch.isnan(loss):
                logger.error("NaN loss during validation at batch %d", batch_idx)
                continue

            total_loss += loss.item()
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / max(1, len(loader))
    acc = 100.0 * total_correct / max(1, total_samples)

    if task.num_classes == 2:
          log_binary_class_accuracy(
                     preds = torch.tensor(all_preds),
                 labels = torch.tensor(all_labels),
             )
    return avg_loss, acc, all_preds, all_labels



def create_train_val_split(
    dataset,
    val_split: float,
    seed: int = 42,
) -> Tuple[Subset, Subset]:
    """
    Stratified train/validation split.

    Assumes dataset[i] returns (..., label) as last element.
    """

    if not 0.0 < val_split < 1.0:
        raise ValueError(f"val_split must be in (0,1), got {val_split}")

    # Extract labels once
    labels = []
    for i in range(len(dataset)):
        sample = dataset[i]
        label = sample[-1]
        labels.append(int(label))

    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=val_split,
        random_state=seed,
    )

    train_idx, val_idx = next(splitter.split(range(len(labels)), labels))

    train_ds = Subset(dataset, train_idx.tolist())
    val_ds = Subset(dataset, val_idx.tolist())

    logger.info(
        "Dataset split | total=%d train=%d val=%d val_ratio=%.2f",
        len(dataset),
        len(train_ds),
        len(val_ds),
        val_split,
    )

    return train_ds, val_ds