# ModelTraining/train/training.py
import logging
import torch

from  ModelTraining.train.losses import build_criterion
from ModelTraining.train.loops import  train_epoch, validate_epoch, create_train_val_split
from ModelTraining.train.metrics import TrainingMetrics, get_class_weights
from ModelTraining.train.checkpointing import save_checkpoint

logger = logging.getLogger(__name__)


def train_model(model, dataset, task, config, device):
    """
    Pure training loop.
    No inference.
    No user interaction.
    """

    train_ds, val_ds = create_train_val_split(dataset, config["val_split"])

    class_weights = get_class_weights(train_ds).to(device)
    criterion = build_criterion(task, class_weights, config["label_smoothing"])

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=config["patience"]
    )

    metrics = TrainingMetrics()
    best_acc = 0.0

    for epoch in range(config["epochs"]):
        logger.info("Epoch %d / %d", epoch + 1, config["epochs"])

        train_loss, train_acc = train_epoch(
            model, train_ds, optimizer, criterion, task, device, config["grad_clip"]
        )

        val_loss, val_acc, preds, labels = validate_epoch(
            model, val_ds, criterion, task, device
        )

        scheduler.step(val_acc)
        metrics.update(
            train_loss,
            train_acc,
            val_loss,
            val_acc,
            optimizer.param_groups[0]["lr"],
        )

        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(
                path=f"{config['checkpoint_dir']}/best.pth",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_acc=best_acc,
                metrics=metrics,
                task=task,
                model_cfg={"num_classes": task.num_classes},
            )

    return model, metrics, best_acc