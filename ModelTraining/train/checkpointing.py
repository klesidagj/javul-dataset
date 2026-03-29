# train/checkpoint.py
import torch, os, logging

logger = logging.getLogger(__name__)

def save_checkpoint(path, model, optimizer, scheduler, epoch, best_acc, metrics, task, model_cfg):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "best_acc": best_acc,
        "task": task.__dict__,
        "model_cfg": model_cfg,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
        "metrics": metrics,
    }, path)
    logger.info(f"💾 Saved checkpoint → {path}")