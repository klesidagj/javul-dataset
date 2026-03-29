# ModelTraining/dataloading/inference.py
import gc
import time
import logging
import torch
import numpy as np
from collections import Counter
from torch.utils.data import DataLoader
from tqdm import tqdm

from .collate import optimized_collate

logger = logging.getLogger(__name__)


def run_optimized_inference(
    model,
    dataset,
    batch_size: int,
    device: str | None = None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(
        "Starting inference | device=%s samples=%d batch=%d params=%d",
        device,
        len(dataset),
        batch_size,
        sum(p.numel() for p in model.parameters()),
    )

    model.to(device)
    model.eval()

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=optimized_collate,
        pin_memory=(device == "cuda"),
        drop_last=False,
    )

    total_correct = 0
    total_samples = 0
    batch_times = []

    all_preds = []
    all_labels = []

    start = time.time()
    pbar = tqdm(total=len(loader), desc="Inference")

    with torch.no_grad():
        for idx, (ast_b, cfg_b, dfg_b, css_b, labels) in enumerate(loader):
            t0 = time.time()

            ast_x, _, ast_m = ast_b
            cfg_x, _, cfg_m = cfg_b
            dfg_x, _, dfg_m = dfg_b

            ast_x = ast_x.to(device)
            ast_m = ast_m.to(device)
            cfg_x = cfg_x.to(device)
            cfg_m = cfg_m.to(device)
            dfg_x = dfg_x.to(device)
            dfg_m = dfg_m.to(device)
            css_b = css_b.to(device)
            labels = labels.to(device)

            assert not torch.isnan(ast_x).any()
            assert not torch.isnan(css_b).any()

            logits = model(
                ast_x, ast_m,
                cfg_x, cfg_m,
                dfg_x, dfg_m,
                css_b,
            )

            assert not torch.isnan(logits).any()

            preds = logits.argmax(dim=-1)

            correct = (preds == labels).sum().item()
            total_correct += correct
            total_samples += labels.size(0)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

            dt = time.time() - t0
            batch_times.append(dt)

            acc = 100.0 * total_correct / total_samples
            eta = np.mean(batch_times[-10:]) * (len(loader) - idx - 1)

            if idx % 50 == 0:
                logger.info(
                    "Batch=%d Acc=%.2f%% ETA=%.1fm pred_dist=%s",
                    idx,
                    acc,
                    eta / 60,
                    dict(Counter(preds.cpu().tolist())),
                )

            if idx % 100 == 0:
                gc.collect()
                if device == "cuda":
                    torch.cuda.empty_cache()

            pbar.set_postfix(acc=f"{acc:.2f}%", eta=f"{eta/60:.1f}m")
            pbar.update(1)

    pbar.close()

    total_time = time.time() - start
    final_acc = 100.0 * total_correct / total_samples

    logger.info(
        "Inference complete | acc=%.2f%% samples=%d time=%.1fm",
        final_acc,
        total_samples,
        total_time / 60,
    )

    return {
        "predictions": all_preds,
        "labels": all_labels,
        "accuracy": final_acc,
        "samples_processed": total_samples,
        "total_time": total_time,
    }