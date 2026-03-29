# ModelTraining/train/task.py
from dataclasses import dataclass
from typing import Literal, Optional, Dict
from ModelTraining.data.labels import binary_vulnerability_labels, topk_cwe_labels


@dataclass(frozen=True)
class TaskConfig:
    """
    Pure task description.
    No dataset logic.
    No DB logic.
    """

    task_type: Literal["binary", "multiclass"]
    num_classes: int
    loss_type: Literal["ce", "bce", "focal"]
    label_names: Optional[list[str]] = None

    # Optional: only used for multiclass
    label_to_id: Optional[Dict[str, int]] = None

    @property
    def uses_bce(self) -> bool:
        return False

def build_task_config(
    mode: Literal["binary", "multiclass"],
    db_cfg,
    top_k: int = 3,
) -> TaskConfig:

    if mode == "binary":
        label_to_id = binary_vulnerability_labels(db_cfg)

        return TaskConfig(
            task_type="binary",
            num_classes=2,
            loss_type="ce",
            label_names=list(label_to_id.keys()),
            label_to_id=label_to_id,
        )

    if mode == "multiclass":
        label_to_id = topk_cwe_labels(db_cfg, k=top_k)

        return TaskConfig(
            task_type="multiclass",
            num_classes=len(label_to_id),
            loss_type="ce",
            label_names=list(label_to_id.keys()),
            label_to_id=label_to_id,
        )

    raise ValueError(f"Unknown task mode: {mode}")