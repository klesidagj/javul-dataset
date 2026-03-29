# ModelTraining/data/config.py

from dataclasses import dataclass
import logging
from typing import Dict


# ───────────────── DB CONFIG ─────────────────

@dataclass(frozen=True)
class DBConfig:
    host: str = "localhost"
    port: int = 5432
    dbname: str = "postgres"
    user: str = "klesi"
    password: str = ""

# Filter to access only rows with all graphs
REQUIRED_GRAPHS = (
    "ast_graph IS NOT NULL",
    "cfg_graph IS NOT NULL",
    "dfg_graph IS NOT NULL",
)

TABLE_NAME = "javul_cl"

# ───────────────── LOGGING ─────────────────

LOG_FORMAT = "[%(asctime)s] %(levelname)s | %(name)s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

# ───────────────── TRAINING CONFIG ─────────────────

CUSTOM_TRAINING_CONFIG = {
    # ── Task ─────────────────────────
    # "task_mode": "binary",
    "task_mode": "multiclass",
    "top_k": 10,

    # ── DB ───────────────────────────
    "db": DBConfig(),

    # ── Vocabularies (REQUIRED) ──────

    "ast_vocab": Dict[str, int],
    "cfg_vocab": Dict[str, int],
    "dfg_vocab": Dict[str, int],

    # ── Model ────────────────────────
    "d_model": 768,
    "n_heads": 8,

    # ── Data ─────────────────────────
    "batch_size": 16,
    "val_split": 0.2,
    "max_nodes": 512,

    # ── Training ─────────────────────
    "epochs": 10,
    "learning_rate": 3e-4,
    "weight_decay": 1e-2,
    "patience": 5,
    "grad_clip": 1.0,
    "label_smoothing": 0.0,

    # ── Loss ─────────────────────────
    "loss_type": "ce",

    # ── Checkpoints ──────────────────
    "checkpoint_dir": "./checkpoints",
    "checkpoint_path": None,
}