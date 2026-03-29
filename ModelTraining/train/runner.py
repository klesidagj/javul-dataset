import os
import logging
import torch

logger = logging.getLogger(__name__)


class TrainingPipelineRunner:
    """
    Orchestrates:
      - dataset creation (via factory)
      - model creation
      - training
      - inference
    """

    def __init__(
        self,
        task_config,
        dataset_factory,
        model_cls,
        train_fn,
        inference_fn,
        training_config,
        device=None,
    ):
        self.task = task_config
        self.dataset_factory = dataset_factory
        self.model_cls = model_cls
        self.train_fn = train_fn
        self.inference_fn = inference_fn
        self.cfg = training_config
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # ──────────────────────────────
    def build_dataset(self):
        logger.info("📦 Building dataset via factory | task=%s", self.task.task_type)
        return self.dataset_factory(self.task)

    # ──────────────────────────────
    def build_model(self):
        logger.info(
            "🤖 Building model | classes=%d | task=%s",
            self.task.num_classes,
            self.task.task_type,
        )

        model = self.model_cls(
            self.cfg["ast_vocab"],
            self.cfg["cfg_vocab"],
            self.cfg["dfg_vocab"],
            self.cfg["d_model"],
            self.cfg["n_heads"],
            self.task.num_classes,
        )

        return model.to(self.device)

    # ──────────────────────────────
    def load_checkpoint(self, path, model):
        logger.info("📂 Loading checkpoint: %s", path)
        ckpt = torch.load(path, map_location=self.device)
        model.load_state_dict(ckpt["model"])
        return model

    # ──────────────────────────────
    def train(self, model, dataset):
        return self.train_fn(
            model=model,
            dataset=dataset,
            task=self.task,
            config=self.cfg,
            device=self.device,
        )

    def infer(self, model, dataset):
        return self.inference_fn(
            model=model,
            dataset=dataset,
            batch_size=self.cfg["batch_size"],
            device=self.device,
        )

    # ──────────────────────────────
    def run(self, mode="train"):
        dataset = self.build_dataset()
        model = self.build_model()

        logger.info(
            "Model parameters: %d",
            sum(p.numel() for p in model.parameters()),
        )

        if mode == "inference":
            return model, None, self.infer(model, dataset)

        if mode == "resume":
            ckpt = self.cfg.get("checkpoint_path")
            if not ckpt or not os.path.exists(ckpt):
                raise FileNotFoundError("Checkpoint not found")
            model = self.load_checkpoint(ckpt, model)

        model, metrics, best_acc = self.train(model, dataset)
        logger.info("🏆 Best validation accuracy: %.2f%%", best_acc)

        inference = self.infer(model, dataset)
        return model, metrics, inference