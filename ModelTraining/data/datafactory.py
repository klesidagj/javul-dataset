# ModelTraining/data/factory.py
import logging

from ModelTraining.datasets import (
    BinaryCodeSnippetDataset,
    MulticlassCodeSnippetDataset,
)
from ModelTraining.train.task import TaskConfig

logger = logging.getLogger(__name__)


class CodeSnippetDatasetFactory:
    def __init__(self, training_config):

        self.cfg = training_config
        self.db_cfg = training_config["db"]

    #Checks task type Instantiates BinaryCodeSnippetDataset or MulticlassCodeSnippetDataset
    def __call__(self, task: TaskConfig):
        logger.info(
            "Building dataset | task=%s classes=%d",
            task.num_classes,
        )

        if task.task_type == "binary":
            return BinaryCodeSnippetDataset(
                db_params=self.db_cfg,
                ast_type2id=self.cfg["ast_vocab"],
                cfg_type2id=self.cfg["cfg_vocab"],
                dfg_type2id=self.cfg["dfg_vocab"],
                max_nodes=self.cfg["max_nodes"],
            )

        if task.task_type == "multiclass":
            return MulticlassCodeSnippetDataset(
                cwe2id=task.label_to_id,
                db_params=self.db_cfg,
                ast_type2id=self.cfg["ast_vocab"],
                cfg_type2id=self.cfg["cfg_vocab"],
                dfg_type2id=self.cfg["dfg_vocab"],
                max_nodes=self.cfg["max_nodes"],
            )