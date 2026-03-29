# train.py
import logging
from ModelTraining.data.db import fetch_distinct_node_types
from ModelTraining.data.vocab import build_vocab
from ModelTraining.train.runner import TrainingPipelineRunner
from ModelTraining.train.task import build_task_config
from ModelTraining.data.datafactory import CodeSnippetDatasetFactory
from ModelTraining.models.multiview import OptimizedMultiViewCWEModel
from ModelTraining.train.training import train_model
from ModelTraining.dataloading.inference import run_optimized_inference
from ModelTraining.data.config import CUSTOM_TRAINING_CONFIG

logger = logging.getLogger(__name__)


def main():
    logger.info("🔄 Initializing training pipeline")
    db_cfg = CUSTOM_TRAINING_CONFIG["db"]

    logger.info("🔤 Building AST/CFG/DFG vocabularies from database")
    #scans the database to learn what node types exist.
    ast_types = fetch_distinct_node_types(db_cfg, "ast_graph")
    cfg_types = fetch_distinct_node_types(db_cfg, "cfg_graph")
    dfg_types = fetch_distinct_node_types(db_cfg, "dfg_graph")
    #builds vocabs
    CUSTOM_TRAINING_CONFIG["ast_vocab"] = build_vocab(ast_types)
    CUSTOM_TRAINING_CONFIG["cfg_vocab"] = build_vocab(cfg_types)
    CUSTOM_TRAINING_CONFIG["dfg_vocab"] = build_vocab(dfg_types)

    logger.info(
        "Vocab sizes | AST=%d CFG=%d DFG=%d",
        len(CUSTOM_TRAINING_CONFIG["ast_vocab"]),
        len(CUSTOM_TRAINING_CONFIG["cfg_vocab"]),
        len(CUSTOM_TRAINING_CONFIG["dfg_vocab"]),
    )

    task = build_task_config(
        mode=CUSTOM_TRAINING_CONFIG["task_mode"],  # binary | multiclass
        db_cfg=CUSTOM_TRAINING_CONFIG["db"],
        top_k=CUSTOM_TRAINING_CONFIG.get("top_k", 3),
    )

    #Pass the dataset and model class
    runner = TrainingPipelineRunner(
        task_config=task,
        dataset_factory=CodeSnippetDatasetFactory(CUSTOM_TRAINING_CONFIG),

        model_cls=OptimizedMultiViewCWEModel,
        train_fn=train_model,
        inference_fn=run_optimized_inference,
        training_config=CUSTOM_TRAINING_CONFIG,
    )

    print("\n🎯 Choose mode:")
    print("1. Train + Inference")
    print("2. Inference only")
    print("3. Resume from checkpoint")

    choice = input("Choice (1-3): ").strip()

    if choice == "2":
        model, metrics, inference = runner.run(mode="inference")
    elif choice == "3":
        model, metrics, inference = runner.run(mode="resume")
    else:
        model, metrics, inference = runner.run(mode="train")

    # ─── Final reporting ───
    print("\n" + "=" * 60)
    print("🎉 PIPELINE EXECUTION COMPLETE")
    print("=" * 60)

    if metrics:
        print(f"🏆 Best validation accuracy: {max(metrics.val_accuracies):.2f}%")

    if inference:
        print(f"📊 Samples processed: {inference['samples_processed']:,}")
        print(f"🎯 Final inference accuracy: {inference['accuracy']:.2f}%")
        print(f"⏱️ Time: {inference['total_time']/60:.1f} min")


if __name__ == "__main__":
    main()