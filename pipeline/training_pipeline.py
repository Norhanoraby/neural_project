"""
pipeline/training_pipeline.py
------------------------------
Chains the full model training flow in one callable:
  load model → build dataloaders → train → evaluate → save weights

Call run_training_pipeline() from app.py.
"""

import torch

from utils.config import (
    MODEL_NAME,
    WORK_DIR,
    BATCH_SIZE,
    NUM_WORKERS,
    LEARNING_RATE,
    NUM_EPOCHS,
    EXPERIMENT_TAG,
)
from utils.logging_setup import get_logger
from utils.exception_handling import TrainingPipelineError
from components.model_builder import build_model, build_dataloaders
from components.evaluation import evaluate, save_confusion_matrix, save_metrics_txt
from model.train import train

logger = get_logger(__name__)


def run_training_pipeline(train_data, val_data, test_data) -> dict:
    """
    Execute the full training pipeline.

    Parameters
    ----------
    train_data, val_data, test_data : pd.DataFrame
        Split DataFrames produced by run_data_pipeline().

    Returns
    -------
    dict with keys: model, processor, test_metrics
    """
    try:
        logger.info("=== Training Pipeline Start ===")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        # 1. Build model + processor
        model, processor = build_model(MODEL_NAME, device)

        # 2. Build dataloaders
        train_loader, val_loader, test_loader = build_dataloaders(
            train_data, val_data, test_data, processor,
            batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
        )

        # 3. Train
        model = train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            lr=LEARNING_RATE,
            epochs=NUM_EPOCHS,
            save_path=f"{WORK_DIR}/best_vit_{EXPERIMENT_TAG}.pth",
            evaluate_fn=evaluate,
        )

        # 4. Load best weights and evaluate on test set
        model.load_state_dict(
            torch.load(f"{WORK_DIR}/best_vit_{EXPERIMENT_TAG}.pth", map_location=device)
        )
        model.eval()

        test_metrics = evaluate(model, test_loader, device)
        logger.info(f"Test metrics: {test_metrics}")

        # 5. Save evaluation artefacts
        save_metrics_txt(test_metrics, WORK_DIR, tag=EXPERIMENT_TAG)
        save_confusion_matrix(model, test_loader, device, WORK_DIR, tag=EXPERIMENT_TAG)

        logger.info("=== Training Pipeline Complete ===")
        return {"model": model, "processor": processor, "test_metrics": test_metrics}

    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        raise TrainingPipelineError(f"Training pipeline failed: {e}") from e
