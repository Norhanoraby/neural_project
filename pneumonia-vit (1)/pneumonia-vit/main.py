"""
Main entry point.

Runs the full pipeline end-to-end: prepare data -> load model ->
train -> evaluate -> produce confusion matrix, prediction grid, and
Grad-CAM visualizations.

Run with:
    python main.py
"""

import os

import config
from src.preprocessing import prepare_data
from src.dataset import build_loaders
from src.model import load_model_and_processor, load_weights
from src.train import train
from src.evaluate import evaluate, save_metrics
from src.visualize import plot_confusion_matrix, plot_prediction_grid, generate_gradcam


def run_experiment(n_per_class=1500, clean=True, tag="3000_clean"):
    """Run one full experiment and save all outputs to WORK_DIR."""
    os.makedirs(config.WORK_DIR, exist_ok=True)
    checkpoint = os.path.join(config.WORK_DIR, f"best_cxrpretrained_vit_{tag}.pth")

    # 1. Data
    train_data, val_data, test_data = prepare_data(n_per_class=n_per_class, clean=clean)
    print(f"Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")

    # 2. Model + loaders
    model, processor = load_model_and_processor()
    train_loader, val_loader, test_loader = build_loaders(
        train_data, val_data, test_data, processor
    )

    # 3. Train (best checkpoint chosen by validation AUC)
    train(model, train_loader, val_loader, checkpoint)

    # 4. Evaluate the best checkpoint on the test set
    model = load_weights(model, checkpoint)
    test_metrics = evaluate(model, test_loader)
    print("Test metrics:", test_metrics)

    save_metrics(
        test_metrics,
        os.path.join(config.WORK_DIR, f"cxrpretrained_{tag}_metrics.txt"),
        header=f"Experiment: CXR-pretrained ViT ({tag})\nModel: {config.MODEL_NAME}",
    )

    # 5. Visualizations
    plot_confusion_matrix(
        model, test_loader,
        os.path.join(config.WORK_DIR, f"cxrpretrained_{tag}_confusion_matrix.png"),
        title=f"CXR-pretrained ViT - {tag}",
    )
    plot_prediction_grid(
        model, processor, test_data,
        os.path.join(config.WORK_DIR, f"cxrpretrained_{tag}_prediction_grid.png"),
    )
    generate_gradcam(
        model, processor, test_data,
        os.path.join(config.WORK_DIR, f"cxrpretrained_{tag}_gradcam"),
    )

    return test_metrics


if __name__ == "__main__":
    # The two experiments from the notebook:
    # run_experiment(n_per_class=500, clean=False, tag="1000")
    run_experiment(n_per_class=1500, clean=True, tag="3000_clean")
