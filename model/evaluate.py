"""
Evaluation utilities: compute classification metrics on a DataLoader.
"""

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

import config


def get_predictions(model, loader):
    """Run the model over a loader and return (true_labels, pneumonia_probs)."""
    model.eval()
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(config.DEVICE)
            logits = model(pixel_values=images).logits
            probs = torch.softmax(logits, dim=1)[:, 1]

            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_labels), np.array(all_probs)


def evaluate(model, loader, threshold=0.5):
    """Compute accuracy, precision, recall, F1, and AUC on a loader."""
    labels, probs = get_predictions(model, loader)
    preds = (probs >= threshold).astype(int)

    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
        "auc": roc_auc_score(labels, probs),
    }


def save_metrics(metrics, path, header=""):
    """Write a metrics dict to a text file with an optional header block."""
    with open(path, "w") as f:
        if header:
            f.write(header.rstrip() + "\n\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value:.4f}\n")
