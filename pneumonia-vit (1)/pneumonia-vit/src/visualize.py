"""
Visualization and explainability.

Produces the confusion matrix, sample prediction grids, and Grad-CAM
heatmaps that highlight the regions the model focused on.
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

import config
from src.evaluate import get_predictions
from src.model import HFViTWrapper


def plot_confusion_matrix(model, test_loader, save_path, title="Confusion Matrix"):
    labels, probs = get_predictions(model, test_loader)
    preds = (probs >= 0.5).astype(int)

    cm = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=config.CLASS_NAMES)
    disp.plot(cmap="Blues")
    plt.title(title)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    return cm


def _predict_single(model, processor, image_path):
    """Return (pneumonia_prob, pred_label) for one image."""
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(config.DEVICE)

    with torch.no_grad():
        logits = model(pixel_values=pixel_values).logits
        prob = torch.softmax(logits, dim=1)[0, 1].item()

    label = "Pneumonia" if prob >= 0.5 else "Non-Pneumonia"
    return prob, label


def plot_prediction_grid(model, processor, test_data, save_path,
                         n=9, random_state=7):
    """Plot a 3x3 grid of sample predictions vs. true labels."""
    sample = test_data.sample(n, random_state=random_state).reset_index(drop=True)
    plt.figure(figsize=(12, 12))

    for i, row in sample.iterrows():
        prob, pred_label = _predict_single(model, processor, row["image_path"])
        true_label = "Pneumonia" if row["label"] == 1 else "Non-Pneumonia"
        status = "Correct" if pred_label == true_label else "Wrong"

        plt.subplot(3, 3, i + 1)
        plt.imshow(Image.open(row["image_path"]).convert("RGB"), cmap="gray")
        plt.axis("off")
        plt.title(f"T: {true_label}\nP: {pred_label}\nProb: {prob:.2f}\n{status}",
                  fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def _reshape_transform(tensor, height=14, width=14):
    """Reshape ViT patch tokens into a 2D feature map for Grad-CAM."""
    tensor = tensor[:, 1:, :]  # drop CLS token
    tensor = tensor.reshape(tensor.size(0), height, width, tensor.size(2))
    return tensor.permute(0, 3, 1, 2)


def generate_gradcam(model, processor, test_data, out_dir,
                     n=12, random_state=42):
    """Save Grad-CAM overlays for a sample of test images."""
    os.makedirs(out_dir, exist_ok=True)

    wrapped_model = HFViTWrapper(model).to(config.DEVICE).eval()
    target_layers = [model.vit.encoder.layer[-1].layernorm_before]
    cam = GradCAM(model=wrapped_model, target_layers=target_layers,
                  reshape_transform=_reshape_transform)

    sample = test_data.sample(n, random_state=random_state).reset_index(drop=True)

    for i, row in sample.iterrows():
        image = Image.open(row["image_path"]).convert("RGB")
        rgb_img = np.array(image.resize((224, 224))).astype(np.float32) / 255.0

        inputs = processor(images=image, return_tensors="pt")
        input_tensor = inputs["pixel_values"].to(config.DEVICE)

        prob, pred_label = _predict_single(model, processor, row["image_path"])
        true_label = "Pneumonia" if row["label"] == 1 else "Non-Pneumonia"
        status = "Correct" if pred_label == true_label else "Wrong"

        grayscale_cam = cam(input_tensor=input_tensor,
                            targets=[ClassifierOutputTarget(1)])[0, :]
        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(rgb_img)
        plt.axis("off")
        plt.title("Original X-ray")

        plt.subplot(1, 2, 2)
        plt.imshow(cam_image)
        plt.axis("off")
        plt.title(f"Grad-CAM\nTrue: {true_label}\n"
                  f"Pred: {pred_label} | Prob: {prob:.2f}\n{status}")

        plt.savefig(f"{out_dir}/gradcam_{i + 1}_{status}.png",
                    dpi=300, bbox_inches="tight")
        plt.close()
