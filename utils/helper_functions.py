"""
utils/helper_functions.py
--------------------------
Reusable utility functions shared across the project:
  - Grad-CAM wrapper and reshape transform
  - Confidence label helper
  - Image conversion helpers
  - Prediction grid / QA image saving
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from PIL import Image

from utils.config import (
    PNEUMONIA_THRESHOLD,
    HIGH_CONFIDENCE_THRESHOLD,
    LOW_CONFIDENCE_THRESHOLD,
    LABEL_MAP,
    WORK_DIR,
)
from utils.logging_setup import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Grad-CAM helpers
# ---------------------------------------------------------------------------

class HFViTWrapper(nn.Module):
    """
    Thin wrapper around a HuggingFace ViT model so that pytorch-grad-cam
    can treat it as a standard nn.Module that returns raw logits.
    """
    def __init__(self, hf_model):
        super().__init__()
        self.hf_model = hf_model

    def forward(self, pixel_values):
        outputs = self.hf_model(pixel_values=pixel_values)
        return outputs.logits


def reshape_transform(tensor, height=14, width=14):
    """
    Reshape ViT attention tensors for Grad-CAM visualisation.
    Removes the CLS token and converts [B, 196, C] → [B, C, 14, 14].
    """
    tensor = tensor[:, 1:, :]  # drop CLS token
    tensor = tensor.reshape(tensor.size(0), height, width, tensor.size(2))
    tensor = tensor.permute(0, 3, 1, 2)
    return tensor


def get_gradcam_object(model, device):
    """
    Build and return a GradCAM object targeting the last ViT encoder layer.
    Requires pytorch-grad-cam to be installed.
    """
    from pytorch_grad_cam import GradCAM

    wrapped = HFViTWrapper(model).to(device)
    wrapped.eval()
    target_layers = [model.vit.encoder.layer[-1].layernorm_before]
    cam = GradCAM(model=wrapped, target_layers=target_layers, reshape_transform=reshape_transform)
    return cam


def generate_gradcam_image(cam, processor, model, image: Image.Image, device) -> tuple:
    """
    Run Grad-CAM on a single PIL image.

    Returns
    -------
    (cam_image, pneumonia_prob, pred_label, true_label_str)
    where cam_image is a uint8 numpy array (H x W x 3).
    """
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    image_resized = image.resize((224, 224))
    rgb_img = np.array(image_resized).astype(np.float32) / 255.0

    inputs = processor(images=image, return_tensors="pt")
    input_tensor = inputs["pixel_values"].to(device)

    with torch.no_grad():
        outputs = model(pixel_values=input_tensor)
        probs = torch.softmax(outputs.logits, dim=1)[0]
        pneumonia_prob = probs[1].item()

    pred_label = LABEL_MAP[1 if pneumonia_prob >= PNEUMONIA_THRESHOLD else 0]
    targets = [ClassifierOutputTarget(1)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
    cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    return cam_image, pneumonia_prob, pred_label


# ---------------------------------------------------------------------------
# Confidence label
# ---------------------------------------------------------------------------

def get_confidence_label(pneumonia_prob: float) -> str:
    """Return a human-readable confidence string for a given probability."""
    if pneumonia_prob >= HIGH_CONFIDENCE_THRESHOLD:
        return "High"
    elif pneumonia_prob >= LOW_CONFIDENCE_THRESHOLD:
        return "Medium / Uncertain"
    else:
        return "Low pneumonia probability"


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def save_prediction_grid(model, processor, sample_df, device, save_path: str):
    """
    Save a 3×3 grid of test images with prediction overlays.
    Matches the notebook's prediction grid cell.
    """
    plt.figure(figsize=(12, 12))

    for i, row in sample_df.iterrows():
        image = Image.open(row["image_path"]).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)

        with torch.no_grad():
            outputs = model(pixel_values=pixel_values)
            probs = torch.softmax(outputs.logits, dim=1)[0]
            pneumonia_prob = probs[1].item()

        pred_label = LABEL_MAP[1 if pneumonia_prob >= PNEUMONIA_THRESHOLD else 0]
        true_label = LABEL_MAP[int(row["label"])]
        status = "Correct" if pred_label == true_label else "Wrong"

        plt.subplot(3, 3, i + 1)
        plt.imshow(image, cmap="gray")
        plt.axis("off")
        plt.title(
            f"T: {true_label}\nP: {pred_label}\nProb: {pneumonia_prob:.2f}\n{status}",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Prediction grid saved to {save_path}")


def save_qa_predictions(model, processor, sample_df, device, output_dir: str):
    """
    Save individual QA-style prediction cards (image + text panel) to output_dir.
    Matches the notebook's QA prediction cell.
    """
    os.makedirs(output_dir, exist_ok=True)

    for i, row in sample_df.iterrows():
        image = Image.open(row["image_path"]).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)

        with torch.no_grad():
            outputs = model(pixel_values=pixel_values)
            probs = torch.softmax(outputs.logits, dim=1)[0]
            pneumonia_prob = probs[1].item()

        pred_label = LABEL_MAP[1 if pneumonia_prob >= PNEUMONIA_THRESHOLD else 0]
        true_label = LABEL_MAP[int(row["label"])]
        status = "Correct" if pred_label == true_label else "Wrong"
        confidence = get_confidence_label(pneumonia_prob)

        qa_text = (
            f"Q1: Is pneumonia predicted?\nA: {pred_label}\n\n"
            f"Q2: Pneumonia probability?\nA: {pneumonia_prob:.2f}\n\n"
            f"Q3: Confidence?\nA: {confidence}\n\n"
            f"True label: {true_label}\n"
            f"Result: {status}"
        )

        plt.figure(figsize=(9, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(image, cmap="gray")
        plt.axis("off")
        plt.title("Chest X-ray")
        plt.subplot(1, 2, 2)
        plt.axis("off")
        plt.text(0, 0.95, qa_text, fontsize=11, va="top")

        save_path = f"{output_dir}/qa_prediction_{i+1}_{status}.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

    logger.info(f"QA predictions saved to {output_dir}")


# ---------------------------------------------------------------------------
# Image conversion
# ---------------------------------------------------------------------------

def pil_to_base64(image: Image.Image, fmt: str = "JPEG") -> str:
    """Convert a PIL image to a base64-encoded string (used by the Lambda handler)."""
    import io, base64
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def base64_to_pil(b64_string: str) -> Image.Image:
    """Convert a base64-encoded string back to a PIL image."""
    import io, base64
    image_data = base64.b64decode(b64_string)
    return Image.open(io.BytesIO(image_data))
