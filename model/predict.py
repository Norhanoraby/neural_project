"""
model/predict.py
----------------
Single-image inference for the pneumonia detection ViT model.
Used by the Lambda handler and any downstream consumer that needs
a prediction without running the full training pipeline.
"""

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

from utils.config import (
    MODEL_NAME,
    PNEUMONIA_THRESHOLD,
    HIGH_CONFIDENCE_THRESHOLD,
    LOW_CONFIDENCE_THRESHOLD,
    LABEL_MAP,
)
from utils.logging_setup import get_logger
from utils.exception_handling import ModelNotLoadedError, InvalidImageError

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Model loading (cached — called once at cold start)
# ---------------------------------------------------------------------------

_model = None
_processor = None


def load_model(model_name: str = MODEL_NAME, device: torch.device = None):
    """
    Load the HuggingFace ViT model and processor.
    Caches the result in module-level globals so subsequent calls are free.
    """
    global _model, _processor

    if _model is not None and _processor is not None:
        return _model, _processor

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Loading model '{model_name}' on {device} ...")
    _processor = AutoImageProcessor.from_pretrained(model_name)
    _model = AutoModelForImageClassification.from_pretrained(model_name)
    _model = _model.to(device)
    _model.eval()
    logger.info("Model loaded successfully.")

    return _model, _processor


# ---------------------------------------------------------------------------
# Core prediction
# ---------------------------------------------------------------------------

def predict_image(image: Image.Image, model=None, processor=None, device=None) -> dict:
    """
    Run inference on a single PIL image.

    Parameters
    ----------
    image     : PIL.Image — the chest X-ray (any mode; converted to RGB internally)
    model     : loaded HF model (loaded automatically if None)
    processor : loaded HF processor (loaded automatically if None)
    device    : torch.device (auto-detected if None)

    Returns
    -------
    dict with keys:
        prediction      : "Pneumonia" | "Non-Pneumonia"
        pneumonia_prob  : float  (0–1)
        confidence_level: "High" | "Medium / Uncertain" | "Low pneumonia probability"
        label           : int  (1 = Pneumonia, 0 = Non-Pneumonia)
    """
    if image is None:
        raise InvalidImageError("Received a None image object.")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model is None or processor is None:
        model, processor = load_model(device=device)

    if model is None:
        raise ModelNotLoadedError("Model is not loaded. Call load_model() first.")

    try:
        image = image.convert("RGB")
    except Exception as e:
        raise InvalidImageError(f"Could not convert image to RGB: {e}")

    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)
        probs = torch.softmax(outputs.logits, dim=1)[0]

    pneumonia_prob = probs[1].item()
    label = 1 if pneumonia_prob >= PNEUMONIA_THRESHOLD else 0
    prediction = LABEL_MAP[label]

    if pneumonia_prob >= HIGH_CONFIDENCE_THRESHOLD:
        confidence_level = "High"
    elif pneumonia_prob >= LOW_CONFIDENCE_THRESHOLD:
        confidence_level = "Medium / Uncertain"
    else:
        confidence_level = "Low pneumonia probability"

    result = {
        "prediction": prediction,
        "pneumonia_prob": round(pneumonia_prob, 4),
        "confidence_level": confidence_level,
        "label": label,
    }

    logger.info(f"Prediction: {result}")
    return result


def predict_from_path(image_path: str, model=None, processor=None, device=None) -> dict:
    """
    Convenience wrapper — load image from disk then call predict_image().
    """
    try:
        image = Image.open(image_path)
    except Exception as e:
        raise InvalidImageError(f"Could not open image at '{image_path}': {e}")

    return predict_image(image, model=model, processor=processor, device=device)
