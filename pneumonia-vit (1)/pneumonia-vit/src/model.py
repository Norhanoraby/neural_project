"""
Model definition and loading.

Loads the pretrained Vision Transformer from Hugging Face and provides a
thin wrapper used by Grad-CAM (which expects a model that returns raw
logits rather than a Hugging Face output object).
"""

import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModelForImageClassification

import config


def load_model_and_processor():
    """Load a fresh pretrained ViT and its image processor onto the device."""
    processor = AutoImageProcessor.from_pretrained(config.MODEL_NAME)
    model = AutoModelForImageClassification.from_pretrained(config.MODEL_NAME)
    model = model.to(config.DEVICE)
    return model, processor


class HFViTWrapper(nn.Module):
    """Returns raw logits so libraries like Grad-CAM can hook into it."""

    def __init__(self, hf_model):
        super().__init__()
        self.hf_model = hf_model

    def forward(self, pixel_values):
        return self.hf_model(pixel_values=pixel_values).logits


def save_model(model, path):
    torch.save(model.state_dict(), path)


def load_weights(model, path):
    model.load_state_dict(torch.load(path, map_location=config.DEVICE))
    return model.to(config.DEVICE)
