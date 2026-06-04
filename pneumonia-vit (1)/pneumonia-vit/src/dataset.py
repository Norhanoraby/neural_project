"""
PyTorch Dataset and DataLoader construction.

Wraps a pandas DataFrame of image paths + labels into a torch Dataset
that runs the Hugging Face image processor on each sample.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

import config


class PneumoniaHFDataset(Dataset):
    """Dataset that loads a chest X-ray and processes it for the ViT model."""

    def __init__(self, dataframe, processor):
        self.df = dataframe.reset_index(drop=True)
        self.processor = processor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image = Image.open(row["image_path"]).convert("RGB")
        label = torch.tensor(int(row["label"]), dtype=torch.long)

        # The HF image processor handles preprocessing of the model input:
        # it resizes the X-ray to 224x224 and normalizes pixel values using
        # the mean/std the pretrained ViT was trained with. (No separate
        # transforms.Normalize is needed — matching the pretrained model's
        # normalization is required for correct results.)
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)

        return pixel_values, label


def build_loaders(train_data, val_data, test_data, processor):
    """Build train / val / test DataLoaders from the split DataFrames."""
    train_dataset = PneumoniaHFDataset(train_data, processor)
    val_dataset = PneumoniaHFDataset(val_data, processor)
    test_dataset = PneumoniaHFDataset(test_data, processor)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
    )
    return train_loader, val_loader, test_loader
