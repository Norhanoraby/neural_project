"""
Training loop.

Fine-tunes the ViT, validates after each epoch, and keeps the checkpoint
with the best validation AUC.
"""

import torch
import torch.nn as nn
from tqdm import tqdm

import config
from src.evaluate import evaluate
from src.model import save_model


def train(model, train_loader, val_loader, checkpoint_path,
          num_epochs=config.NUM_EPOCHS, weight_decay=config.WEIGHT_DECAY):
    """Fine-tune the model, saving the best checkpoint by validation AUC."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=weight_decay,
    )

    best_val_auc = 0.0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        for images, labels in loop:
            images = images.to(config.DEVICE)
            labels = labels.to(config.DEVICE)

            optimizer.zero_grad()
            logits = model(pixel_values=images).logits
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        val_metrics = evaluate(model, val_loader)

        print(f"\nEpoch {epoch + 1}/{num_epochs} | Train loss: {avg_loss:.4f}")
        print("Validation:", val_metrics)

        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            save_model(model, checkpoint_path)
            print("Best model saved!")

    print("Training finished. Best validation AUC:", best_val_auc)
    return best_val_auc
