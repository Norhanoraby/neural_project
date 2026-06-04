"""
Central configuration for the pneumonia detection project.
All paths, hyperparameters, and constants live here so they can be
changed in one place instead of being scattered across the code.
"""

import torch

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
DATASET_DIR = "/kaggle/input/datasets/phuong20052/mimic-cxr-jpg-lite"
WORK_DIR = "/kaggle/working"

# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------
MODEL_NAME = "nickmuchi/vit-finetuned-chest-xray-pneumonia"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------
BATCH_SIZE = 8
NUM_WORKERS = 2
NUM_EPOCHS = 5
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-4          # used in the cleaned 3000-image run
RANDOM_STATE = 42

# ---------------------------------------------------------------------
# Data splitting
# ---------------------------------------------------------------------
# 70% train, 15% validation, 15% test (stratified by label)
TEST_SIZE = 0.30             # first split: 70% train / 30% temp
VAL_TEST_SIZE = 0.50         # second split: half of temp -> val, half -> test

# Columns that indicate pneumonia-like findings. Negative (non-pneumonia)
# images that are positive for any of these are removed as label noise.
PNEUMONIA_LIKE_COLS = ["Consolidation", "Edema", "Lung Opacity"]

# Only keep frontal chest X-rays
FRONTAL_VIEWS = ["AP", "PA"]

# Class names for plots / reports
CLASS_NAMES = ["Non-Pneumonia", "Pneumonia"]
