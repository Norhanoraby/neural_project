"""
utils/config.py
----------------
Single source of truth for all paths, hyperparameters, and constants.
Change values here; everything else imports from this file.
"""

import os

# ── Paths ────────────────────────────────────────────────────────────────────
DATASET_DIR = os.getenv("DATASET_DIR", "/kaggle/input/datasets/phuong20052/mimic-cxr-jpg-lite")
WORK_DIR    = os.getenv("WORK_DIR",    "/kaggle/working")
LOG_FILE    = os.getenv("LOG_FILE",    None)   # e.g. "/kaggle/working/run.log"

# ── Model ────────────────────────────────────────────────────────────────────
MODEL_NAME = "nickmuchi/vit-finetuned-chest-xray-pneumonia"

# ── Experiment ───────────────────────────────────────────────────────────────
EXPERIMENT_TAG        = "cxrpretrained_3000_clean"
SAMPLE_SIZE_PER_CLASS = 1500       # 1500 pneumonia + 1500 non-pneumonia = 3000 total
RANDOM_STATE          = 42

# ── Data split ratios ────────────────────────────────────────────────────────
TEST_SIZE = 0.30    # 30% held out from full subset → split equally into val + test
VAL_SIZE  = 0.50    # 50% of the 30% → 15% val, 15% test

# ── Training ─────────────────────────────────────────────────────────────────
BATCH_SIZE    = 8
NUM_WORKERS   = 2
LEARNING_RATE = 1e-5
NUM_EPOCHS    = 5

# ── Inference thresholds ─────────────────────────────────────────────────────
PNEUMONIA_THRESHOLD      = 0.50   # >= this → "Pneumonia"
HIGH_CONFIDENCE_THRESHOLD = 0.70  # >= this → "High" confidence
LOW_CONFIDENCE_THRESHOLD  = 0.40  # >= this → "Medium / Uncertain"; below → "Low"

# ── Label map ────────────────────────────────────────────────────────────────
LABEL_MAP = {0: "Non-Pneumonia", 1: "Pneumonia"}

# ── Logging ──────────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")   # DEBUG | INFO | WARNING | ERROR
