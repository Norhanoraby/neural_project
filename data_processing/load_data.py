"""
Data loading.

Finds the raw CSV tables on disk, loads them, merges metadata + CheXpert +
split into one DataFrame, builds the binary pneumonia label, and matches
each row to its chest X-ray image file. (Cleaning / filtering / splitting
lives in preprocessing.py.)
"""

import os
import pandas as pd

import config


def find_csv_files(dataset_dir=config.DATASET_DIR):
    """Walk the dataset directory and return all CSV / CSV.GZ paths."""
    csv_files = []
    for root, _dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith(".csv") or file.endswith(".csv.gz"):
                csv_files.append(os.path.join(root, file))
    return csv_files


def load_tables(csv_files):
    """Locate and load the CheXpert, metadata, and split CSV tables."""
    chexpert_path = [p for p in csv_files if "chexpert" in p.lower()][0]
    metadata_path = [p for p in csv_files if "metadata" in p.lower()][0]
    split_path = [p for p in csv_files if "split" in p.lower()][0]

    chexpert = pd.read_csv(chexpert_path)
    metadata = pd.read_csv(metadata_path)
    split = pd.read_csv(split_path)
    return chexpert, metadata, split


def merge_and_label(chexpert, metadata, split):
    """Merge the three tables and build the binary pneumonia label."""
    df = metadata.merge(chexpert, on=["subject_id", "study_id"], how="inner")
    df = df.merge(split, on=["subject_id", "study_id", "dicom_id"], how="inner")

    # Remove uncertain pneumonia cases (label == -1)
    df = df[df["Pneumonia"] != -1].copy()

    # 1 = Pneumonia, 0 = Non-Pneumonia
    df["label"] = df["Pneumonia"].fillna(0).astype(int)

    df.columns = df.columns.str.strip()
    return df


def attach_image_paths(df, dataset_dir=config.DATASET_DIR):
    """Map each dicom_id to its .jpg file path and drop rows without images."""
    image_map = {}
    for root, _dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.lower().endswith(".jpg"):
                dicom_id = file.replace(".jpg", "")
                image_map[dicom_id] = os.path.join(root, file)

    df["image_path"] = df["dicom_id"].map(image_map)
    df = df[df["image_path"].notna()].copy()
    return df
