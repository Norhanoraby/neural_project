"""
Data cleaning and preprocessing.

Takes the loaded/merged DataFrame and applies the cleaning steps:
keep frontal (AP/PA) views only, remove negative-class label noise, build
a balanced subset, and produce a stratified train/val/test split.

Also exposes prepare_data(), the end-to-end pipeline that chains the
loading steps (from data_loading.py) with the preprocessing steps here.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

import config
from src.data_loading import (
    find_csv_files,
    load_tables,
    merge_and_label,
    attach_image_paths,
)


def filter_frontal_and_noise(df):
    """
    Keep only frontal (AP/PA) views and reduce negative-class label noise
    by dropping non-pneumonia images that are positive for pneumonia-like
    findings (Consolidation, Edema, Lung Opacity).
    """
    df_work = df[df["ViewPosition"].isin(config.FRONTAL_VIEWS)].copy()

    positive_df = df_work[df_work["label"] == 1].copy()
    negative_df = df_work[df_work["label"] == 0].copy()

    for col in config.PNEUMONIA_LIKE_COLS:
        if col in negative_df.columns:
            negative_df = negative_df[
                (negative_df[col].isna()) | (negative_df[col] == 0)
            ].copy()

    df_clean = pd.concat([positive_df, negative_df]).reset_index(drop=True)
    return df_clean


def make_balanced_subset(df, n_per_class, random_state=config.RANDOM_STATE):
    """Sample an equal number of pneumonia / non-pneumonia rows and shuffle."""
    pneumonia = df[df["label"] == 1].sample(n=n_per_class, random_state=random_state)
    non_pneumonia = df[df["label"] == 0].sample(n=n_per_class, random_state=random_state)

    subset = pd.concat([pneumonia, non_pneumonia])
    subset = subset.sample(frac=1, random_state=random_state).reset_index(drop=True)
    return subset


def stratified_split(subset, random_state=config.RANDOM_STATE):
    """Split into train / val / test, stratified by label (70/15/15)."""
    train_data, temp_data = train_test_split(
        subset,
        test_size=config.TEST_SIZE,
        stratify=subset["label"],
        random_state=random_state,
    )
    val_data, test_data = train_test_split(
        temp_data,
        test_size=config.VAL_TEST_SIZE,
        stratify=temp_data["label"],
        random_state=random_state,
    )
    return train_data, val_data, test_data


def prepare_data(n_per_class=1500, clean=True):
    """
    End-to-end data pipeline: load -> merge/label -> attach images ->
    (optional) clean -> balance -> stratified split.

    Returns (train_data, val_data, test_data).
    Set clean=True to apply frontal + noise filtering (the 3000-image run),
    or clean=False for the simpler 1000-image run.
    """
    # --- loading (data_loading.py) ---
    csv_files = find_csv_files()
    chexpert, metadata, split = load_tables(csv_files)
    df = merge_and_label(chexpert, metadata, split)
    df = attach_image_paths(df)

    # --- preprocessing (this file) ---
    if clean:
        df = filter_frontal_and_noise(df)

    subset = make_balanced_subset(df, n_per_class=n_per_class)
    return stratified_split(subset)
