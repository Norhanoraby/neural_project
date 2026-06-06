"""
pipeline/data_pipeline.py
--------------------------
Chains the full data preparation flow in one callable:
  load_data → preprocess → feature_engineering → split → save CSVs

Call run_data_pipeline() from app.py or training_pipeline.py.
"""

from utils.config import (
    DATASET_DIR,
    WORK_DIR,
    SAMPLE_SIZE_PER_CLASS,
    RANDOM_STATE,
    TEST_SIZE,
    VAL_SIZE,
    EXPERIMENT_TAG,
)
from utils.logging_setup import get_logger
from utils.exception_handling import DataPipelineError
from data_processing.load_data import load_csvs, build_image_map, merge_dataframes
from data_processing.preprocess import clean_labels, filter_frontal, filter_label_noise
from data_processing.feature_engineering import sample_balanced, stratified_split, save_splits

logger = get_logger(__name__)


def run_data_pipeline() -> dict:
    """
    Execute the full data preparation pipeline.

    Returns
    -------
    dict with keys: train_data, val_data, test_data, full_subset (DataFrames)
    """
    try:
        logger.info("=== Data Pipeline Start ===")

        # 1. Load raw CSVs
        chexpert, metadata, split_df = load_csvs(DATASET_DIR)

        # 2. Merge into single DataFrame
        df = merge_dataframes(chexpert, metadata, split_df)

        # 3. Build image path map and attach
        image_map = build_image_map(DATASET_DIR)
        df["image_path"] = df["dicom_id"].map(image_map)
        df = df[df["image_path"].notna()].copy()
        logger.info(f"Rows after image matching: {len(df)}")

        # 4. Preprocess — clean labels, frontal filter, noise reduction
        df = clean_labels(df)
        df = filter_frontal(df)
        df = filter_label_noise(df)

        # 5. Feature engineering — balanced sampling + stratified split
        subset = sample_balanced(df, n_per_class=SAMPLE_SIZE_PER_CLASS, random_state=RANDOM_STATE)
        train_data, val_data, test_data = stratified_split(
            subset,
            test_size=TEST_SIZE,
            val_size=VAL_SIZE,
            random_state=RANDOM_STATE,
        )

        # 6. Save CSVs to WORK_DIR
        save_splits(subset, train_data, val_data, test_data, WORK_DIR, tag=EXPERIMENT_TAG)

        logger.info("=== Data Pipeline Complete ===")
        return {
            "train_data": train_data,
            "val_data": val_data,
            "test_data": test_data,
            "full_subset": subset,
        }

    except Exception as e:
        logger.error(f"Data pipeline failed: {e}")
        raise DataPipelineError(f"Data pipeline failed: {e}") from e
