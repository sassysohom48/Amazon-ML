"""
Model Training, Hard Negative Mining & Validation Macro F0.5 Calibration.
"""

import sys
import time
from pathlib import Path
import polars as pl

# Ensure package import works
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PROCESSED_DIR
from src.train_model import build_training_dataset, train_and_calibrate_model


def main():
    print("=" * 70)
    print("PHASE 4 & 5: FEATURE EXTRACTION, MODEL TRAINING & MACRO F0.5 CALIBRATION")
    print("=" * 70)

    val_s1_path = PROCESSED_DIR / "val_source1_cleaned.parquet"
    train_s2_path = PROCESSED_DIR / "train_source2_cleaned.parquet"
    train_s3_path = PROCESSED_DIR / "train_source3_cleaned.parquet"
    val_gt_path = PROCESSED_DIR / "val_ground_truth.parquet"
    cand_pairs_path = PROCESSED_DIR / "val_candidate_pairs.parquet"

    print("Loading datasets...")
    t0 = time.time()
    val_s1 = pl.read_parquet(val_s1_path)
    train_s2 = pl.read_parquet(train_s2_path)
    train_s3 = pl.read_parquet(train_s3_path)
    val_gt = pl.read_parquet(val_gt_path)
    val_cand_df = pl.read_parquet(cand_pairs_path)
    print(f"Loaded all Parquet tables in {time.time() - t0:.2f}s.")

    # 1. Build training dataset (300k balanced pairs: 60k Positives, 240k Negatives)
    X_train, y_train, extractor = build_training_dataset(
        s1_df=val_s1,
        s2_df=train_s2,
        s3_df=train_s3,
        gt_df=val_gt,
        cand_pairs_path=cand_pairs_path,
        max_samples=300000,
        neg_to_pos_ratio=4,
    )

    # 2. Build Ground Truth lookup
    val_gt_map = {}
    for row in val_gt.iter_rows():
        s1_id, matches = row[0], row[1]
        if matches and str(matches).strip():
            val_gt_map[s1_id] = set(str(matches).strip().split(","))
        else:
            val_gt_map[s1_id] = set()

    # 3. Train LightGBM & Calibrate Threshold
    model, best_thresh, best_metrics = train_and_calibrate_model(
        X_train=X_train,
        y_train=y_train,
        val_cand_df=val_cand_df,
        val_gt_map=val_gt_map,
        extractor=extractor,
    )


if __name__ == "__main__":
    main()
