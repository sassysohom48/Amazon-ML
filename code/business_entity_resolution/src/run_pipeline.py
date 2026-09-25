"""
End-to-End Orchestrator Pipeline (Amazon ML Challenge 2026).
Runs Blocking -> Feature Engineering -> LightGBM Training & F0.5 Calibration -> Test Inference -> Official Validation.
"""

import os
import sys
import time
from pathlib import Path
import polars as pl

# Ensure package import works
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PROCESSED_DIR
from src.run_blocking import benchmark_validation_blocking
from src.train_model import build_training_dataset, train_and_calibrate_model
from src.inference import run_full_inference


def main():
    print("=" * 70)
    print("AMAZON ML CHALLENGE 2026: END-TO-END ENTITY RESOLUTION PIPELINE")
    print("=" * 70)

    # Step 1: Run Validation Blocking Benchmark
    cand_pairs_path = PROCESSED_DIR / "val_candidate_pairs.parquet"
    if not cand_pairs_path.exists():
        print("\n--- STEP 1: VALIDATION BLOCKING ---")
        benchmark_validation_blocking(max_candidates=15)
    else:
        print(f"\n[Found] Existing validation candidates at {cand_pairs_path.name}.")

    # Step 2: Load Data for Model Training
    print("\n--- STEP 2: TRAINING DATASET CONSTRUCTION & FEATURE EXTRACTION ---")
    val_s1 = pl.read_parquet(PROCESSED_DIR / "val_source1_cleaned.parquet")
    train_s2 = pl.read_parquet(PROCESSED_DIR / "train_source2_cleaned.parquet")
    train_s3 = pl.read_parquet(PROCESSED_DIR / "train_source3_cleaned.parquet")
    val_gt = pl.read_parquet(PROCESSED_DIR / "val_ground_truth.parquet")

    X_train, y_train, extractor = build_training_dataset(
        s1_df=val_s1,
        s2_df=train_s2,
        s3_df=train_s3,
        gt_df=val_gt,
        cand_pairs_path=cand_pairs_path,
        max_samples=500000,
        neg_to_pos_ratio=4,
    )

    # Step 3: Train LightGBM & Calibrate Macro F0.5
    print("\n--- STEP 3: MODEL TRAINING & MACRO F0.5 CALIBRATION ---")
    val_cand_df = pl.read_parquet(cand_pairs_path)

    # Build validation ground truth map
    val_gt_map = {}
    for row in val_gt.iter_rows():
        s1_id, matched = row[0], row[1]
        if matched and str(matched).strip():
            val_gt_map[s1_id] = set(str(matched).strip().split(","))
        else:
            val_gt_map[s1_id] = set()

    model, best_thresh, metrics = train_and_calibrate_model(
        X_train=X_train,
        y_train=y_train,
        val_cand_df=val_cand_df,
        val_gt_map=val_gt_map,
        extractor=extractor,
    )

    # Step 4: Run Full Test Inference & Validation
    print("\n--- STEP 4: FULL TEST SET INFERENCE & VALIDATION HARNESS ---")
    run_full_inference()

    print("\n" + "=" * 70)
    print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
