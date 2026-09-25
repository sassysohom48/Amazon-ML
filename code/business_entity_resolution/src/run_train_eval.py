"""
Phase 5: LightGBM Model Training, Validation Calibration & Metric Evaluation.
"""

import sys
import time
from pathlib import Path
import numpy as np
import polars as pl

# Ensure package import works
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PROCESSED_DIR
from src.feature_engineering import FeatureExtractor, FEATURE_NAMES
from src.train_model import build_training_dataset, train_and_calibrate_model


def main():
    print("=" * 70)
    print("PHASE 5: LIGHTGBM MODEL TRAINING & MACRO F0.5 CALIBRATION")
    print("=" * 70)

    val_s1_path = PROCESSED_DIR / "val_source1_cleaned.parquet"
    train_s2_path = PROCESSED_DIR / "train_source2_cleaned.parquet"
    train_s3_path = PROCESSED_DIR / "train_source3_cleaned.parquet"
    val_gt_path = PROCESSED_DIR / "val_ground_truth.parquet"
    cand_pairs_path = PROCESSED_DIR / "val_candidate_pairs.parquet"
    train_features_path = PROCESSED_DIR / "train_features.parquet"

    # 1. Load or Build Training Features
    if train_features_path.exists():
        print(f"\nLoading precomputed training features from {train_features_path.name}...")
        t0 = time.time()
        df_train = pl.read_parquet(train_features_path)
        X_train = df_train.select(FEATURE_NAMES).to_numpy().astype(np.float32)
        y_train = df_train["label"].to_numpy().astype(np.int32)
        print(f"Loaded {len(X_train):,} training pairs in {time.time() - t0:.2f}s.")
        print(f"  • Positives (Matches):     {int((y_train == 1).sum()):,} ({(y_train == 1).mean()*100:.1f}%)")
        print(f"  • Negatives (Non-matches): {int((y_train == 0).sum()):,} ({(y_train == 0).mean()*100:.1f}%)")
    else:
        print("\nPrecomputed train_features.parquet not found. Building dataset...")
        val_s1 = pl.read_parquet(val_s1_path)
        train_s2 = pl.read_parquet(train_s2_path)
        train_s3 = pl.read_parquet(train_s3_path)
        val_gt = pl.read_parquet(val_gt_path)
        X_train, y_train, _ = build_training_dataset(
            s1_df=val_s1,
            s2_df=train_s2,
            s3_df=train_s3,
            gt_df=val_gt,
            cand_pairs_path=cand_pairs_path,
            max_samples=350000,
            neg_to_pos_ratio=4,
        )

    # 2. Load Validation Candidate Pairs and Ground Truth
    print("\nLoading validation candidates & ground truth for threshold calibration...")
    val_cand_df = pl.read_parquet(cand_pairs_path)
    val_gt = pl.read_parquet(val_gt_path)

    val_gt_map = {}
    for row in val_gt.iter_rows():
        s1_id, matches = row[0], row[1]
        if matches and str(matches).strip():
            val_gt_map[s1_id] = set(str(matches).strip().split(","))
        else:
            val_gt_map[s1_id] = set()

    # 3. Setup fast selective validation extractor for calibration
    # Calibrate on a representative sample of validation entities for speed and accuracy
    calib_size = min(25000, len(val_cand_df))
    print(f"\nSetting up feature extractor for {calib_size:,} validation calibration entities...")
    calib_cand_df = val_cand_df.head(calib_size)
    calib_s1_ids = calib_cand_df["source1_entity_id"].to_list()
    calib_tgt_ids = set()
    for cands_str in calib_cand_df["candidate_entity_ids"].to_list():
        if cands_str:
            for tgt in str(cands_str).split(","):
                tgt = tgt.strip()
                if tgt:
                    calib_tgt_ids.add(tgt)

    needed_s1 = pl.Series("id", calib_s1_ids)
    needed_tgt = pl.Series("id", list(calib_tgt_ids))

    val_s1 = pl.read_parquet(val_s1_path).filter(pl.col("entity_id").is_in(needed_s1))
    train_s2 = pl.read_parquet(train_s2_path).filter(pl.col("entity_id").is_in(needed_tgt))
    train_s3 = pl.read_parquet(train_s3_path).filter(pl.col("entity_id").is_in(needed_tgt))

    extractor = FeatureExtractor()
    extractor.register_dataset(val_s1)
    extractor.register_dataset(train_s2)
    extractor.register_dataset(train_s3)
    print(f"Cached {len(extractor.entity_lookup):,} calibration entities.")

    calib_gt_map = {s1_id: val_gt_map.get(s1_id, set()) for s1_id in calib_s1_ids}

    # 4. Train LightGBM & Calibrate Macro F0.5
    model, best_thresh, best_metrics = train_and_calibrate_model(
        X_train=X_train,
        y_train=y_train,
        val_cand_df=calib_cand_df,
        val_gt_map=calib_gt_map,
        extractor=extractor,
    )

    print("\n" + "=" * 70)
    print("PHASE 5 MODEL TRAINING & CALIBRATION COMPLETE!")
    print(f"Optimal Decision Threshold θ*: {best_thresh:.2f}")
    print(f"Peak Validation Macro F0.5:     {best_metrics.get('macro_f05', 0):.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
