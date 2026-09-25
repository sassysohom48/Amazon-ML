"""
Phase 4: Pairwise High-Dimensional Feature Engineering Runner (Amazon ML Challenge 2026).
Constructs balanced training pairs (1 Positive : 4 Hard Negatives), extracts 27 RapidFuzz
and structural similarity features, and exports train_features.parquet for model training.
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple
import numpy as np
import polars as pl

# Ensure package import works
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PROCESSED_DIR
from src.feature_engineering import FeatureExtractor, FEATURE_NAMES
from src.dataset import create_clean_validation_split
from src.run_blocking import benchmark_validation_blocking


def run_phase_4_feature_engineering(
    max_samples: int = 350000,
    neg_to_pos_ratio: int = 4,
    random_seed: int = 42,
):
    print("=" * 70)
    print("PHASE 4: PAIRWISE FEATURE ENGINEERING & TRAINING DATASET GENERATION")
    print("=" * 70)

    val_s1_path = PROCESSED_DIR / "val_source1_cleaned.parquet"
    val_gt_path = PROCESSED_DIR / "val_ground_truth.parquet"
    train_s2_path = PROCESSED_DIR / "train_source2_cleaned.parquet"
    train_s3_path = PROCESSED_DIR / "train_source3_cleaned.parquet"
    cand_pairs_path = PROCESSED_DIR / "val_candidate_pairs.parquet"
    output_train_features = PROCESSED_DIR / "train_features.parquet"

    # 1. Ensure validation split exists
    if not val_s1_path.exists() or not val_gt_path.exists():
        print("Creating validation split...")
        create_clean_validation_split()

    # 2. Ensure candidate blocking pairs exist
    if not cand_pairs_path.exists():
        print("Running candidate blocking to generate training candidates...")
        benchmark_validation_blocking(max_candidates=35, max_token_freq=35000)

    # 3. Load Ground Truth & Candidate Pool
    print("\nLoading Ground Truth and candidate blocking pairs...")
    t0 = time.time()
    val_gt = pl.read_parquet(val_gt_path)
    cand_df = pl.read_parquet(cand_pairs_path)
    print(f"Loaded validation pairs in {time.time() - t0:.2f}s.")

    # 4. Build Ground Truth lookup set
    print("\nIndexing Ground Truth true match pairs...")
    gt_pairs: Set[Tuple[str, str]] = set()
    for row in val_gt.iter_rows():
        s1_id, matches = row[0], row[1]
        if matches and str(matches).strip():
            for tgt_id in str(matches).strip().split(","):
                tgt_id = tgt_id.strip()
                if tgt_id:
                    gt_pairs.add((s1_id, tgt_id))

    print(f"Total True Positive Pairs in Validation GT: {len(gt_pairs):,}")

    # 5. Mine Positives and Hard Negatives from Blocker Candidates
    print("\nMining Positives and Hard Negatives from candidate pool...")
    pos_pairs: List[Tuple[str, str]] = []
    neg_pairs: List[Tuple[str, str]] = []

    np.random.seed(random_seed)

    for row in cand_df.iter_rows():
        s1_id, cands_str = row[0], row[1]
        if not cands_str:
            continue
        cands = str(cands_str).split(",")
        for tgt_id in cands:
            tgt_id = tgt_id.strip()
            if not tgt_id:
                continue
            pair = (s1_id, tgt_id)
            if pair in gt_pairs:
                pos_pairs.append(pair)
            else:
                neg_pairs.append(pair)

    print(f"Candidate pool composition: {len(pos_pairs):,} Positives (True Matches), {len(neg_pairs):,} Hard Negatives.")

    # 6. Balanced Subsampling
    n_pos = min(len(pos_pairs), max_samples // (1 + neg_to_pos_ratio))
    n_neg = min(len(neg_pairs), n_pos * neg_to_pos_ratio)

    if len(pos_pairs) > n_pos:
        pos_idx = np.random.choice(len(pos_pairs), size=n_pos, replace=False)
        selected_pos = [pos_pairs[i] for i in pos_idx]
    else:
        selected_pos = pos_pairs

    if len(neg_pairs) > n_neg:
        neg_idx = np.random.choice(len(neg_pairs), size=n_neg, replace=False)
        selected_neg = [neg_pairs[i] for i in neg_idx]
    else:
        selected_neg = neg_pairs

    print(f"Balanced sample: {len(selected_pos):,} Positives (y=1) and {len(selected_neg):,} Hard Negatives (y=0).")
    print(f"Sampling ratio: 1 : {len(selected_neg)/len(selected_pos):.1f}")

    all_pairs = selected_pos + selected_neg
    all_labels = [1] * len(selected_pos) + [0] * len(selected_neg)

    # Shuffle
    shuffle_order = np.random.permutation(len(all_pairs))
    shuffled_pairs = [all_pairs[i] for i in shuffle_order]
    shuffled_labels = [all_labels[i] for i in shuffle_order]

    # 7. Fast Selective Entity Registration
    needed_s1 = pl.Series("id", list({p[0] for p in shuffled_pairs}))
    needed_tgt = pl.Series("id", list({p[1] for p in shuffled_pairs}))
    print(f"\nLoading entity attributes for {len(needed_s1) + len(needed_tgt):,} active entities...")
    t_reg = time.time()
    
    val_s1 = pl.read_parquet(val_s1_path).filter(pl.col("entity_id").is_in(needed_s1))
    train_s2 = pl.read_parquet(train_s2_path).filter(pl.col("entity_id").is_in(needed_tgt))
    train_s3 = pl.read_parquet(train_s3_path).filter(pl.col("entity_id").is_in(needed_tgt))

    extractor = FeatureExtractor()
    extractor.register_dataset(val_s1)
    extractor.register_dataset(train_s2)
    extractor.register_dataset(train_s3)
    print(f"Cached {len(extractor.entity_lookup):,} entities for O(1) attribute lookup in {time.time() - t_reg:.2f}s.")

    # 8. Extract 27-Dimensional RapidFuzz Features
    print(f"\nExtracting 27 RapidFuzz similarity features for {len(shuffled_pairs):,} pairs...")
    t_feat_start = time.time()
    X, y = extractor.extract_features_for_pairs(shuffled_pairs, shuffled_labels)
    feat_elapsed = time.time() - t_feat_start
    print(f"Feature extraction completed in {feat_elapsed:.2f}s ({len(X)/feat_elapsed:,.0f} pairs/s)!")

    # 9. Build and Export Feature DataFrame
    print(f"\nBuilding feature table and exporting to {output_train_features.name}...")
    s1_col = [p[0] for p in shuffled_pairs]
    tgt_col = [p[1] for p in shuffled_pairs]

    feature_dict = {
        "source1_id": s1_col,
        "target_id": tgt_col,
        "label": y,
    }

    for col_idx, feat_name in enumerate(FEATURE_NAMES):
        feature_dict[feat_name] = X[:, col_idx]

    df_features = pl.DataFrame(feature_dict)
    df_features.write_parquet(output_train_features, compression="snappy")
    file_size_mb = output_train_features.stat().st_size / (1024 * 1024)

    # 10. Summary & Feature Statistics
    print("\n" + "=" * 70)
    print("PHASE 4 FEATURE ENGINEERING SUMMARY:")
    print("=" * 70)
    print(f"  • Exported File:            {output_train_features.name} ({file_size_mb:.2f} MB)")
    print(f"  • Total Training Pairs:     {len(df_features):,}")
    print(f"  • Positives (Matches):      {int((y == 1).sum()):,} ({(y == 1).mean()*100:.1f}%)")
    print(f"  • Negatives (Non-matches):  {int((y == 0).sum()):,} ({(y == 0).mean()*100:.1f}%)")
    print(f"  • Feature Dimensions:       {len(FEATURE_NAMES)} Pairwise Similarity Metrics")
    print(f"  • Feature Extraction Rate:  {len(X)/feat_elapsed:,.0f} pairs/sec")
    print("=" * 70)

    # Print mean feature separation (Positives vs Negatives)
    print("\nKey Feature Discriminative Power (Mean Positive vs Mean Negative):")
    pos_mask = (y == 1)
    neg_mask = (y == 0)
    for feat_name in [
        "name_fuzz_token_set_ratio",
        "name_fuzz_wratio",
        "name_jaro_winkler",
        "name_tok_jaccard",
        "addr_fuzz_ratio",
        "addr_tok_jaccard",
        "postal_exact_match",
        "name_set_x_addr_jaccard",
    ]:
        col_data = df_features[feat_name].to_numpy()
        pos_mean = col_data[pos_mask].mean()
        neg_mean = col_data[neg_mask].mean()
        print(f"  • {feat_name:<28} | Pos: {pos_mean:.4f} | Neg: {neg_mean:.4f} | Diff: {pos_mean - neg_mean:+.4f}")

    print("=" * 70)
    print("PHASE 4 COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    return df_features


if __name__ == "__main__":
    run_phase_4_feature_engineering()
