"""
Blocking Benchmark & Candidate Generation Script (Amazon ML Challenge 2026).
Runs the MultiIndexBlocker on the 10% validation split, evaluates candidate recall,
and exports candidate pairs for feature engineering.
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, Set, List
import polars as pl

# Ensure package import works
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PROCESSED_DIR
from src.blocking import MultiIndexBlocker
from src.evaluator import evaluate_blocking_recall


def benchmark_validation_blocking(max_candidates: int = 35, max_token_freq: int = 35000):
    """
    Evaluates blocking quality on the 10% validation split.
    """
    print("=" * 70)
    print("PHASE 3: BLOCKING BENCHMARK ON 10% VALIDATION SPLIT")
    print("=" * 70)

    val_s1_path = PROCESSED_DIR / "val_source1_cleaned.parquet"
    val_gt_path = PROCESSED_DIR / "val_ground_truth.parquet"
    train_s2_path = PROCESSED_DIR / "train_source2_cleaned.parquet"
    train_s3_path = PROCESSED_DIR / "train_source3_cleaned.parquet"
    output_cand_path = PROCESSED_DIR / "val_candidate_pairs.parquet"

    # 1. Load validation ground truth
    print(f"Loading validation ground truth from {val_gt_path}...")
    val_gt_df = pl.read_parquet(val_gt_path)
    val_gt_map: Dict[str, Set[str]] = {}
    for row in val_gt_df.iter_rows():
        s1_id, matched = row[0], row[1]
        if matched and str(matched).strip():
            val_gt_map[s1_id] = set(str(matched).strip().split(","))
        else:
            val_gt_map[s1_id] = set()

    total_true_links = sum(len(ids) for ids in val_gt_map.values())
    print(f"Validation Ground Truth: {len(val_gt_map):,} entities, {total_true_links:,} true matches.")

    # 2. Load validation S1 and target S2/S3
    print("Loading preprocessed clean datasets with Polars...")
    t0 = time.time()
    val_s1_df = pl.read_parquet(val_s1_path)
    train_s2_df = pl.read_parquet(train_s2_path)
    train_s3_df = pl.read_parquet(train_s3_path)
    print(f"Loaded all Parquet tables in {time.time() - t0:.2f}s.")

    # 3. Fit blocker on target pool
    blocker = MultiIndexBlocker(max_candidates=max_candidates, max_token_freq=max_token_freq)
    blocker.fit(train_s2_df, train_s3_df)

    # 4. Generate candidates for validation S1
    cand_map: Dict[str, List[str]] = blocker.block_s1(val_s1_df)

    # 5. Evaluate Candidate Recall
    cand_set_map: Dict[str, Set[str]] = {k: set(v) for k, v in cand_map.items()}
    metrics = evaluate_blocking_recall(val_gt_map, cand_set_map)

    print("\n" + "=" * 70)
    print("BLOCKING EVALUATION METRICS (10% VALIDATION SPLIT)")
    print("=" * 70)
    print(f"  • Candidate Recall (Upper Bound):  {metrics['candidate_recall'] * 100:.2f}%")
    print(f"  • Captured True Matches:          {metrics['captured_true_pairs']:,} / {metrics['total_true_pairs']:,}")
    print(f"  • Missed Matches:                  {metrics['total_true_pairs'] - metrics['captured_true_pairs']:,}")
    print(f"  • Average Candidates per S1:       {metrics['avg_candidates_per_s1']:.2f}")
    print(f"  • Total Candidates Generated:      {metrics['total_candidates']:,}")
    print("=" * 70)

    # 6. Save validation candidates to Parquet
    print(f"\nSaving validation candidate pairs to {output_cand_path}...")
    s1_ids = []
    cand_lists = []
    for s1_id, c_list in cand_map.items():
        s1_ids.append(s1_id)
        cand_lists.append(",".join(c_list))

    cand_df = pl.DataFrame({
        "source1_entity_id": s1_ids,
        "candidate_entity_ids": cand_lists
    })
    cand_df.write_parquet(output_cand_path, compression="snappy")
    print(f"Saved {output_cand_path.name} ({len(cand_df):,} rows, {output_cand_path.stat().st_size / (1024*1024):.2f} MB).")

    return metrics


if __name__ == "__main__":
    benchmark_validation_blocking()
