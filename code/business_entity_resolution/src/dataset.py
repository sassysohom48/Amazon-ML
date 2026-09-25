"""
Dataset and Validation Splitting Module (Amazon ML Challenge 2026).
Implements fast Polars loading, stratified validation splitting, and pair sampling.
"""

import os
from pathlib import Path
from typing import Dict, Set, Tuple
import polars as pl

from .config import (
    TRAIN_S1, TRAIN_S2, TRAIN_S3, TRAIN_GT,
    TEST_S1, TEST_S2, TEST_S3,
    PROCESSED_DIR, VAL_FRACTION, RANDOM_SEED
)


def load_ground_truth(gt_path: Path) -> Dict[str, Set[str]]:
    """
    Load ground truth TSV into an in-memory mapping:
    source1_entity_id -> set of matched entity IDs.
    """
    print(f"Loading ground truth from {gt_path}...")
    gt_df = pl.read_csv(gt_path, separator="\t", has_header=True)
    gt_map: Dict[str, Set[str]] = {}
    for row in gt_df.iter_rows():
        s1_id, matched = row[0], row[1]
        if matched and str(matched).strip():
            gt_map[s1_id] = set(str(matched).strip().split(","))
        else:
            gt_map[s1_id] = set()
    print(f"Loaded ground truth for {len(gt_map):,} S1 entities.")
    return gt_map


def create_validation_split(
    val_fraction: float = VAL_FRACTION,
    random_seed: int = RANDOM_SEED,
    output_dir: Path = PROCESSED_DIR
) -> Tuple[Path, Path]:
    """
    Creates a stratified 10% holdout validation split from train_source1.tsv
    stratified by country, and saves partitioned parquet files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    val_s1_path = output_dir / "val_source1.parquet"
    train_s1_path = output_dir / "train_source1_split.parquet"
    val_gt_path = output_dir / "val_ground_truth.parquet"
    train_gt_path = output_dir / "train_ground_truth_split.parquet"

    if val_s1_path.exists() and train_s1_path.exists():
        print(f"Validation split already exists at {val_s1_path}")
        return val_s1_path, train_s1_path

    print(f"Reading {TRAIN_S1} with Polars...")
    df_s1 = pl.read_csv(TRAIN_S1, separator="\t", has_header=True)
    print(f"Total S1 records: {len(df_s1):,}")

    # Stratified sampling by country
    val_dfs = []
    train_dfs = []
    for country in df_s1["country"].unique().to_list():
        subset = df_s1.filter(pl.col("country") == country)
        shuffled = subset.sample(fraction=1.0, shuffle=True, seed=random_seed)
        n_val = int(len(subset) * val_fraction)
        val_dfs.append(shuffled.slice(0, n_val))
        train_dfs.append(shuffled.slice(n_val))

    df_val_s1 = pl.concat(val_dfs)
    df_train_s1 = pl.concat(train_dfs)

    print(f"Validation S1: {len(df_val_s1):,} rows (Countries: {df_val_s1['country'].value_counts().to_dicts()})")
    print(f"Train S1 Split: {len(df_train_s1):,} rows (Countries: {df_train_s1['country'].value_counts().to_dicts()})")

    # Save S1 splits
    df_val_s1.write_parquet(val_s1_path)
    df_train_s1.write_parquet(train_s1_path)

    # Split Ground Truth
    print(f"Reading {TRAIN_GT}...")
    df_gt = pl.read_csv(TRAIN_GT, separator="\t", has_header=True)
    val_s1_set = set(df_val_s1["entity_id"].to_list())

    df_val_gt = df_gt.filter(pl.col("source1_entity_id").is_in(val_s1_set))
    df_train_gt = df_gt.filter(~pl.col("source1_entity_id").is_in(val_s1_set))

    df_val_gt.write_parquet(val_gt_path)
    df_train_gt.write_parquet(train_gt_path)

    print("Validation split successfully created and saved to Parquet!")
    return val_s1_path, train_s1_path


if __name__ == "__main__":
    create_validation_split()
