"""
Dataset and Validation Splitting Module (Amazon ML Challenge 2026).
Implements fast Polars loading, stratified validation splitting, and pair sampling.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Set, Tuple
import polars as pl

# Ensure package import works
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    TRAIN_GT,
    PROCESSED_DIR,
    VAL_FRACTION,
    RANDOM_SEED,
)


def create_clean_validation_split(
    val_fraction: float = VAL_FRACTION,
    random_seed: int = RANDOM_SEED,
    output_dir: Path = PROCESSED_DIR,
) -> Tuple[Path, Path]:
    """
    Creates a stratified 10% holdout validation split directly from train_source1_cleaned.parquet
    and partitions the ground truth into val_ground_truth.parquet.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    val_s1_clean_path = output_dir / "val_source1_cleaned.parquet"
    val_gt_path = output_dir / "val_ground_truth.parquet"

    if val_s1_clean_path.exists() and val_gt_path.exists():
        print(f"Validation split already exists at {val_s1_clean_path.name} and {val_gt_path.name}.")
        return val_s1_clean_path, val_gt_path

    train_s1_clean_path = output_dir / "train_source1_cleaned.parquet"
    print(f"Creating validation split from {train_s1_clean_path}...")
    df_s1 = pl.read_parquet(train_s1_clean_path)
    print(f"Total clean S1 records: {len(df_s1):,}")

    # Stratified sampling by country
    val_dfs = []
    for country in df_s1["country"].unique().to_list():
        subset = df_s1.filter(pl.col("country") == country)
        shuffled = subset.sample(fraction=1.0, shuffle=True, seed=random_seed)
        n_val = int(len(subset) * val_fraction)
        val_dfs.append(shuffled.slice(0, n_val))

    df_val_s1 = pl.concat(val_dfs)
    print(f"Validation S1: {len(df_val_s1):,} rows.")
    df_val_s1.write_parquet(val_s1_clean_path, compression="snappy")

    # Split Ground Truth
    gt_tsv = TRAIN_GT
    if not gt_tsv.exists():
        # Look in dataset/raw/train/ or dataset/train/
        for cand_path in [
            output_dir.parent / "raw" / "train" / "train_ground_truth.tsv",
            output_dir.parent / "train" / "train_ground_truth.tsv",
        ]:
            if cand_path.exists():
                gt_tsv = cand_path
                break

    print(f"Reading Ground Truth from {gt_tsv}...")
    df_gt = pl.read_csv(gt_tsv, separator="\t", has_header=True)
    val_s1_set = set(df_val_s1["entity_id"].to_list())

    df_val_gt = df_gt.filter(pl.col("source1_entity_id").is_in(val_s1_set))
    df_val_gt.write_parquet(val_gt_path, compression="snappy")

    print(f"Saved {val_s1_clean_path.name} ({len(df_val_s1):,} rows) and {val_gt_path.name} ({len(df_val_gt):,} rows).")
    return val_s1_clean_path, val_gt_path


if __name__ == "__main__":
    create_clean_validation_split()
