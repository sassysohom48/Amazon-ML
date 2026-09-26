"""
Step 1.1: Deterministic 5-Fold Stratified Split Generator (Amazon ML Challenge 2026).
Generates train_folds.parquet stratified by (country, match_cardinality, has_address).
Enables leak-free out-of-fold (OOF) cross-validation and stacking.
"""

import sys
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
import polars as pl
from sklearn.model_selection import StratifiedKFold

# Ensure package import works
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PROCESSED_DIR, TRAIN_DIR, RANDOM_SEED
from src.validation_contract import NUM_FOLDS, export_validation_contract


def generate_stratified_folds(
    s1_path: Path = PROCESSED_DIR / "train_source1_cleaned.parquet",
    gt_path: Optional[Path] = None,
    output_path: Path = PROCESSED_DIR / "train_folds.parquet",
    num_folds: int = NUM_FOLDS,
    random_state: int = RANDOM_SEED,
) -> pl.DataFrame:
    """
    Generates permanent, deterministic 5-fold stratification mapping.
    Stratifies by joint distribution: (country, match_cardinality, has_address).
    """
    print("=" * 75)
    print("STEP 1.1: GENERATING 5-FOLD STRATIFIED SPLITS")
    print("=" * 75)

    # 1. Load Cleaned Training S1 Entities
    print(f"Reading training Source 1 entities from {s1_path}...")
    t0 = time.time()
    s1_df = pl.read_parquet(
        s1_path,
        columns=["entity_id", "country", "has_address"]
    )
    n_s1 = len(s1_df)
    print(f"Loaded {n_s1:,} training S1 entities in {time.time() - t0:.2f}s")

    # 2. Compute Match Cardinality from Ground Truth
    if gt_path is None:
        if (PROCESSED_DIR / "train_ground_truth.parquet").exists():
            gt_path = PROCESSED_DIR / "train_ground_truth.parquet"
        elif (TRAIN_DIR / "train_ground_truth.tsv").exists():
            gt_path = TRAIN_DIR / "train_ground_truth.tsv"
        else:
            gt_path = PROCESSED_DIR / "val_ground_truth.parquet"

    print(f"\nComputing match cardinality from ground truth: {gt_path}...")
    if str(gt_path).endswith(".parquet"):
        gt_df = pl.read_parquet(gt_path)
    else:
        gt_df = pl.read_csv(gt_path, separator="\t")

    # Identify column names
    s1_col = "source1_entity_id" if "source1_entity_id" in gt_df.columns else gt_df.columns[0]
    match_col = "matched_entity_id" if "matched_entity_id" in gt_df.columns else ("matching_entity_ids" if "matching_entity_ids" in gt_df.columns else gt_df.columns[1])

    # Check if ground truth is list-formatted or pair-formatted
    sample_val = str(gt_df[match_col][0])
    if "," in sample_val or len(sample_val) > 20:
        # List formatted TSV: count comma-separated matches
        cardinality_df = gt_df.select([
            pl.col(s1_col).alias("entity_id"),
            pl.col(match_col).fill_null("").map_elements(
                lambda s: len([x for x in str(s).split(",") if x.strip()]),
                return_dtype=pl.Int32
            ).alias("match_count")
        ])
    else:
        # Pair formatted TSV: count rows per entity_id
        cardinality_df = gt_df.group_by(pl.col(s1_col).alias("entity_id")).agg(
            pl.len().alias("match_count")
        )

    print(f"Computed cardinality for {len(cardinality_df):,} entities with matches.")

    # 3. Join Cardinality to S1 Entities
    merged_df = s1_df.join(cardinality_df, on="entity_id", how="left")
    merged_df = merged_df.with_columns(
        pl.col("match_count").fill_null(0).cast(pl.Int32)
    )

    # 4. Bin Cardinality into 4 Meaningful Strata
    # Bins: 0 (singleton), 1 (1-to-1), 2_to_5 (small cluster), 6_plus (dense)
    merged_df = merged_df.with_columns(
        pl.when(pl.col("match_count") == 0)
        .then(pl.lit("singleton"))
        .when(pl.col("match_count") == 1)
        .then(pl.lit("1_to_1"))
        .when((pl.col("match_count") >= 2) & (pl.col("match_count") <= 5))
        .then(pl.lit("2_to_5"))
        .otherwise(pl.lit("6_plus"))
        .alias("match_cardinality")
    )

    # 5. Build Stratification Key
    merged_df = merged_df.with_columns(
        pl.concat_str([
            pl.col("country"),
            pl.lit("_"),
            pl.col("match_cardinality"),
            pl.lit("_addr"),
            pl.col("has_address").cast(pl.String)
        ]).alias("strata_key")
    )

    print("\nStrata Distribution across Training Data:")
    for row in merged_df["strata_key"].value_counts().sort("count", descending=True).iter_rows(named=True):
        pct = (row["count"] / n_s1) * 100
        print(f"  • {row['strata_key']:<30}: {row['count']:>10,} ({pct:>5.2f}%)")

    # 6. Apply Stratified K-Fold
    print(f"\nApplying Stratified {num_folds}-Fold Split (seed={random_state})...")
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=random_state)
    
    strata_labels = merged_df["strata_key"].to_list()
    fold_assignments = [0] * n_s1

    for fold_id, (_, val_idx) in enumerate(skf.split(merged_df, strata_labels)):
        for idx in val_idx:
            fold_assignments[idx] = fold_id

    # 7. Add fold_id and export lightweight parquet mapping
    folds_df = merged_df.with_columns(
        pl.Series("fold_id", fold_assignments, dtype=pl.Int8)
    ).select([
        "entity_id",
        "country",
        "match_cardinality",
        "has_address",
        "match_count",
        "fold_id",
    ])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    folds_df.write_parquet(output_path, compression="snappy")
    print(f"\n[SUCCESS] Successfully exported permanent 5-fold mapping to: {output_path}")

    # Print Fold Verification
    print("\nFold Verification Matrix:")
    for f_id in range(num_folds):
        f_slice = folds_df.filter(pl.col("fold_id") == f_id)
        singletons = len(f_slice.filter(pl.col("match_cardinality") == "singleton"))
        us_count = len(f_slice.filter(pl.col("country") == "US"))
        in_count = len(f_slice.filter(pl.col("country") == "India"))
        print(f"  Fold {f_id}: {len(f_slice):,} entities (US: {us_count:,}, IN: {in_count:,}, Singletons: {singletons:,})")

    # Export contract metadata alongside
    export_validation_contract(output_path.parent / "validation_contract.json")

    return folds_df


def load_fold_split(
    fold_id: int,
    folds_path: Path = PROCESSED_DIR / "train_folds.parquet"
) -> Tuple[Set[str], Set[str]]:
    """
    Returns (train_entity_ids, val_entity_ids) for the specified fold_id.
    """
    df = pl.read_parquet(folds_path, columns=["entity_id", "fold_id"])
    val_ids = set(df.filter(pl.col("fold_id") == fold_id)["entity_id"].to_list())
    train_ids = set(df.filter(pl.col("fold_id") != fold_id)["entity_id"].to_list())
    return train_ids, val_ids


if __name__ == "__main__":
    generate_stratified_folds()
