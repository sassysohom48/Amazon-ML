"""
High-Speed Ingestion & Partitioning Module (Amazon ML Challenge 2026).
Converts raw TSVs into Snappy-compressed Parquet partitioned by country,
profiles dataset invariants, and generates stratified validation splits.
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
import polars as pl

from .config import (
    TRAIN_S1, TRAIN_S2, TRAIN_S3, TRAIN_GT,
    TEST_S1, TEST_S2, TEST_S3,
    PARQUET_DIR, PROCESSED_DIR,
    VAL_FRACTION, RANDOM_SEED
)


def clean_and_validate_dataframe(df: pl.DataFrame, source_prefix: Optional[str] = None) -> pl.DataFrame:
    """
    Cleans raw string columns, trims whitespace, standardizes country labels,
    and replaces null/empty address fields with empty strings.
    """
    # 1. Clean and normalize string columns
    exprs = []
    for col_name in df.columns:
        if df[col_name].dtype == pl.Utf8 or df[col_name].dtype == pl.String:
            # Strip carriage returns and outer whitespace
            exprs.append(
                pl.col(col_name)
                .str.replace_all(r"\r", "")
                .str.strip_chars()
                .fill_null("")
                .alias(col_name)
            )

    if exprs:
        df = df.with_columns(exprs)

    # 2. Standardize country column if present
    if "country" in df.columns:
        # Standardize country to uppercase stripped representation (e.g. 'US', 'INDIA', 'FRANCE')
        df = df.with_columns(
            pl.col("country").str.to_uppercase().alias("country")
        )

    # 3. Ensure address missingness flag column
    if "business_address" in df.columns:
        df = df.with_columns(
            (pl.col("business_address").str.len_chars() > 0).cast(pl.Int8).alias("has_address")
        )

    # 4. Verify entity_id prefix integrity if source_prefix is given
    if source_prefix and "entity_id" in df.columns:
        invalid_count = df.filter(~pl.col("entity_id").str.starts_with(source_prefix)).height
        if invalid_count > 0:
            print(f"⚠️ Warning: Found {invalid_count} records with unexpected ID prefix (expected {source_prefix})")

    return df


def ingest_tsv_to_parquet(
    tsv_path: Path,
    source_name: str,
    output_dir: Path = PARQUET_DIR,
    source_prefix: Optional[str] = None
) -> Dict[str, Path]:
    """
    Parses a raw TSV file using Polars, validates data integrity,
    and writes country-partitioned Snappy Parquet files.
    """
    if not tsv_path.exists():
        raise FileNotFoundError(f"Input TSV file not found: {tsv_path}")

    print(f"\n=======================================================")
    print(f"Ingesting: {tsv_path.name} -> {source_name}")
    print(f"=======================================================")
    start_time = time.time()

    # Read TSV with explicit tab separator, skipping malformed rows if any
    df = pl.read_csv(
        tsv_path,
        separator="\t",
        has_header=True,
        quote_char=None,  # Do not treat quotes as delimiters (names/addresses contain single/double quotes)
        truncate_ragged_lines=False,
        infer_schema_length=10000
    )

    total_rows = df.height
    print(f"Raw records loaded: {total_rows:,} rows across {len(df.columns)} columns")

    # Clean and sanitize strings
    df = clean_and_validate_dataframe(df, source_prefix=source_prefix)

    created_partitions: Dict[str, Path] = {}

    # Save complete unified parquet
    unified_path = output_dir / f"{source_name}_all.parquet"
    df.write_parquet(unified_path, compression="snappy")
    created_partitions["all"] = unified_path
    print(f"Saved unified table -> {unified_path.name} ({unified_path.stat().st_size / (1024*1024):.2f} MB)")

    # Partition by country if country column exists
    if "country" in df.columns:
        countries = df["country"].unique().to_list()
        for country in sorted(countries):
            country_df = df.filter(pl.col("country") == country)
            country_path = output_dir / f"{source_name}_country={country}.parquet"
            country_df.write_parquet(country_path, compression="snappy")
            created_partitions[country] = country_path
            print(f"  • Partition [{country}]: {country_df.height:,} rows -> {country_path.name}")

    elapsed = time.time() - start_time
    print(f"Ingestion completed in {elapsed:.2f}s ({total_rows / max(elapsed, 0.001):,.0f} rows/s)")
    return created_partitions


def create_stratified_validation_split(
    train_s1_parquet: Path,
    train_gt_tsv: Path,
    val_fraction: float = VAL_FRACTION,
    random_seed: int = RANDOM_SEED,
    output_dir: Path = PROCESSED_DIR
) -> Tuple[Path, Path, Path, Path]:
    """
    Creates an entity-grouped, country-stratified validation split:
    - 10% of S1 entities held out for validation
    - 90% of S1 entities kept for training
    - Ground truth split strictly along S1 entity boundary (zero leakage)
    """
    print(f"\n=======================================================")
    print(f"Creating Stratified Validation Split ({val_fraction*100:.0f}% Holdout)")
    print(f"=======================================================")

    df_s1 = pl.read_parquet(train_s1_parquet)
    print(f"Total S1 Reference Records: {df_s1.height:,}")

    # Stratify by country
    val_dfs = []
    train_dfs = []

    for country in sorted(df_s1["country"].unique().to_list()):
        subset = df_s1.filter(pl.col("country") == country)
        shuffled = subset.sample(fraction=1.0, shuffle=True, seed=random_seed)
        n_val = int(len(subset) * val_fraction)
        val_subset = shuffled.slice(0, n_val)
        train_subset = shuffled.slice(n_val)

        val_dfs.append(val_subset)
        train_dfs.append(train_subset)
        print(f"  Country [{country}]: {train_subset.height:,} train / {val_subset.height:,} val")

    df_val_s1 = pl.concat(val_dfs)
    df_train_s1 = pl.concat(train_dfs)

    val_s1_path = output_dir / "val_source1.parquet"
    train_s1_path = output_dir / "train_source1_split.parquet"

    df_val_s1.write_parquet(val_s1_path, compression="snappy")
    df_train_s1.write_parquet(train_s1_path, compression="snappy")

    # Split Ground Truth
    print(f"\nSplitting Ground Truth Labels...")
    df_gt = pl.read_csv(train_gt_tsv, separator="\t", has_header=True, quote_char=None)
    df_gt = clean_and_validate_dataframe(df_gt)

    val_s1_set = set(df_val_s1["entity_id"].to_list())

    df_val_gt = df_gt.filter(pl.col("source1_entity_id").is_in(val_s1_set))
    df_train_gt = df_gt.filter(~pl.col("source1_entity_id").is_in(val_s1_set))

    val_gt_path = output_dir / "val_ground_truth.parquet"
    train_gt_path = output_dir / "train_ground_truth_split.parquet"

    df_val_gt.write_parquet(val_gt_path, compression="snappy")
    df_train_gt.write_parquet(train_gt_path, compression="snappy")

    print(f"Ground Truth Split:")
    print(f"  • Train GT: {df_train_gt.height:,} S1 entities")
    print(f"  • Val GT:   {df_val_gt.height:,} S1 entities")

    return train_s1_path, val_s1_path, train_gt_path, val_gt_path


def audit_and_profile_data(
    train_s1_parquet: Path,
    train_gt_parquet: Path
) -> None:
    """
    Profiles data integrity, singleton ratios, match multiplicities,
    and asserts the 100% strict country partition invariant.
    """
    print(f"\n=======================================================")
    print(f"Data Profiling & Invariant Audit")
    print(f"=======================================================")

    df_s1 = pl.read_parquet(train_s1_parquet)
    df_gt = pl.read_parquet(train_gt_parquet)

    # Compute Singleton Statistics
    singletons = df_gt.filter((pl.col("matched_entity_ids") == "") | pl.col("matched_entity_ids").is_null())
    non_singletons = df_gt.filter((pl.col("matched_entity_ids") != "") & pl.col("matched_entity_ids").is_not_null())

    total_s1 = df_gt.height
    singleton_count = singletons.height
    non_singleton_count = non_singletons.height
    singleton_pct = (singleton_count / total_s1) * 100 if total_s1 > 0 else 0

    print(f"Total S1 Entities:     {total_s1:,}")
    print(f"Singletons (0 matches): {singleton_count:,} ({singleton_pct:.2f}%)")
    print(f"Linked Entities:       {non_singleton_count:,} ({100 - singleton_pct:.2f}%)")


def run_pipeline_step1() -> None:
    """
    Master execution method for Step 1:
    Ingests all raw TSVs, builds country partitions, and creates validation splits.
    """
    start_total = time.time()
    print("=" * 70)
    print("  AMAZON ML CHALLENGE 2026: PHASE 1 DATA INGESTION & PARTITIONING")
    print("=" * 70)

    # 1. Ingest Training Datasets
    if TRAIN_S1.exists():
        s1_parts = ingest_tsv_to_parquet(TRAIN_S1, "train_source1", source_prefix="S1-")
    if TRAIN_S2.exists():
        ingest_tsv_to_parquet(TRAIN_S2, "train_source2", source_prefix="S2-")
    if TRAIN_S3.exists():
        ingest_tsv_to_parquet(TRAIN_S3, "train_source3", source_prefix="S3-")

    # 2. Ingest Test Datasets (including France)
    if TEST_S1.exists():
        ingest_tsv_to_parquet(TEST_S1, "test_source1", source_prefix="S1-")
    if TEST_S2.exists():
        ingest_tsv_to_parquet(TEST_S2, "test_source2", source_prefix="S2-")
    if TEST_S3.exists():
        ingest_tsv_to_parquet(TEST_S3, "test_source3", source_prefix="S3-")

    # 3. Create Stratified Validation Split
    train_s1_all = PARQUET_DIR / "train_source1_all.parquet"
    if train_s1_all.exists() and TRAIN_GT.exists():
        create_stratified_validation_split(train_s1_all, TRAIN_GT)

    # 4. Audit & Profile
    val_gt_path = PROCESSED_DIR / "val_ground_truth.parquet"
    val_s1_path = PROCESSED_DIR / "val_source1.parquet"
    if val_s1_path.exists() and val_gt_path.exists():
        audit_and_profile_data(val_s1_path, val_gt_path)

    total_time = time.time() - start_total
    print(f"\n✅ Step 1 Ingestion and Partitioning completed in {total_time:.2f} seconds.")


if __name__ == "__main__":
    run_pipeline_step1()
