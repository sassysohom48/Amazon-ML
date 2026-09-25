"""
High-Speed Multilingual Dataset Preprocessing Pipeline (Amazon ML Challenge 2026).
Processes raw TSV datasets using multi-core multiprocessing and saves
optimized Snappy-compressed Parquet files.
"""

import os
import sys
import time
from pathlib import Path
from typing import List, Tuple
from concurrent.futures import ProcessPoolExecutor
import polars as pl

# Ensure package import works
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    TRAIN_S1, TRAIN_S2, TRAIN_S3,
    TEST_S1, TEST_S2, TEST_S3,
    PROCESSED_DIR
)
from src.text_normalizer import normalize_record


def process_chunk(
    rows: List[Tuple[str, str, str, str]]
) -> List[Tuple[str, str, str, str, str, str, str, str, str, int]]:
    """Process a chunk of (eid, bname, baddr, country) tuples."""
    return [normalize_record(r[0], r[1], r[2], r[3]) for r in rows]


def preprocess_tsv_file(
    input_tsv_path: Path,
    output_parquet_path: Path,
    chunk_size: int = 50000,
    max_workers: int = None
):
    """
    Reads a raw TSV file, cleans and normalizes all fields using multi-core workers,
    and writes the result to a Snappy-compressed Parquet file.
    """
    if output_parquet_path.exists():
        print(f"File already exists at {output_parquet_path}. Skipping.")
        return

    print(f"\n{'='*70}")
    print(f"PREPROCESSING: {input_tsv_path.name}")
    print(f"Output: {output_parquet_path.name}")
    print(f"{'='*70}")

    start_time = time.time()
    
    # Read raw TSV using Polars for high speed
    print("Reading raw TSV with Polars...")
    df_raw = pl.read_csv(
        input_tsv_path,
        separator="\t",
        has_header=True,
        schema_overrides={"entity_id": pl.Utf8, "business_name": pl.Utf8, "business_address": pl.Utf8, "country": pl.Utf8}
    )
    total_rows = len(df_raw)
    print(f"Loaded {total_rows:,} raw rows in {time.time() - start_time:.2f}s.")

    # Convert to list of tuples for parallel batching
    eids = df_raw["entity_id"].to_list()
    bnames = df_raw["business_name"].to_list()
    baddrs = df_raw["business_address"].to_list()
    countries = df_raw["country"].to_list()

    rows = list(zip(eids, bnames, baddrs, countries))

    # Split into chunks
    chunks = [rows[i:i + chunk_size] for i in range(0, total_rows, chunk_size)]
    cpu_count = max_workers or os.cpu_count() or 4
    print(f"Dispatched into {len(chunks)} chunks across {cpu_count} CPU cores...")

    processed_records = []
    norm_start = time.time()
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for i, chunk_result in enumerate(executor.map(process_chunk, chunks)):
            processed_records.extend(chunk_result)
            if (i + 1) % 10 == 0 or (i + 1) == len(chunks):
                progress_pct = ((i + 1) / len(chunks)) * 100
                elapsed = time.time() - norm_start
                rate = len(processed_records) / elapsed if elapsed > 0 else 0
                print(f"  Progress: {progress_pct:.1f}% ({len(processed_records):,}/{total_rows:,} rows) [{rate:,.0f} rows/s]")

    print(f"Normalization complete in {time.time() - norm_start:.2f}s.")

    # Build Polars DataFrame
    print("Creating Parquet DataFrame...")
    schema = {
        "entity_id": pl.Utf8,
        "country": pl.Categorical,
        "business_name_raw": pl.Utf8,
        "business_address_raw": pl.Utf8,
        "name_clean": pl.Utf8,
        "addr_clean": pl.Utf8,
        "name_tokens": pl.Utf8,
        "core_stem": pl.Utf8,
        "postal_digits": pl.Utf8,
        "has_address": pl.Int8,
    }

    # Transpose lists for fast columnar DataFrame construction
    cols = list(zip(*processed_records))
    df_clean = pl.DataFrame(
        {
            "entity_id": cols[0],
            "country": cols[1],
            "business_name_raw": cols[2],
            "business_address_raw": cols[3],
            "name_clean": cols[4],
            "addr_clean": cols[5],
            "name_tokens": cols[6],
            "core_stem": cols[7],
            "postal_digits": cols[8],
            "has_address": cols[9],
        },
        schema=schema
    )

    # Save to Parquet with snappy compression
    output_parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df_clean.write_parquet(output_parquet_path, compression="snappy")
    
    file_size_mb = output_parquet_path.stat().st_size / (1024 * 1024)
    total_elapsed = time.time() - start_time
    print(f"SAVED: {output_parquet_path.name} ({file_size_mb:.2f} MB) in {total_elapsed:.2f}s total.")


def run_full_preprocessing():
    """Run preprocessing across all train and test files."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    files_to_process = [
        # (Input TSV, Output Cleaned Parquet)
        (TRAIN_S1, PROCESSED_DIR / "train_source1_cleaned.parquet"),
        (TRAIN_S2, PROCESSED_DIR / "train_source2_cleaned.parquet"),
        (TRAIN_S3, PROCESSED_DIR / "train_source3_cleaned.parquet"),
        (TEST_S1, PROCESSED_DIR / "test_source1_cleaned.parquet"),
        (TEST_S2, PROCESSED_DIR / "test_source2_cleaned.parquet"),
        (TEST_S3, PROCESSED_DIR / "test_source3_cleaned.parquet"),
    ]

    total_start = time.time()
    for input_tsv, output_parquet in files_to_process:
        preprocess_tsv_file(input_tsv, output_parquet)

    # Also build validation split cleaned parquet
    val_s1_parquet = PROCESSED_DIR / "val_source1.parquet"
    val_s1_clean_parquet = PROCESSED_DIR / "val_source1_cleaned.parquet"
    if val_s1_parquet.exists() and not val_s1_clean_parquet.exists():
        print("\nCreating val_source1_cleaned.parquet...")
        val_s1_df = pl.read_parquet(val_s1_parquet)
        val_eids = set(val_s1_df["entity_id"].to_list())
        train_s1_clean = pl.read_parquet(PROCESSED_DIR / "train_source1_cleaned.parquet")
        val_clean_df = train_s1_clean.filter(pl.col("entity_id").is_in(val_eids))
        val_clean_df.write_parquet(val_s1_clean_parquet, compression="snappy")
        print(f"Saved {val_s1_clean_parquet.name} ({len(val_clean_df):,} rows).")

    print(f"\nALL DATASETS SUCCESSFULLY PREPROCESSED IN {time.time() - total_start:.2f}s TOTAL!")


if __name__ == "__main__":
    run_full_preprocessing()
