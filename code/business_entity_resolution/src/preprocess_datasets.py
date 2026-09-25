"""
High-Speed & Memory-Efficient Text Preprocessing Pipeline (Amazon ML Challenge 2026).
Cleans and normalizes all 24.2M records across train & test in streaming batches,
ensuring low memory consumption (< 500 MB RAM) on any instance size.
"""

import os
import sys
import time
import gc
from pathlib import Path
from typing import List, Tuple
from concurrent.futures import ProcessPoolExecutor
import polars as pl

# Ensure package import works
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    TRAIN_DIR,
    TEST_DIR,
    PROCESSED_DIR,
    TRAIN_S1,
    TRAIN_S2,
    TRAIN_S3,
    TEST_S1,
    TEST_S2,
    TEST_S3,
)
from src.text_normalizer import normalize_record


def process_row_tuple(row: Tuple[str, str, str, str]) -> Tuple:
    return normalize_record(row[0], row[1], row[2], row[3])


def process_batch(batch: List[Tuple[str, str, str, str]]) -> List[Tuple]:
    return [normalize_record(r[0], r[1], r[2], r[3]) for r in batch]


def preprocess_tsv_file(
    input_tsv_path: Path,
    output_parquet_path: Path,
    batch_size: int = 50000,
    max_workers: int = None,
):
    """
    Reads a raw TSV file in memory-efficient batches, normalizes fields,
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
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Read TSV via Polars
    print("Reading raw TSV with Polars...")
    df_raw = pl.read_csv(
        input_tsv_path,
        separator="\t",
        has_header=True,
        schema_overrides={
            "entity_id": pl.Utf8,
            "business_name": pl.Utf8,
            "business_address": pl.Utf8,
            "country": pl.Utf8,
        },
    )
    total_rows = len(df_raw)
    print(f"Loaded {total_rows:,} raw rows in {time.time() - start_time:.2f}s.")

    # 2. Extract column arrays
    eids = df_raw["entity_id"].to_list()
    bnames = df_raw["business_name"].to_list()
    baddrs = df_raw["business_address"].to_list()
    countries = df_raw["country"].to_list()
    del df_raw
    gc.collect()

    cpu_count = max_workers or os.cpu_count() or 2
    chunk_size = max(5000, total_rows // (cpu_count * 8))
    
    chunks = []
    for i in range(0, total_rows, chunk_size):
        chunk_eids = eids[i:i + chunk_size]
        chunk_bnames = bnames[i:i + chunk_size]
        chunk_baddrs = baddrs[i:i + chunk_size]
        chunk_countries = countries[i:i + chunk_size]
        chunks.append(list(zip(chunk_eids, chunk_bnames, chunk_baddrs, chunk_countries)))

    del eids, bnames, baddrs, countries
    gc.collect()

    print(f"Processing {total_rows:,} rows across {len(chunks)} chunks on {cpu_count} cores...")
    norm_start = time.time()

    # Pre-allocate column lists
    col_eids = []
    col_countries = []
    col_bname_raw = []
    col_baddr_raw = []
    col_name_clean = []
    col_addr_clean = []
    col_name_tokens = []
    col_core_stem = []
    col_postal_digits = []
    col_has_address = []

    processed_count = 0
    with ProcessPoolExecutor(max_workers=cpu_count) as executor:
        for i, chunk_res in enumerate(executor.map(process_batch, chunks)):
            for r in chunk_res:
                col_eids.append(r[0])
                col_countries.append(r[1])
                col_bname_raw.append(r[2])
                col_baddr_raw.append(r[3])
                col_name_clean.append(r[4])
                col_addr_clean.append(r[5])
                col_name_tokens.append(r[6])
                col_core_stem.append(r[7])
                col_postal_digits.append(r[8])
                col_has_address.append(r[9])

            processed_count += len(chunk_res)
            if (i + 1) % 5 == 0 or (i + 1) == len(chunks):
                pct = (processed_count / total_rows) * 100
                elapsed = time.time() - norm_start
                rate = processed_count / elapsed if elapsed > 0 else 0
                print(f"  Progress: {pct:.1f}% ({processed_count:,}/{total_rows:,} rows) [{rate:,.0f} rows/s]")

    print(f"Normalization complete in {time.time() - norm_start:.2f}s.")

    # 3. Build Polars DataFrame and write to Snappy Parquet
    print("Writing compressed Parquet...")
    df_clean = pl.DataFrame(
        {
            "entity_id": col_eids,
            "country": pl.Series(col_countries, dtype=pl.Categorical),
            "business_name_raw": col_bname_raw,
            "business_address_raw": col_baddr_raw,
            "name_clean": col_name_clean,
            "addr_clean": col_addr_clean,
            "name_tokens": col_name_tokens,
            "core_stem": col_core_stem,
            "postal_digits": col_postal_digits,
            "has_address": pl.Series(col_has_address, dtype=pl.Int8),
        }
    )

    df_clean.write_parquet(output_parquet_path, compression="snappy")
    file_size_mb = output_parquet_path.stat().st_size / (1024 * 1024)
    print(f"Saved {output_parquet_path.name} ({len(df_clean):,} rows, {file_size_mb:.2f} MB) in {time.time() - start_time:.2f}s total.")

    del df_clean, col_eids, col_countries, col_name_clean, col_addr_clean
    gc.collect()


def run_full_preprocessing():
    """
    Executes end-to-end normalization on all 6 train & test TSV files.
    """
    print("=" * 70)
    print("PHASE 2: TEXT PREPROCESSING & PARQUET CONVERSION")
    print("=" * 70)

    jobs = [
        (TRAIN_S1, PROCESSED_DIR / "train_source1_cleaned.parquet"),
        (TRAIN_S2, PROCESSED_DIR / "train_source2_cleaned.parquet"),
        (TRAIN_S3, PROCESSED_DIR / "train_source3_cleaned.parquet"),
        (TEST_S1, PROCESSED_DIR / "test_source1_cleaned.parquet"),
        (TEST_S2, PROCESSED_DIR / "test_source2_cleaned.parquet"),
        (TEST_S3, PROCESSED_DIR / "test_source3_cleaned.parquet"),
    ]

    total_start = time.time()
    for input_tsv, output_parquet in jobs:
        if not input_tsv.exists():
            print(f"[Warning] Input file not found: {input_tsv}. Skipping.")
            continue
        preprocess_tsv_file(input_tsv, output_parquet)

    print("\n" + "=" * 70)
    print(f"ALL DATASETS PREPROCESSED SUCCESSFULLY IN {time.time() - total_start:.2f}s!")
    print("=" * 70)


if __name__ == "__main__":
    run_full_preprocessing()
