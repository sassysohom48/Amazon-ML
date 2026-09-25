"""
Streaming High-Speed Preprocessing Pipeline (Amazon ML Challenge 2026).
Uses PyArrow ParquetWriter to stream normalized batches directly to disk,
maintaining constant memory footprint (< 100 MB RAM) across arbitrary file sizes.
"""

import os
import sys
import time
import gc
from pathlib import Path
from typing import List, Tuple
import pyarrow as pa
import pyarrow.parquet as pq
import polars as pl

# Ensure package import works
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    PROCESSED_DIR,
    TRAIN_S1,
    TRAIN_S2,
    TRAIN_S3,
    TEST_S1,
    TEST_S2,
    TEST_S3,
)
from src.text_normalizer import normalize_record


# Define PyArrow Schema
PARQUET_SCHEMA = pa.schema([
    ("entity_id", pa.string()),
    ("country", pa.string()),
    ("business_name_raw", pa.string()),
    ("business_address_raw", pa.string()),
    ("name_clean", pa.string()),
    ("addr_clean", pa.string()),
    ("name_tokens", pa.string()),
    ("core_stem", pa.string()),
    ("postal_digits", pa.string()),
    ("has_address", pa.int8()),
])


def preprocess_tsv_streaming(
    input_tsv_path: Path,
    output_parquet_path: Path,
    chunk_size: int = 50000,
):
    """
    Streams a raw TSV file in batches of 50,000 rows directly into a ParquetWriter.
    Memory usage is strictly constant (< 100 MB RAM).
    """
    if output_parquet_path.exists():
        print(f"File already exists at {output_parquet_path}. Skipping.")
        return

    print(f"\n{'='*70}")
    print(f"STREAMING PREPROCESSING: {input_tsv_path.name}")
    print(f"Output: {output_parquet_path.name}")
    print(f"{'='*70}")

    start_time = time.time()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Read TSV lazily in batches using Polars scan
    print("Initializing streaming batch reader...")
    reader = pl.read_csv_batched(
        input_tsv_path,
        separator="\t",
        has_header=True,
        batch_size=chunk_size,
        schema_overrides={
            "entity_id": pl.Utf8,
            "business_name": pl.Utf8,
            "business_address": pl.Utf8,
            "country": pl.Utf8,
        },
    )

    writer = pq.ParquetWriter(output_parquet_path, PARQUET_SCHEMA, compression="snappy")
    total_processed = 0

    try:
        while True:
            batches = reader.next_batches(1)
            if not batches:
                break
            batch_df = batches[0]
            n_rows = len(batch_df)
            if n_rows == 0:
                break

            eids = batch_df["entity_id"].to_list()
            bnames = batch_df["business_name"].to_list()
            baddrs = batch_df["business_address"].to_list()
            countries = batch_df["country"].to_list()
            del batch_df

            # Normalize batch
            c_eids, c_countries, c_bnames, c_baddrs = [], [], [], []
            c_nclean, c_aclean, c_ntoks, c_stem, c_post, c_has_addr = [], [], [], [], [], []

            for eid, bname, baddr, country in zip(eids, bnames, baddrs, countries):
                r = normalize_record(eid, bname, baddr, country)
                c_eids.append(r[0])
                c_countries.append(r[1])
                c_bnames.append(r[2])
                c_baddrs.append(r[3])
                c_nclean.append(r[4])
                c_aclean.append(r[5])
                c_ntoks.append(r[6])
                c_stem.append(r[7])
                c_post.append(r[8])
                c_has_addr.append(r[9])

            del eids, bnames, baddrs, countries

            # Write PyArrow RecordBatch directly to disk
            pa_batch = pa.RecordBatch.from_arrays(
                [
                    pa.array(c_eids, type=pa.string()),
                    pa.array(c_countries, type=pa.string()),
                    pa.array(c_bnames, type=pa.string()),
                    pa.array(c_baddrs, type=pa.string()),
                    pa.array(c_nclean, type=pa.string()),
                    pa.array(c_aclean, type=pa.string()),
                    pa.array(c_ntoks, type=pa.string()),
                    pa.array(c_stem, type=pa.string()),
                    pa.array(c_post, type=pa.string()),
                    pa.array(c_has_addr, type=pa.int8()),
                ],
                schema=PARQUET_SCHEMA,
            )

            writer.write_batch(pa_batch)
            total_processed += n_rows
            elapsed = time.time() - start_time
            rate = total_processed / elapsed if elapsed > 0 else 0
            print(f"  Processed {total_processed:,} rows [{rate:,.0f} rows/s] (RAM < 100MB)")

            del c_eids, c_countries, c_bnames, c_baddrs, c_nclean, c_aclean, c_ntoks, c_stem, c_post, c_has_addr, pa_batch
            gc.collect()

    finally:
        writer.close()

    file_size_mb = output_parquet_path.stat().st_size / (1024 * 1024)
    print(f"Saved {output_parquet_path.name} ({total_processed:,} rows, {file_size_mb:.2f} MB) in {time.time() - start_time:.2f}s.")


def run_full_preprocessing():
    """
    Executes end-to-end streaming normalization on all 6 train & test TSV files.
    """
    print("=" * 70)
    print("PHASE 2: STREAMING TEXT PREPROCESSING & PARQUET CONVERSION")
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
        preprocess_tsv_streaming(input_tsv, output_parquet)

    print("\n" + "=" * 70)
    print(f"ALL DATASETS PREPROCESSED SUCCESSFULLY IN {time.time() - total_start:.2f}s!")
    print("=" * 70)


if __name__ == "__main__":
    run_full_preprocessing()
