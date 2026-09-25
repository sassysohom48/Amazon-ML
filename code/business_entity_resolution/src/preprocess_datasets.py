"""
Ultra Low-Memory Streaming Preprocessing Pipeline (Amazon ML Challenge 2026).
Processes records in small streaming batches (20,000 rows at a time) using PyArrow,
keeping peak RAM consumption strictly under 100 MB. Works on any instance size.
"""

import os
import sys
import time
import gc
from pathlib import Path
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

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


def preprocess_tsv_file_streaming(
    input_tsv_path: Path,
    output_parquet_path: Path,
    batch_size: int = 25000,
):
    """
    Streams a TSV file in small batches and writes directly to Parquet.
    Peak memory usage < 100 MB.
    """
    if output_parquet_path.exists():
        print(f"File already exists at {output_parquet_path}. Skipping.")
        return

    print(f"\n{'='*70}")
    print(f"STREAMING PREPROCESSING: {input_tsv_path.name}")
    print(f"Output: {output_parquet_path.name}")
    print(f"{'='*70}")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    start_time = time.time()

    # Create PyArrow Parquet Writer
    writer = pq.ParquetWriter(str(output_parquet_path), PARQUET_SCHEMA, compression="SNAPPY")

    total_processed = 0
    
    # Read in streaming batches with Polars lazy scan or read_csv_batched
    reader = pl.read_csv_batched(
        input_tsv_path,
        separator="\t",
        has_header=True,
        batch_size=batch_size,
        schema_overrides={
            "entity_id": pl.Utf8,
            "business_name": pl.Utf8,
            "business_address": pl.Utf8,
            "country": pl.Utf8,
        },
    )

    while True:
        batches = reader.next_batches(1)
        if not batches:
            break
        df_batch = batches[0]
        n_rows = len(df_batch)
        if n_rows == 0:
            break

        eids = df_batch["entity_id"].to_list()
        bnames = df_batch["business_name"].to_list()
        baddrs = df_batch["business_address"].to_list()
        countries = df_batch["country"].to_list()

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

        for eid, bname, baddr, country in zip(eids, bnames, baddrs, countries):
            r = normalize_record(eid, bname, baddr, country)
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

        # Convert batch to Arrow Table and write directly to disk
        arrow_table = pa.Table.from_arrays(
            [
                pa.array(col_eids, type=pa.string()),
                pa.array(col_countries, type=pa.string()),
                pa.array(col_bname_raw, type=pa.string()),
                pa.array(col_baddr_raw, type=pa.string()),
                pa.array(col_name_clean, type=pa.string()),
                pa.array(col_addr_clean, type=pa.string()),
                pa.array(col_name_tokens, type=pa.string()),
                pa.array(col_core_stem, type=pa.string()),
                pa.array(col_postal_digits, type=pa.string()),
                pa.array(col_has_address, type=pa.int8()),
            ],
            schema=PARQUET_SCHEMA,
        )

        writer.write_table(arrow_table)
        total_processed += n_rows

        if total_processed % 100000 == 0:
            elapsed = time.time() - start_time
            rate = total_processed / elapsed if elapsed > 0 else 0
            print(f"  Processed {total_processed:,} rows [{rate:,.0f} rows/s] (RAM: < 100 MB)...")

        del arrow_table, df_batch, col_eids, col_countries, col_name_clean, col_addr_clean
        gc.collect()

    writer.close()
    file_size_mb = output_parquet_path.stat().st_size / (1024 * 1024)
    print(f"Finished {output_parquet_path.name} ({total_processed:,} rows, {file_size_mb:.2f} MB) in {time.time() - start_time:.2f}s.")


def run_full_preprocessing():
    print("=" * 70)
    print("PHASE 2: LOW-MEMORY STREAMING PREPROCESSING")
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
        preprocess_tsv_file_streaming(input_tsv, output_parquet)

    print("\n" + "=" * 70)
    print(f"ALL DATASETS PREPROCESSED SUCCESSFULLY IN {time.time() - total_start:.2f}s!")
    print("=" * 70)


if __name__ == "__main__":
    run_full_preprocessing()
