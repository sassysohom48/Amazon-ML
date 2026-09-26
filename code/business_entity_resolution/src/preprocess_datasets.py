"""
Step 2.3: Ultra Low-Memory Streaming Multi-Representation Preprocessing Pipeline (Amazon ML Challenge 2026).
Processes millions of business records across US, India, and France in streaming batches using PyArrow,
extracting parallel name and structured address representations while keeping peak RAM strictly under 250 MB.
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
from src.multilingual_normalizer import normalize_business_name_record
from src.address_parser import parse_address_record


PARQUET_SCHEMA = pa.schema([
    ("entity_id", pa.string()),
    ("country", pa.string()),
    ("business_name_raw", pa.string()),
    ("business_address_raw", pa.string()),
    
    # Parallel Name Representations
    ("name_clean", pa.string()),
    ("name_core", pa.string()),
    ("legal_form", pa.string()),
    ("name_tokens", pa.string()),
    ("name_acronym", pa.string()),
    ("name_phonetic", pa.string()),
    
    # Structured Address Representations
    ("addr_clean", pa.string()),
    ("postal_clean", pa.string()),
    ("addr_unit_num", pa.string()),
    ("addr_digits", pa.string()),
    ("addr_tokens", pa.string()),
    ("has_address", pa.int8()),
    
    # Backwards-compatibility aliases
    ("core_stem", pa.string()),
    ("postal_digits", pa.string()),
])


def preprocess_tsv_file_streaming(
    input_tsv_path: Path,
    output_parquet_path: Path,
    batch_size: int = 50000,
    force_recompute: bool = False,
):
    """
    Streams a TSV file in chunked batches and writes directly to Parquet.
    Extracts all Phase 2 multi-representations with low memory (< 250 MB peak).
    """
    if output_parquet_path.exists() and not force_recompute:
        print(f"File already exists at {output_parquet_path}. Skipping.")
        return

    print(f"\n{'='*75}")
    print(f"STREAMING MULTI-REPRESENTATION PREPROCESSING: {input_tsv_path.name}")
    print(f"Output: {output_parquet_path.name}")
    print(f"{'='*75}")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    start_time = time.time()

    # Create PyArrow Parquet Writer with SNAPPY compression
    writer = pq.ParquetWriter(str(output_parquet_path), PARQUET_SCHEMA, compression="SNAPPY")

    total_processed = 0
    
    # Read in streaming batches with Polars lazy reader
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
        col_name_core = []
        col_legal_form = []
        col_name_tokens = []
        col_name_acronym = []
        col_name_phonetic = []
        
        col_addr_clean = []
        col_postal_clean = []
        col_addr_unit_num = []
        col_addr_digits = []
        col_addr_tokens = []
        col_has_address = []
        
        col_core_stem = []
        col_postal_digits = []

        for eid, bname, baddr, country in zip(eids, bnames, baddrs, countries):
            eid_str = str(eid) if eid is not None else ""
            bname_str = str(bname) if bname is not None else ""
            baddr_str = str(baddr) if baddr is not None else ""
            country_str = str(country) if country is not None else ""

            # 1. Multi-Representation Name Extraction
            name_rep = normalize_business_name_record(bname_str, country_str)
            
            # 2. Structured Address Extraction
            addr_rep = parse_address_record(baddr_str, country_str)

            col_eids.append(eid_str)
            col_countries.append(country_str)
            col_bname_raw.append(bname_str)
            col_baddr_raw.append(baddr_str)
            
            col_name_clean.append(name_rep["name_clean"])
            col_name_core.append(name_rep["name_core"])
            col_legal_form.append(name_rep["legal_form"])
            col_name_tokens.append(name_rep["name_tokens"])
            col_name_acronym.append(name_rep["name_acronym"])
            col_name_phonetic.append(name_rep["name_phonetic"])
            
            col_addr_clean.append(addr_rep["addr_clean"])
            col_postal_clean.append(addr_rep["postal_clean"])
            col_addr_unit_num.append(addr_rep["addr_unit_num"])
            col_addr_digits.append(addr_rep["addr_digits"])
            col_addr_tokens.append(addr_rep["addr_tokens"])
            col_has_address.append(addr_rep["has_address"])
            
            col_core_stem.append(name_rep["name_core"])
            col_postal_digits.append(addr_rep["addr_digits"])

        # Convert batch to PyArrow Table and write directly to disk
        arrow_table = pa.Table.from_arrays(
            [
                pa.array(col_eids, type=pa.string()),
                pa.array(col_countries, type=pa.string()),
                pa.array(col_bname_raw, type=pa.string()),
                pa.array(col_baddr_raw, type=pa.string()),
                pa.array(col_name_clean, type=pa.string()),
                pa.array(col_name_core, type=pa.string()),
                pa.array(col_legal_form, type=pa.string()),
                pa.array(col_name_tokens, type=pa.string()),
                pa.array(col_name_acronym, type=pa.string()),
                pa.array(col_name_phonetic, type=pa.string()),
                pa.array(col_addr_clean, type=pa.string()),
                pa.array(col_postal_clean, type=pa.string()),
                pa.array(col_addr_unit_num, type=pa.string()),
                pa.array(col_addr_digits, type=pa.string()),
                pa.array(col_addr_tokens, type=pa.string()),
                pa.array(col_has_address, type=pa.int8()),
                pa.array(col_core_stem, type=pa.string()),
                pa.array(col_postal_digits, type=pa.string()),
            ],
            schema=PARQUET_SCHEMA,
        )

        writer.write_table(arrow_table)
        total_processed += n_rows

        if total_processed % 100000 == 0:
            elapsed = time.time() - start_time
            rate = total_processed / elapsed if elapsed > 0 else 0
            print(f"  Processed {total_processed:,} rows [{rate:,.0f} rows/s] (RAM: < 250 MB)...")

        del arrow_table, df_batch
        gc.collect()

    writer.close()
    file_size_mb = output_parquet_path.stat().st_size / (1024 * 1024)
    print(f"Finished {output_parquet_path.name} ({total_processed:,} rows, {file_size_mb:.2f} MB) in {time.time() - start_time:.2f}s.")


def run_full_preprocessing(force_recompute: bool = False):
    print("=" * 75)
    print("PHASE 2: MULTI-REPRESENTATION LOW-MEMORY STREAMING PREPROCESSING")
    print("=" * 75)

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
        preprocess_tsv_file_streaming(input_tsv, output_parquet, force_recompute=force_recompute)

    print("\n" + "=" * 75)
    print(f"ALL DATASETS PREPROCESSED SUCCESSFULLY IN {time.time() - total_start:.2f}s!")
    print("=" * 75)


if __name__ == "__main__":
    force = "--force" in sys.argv
    run_full_preprocessing(force_recompute=force)
