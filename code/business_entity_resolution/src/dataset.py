"""
Dataset Utilities & Fast In-Memory Data Access (Amazon ML Challenge 2026).
Provides fast Polars-based streaming and dict loading for downstream blocking,
feature engineering, training, and evaluation pipelines.
"""

from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
import polars as pl

from .config import (
    PARQUET_DIR, PROCESSED_DIR,
    TRAIN_S1, TRAIN_S2, TRAIN_S3, TRAIN_GT,
    TEST_S1, TEST_S2, TEST_S3
)


def load_ground_truth(gt_input: Path) -> Dict[str, Set[str]]:
    """
    Loads ground truth mapping from Parquet or TSV:
    source1_entity_id -> set of matched entity IDs.
    """
    if str(gt_input).endswith(".parquet"):
        df_gt = pl.read_parquet(gt_input)
    else:
        df_gt = pl.read_csv(gt_input, separator="\t", has_header=True, quote_char=None)

    gt_map: Dict[str, Set[str]] = {}
    for row in df_gt.iter_rows():
        s1_id, matched = str(row[0]).strip(), row[1]
        if matched and str(matched).strip():
            gt_map[s1_id] = set(str(matched).strip().split(","))
        else:
            gt_map[s1_id] = set()
    return gt_map


def load_parquet_table(
    table_name: str,
    country: Optional[str] = None,
    columns: Optional[List[str]] = None
) -> pl.DataFrame:
    """
    Loads a specific source table from Parquet, optionally filtered by country partition.
    
    Args:
        table_name: e.g. "train_source1", "train_source2", "test_source1", etc.
        country: e.g. "US", "INDIA", "FRANCE"
        columns: list of columns to load (e.g. ['entity_id', 'business_name'])
    """
    if country:
        country_clean = country.strip().upper()
        path = PARQUET_DIR / f"{table_name}_country={country_clean}.parquet"
    else:
        path = PARQUET_DIR / f"{table_name}_all.parquet"

    if not path.exists():
        raise FileNotFoundError(f"Parquet table not found at {path}. Run Step 1 ingestion first.")

    if columns:
        return pl.read_parquet(path, columns=columns)
    return pl.read_parquet(path)


def get_available_countries(source_name: str = "train_source1") -> List[str]:
    """Returns all available country partition keys for a given table."""
    matching_files = list(PARQUET_DIR.glob(f"{source_name}_country=*.parquet"))
    countries = []
    for f in matching_files:
        country = f.stem.split("country=")[-1]
        countries.append(country)
    return sorted(countries)
