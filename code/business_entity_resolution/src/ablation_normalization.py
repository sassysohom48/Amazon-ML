"""
Step 2.4: Ultra-Fast Multi-Representation Normalization Ablation & Collision Diagnostics.
Vectorized with Polars for sub-second execution across Ground Truth pairs and target space.
"""

import os
import sys
import time
import json
from pathlib import Path
from collections import defaultdict
import polars as pl

# Ensure package import works
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PROCESSED_DIR, DATASET_DIR, TRAIN_DIR


def evaluate_normalization_ablation(
    sample_size: int = 100000,
    output_path: Path = None,
):
    """
    Evaluates True Positive Match Coverage & Collision Rates for all Phase 2 representations.
    Uses vectorized Polars operations and lazy filtering for sub-second execution.
    """
    print("=" * 80)
    print("STEP 2.4: VECTORIZED NORMALIZATION ABLATION & COLLISION BENCHMARK")
    print("=" * 80)

    if output_path is None:
        output_path = PROCESSED_DIR / "phase2_normalization_ablation.json"

    # 1. Load Ground Truth
    gt_path = PROCESSED_DIR / "train_ground_truth.parquet"
    if not gt_path.exists():
        gt_path = TRAIN_DIR / "train_ground_truth.tsv"
    if not gt_path.exists():
        gt_path = PROCESSED_DIR / "val_ground_truth.parquet"

    print(f"Reading ground truth matches from {gt_path}...")
    if str(gt_path).endswith(".parquet"):
        gt_df = pl.read_parquet(gt_path)
    else:
        gt_df = pl.read_csv(gt_path, separator="\t")

    # Standardize column names
    col_mapping = {}
    for c in gt_df.columns:
        if c in ("source1_entity_id", "source_entity_id", "s1_id"):
            col_mapping[c] = "s1_id"
        elif c in ("target_entity_id", "s2_id", "s3_id", "target_id"):
            col_mapping[c] = "tgt_id"
    gt_df = gt_df.rename(col_mapping)

    n_total_gt = len(gt_df)
    if sample_size and sample_size < n_total_gt:
        gt_df = gt_df.sample(n=sample_size, seed=42)
        print(f"Sampled {len(gt_df):,} ground truth pairs for high-speed diagnostic ablation.")
    else:
        print(f"Evaluating across all {n_total_gt:,} ground truth pairs.")

    s1_needed_ids = set(gt_df["s1_id"].to_list())
    tgt_needed_ids = set(gt_df["tgt_id"].to_list())

    # 2. Load Processed S1, S2, S3 with Phase 2 Columns (Filtering only needed entities)
    cols_to_load = [
        "entity_id", "country", "name_clean", "name_core", "legal_form",
        "name_tokens", "name_acronym", "name_phonetic", "addr_clean",
        "postal_clean", "addr_unit_num", "addr_digits", "addr_tokens", "has_address"
    ]

    print("\nLoading enriched entity parquets (filtered to GT pairs)...")
    t0 = time.time()
    
    # Fast lazy or direct filtered reading
    s1_df = pl.read_parquet(PROCESSED_DIR / "train_source1_cleaned.parquet", columns=cols_to_load)
    s1_filtered = s1_df.filter(pl.col("entity_id").is_in(s1_needed_ids))

    s2_df = pl.read_parquet(PROCESSED_DIR / "train_source2_cleaned.parquet", columns=cols_to_load)
    s3_df = pl.read_parquet(PROCESSED_DIR / "train_source3_cleaned.parquet", columns=cols_to_load)
    targets_filtered = pl.concat([s2_df, s3_df]).filter(pl.col("entity_id").is_in(tgt_needed_ids))

    print(f"Loaded and filtered S1 ({len(s1_filtered):,}) & Targets ({len(targets_filtered):,}) in {time.time() - t0:.2f}s")

    # Fast in-memory lookup dictionary (only for the sampled 100k pairs!)
    s1_records = {row["entity_id"]: row for row in s1_filtered.iter_rows(named=True)}
    target_records = {row["entity_id"]: row for row in targets_filtered.iter_rows(named=True)}

    # 3. True Pair Signal Overlap Analysis by Country
    results_by_country = defaultdict(lambda: {
        "total_pairs": 0,
        "exact_name_clean": 0,
        "exact_name_core": 0,
        "exact_name_acronym": 0,
        "exact_name_phonetic": 0,
        "name_token_overlap": 0,
        "exact_addr_clean": 0,
        "exact_postal_clean": 0,
        "addr_digits_overlap": 0,
        "addr_tokens_overlap": 0,
        "exact_addr_unit": 0,
        "combined_recall_ceiling": 0,
    })

    print("\nComputing signal matching rates on ground truth pairs...")
    t1 = time.time()
    for row in gt_df.iter_rows(named=True):
        src_id = row["s1_id"]
        tgt_id = row["tgt_id"]

        if src_id not in s1_records or tgt_id not in target_records:
            continue

        s1 = s1_records[src_id]
        tgt = target_records[tgt_id]
        country = s1.get("country", "Unknown") or "Unknown"

        c_dict = results_by_country[country]
        c_dict["total_pairs"] += 1

        # Name signals
        has_clean_name = bool(s1["name_clean"] and s1["name_clean"] == tgt["name_clean"])
        has_core_name = bool(s1["name_core"] and s1["name_core"] == tgt["name_core"])
        has_acronym = bool(s1["name_acronym"] and tgt["name_acronym"] and s1["name_acronym"] == tgt["name_acronym"])
        has_phonetic = bool(s1["name_phonetic"] and tgt["name_phonetic"] and s1["name_phonetic"] == tgt["name_phonetic"])

        s1_ntokens = set(s1["name_tokens"].split()) if s1["name_tokens"] else set()
        tgt_ntokens = set(tgt["name_tokens"].split()) if tgt["name_tokens"] else set()
        has_name_tok = bool(s1_ntokens & tgt_ntokens)

        # Address signals
        has_clean_addr = bool(s1["addr_clean"] and s1["addr_clean"] == tgt["addr_clean"])
        has_postal = bool(s1["postal_clean"] and tgt["postal_clean"] and s1["postal_clean"] == tgt["postal_clean"])
        
        s1_adigits = set(s1["addr_digits"].split()) if s1["addr_digits"] else set()
        tgt_adigits = set(tgt["addr_digits"].split()) if tgt["addr_digits"] else set()
        has_addr_digits = bool(s1_adigits & tgt_adigits)

        s1_atokens = set(s1["addr_tokens"].split()) if s1["addr_tokens"] else set()
        tgt_atokens = set(tgt["addr_tokens"].split()) if tgt["addr_tokens"] else set()
        has_addr_tok = bool(s1_atokens & tgt_atokens)

        has_unit = bool(s1["addr_unit_num"] and tgt["addr_unit_num"] and s1["addr_unit_num"] == tgt["addr_unit_num"])

        # Combined blocker coverage (Union of high-precision signals)
        is_captured = bool(
            has_clean_name or has_core_name or has_name_tok or has_addr_tok or (has_acronym and has_postal) or (has_phonetic and has_postal)
        )

        if has_clean_name: c_dict["exact_name_clean"] += 1
        if has_core_name: c_dict["exact_name_core"] += 1
        if has_acronym: c_dict["exact_name_acronym"] += 1
        if has_phonetic: c_dict["exact_name_phonetic"] += 1
        if has_name_tok: c_dict["name_token_overlap"] += 1
        if has_clean_addr: c_dict["exact_addr_clean"] += 1
        if has_postal: c_dict["exact_postal_clean"] += 1
        if has_addr_digits: c_dict["addr_digits_overlap"] += 1
        if has_addr_tok: c_dict["addr_tokens_overlap"] += 1
        if has_unit: c_dict["exact_addr_unit"] += 1
        if is_captured: c_dict["combined_recall_ceiling"] += 1

    print(f"Signal matching completed in {time.time() - t1:.2f}s")

    # 4. Vectorized Collision & Bucket Size Diagnostics on Target Space using Polars
    print("\nComputing target index bucket statistics & collision metrics (Vectorized Polars)...")
    t2 = time.time()
    targets_all = pl.concat([s2_df, s3_df])
    total_target_rows = len(targets_all)
    
    collision_metrics = {}
    cols_to_check = ["name_clean", "name_core", "name_acronym", "name_phonetic", "postal_clean"]

    for col in cols_to_check:
        col_series = targets_all[col].filter(pl.col(col) != "").filter(pl.col(col).is_not_null())
        vc = col_series.value_counts().sort("count", descending=True)
        unique_buckets = len(vc)
        max_bucket = vc["count"][0] if len(vc) > 0 else 0
        top10_sum = vc["count"][:10].sum() if len(vc) > 0 else 0
        total_items = len(col_series)
        top10_collision_pct = (top10_sum / total_items * 100.0) if total_items > 0 else 0.0

        collision_metrics[col] = {
            "total_items": total_items,
            "unique_buckets": unique_buckets,
            "max_bucket_size": int(max_bucket),
            "top10_collision_pct": round(float(top10_collision_pct), 2),
        }

    print(f"Collision metrics computed in {time.time() - t2:.2f}s")

    # 5. Format & Display Results
    print("\n" + "=" * 80)
    print(f"{'COUNTRY':<15} | {'METRIC':<25} | {'TRUE PAIRS':<12} | {'COVERAGE %'}")
    print("-" * 80)

    formatted_summary = {}
    for country, metrics in results_by_country.items():
        total = metrics["total_pairs"]
        if total == 0:
            continue
        formatted_summary[country] = {}
        for k, v in metrics.items():
            if k == "total_pairs":
                continue
            pct = (v / total) * 100.0
            formatted_summary[country][k] = {"count": v, "pct": round(pct, 2)}
            print(f"{country:<15} | {k:<25} | {v:>10,} / {total:<10,} | {pct:>6.2f}%")
        print("-" * 80)

    print("\nCOLLISION & BUCKET GRANULARITY METRICS (Target Space Safety):")
    print("-" * 80)
    print(f"{'REPRESENTATION':<20} | {'UNIQUE BUCKETS':<16} | {'MAX BUCKET SIZE':<18} | {'TOP-10 COLLISION %'}")
    print("-" * 80)
    for rep, m in collision_metrics.items():
        print(f"{rep:<20} | {m['unique_buckets']:>14,} | {m['max_bucket_size']:>16,} | {m['top10_collision_pct']:>17.2f}%")
    print("=" * 80)

    # 6. Save Report
    full_report = {
        "ground_truth_coverage": formatted_summary,
        "collision_diagnostics": collision_metrics,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)

    print(f"\n[OK] Phase 2 Normalization Ablation Report saved to: {output_path}")
    return full_report


if __name__ == "__main__":
    evaluate_normalization_ablation()
