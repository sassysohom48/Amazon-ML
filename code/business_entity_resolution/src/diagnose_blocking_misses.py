"""
Fast Diagnostic Script: Analyze Missed True Matches in Blocking.
"""

import sys
from pathlib import Path
from typing import Dict, Set, List
import polars as pl

# Ensure package import works
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PROCESSED_DIR


def diagnose_misses(sample_size: int = 25):
    print("=" * 70)
    print("DIAGNOSING MISSED TRUE MATCHES IN BLOCKING")
    print("=" * 70)

    val_s1_path = PROCESSED_DIR / "val_source1_cleaned.parquet"
    val_gt_path = PROCESSED_DIR / "val_ground_truth.parquet"
    cand_pairs_path = PROCESSED_DIR / "val_candidate_pairs.parquet"

    val_gt = pl.read_parquet(val_gt_path)
    cand_df = pl.read_parquet(cand_pairs_path)

    cand_map = {}
    for row in cand_df.iter_rows():
        s1_id, cands = row[0], row[1]
        cand_map[s1_id] = set(cands.split(",")) if cands else set()

    missed_pairs = []
    for row in val_gt.iter_rows():
        s1_id, matches = row[0], row[1]
        if not matches or not str(matches).strip():
            continue
        cands = cand_map.get(s1_id, set())
        for tgt_id in str(matches).strip().split(","):
            tgt_id = tgt_id.strip()
            if tgt_id and tgt_id not in cands:
                missed_pairs.append((s1_id, tgt_id))

    print(f"Total Missed True Matches: {len(missed_pairs):,} / 763,889 ({len(missed_pairs)/763889*100:.2f}%)")

    # Sample missed pairs
    sampled = missed_pairs[:sample_size]
    sample_s1_ids = [p[0] for p in sampled]
    sample_tgt_ids = [p[1] for p in sampled]

    val_s1 = pl.read_parquet(val_s1_path).filter(pl.col("entity_id").is_in(sample_s1_ids))
    s1_dict = {row["entity_id"]: row for row in val_s1.iter_rows(named=True)}

    train_s2 = pl.read_parquet(PROCESSED_DIR / "train_source2_cleaned.parquet").filter(pl.col("entity_id").is_in(sample_tgt_ids))
    train_s3 = pl.read_parquet(PROCESSED_DIR / "train_source3_cleaned.parquet").filter(pl.col("entity_id").is_in(sample_tgt_ids))

    target_dict = {}
    for row in train_s2.iter_rows(named=True):
        target_dict[row["entity_id"]] = row
    for row in train_s3.iter_rows(named=True):
        target_dict[row["entity_id"]] = row

    print(f"\nDetailed Inspection of {len(sampled)} Missed True Pairs:")
    print("=" * 70)

    for i, (s1_id, tgt_id) in enumerate(sampled, 1):
        s1_rec = s1_dict.get(s1_id, {})
        tgt_rec = target_dict.get(tgt_id, {})

        print(f"\n[{i}] S1 ID: {s1_id} | Country: {s1_rec.get('country')}")
        print(f"    S1 Name Clean:       '{s1_rec.get('name_clean')}'")
        print(f"    S1 Name Raw:         '{s1_rec.get('name')}'")
        print(f"    S1 Tokens:           {s1_rec.get('name_tokens')}")
        print(f"    S1 Address Clean:    '{s1_rec.get('address_clean')}'")
        print(f"    S1 Postal:           '{s1_rec.get('postal_digits')}'")
        print(f"    --- vs ---")
        print(f"    Target ID:           {tgt_id}")
        print(f"    Target Name Clean:   '{tgt_rec.get('name_clean')}'")
        print(f"    Target Name Raw:     '{tgt_rec.get('name')}'")
        print(f"    Target Tokens:       {tgt_rec.get('name_tokens')}")
        print(f"    Target Address Clean:'{tgt_rec.get('address_clean')}'")
        print(f"    Target Postal:       '{tgt_rec.get('postal_digits')}'")

        s1_toks = set(s1_rec.get('name_tokens', '').split())
        tgt_toks = set(tgt_rec.get('name_tokens', '').split())
        shared_name_toks = s1_toks & tgt_toks

        s1_addr_toks = set(s1_rec.get('address_clean', '').split())
        tgt_addr_toks = set(tgt_rec.get('address_clean', '').split())
        shared_addr_toks = s1_addr_toks & tgt_addr_toks

        print(f"    Shared Name Tokens:    {shared_name_toks}")
        print(f"    Shared Address Tokens: {list(shared_addr_toks)[:6]}")


if __name__ == "__main__":
    diagnose_misses(25)
