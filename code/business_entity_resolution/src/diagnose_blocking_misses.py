"""
Step 3.5: Forensic Miss Diagnostic Script (Amazon ML Challenge 2026).
Loads generated validation candidate pairs and identifies the root cause of any missed Ground Truth links:
- Missing Token overlap?
- Truncation by channel cap?
- Acronym / Phonetic gap?
"""

import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, Set, List
import polars as pl
from rapidfuzz import fuzz

# Ensure package import works
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PROCESSED_DIR, TRAIN_DIR


def diagnose_misses(sample_size: int = 20):
    print("=" * 85)
    print("FORENSIC INSPECTION OF MISSED GROUND TRUTH TARGET PAIRS")
    print("=" * 85)

    cand_pairs_path = PROCESSED_DIR / "val_candidate_pairs.parquet"
    gt_path = PROCESSED_DIR / "train_ground_truth.parquet"
    if not gt_path.exists():
        gt_path = TRAIN_DIR / "train_ground_truth.tsv"

    print(f"Reading Candidates from {cand_pairs_path.name}...")
    cand_df = pl.read_parquet(cand_pairs_path, columns=["source1_entity_id", "target_entity_id"])

    cand_map = defaultdict(set)
    for row in cand_df.iter_rows():
        cand_map[row[0]].add(row[1])

    val_s1_eids = set(cand_map.keys())
    print(f"Total Validation Entities with Candidates: {len(val_s1_eids):,}")

    # Load Ground Truth
    print(f"Loading Ground Truth from {gt_path}...")
    if str(gt_path).endswith(".parquet"):
        gt_df = pl.read_parquet(gt_path)
    else:
        gt_df = pl.read_csv(gt_path, separator="\t")

    s1_col = gt_df.columns[0]
    tgt_col = gt_df.columns[1]
    for c in gt_df.columns:
        if c in ("source1_entity_id", "source_entity_id", "s1_id"):
            s1_col = c
        elif c in ("matched_entity_ids", "matched_entity_id", "target_entity_id", "s2_id", "s3_id", "target_id"):
            tgt_col = c

    # Filter GT to only validation entities
    missed_pairs = []
    total_val_gt_pairs = 0
    captured_val_gt_pairs = 0
    
    val_eids_with_gt = set()
    val_eids_with_captured_gt = set()

    for row in gt_df.iter_rows(named=True):
        s1 = str(row[s1_col]).strip()
        if s1 not in val_s1_eids:
            continue

        val_eids_with_gt.add(s1)
        cands = cand_map[s1]
        targets_raw = str(row[tgt_col]).strip()
        for tid in targets_raw.split(","):
            tid_clean = tid.strip()
            if not tid_clean:
                continue
            total_val_gt_pairs += 1
            if tid_clean in cands:
                captured_val_gt_pairs += 1
                val_eids_with_captured_gt.add(s1)
            else:
                missed_pairs.append((s1, tid_clean))

    pair_recall = (captured_val_gt_pairs / total_val_gt_pairs * 100.0) if total_val_gt_pairs > 0 else 0
    entity_recall = (len(val_eids_with_captured_gt) / len(val_eids_with_gt) * 100.0) if val_eids_with_gt else 0

    print("\n" + "=" * 85)
    print(f"VALIDATION CANDIDATE RECALL METRICS:")
    print(f"  • Validation Entities with GT:    {len(val_eids_with_gt):,}")
    print(f"  • Entities with >=1 Match Found:  {len(val_eids_with_captured_gt):,} ({entity_recall:.2f}% Entity Recall)")
    print(f"  • Total True Target Pairs in Val: {total_val_gt_pairs:,}")
    print(f"  • Captured True Target Pairs:     {captured_val_gt_pairs:,} ({pair_recall:.2f}% Pair Recall)")
    print(f"  • Missed True Target Pairs:       {len(missed_pairs):,} ({100.0 - pair_recall:.2f}%)")
    print("=" * 85)

    # Detailed Inspection of Sample Misses
    sample_misses = missed_pairs[:sample_size]
    sample_s1_ids = [p[0] for p in sample_misses]
    sample_tgt_ids = [p[1] for p in sample_misses]

    cols_to_load = ["entity_id", "country", "name_clean", "name_core", "name_tokens", "addr_clean", "postal_clean", "addr_tokens", "addr_digits"]
    
    s1_records = {row["entity_id"]: row for row in pl.read_parquet(PROCESSED_DIR / "train_source1_cleaned.parquet", columns=cols_to_load).filter(pl.col("entity_id").is_in(sample_s1_ids)).iter_rows(named=True)}
    
    s2_records = {row["entity_id"]: row for row in pl.read_parquet(PROCESSED_DIR / "train_source2_cleaned.parquet", columns=cols_to_load).filter(pl.col("entity_id").is_in(sample_tgt_ids)).iter_rows(named=True)}
    s3_records = {row["entity_id"]: row for row in pl.read_parquet(PROCESSED_DIR / "train_source3_cleaned.parquet", columns=cols_to_load).filter(pl.col("entity_id").is_in(sample_tgt_ids)).iter_rows(named=True)}
    tgt_records = {**s2_records, **s3_records}

    print(f"\nDETAILED FORENSIC AUDIT OF {len(sample_misses)} MISSED TARGET PAIRS:")
    print("=" * 85)

    for i, (s1_id, tgt_id) in enumerate(sample_misses, 1):
        s1 = s1_records.get(s1_id, {})
        tgt = tgt_records.get(tgt_id, {})

        s1_name = s1.get("name_clean", "")
        tgt_name = tgt.get("name_clean", "")
        s1_core = s1.get("name_core", "")
        tgt_core = tgt.get("name_core", "")
        s1_addr = s1.get("addr_clean", "")
        tgt_addr = tgt.get("addr_clean", "")
        s1_post = s1.get("postal_clean", "")
        tgt_post = tgt.get("postal_clean", "")

        name_fuzz = fuzz.token_set_ratio(s1_name, tgt_name) if s1_name and tgt_name else 0
        addr_fuzz = fuzz.token_set_ratio(s1_addr, tgt_addr) if s1_addr and tgt_addr else 0

        s1_toks = set(s1.get("name_tokens", "").split())
        tgt_toks = set(tgt.get("name_tokens", "").split())
        shared_name = s1_toks & tgt_toks

        s1_atoks = set(s1.get("addr_tokens", "").split())
        tgt_atoks = set(tgt.get("addr_tokens", "").split())
        shared_addr = s1_atoks & tgt_atoks

        print(f"\n[{i}] S1: {s1_id} vs Target: {tgt_id} [{s1.get('country')}]")
        print(f"    S1 Name:     '{s1_name}'  (Core: '{s1_core}')")
        print(f"    Tgt Name:    '{tgt_name}' (Core: '{tgt_core}')  [Fuzz Ratio: {name_fuzz}%]")
        print(f"    S1 Address:  '{s1_addr}'  (Postal: '{s1_post}')")
        print(f"    Tgt Address: '{tgt_addr}' (Postal: '{tgt_post}') [Fuzz Ratio: {addr_fuzz}%]")
        print(f"    Shared Tokens: Name={shared_name or 'None'} | Addr={list(shared_addr)[:4] or 'None'}")


if __name__ == "__main__":
    n = 20
    if len(sys.argv) > 1:
        n = int(sys.argv[1])
    diagnose_misses(n)
