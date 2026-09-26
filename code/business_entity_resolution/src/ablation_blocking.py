"""
Step 3.4: Candidate Blocker Diagnostic Benchmark & Oracle F0.5 Evaluator (Amazon ML Challenge 2026).
Measures:
  1. Pre-capping Union Recall vs Top-K Recall Curves (K = 5..100)
  2. Oracle F0.5 Downstream Ceiling per K
  3. 8-Channel Attribution & Unique Recovery Matrix
  4. Forensic Miss Attribution
"""

import os
import sys
import time
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple
import polars as pl

# Ensure package import works
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PROCESSED_DIR, DATASET_DIR, TRAIN_DIR
from src.blocking import MultiChannelBlocker


def evaluate_blocking_benchmark(
    sample_size: int = 50000,
    k_list: List[int] = None,
    output_json_path: Path = None,
) -> Dict[str, any]:
    """
    Evaluates the Multi-Channel Blocker against Ground Truth on validation entities.
    """
    if k_list is None:
        k_list = [5, 10, 15, 20, 25, 35, 50, 75, 100]

    if output_json_path is None:
        output_json_path = PROCESSED_DIR / "phase3_blocking_ablation.json"

    print("=" * 80)
    print("STEP 3.4: CANDIDATE BLOCKER BENCHMARK & ORACLE F0.5 EVALUATOR")
    print("=" * 80)

    # 1. Load Ground Truth
    gt_path = PROCESSED_DIR / "train_ground_truth.parquet"
    if not gt_path.exists():
        gt_path = TRAIN_DIR / "train_ground_truth.tsv"
    if not gt_path.exists():
        gt_path = PROCESSED_DIR / "val_ground_truth.parquet"

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
        elif c in ("matched_entity_id", "target_entity_id", "s2_id", "s3_id", "target_id"):
            tgt_col = c
    gt_df = gt_df.rename({s1_col: "s1_id", tgt_col: "tgt_id"})

    # Map s1_id -> set of true matched targets
    gt_map = defaultdict(set)
    for row in gt_df.iter_rows(named=True):
        gt_map[row["s1_id"]].add(row["tgt_id"])

    # 2. Load Processed S1, S2, S3 with exact required columns
    cols_to_load = [
        "entity_id", "country", "name_core", "name_tokens", "name_acronym",
        "name_phonetic", "addr_clean", "addr_tokens", "addr_digits", "addr_unit_num", "postal_clean"
    ]

    print("\nLoading Enriched Cleaned Parquets (minimal columns)...")
    s1_all = pl.read_parquet(PROCESSED_DIR / "train_source1_cleaned.parquet", columns=cols_to_load)
    
    # Sample S1 validation entities that have ground truth
    s1_with_gt = s1_all.filter(pl.col("entity_id").is_in(list(gt_map.keys())))
    if sample_size and sample_size < len(s1_with_gt):
        s1_eval = s1_with_gt.sample(n=sample_size, seed=42)
        print(f"Sampled {len(s1_eval):,} S1 validation entities with Ground Truth matches.")
    else:
        s1_eval = s1_with_gt
        print(f"Evaluating across all {len(s1_eval):,} S1 entities with Ground Truth.")

    del s1_all, s1_with_gt
    import gc
    gc.collect()

    s2_all = pl.read_parquet(PROCESSED_DIR / "train_source2_cleaned.parquet", columns=cols_to_load)
    s3_all = pl.read_parquet(PROCESSED_DIR / "train_source3_cleaned.parquet", columns=cols_to_load)

    # 3. Fit MultiChannelBlocker on Target Pool
    blocker = MultiChannelBlocker(max_candidates=max(k_list))
    blocker.fit(s2_all, s3_all)

    # 4. Generate Candidate Pools with Provenance
    print(f"\nQuerying candidates for evaluation sample (max K = {max(k_list)})...")
    candidate_dict = blocker.block_dataframe(s1_eval, max_k=max(k_list))

    # 5. Evaluate Recall Curves & Oracle F0.5 across K
    print("\nComputing Candidate Recall & Oracle F0.5 Curves...")
    
    country_eval_entities = defaultdict(list)
    for row in s1_eval.iter_rows(named=True):
        c = row.get("country", "Unknown") or "Unknown"
        country_eval_entities[c].append(row["entity_id"])
    country_eval_entities["ALL"] = s1_eval["entity_id"].to_list()

    k_results = {}
    channel_attribution = defaultdict(lambda: {"total_hits": 0, "unique_hits": 0})
    missed_pairs_log = []

    for c_name, eids in country_eval_entities.items():
        total_eval_pairs = sum(len(gt_map[eid]) for eid in eids)
        if total_eval_pairs == 0:
            continue

        k_results[c_name] = {
            "total_entities": len(eids),
            "total_true_pairs": total_eval_pairs,
            "by_k": {},
        }

        for k in k_list:
            captured_true_pairs = 0
            for eid in eids:
                true_targets = gt_map[eid]
                cand_targets = [c["target_id"] for c in candidate_dict.get(eid, [])[:k]]
                captured = len(true_targets & set(cand_targets))
                captured_true_pairs += captured

            recall = (captured_true_pairs / total_eval_pairs) * 100.0
            
            # Oracle F0.5 calculation:
            # An oracle classifier predicts 1 for true targets in candidates and 0 for non-matches.
            # Precision = 1.0 (since it never predicts false positives), Recall = recall
            # F_0.5 = (1 + 0.5^2) * (P * R) / (0.5^2 * P + R) = 1.25 * R / (0.25 + R)
            r_frac = recall / 100.0
            oracle_f05 = (1.25 * r_frac) / (0.25 + r_frac) if (0.25 + r_frac) > 0 else 0.0

            k_results[c_name]["by_k"][k] = {
                "captured_pairs": captured_true_pairs,
                "recall_pct": round(recall, 2),
                "oracle_f05": round(oracle_f05, 4),
            }

    # 6. Evaluate 8-Channel Attribution & Unique Recovery on ALL entities
    all_eval_eids = country_eval_entities["ALL"]
    for eid in all_eval_eids:
        true_targets = gt_map[eid]
        cands = candidate_dict.get(eid, [])
        for c in cands:
            tgt_id = c["target_id"]
            if tgt_id in true_targets:
                # Identify which channels found this true match
                channels_hit = []
                if c["c_name_core"]: channels_hit.append("c_name_core")
                if c["c_name_token"]: channels_hit.append("c_name_token")
                if c["c_name_contain"]: channels_hit.append("c_name_contain")
                if c["c_acronym"]: channels_hit.append("c_acronym")
                if c["c_addr_token"]: channels_hit.append("c_addr_token")
                if c["c_addr_numeric"]: channels_hit.append("c_addr_numeric")
                if c["c_postal"]: channels_hit.append("c_postal")
                if c["c_phonetic"]: channels_hit.append("c_phonetic")

                for ch in channels_hit:
                    channel_attribution[ch]["total_hits"] += 1
                if len(channels_hit) == 1:
                    channel_attribution[channels_hit[0]]["unique_hits"] += 1

    # 7. Collect Sample Misses for Forensic Inspection
    for eid in all_eval_eids[:5000]:
        true_targets = gt_map[eid]
        cand_targets = {c["target_id"] for c in candidate_dict.get(eid, [])}
        missed = true_targets - cand_targets
        if missed:
            for m_tgt in missed:
                missed_pairs_log.append({
                    "s1_id": eid,
                    "target_id": m_tgt,
                })

    # 8. Print Results
    print("\n" + "=" * 85)
    print("CANDIDATE RECALL & ORACLE F0.5 CEILING CURVE (Validation Sample):")
    print("=" * 85)
    print(f"{'K':<6} | {'US RECALL':<14} | {'INDIA RECALL':<14} | {'OVERALL RECALL':<16} | {'ORACLE F0.5'}")
    print("-" * 85)
    
    us_by_k = k_results.get("US", {}).get("by_k", {})
    in_by_k = k_results.get("India", {}).get("by_k", {})
    all_by_k = k_results.get("ALL", {}).get("by_k", {})

    for k in k_list:
        us_r = f"{us_by_k.get(k, {}).get('recall_pct', 0.0):.2f}%" if us_by_k else "N/A"
        in_r = f"{in_by_k.get(k, {}).get('recall_pct', 0.0):.2f}%" if in_by_k else "N/A"
        all_r = f"{all_by_k.get(k, {}).get('recall_pct', 0.0):.2f}%" if all_by_k else "N/A"
        oracle = f"{all_by_k.get(k, {}).get('oracle_f05', 0.0):.4f}" if all_by_k else "N/A"
        print(f"{k:<6} | {us_r:<14} | {in_r:<14} | {all_r:<16} | {oracle}")
    print("=" * 85)

    print("\n8-CHANNEL RETRIEVAL ATTRIBUTION & UNIQUE RECOVERY MATRIX:")
    print("-" * 85)
    print(f"{'CHANNEL':<20} | {'TOTAL HITS':<16} | {'UNIQUE RECOVERY HITS'}")
    print("-" * 85)
    for ch, data in channel_attribution.items():
        print(f"{ch:<20} | {data['total_hits']:>14,} | {data['unique_hits']:>20,}")
    print("=" * 85)

    # 9. Save Full Diagnostic Output
    summary_report = {
        "k_sweep_results": k_results,
        "channel_attribution": dict(channel_attribution),
        "total_missed_count": len(missed_pairs_log),
        "sample_misses": missed_pairs_log[:50],
    }
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=2)

    print(f"\n[OK] Full Phase 3 Blocking Benchmark Report saved to: {output_json_path}")
    return summary_report


if __name__ == "__main__":
    evaluate_blocking_benchmark()
