"""
Full Test Inference & Submission Generation Pipeline (Amazon ML Challenge 2026).
Streams all test records across US, India, and France using:
1. Multi-Core Threaded Blocking Engine
2. Batched Streaming Feature Engineering (50,000 S1 records per batch)
3. LightGBM GBDT Prediction & Calibrated Thresholding (tau* = 0.78)
4. Stream-writing directly to output TSVs (capping RAM < 3 GB permanently)
5. Automatic validation via validate_submission.py
"""

import os
import sys
import gc
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
import numpy as np
import polars as pl
import lightgbm as lgb

from .config import (
    PROJECT_ROOT, DATASET_DIR, TEST_DIR, TEST_S1, TEST_S2, TEST_S3,
    PARQUET_DIR, MODELS_DIR, OUTPUT_DIR, OUTPUT_MATCHING, OUTPUT_CANDIDATES,
    MAX_CANDIDATES_PER_S1
)
from .normalizer import clean_text
from .data_processor import ingest_tsv_to_parquet
from .blocking import HighRecallCountryIndex
from .feature_builder import build_pairwise_features, FEATURE_COLS


def ensure_test_partitions_exist() -> List[str]:
    """Ensures test Parquet partitions exist for US, India, and France."""
    existing_s1 = list(PARQUET_DIR.glob("test_source1_country=*.parquet"))
    if not existing_s1:
        print("\nTest Parquet partitions not found. Ingesting test TSVs into Snappy Parquet...")
        if TEST_S1.exists():
            ingest_tsv_to_parquet(TEST_S1, "test_source1", source_prefix="S1-")
        if TEST_S2.exists():
            ingest_tsv_to_parquet(TEST_S2, "test_source2", source_prefix="S2-")
        if TEST_S3.exists():
            ingest_tsv_to_parquet(TEST_S3, "test_source3", source_prefix="S3-")
        existing_s1 = list(PARQUET_DIR.glob("test_source1_country=*.parquet"))

    countries = sorted([f.stem.split("country=")[-1] for f in existing_s1])
    return countries


def run_full_inference(
    threshold_override: Optional[float] = None,
    batch_size: int = 100000,
    num_workers: int = 4
) -> Tuple[Path, Path]:
    """
    Executes end-to-end batched streaming inference across all test countries (US, India, France).
    Maintains flat RAM consumption (< 6 GB) by streaming results in batches of 100,000 entities.
    """
    start_total = time.time()
    print("=" * 85)
    print("  AMAZON ML CHALLENGE 2026: PHASE 7 BATCHED STREAMING INFERENCE")
    print("=" * 85)

    # 1. Load Trained Model & Calibrated Optimal Threshold
    model_path = MODELS_DIR / "lgbm_model.txt"
    threshold_path = MODELS_DIR / "optimal_threshold.txt"

    if not model_path.exists():
        raise FileNotFoundError(f"Trained model not found at {model_path}. Run Step 4 first.")

    print(f"Loading trained LightGBM booster from: {model_path.name}")
    booster = lgb.Booster(model_file=str(model_path))

    if threshold_override is not None:
        tau = threshold_override
    elif threshold_path.exists():
        with open(threshold_path, "r", encoding="utf-8") as f:
            tau = float(f.read().strip())
    else:
        tau = 0.78  # Calibrated default

    print(f"Applied Calibrated Match Threshold: τ* = {tau:.4f}")

    # 2. Check and Ingest Test Partitions
    countries = ensure_test_partitions_exist()
    print(f"\nDiscovered Test Country Partitions: {countries}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Open output TSV files and write headers
    f_match = open(OUTPUT_MATCHING, "w", encoding="utf-8", newline="\n")
    f_cand = open(OUTPUT_CANDIDATES, "w", encoding="utf-8", newline="\n")

    f_match.write("source1_entity_id\tmatched_entity_ids\n")
    f_cand.write("source1_entity_id\tcandidate_entity_ids\n")

    total_inferred_entities = 0
    total_matches_found = 0

    try:
        # 3. Process each country partition independently
        for country in countries:
            print(f"\n" + "=" * 70)
            print(f"Processing Test Country Partition: [{country}]")
            print("=" * 70)

            p_s1 = PARQUET_DIR / f"test_source1_country={country}.parquet"
            p_s2 = PARQUET_DIR / f"test_source2_country={country}.parquet"
            p_s3 = PARQUET_DIR / f"test_source3_country={country}.parquet"

            if not p_s1.exists():
                print(f"Warning: {p_s1.name} missing, skipping.")
                continue

            df_s1 = pl.read_parquet(p_s1).select(["entity_id", "business_name", "business_address"])
            df_s2 = pl.read_parquet(p_s2).select(["entity_id", "business_name", "business_address"]) if p_s2.exists() else pl.DataFrame()
            df_s3 = pl.read_parquet(p_s3).select(["entity_id", "business_name", "business_address"]) if p_s3.exists() else pl.DataFrame()

            n_country_s1 = df_s1.height

            if df_s2.is_empty() and df_s3.is_empty():
                print(f"[{country}] No target records in S2/S3. Writing all as singletons.")
                for sid in df_s1["entity_id"].to_list():
                    f_match.write(f"{sid}\t\n")
                    f_cand.write(f"{sid}\t\n")
                total_inferred_entities += n_country_s1
                continue

            # Build in-memory multi-channel index for this country once
            indexer = HighRecallCountryIndex(country=country, max_candidates=MAX_CANDIDATES_PER_S1)
            indexer.fit_target_pool(df_s2, df_s3)

            # Build target lookup dict (O(1))
            target_lookup: Dict[str, Tuple[str, str]] = {}
            for row in df_s2.iter_rows():
                target_lookup[row[0]] = (clean_text(row[1]), clean_text(row[2]))
            for row in df_s3.iter_rows():
                target_lookup[row[0]] = (clean_text(row[1]), clean_text(row[2]))

            # Build S1 lookup dict (O(1))
            s1_lookup: Dict[str, Tuple[str, str]] = {
                row[0]: (clean_text(row[1]), clean_text(row[2])) for row in df_s1.iter_rows()
            }

            print(f"\nStreaming [{country}] inference in batches of {batch_size:,} entities (workers={num_workers})...")

            # 4. Stream S1 entities in memory-safe batches
            for b_start in range(0, n_country_s1, batch_size):
                b_end = min(b_start + batch_size, n_country_s1)
                b_s1_sub = df_s1.slice(b_start, b_end - b_start)

                t_b0 = time.time()
                # A. Generate candidates for batch
                b_cands = indexer.generate_candidates_for_s1(b_s1_sub, max_k=MAX_CANDIDATES_PER_S1, num_workers=num_workers)

                # B. Build pairwise feature inputs using O(1) dict lookups
                pair_s1_ids, pair_target_ids, pair_ranks = [], [], []
                pair_s1_names, pair_s1_addrs = [], []
                pair_target_names, pair_target_addrs = [], []

                b_matches_dict: Dict[str, List[str]] = {sid: [] for sid in b_cands}

                for sid, c_list in b_cands.items():
                    n1, a1 = s1_lookup.get(sid, ("", ""))

                    for rank_idx, tid in enumerate(c_list, 1):
                        n2, a2 = target_lookup.get(tid, ("", ""))
                        pair_s1_ids.append(sid)
                        pair_target_ids.append(tid)
                        pair_ranks.append(rank_idx)
                        pair_s1_names.append(n1)
                        pair_s1_addrs.append(a1)
                        pair_target_names.append(n2)
                        pair_target_addrs.append(a2)

                n_batch_pairs = len(pair_s1_ids)

                # C. Compute features & predict probabilities if pairs exist
                if n_batch_pairs > 0:
                    df_feat = build_pairwise_features(
                        s1_ids=pair_s1_ids,
                        s1_names=pair_s1_names,
                        s1_addrs=pair_s1_addrs,
                        target_ids=pair_target_ids,
                        target_names=pair_target_names,
                        target_addrs=pair_target_addrs,
                        ranks=pair_ranks,
                        gt_map=None
                    )
                    X_batch = df_feat.select(FEATURE_COLS).to_numpy()
                    probs = booster.predict(X_batch)

                    for sid, tid, prob in zip(pair_s1_ids, pair_target_ids, probs):
                        if prob >= tau:
                            b_matches_dict[sid].append(tid)

                # D. Stream-write batch rows immediately to disk
                for sid in b_s1_sub["entity_id"].to_list():
                    matched = b_matches_dict.get(sid, [])
                    cands = b_cands.get(sid, [])
                    f_match.write(f"{sid}\t{','.join(matched)}\n")
                    f_cand.write(f"{sid}\t{','.join(cands)}\n")
                    if matched:
                        total_matches_found += 1

                total_inferred_entities += (b_end - b_start)
                b_elapsed = time.time() - t_b0
                print(f"  Processed batch {b_end:,} / {n_country_s1:,} [{country}] in {b_elapsed:.2f}s "
                      f"({int((b_end - b_start)/max(b_elapsed, 0.001))} ent/s)")

                # E. Free batch memory
                del b_cands, pair_s1_ids, pair_target_ids, pair_s1_names, pair_s1_addrs, b_matches_dict
                gc.collect()

            # Clean country index
            del indexer, target_lookup, df_s1, df_s2, df_s3
            gc.collect()

    finally:
        f_match.close()
        f_cand.close()

    print(f"\nFinished streaming all test partitions:")
    print(f"  • Total S1 Entities Processed: {total_inferred_entities:,}")
    print(f"  • Entities with Matched Records: {total_matches_found:,} ({total_matches_found/max(total_inferred_entities,1)*100:.1f}%)")
    print(f"  • Singletons (Empty matches): {total_inferred_entities - total_matches_found:,} ({(total_inferred_entities - total_matches_found)/max(total_inferred_entities,1)*100:.1f}%)")

    print(f"✅ Generated {OUTPUT_MATCHING.name} ({OUTPUT_MATCHING.stat().st_size / (1024*1024):.2f} MB)")
    print(f"✅ Generated {OUTPUT_CANDIDATES.name} ({OUTPUT_CANDIDATES.stat().st_size / (1024*1024):.2f} MB)")

    # 5. Run Validation Script
    val_script = PROJECT_ROOT / "dataset" / "utils" / "validate_submission.py"
    if val_script.exists():
        print("\n" + "=" * 70)
        print("Running validate_submission.py against generated outputs...")
        print("=" * 70)
        import subprocess
        cmd = [
            sys.executable,
            str(val_script),
            "--matching", str(OUTPUT_MATCHING),
            "--candidate", str(OUTPUT_CANDIDATES),
            "--test-dir", str(TEST_DIR)
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        print(res.stdout)
        if res.stderr:
            print(res.stderr)
        if res.returncode == 0:
            print("🏆 SUBMISSION VALIDATION PASSED: 100% compliant with competition rules!")
        else:
            print("⚠️ Validation issues detected. Review output above.")

    total_time = time.time() - start_total
    print(f"\n✅ Phase 7 Full Inference completed in {total_time:.2f} seconds.")
    return OUTPUT_MATCHING, OUTPUT_CANDIDATES


if __name__ == "__main__":
    run_full_inference()
