"""
Full Test Inference & Submission Generation Pipeline (Amazon ML Challenge 2026).
Streams all test records across US, India, and France using:
1. Multi-Core Fork-Based Multiprocessing Blocking Engine (>2,000 ent/s)
2. High-IDF Posting Caps (5-10x speedup with zero recall loss)
3. Country-Level Checkpointing & Auto-Resumption (preserves France, resumes from India)
4. Batched Streaming Feature Engineering (100,000 S1 records per batch)
5. LightGBM GBDT Prediction & Calibrated Thresholding (tau* = 0.78)
6. Merging & Automatic validation via validate_submission.py
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


def count_tsv_rows(path: Path) -> int:
    """Returns the number of non-empty data rows in a TSV file (excluding header)."""
    if not path.exists():
        return 0
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        next(f, None)  # Skip header
        for line in f:
            if line.strip():
                count += 1
    return count


def run_full_inference(
    threshold_override: Optional[float] = None,
    batch_size: int = 100000,
    num_workers: int = 4
) -> Tuple[Path, Path]:
    """
    Executes end-to-end checkpointed streaming inference across all test countries (US, India, France).
    Automatically skips completed country partitions and resumes where it left off.
    """
    start_total = time.time()
    print("=" * 85)
    print("  AMAZON ML CHALLENGE 2026: PHASE 7 CHECKPOINTED STREAMING INFERENCE")
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

    # 3. Process each country partition with checkpointing
    for country in countries:
        print(f"\n" + "=" * 70)
        print(f"Test Country Partition: [{country}]")
        print("=" * 70)

        p_s1 = PARQUET_DIR / f"test_source1_country={country}.parquet"
        p_s2 = PARQUET_DIR / f"test_source2_country={country}.parquet"
        p_s3 = PARQUET_DIR / f"test_source3_country={country}.parquet"

        if not p_s1.exists():
            print(f"Warning: {p_s1.name} missing, skipping.")
            continue

        df_s1 = pl.read_parquet(p_s1).select(["entity_id", "business_name", "business_address"])
        n_country_s1 = df_s1.height

        country_match_path = OUTPUT_DIR / f"matching_results_{country}.tsv"
        country_cand_path = OUTPUT_DIR / f"candidate_pairs_{country}.tsv"

        # Checkpoint verification: if country already fully processed, skip!
        existing_rows = count_tsv_rows(country_match_path)
        if existing_rows >= n_country_s1:
            print(f"✅ Checkpoint Found: [{country}] is already fully completed ({existing_rows:,} rows).")
            print(f"   Skipping [{country}] and reusing existing results -> {country_match_path.name}")
            continue

        print(f"[{country}] Starting inference for {n_country_s1:,} S1 entities...")

        df_s2 = pl.read_parquet(p_s2).select(["entity_id", "business_name", "business_address"]) if p_s2.exists() else pl.DataFrame()
        df_s3 = pl.read_parquet(p_s3).select(["entity_id", "business_name", "business_address"]) if p_s3.exists() else pl.DataFrame()

        # Open country checkpoint files
        with open(country_match_path, "w", encoding="utf-8", newline="\n") as f_c_match, \
             open(country_cand_path, "w", encoding="utf-8", newline="\n") as f_c_cand:

            f_c_match.write("source1_entity_id\tmatched_entity_ids\n")
            f_c_cand.write("source1_entity_id\tcandidate_entity_ids\n")

            if df_s2.is_empty() and df_s3.is_empty():
                print(f"[{country}] No target records in S2/S3. Writing all as singletons.")
                for sid in df_s1["entity_id"].to_list():
                    f_c_match.write(f"{sid}\t\n")
                    f_c_cand.write(f"{sid}\t\n")
                continue

            # Build in-memory multi-channel index for this country
            indexer = HighRecallCountryIndex(country=country, max_candidates=MAX_CANDIDATES_PER_S1)
            indexer.fit_target_pool(df_s2, df_s3)

            # Build O(1) target lookup dict
            target_lookup: Dict[str, Tuple[str, str]] = {}
            for row in df_s2.iter_rows():
                target_lookup[row[0]] = (clean_text(row[1]), clean_text(row[2]))
            for row in df_s3.iter_rows():
                target_lookup[row[0]] = (clean_text(row[1]), clean_text(row[2]))

            # Build O(1) S1 lookup dict
            s1_lookup: Dict[str, Tuple[str, str]] = {
                row[0]: (clean_text(row[1]), clean_text(row[2])) for row in df_s1.iter_rows()
            }

            print(f"\nStreaming [{country}] in batches of {batch_size:,} entities (workers={num_workers})...")

            # Stream S1 entities in batches
            for b_start in range(0, n_country_s1, batch_size):
                b_end = min(b_start + batch_size, n_country_s1)
                b_s1_sub = df_s1.slice(b_start, b_end - b_start)

                t_b0 = time.time()
                # A. Generate candidates for batch (using multi-process fork & posting caps)
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

                # D. Stream-write batch rows immediately to country checkpoint
                for sid in b_s1_sub["entity_id"].to_list():
                    matched = b_matches_dict.get(sid, [])
                    cands = b_cands.get(sid, [])
                    f_c_match.write(f"{sid}\t{','.join(matched)}\n")
                    f_c_cand.write(f"{sid}\t{','.join(cands)}\n")

                b_elapsed = time.time() - t_b0
                rate = int((b_end - b_start) / max(b_elapsed, 0.001))
                print(f"  Processed batch {b_end:,} / {n_country_s1:,} [{country}] in {b_elapsed:.2f}s ({rate} ent/s)")

                # E. Free batch memory
                del b_cands, pair_s1_ids, pair_target_ids, pair_s1_names, pair_s1_addrs, b_matches_dict
                gc.collect()

            del indexer, target_lookup, df_s1, df_s2, df_s3
            gc.collect()

        print(f"✅ Completed [{country}] partition -> {country_match_path.name}")

    # 4. Merge All Country Checkpoints into Final Submission Files
    print("\n" + "=" * 70)
    print("Merging country checkpoints into final submission TSVs...")
    print("=" * 70)

    # Read complete ordered list of test S1 IDs
    if TEST_S1.exists():
        raw_test_s1_df = pl.read_csv(TEST_S1, separator="\t", has_header=True, quote_char=None)
        final_s1_order = raw_test_s1_df["entity_id"].to_list()
    else:
        final_s1_order = []
        for country in countries:
            p_s1 = PARQUET_DIR / f"test_source1_country={country}.parquet"
            if p_s1.exists():
                final_s1_order.extend(pl.read_parquet(p_s1)["entity_id"].to_list())

    # Build unified mapping from country checkpoints
    merged_matches: Dict[str, str] = {}
    merged_cands: Dict[str, str] = {}

    for country in countries:
        c_m_path = OUTPUT_DIR / f"matching_results_{country}.tsv"
        c_c_path = OUTPUT_DIR / f"candidate_pairs_{country}.tsv"

        if c_m_path.exists():
            with open(c_m_path, "r", encoding="utf-8") as f:
                next(f, None)
                for line in f:
                    parts = line.rstrip("\r\n").split("\t", 1)
                    if len(parts) == 2:
                        merged_matches[parts[0]] = parts[1]
                    elif len(parts) == 1:
                        merged_matches[parts[0]] = ""

        if c_c_path.exists():
            with open(c_c_path, "r", encoding="utf-8") as f:
                next(f, None)
                for line in f:
                    parts = line.rstrip("\r\n").split("\t", 1)
                    if len(parts) == 2:
                        merged_cands[parts[0]] = parts[1]
                    elif len(parts) == 1:
                        merged_cands[parts[0]] = ""

    # Write final unified matching_results.tsv
    with open(OUTPUT_MATCHING, "w", encoding="utf-8", newline="\n") as f_match:
        f_match.write("source1_entity_id\tmatched_entity_ids\n")
        for sid in final_s1_order:
            f_match.write(f"{sid}\t{merged_matches.get(sid, '')}\n")

    print(f"✅ Unified {OUTPUT_MATCHING.name} ({OUTPUT_MATCHING.stat().st_size / (1024*1024):.2f} MB)")

    # Write final unified candidate_pairs.tsv
    with open(OUTPUT_CANDIDATES, "w", encoding="utf-8", newline="\n") as f_cand:
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")
        for sid in final_s1_order:
            f_cand.write(f"{sid}\t{merged_cands.get(sid, '')}\n")

    print(f"✅ Unified {OUTPUT_CANDIDATES.name} ({OUTPUT_CANDIDATES.stat().st_size / (1024*1024):.2f} MB)")

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
