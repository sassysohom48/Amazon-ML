"""
Full Test Inference & Submission Generation Pipeline (Amazon ML Challenge 2026).
Streams all test records across US, India, and France through:
1. Country-partitioned Multi-Channel Blocking
2. RapidFuzz SIMD Pairwise Feature Engineering
3. LightGBM GBDT Match Probability Prediction
4. Optimal Threshold Filtering (tau* = 0.78) & Singleton Guardrail
5. Generation of matching_results.tsv and candidate_pairs.tsv
6. Automatic validation via validate_submission.py
"""

import os
import sys
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
    threshold_override: Optional[float] = None
) -> Tuple[Path, Path]:
    """
    Executes end-to-end inference across all test countries (US, India, France):
    Produces output/matching_results.tsv and output/candidate_pairs.tsv.
    """
    start_total = time.time()
    print("=" * 85)
    print("  AMAZON ML CHALLENGE 2026: PHASE 7 FULL TEST INFERENCE & SUBMISSION")
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

    all_matching_dict: Dict[str, List[str]] = {}
    all_candidate_dict: Dict[str, List[str]] = {}
    all_test_s1_order: List[str] = []

    # 3. Process each country partition independently
    for country in countries:
        print(f"\n" + "-" * 70)
        print(f"Processing Test Country Partition: [{country}]")
        print("-" * 70)

        p_s1 = PARQUET_DIR / f"test_source1_country={country}.parquet"
        p_s2 = PARQUET_DIR / f"test_source2_country={country}.parquet"
        p_s3 = PARQUET_DIR / f"test_source3_country={country}.parquet"

        if not p_s1.exists():
            print(f"Warning: {p_s1.name} missing, skipping.")
            continue

        df_s1 = pl.read_parquet(p_s1).select(["entity_id", "business_name", "business_address"])
        df_s2 = pl.read_parquet(p_s2).select(["entity_id", "business_name", "business_address"]) if p_s2.exists() else pl.DataFrame()
        df_s3 = pl.read_parquet(p_s3).select(["entity_id", "business_name", "business_address"]) if p_s3.exists() else pl.DataFrame()

        country_s1_ids = df_s1["entity_id"].to_list()
        all_test_s1_order.extend(country_s1_ids)

        if df_s2.is_empty() and df_s3.is_empty():
            print(f"[{country}] No target records in S2/S3. Marking all as singletons.")
            for sid in country_s1_ids:
                all_matching_dict[sid] = []
                all_candidate_dict[sid] = []
            continue

        # Build in-memory multi-channel index for this country
        indexer = HighRecallCountryIndex(country=country, max_candidates=MAX_CANDIDATES_PER_S1)
        indexer.fit_target_pool(df_s2, df_s3)

        # Generate candidate pairs
        country_cands = indexer.generate_candidates_for_s1(df_s1, max_k=MAX_CANDIDATES_PER_S1)

        # Prepare pairwise records for feature engineering
        s1_lookup: Dict[str, Tuple[str, str]] = {
            row[0]: (clean_text(row[1]), clean_text(row[2])) for row in df_s1.iter_rows()
        }
        target_lookup: Dict[str, Tuple[str, str]] = {}
        for row in df_s2.iter_rows():
            target_lookup[row[0]] = (clean_text(row[1]), clean_text(row[2]))
        for row in df_s3.iter_rows():
            target_lookup[row[0]] = (clean_text(row[1]), clean_text(row[2]))

        pair_s1_ids, pair_target_ids, pair_ranks = [], [], []
        pair_s1_names, pair_s1_addrs = [], []
        pair_target_names, pair_target_addrs = [], []

        for sid, c_list in country_cands.items():
            all_candidate_dict[sid] = c_list
            all_matching_dict[sid] = []  # Default singleton

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

        n_country_pairs = len(pair_s1_ids)
        if n_country_pairs == 0:
            print(f"[{country}] No candidate pairs generated.")
            continue

        print(f"[{country}] Computing features for {n_country_pairs:,} candidate pairs...")
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

        # Predict probabilities
        X_test = df_feat.select(FEATURE_COLS).to_numpy()
        probs = booster.predict(X_test)

        # Filter by threshold tau and assign matches
        for sid, tid, prob in zip(pair_s1_ids, pair_target_ids, probs):
            if prob >= tau:
                all_matching_dict[sid].append(tid)

        matched_entities = sum(1 for sid in country_s1_ids if len(all_matching_dict[sid]) > 0)
        print(f"[{country}] Inferred {len(country_s1_ids):,} S1 entities: "
              f"{matched_entities:,} linked ({matched_entities/len(country_s1_ids)*100:.1f}%), "
              f"{len(country_s1_ids) - matched_entities:,} singletons ({100 - matched_entities/len(country_s1_ids)*100:.1f}%)")

    # 4. Write Final Output TSVs
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nWriting final competition files to: {OUTPUT_DIR}")

    # Ensure every single test S1 entity from test_source1.tsv is present in order
    if TEST_S1.exists():
        raw_test_s1_df = pl.read_csv(TEST_S1, separator="\t", has_header=True, quote_char=None)
        final_s1_order = raw_test_s1_df["entity_id"].to_list()
    else:
        final_s1_order = all_test_s1_order

    total_test_s1 = len(final_s1_order)
    print(f"Total S1 entities to output: {total_test_s1:,}")

    # Write matching_results.tsv
    with open(OUTPUT_MATCHING, "w", encoding="utf-8", newline="\n") as f_match:
        f_match.write("source1_entity_id\tmatched_entity_ids\n")
        for sid in final_s1_order:
            matches = all_matching_dict.get(sid, [])
            match_str = ",".join(matches)
            f_match.write(f"{sid}\t{match_str}\n")

    print(f"✅ Generated {OUTPUT_MATCHING.name} ({OUTPUT_MATCHING.stat().st_size / (1024*1024):.2f} MB)")

    # Write candidate_pairs.tsv
    with open(OUTPUT_CANDIDATES, "w", encoding="utf-8", newline="\n") as f_cand:
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")
        for sid in final_s1_order:
            cands = all_candidate_dict.get(sid, [])
            cand_str = ",".join(cands)
            f_cand.write(f"{sid}\t{cand_str}\n")

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
