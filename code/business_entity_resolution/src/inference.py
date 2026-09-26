"""
High-Throughput Test Set Inference & Submission Generation (Amazon ML Challenge 2026).
Runs country-partitioned blocking, extracts pairwise features, applies the calibrated
LightGBM classifier, and outputs validated TSV submissions.
"""

import os
import sys
import gc
import time
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Set, Tuple
import polars as pl
import lightgbm as lgb

# Ensure package import works
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    PROCESSED_DIR,
    OUTPUT_DIR,
    MODELS_DIR,
    SUBMISSION_MATCHING_TSV,
    SUBMISSION_CANDIDATE_TSV,
    VALIDATION_SCRIPT,
    TEST_DIR,
)
from src.blocking import MultiIndexBlocker
from src.feature_engineering import FeatureExtractor


def run_full_inference(
    batch_size: int = 100000,
    model_path: Path = MODELS_DIR / "lgbm_entity_resolver.txt",
    meta_path: Path = MODELS_DIR / "model_config.json",
):
    """
    Executes full test set inference and generates compliant submission TSVs.
    Processes country-by-country (France, India, US) to minimize RAM and maximize throughput.
    """
    print("=" * 75)
    print("PHASE 6: FULL TEST SET INFERENCE & SUBMISSION GENERATION")
    print("=" * 75)

    # 1. Load Model & Optimal Threshold
    print(f"Loading trained LightGBM model from: {model_path}")
    model = lgb.Booster(model_file=str(model_path))

    with open(meta_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    best_thresh = config["optimal_threshold"]
    print(f"Loaded model successfully. Applying optimal decision threshold θ* = {best_thresh:.2f}")

    # 2. Load Test Source 1 in strict original order
    print("\nLoading test Source 1 dataset...")
    t0 = time.time()
    s1_test = pl.read_parquet(PROCESSED_DIR / "test_source1_cleaned.parquet")
    all_s1_ids = s1_test["entity_id"].to_list()
    total_s1 = len(all_s1_ids)
    print(f"Loaded {total_s1:,} Test Source 1 entities in {time.time() - t0:.2f}s")

    # Storage for all entity results (preserving exact row order)
    s1_to_candidates: Dict[str, List[str]] = {}
    s1_to_matches: Dict[str, List[str]] = {}

    countries = ["France", "India", "US"]
    total_candidate_pairs_scored = 0

    # 3. Country-by-Country Partitioned Processing
    for country in countries:
        print("\n" + "-" * 75)
        print(f"PROCESSING PARTITION: {country.upper()}")
        print("-" * 75)
        t_country = time.time()

        # Slice country S1
        s1_country = s1_test.filter(pl.col("country") == country)
        s1_c_ids = s1_country["entity_id"].to_list()
        print(f"  • S1 Entities in {country}: {len(s1_c_ids):,}")

        if len(s1_c_ids) == 0:
            continue

        # Load S2 and S3 for this country only
        t_load = time.time()
        s2_country = pl.read_parquet(
            PROCESSED_DIR / "test_source2_cleaned.parquet",
            columns=["entity_id", "country", "name_clean", "addr_clean", "name_tokens", "postal_digits", "has_address"],
        ).filter(pl.col("country") == country)

        s3_country = pl.read_parquet(
            PROCESSED_DIR / "test_source3_cleaned.parquet",
            columns=["entity_id", "country", "name_clean", "addr_clean", "name_tokens", "postal_digits", "has_address"],
        ).filter(pl.col("country") == country)

        print(f"  • Loaded Targets in {time.time() - t_load:.2f}s: S2={len(s2_country):,}, S3={len(s3_country):,}")

        # Build Inverted Index Blocker for this country
        print(f"  • Building Multi-Index Blocker for {country} targets...")
        blocker = MultiIndexBlocker(max_candidates=35, max_token_freq=4000)
        blocker.fit(s2_country, s3_country)

        # Generate candidates for S1
        print(f"  • Generating candidate pairs for {len(s1_c_ids):,} entities...")
        t_block = time.time()
        cands_map = blocker.block_s1(s1_country)
        print(f"  • Blocking completed in {time.time() - t_block:.2f}s ({len(s1_c_ids)/(time.time() - t_block):,.0f} ent/s)")

        # Register in Feature Extractor
        print(f"  • Initializing feature lookup cache...")
        extractor = FeatureExtractor()
        extractor.register_dataset(s1_country)
        extractor.register_dataset(s2_country)
        extractor.register_dataset(s3_country)

        # Flatten pairs for batched feature extraction
        pairs_to_score: List[Tuple[str, str]] = []
        for s1_id in s1_c_ids:
            cands = cands_map.get(s1_id, [])
            s1_to_candidates[s1_id] = cands
            for tgt_id in cands:
                pairs_to_score.append((s1_id, tgt_id))

        print(f"  • Total candidate pairs to score in {country}: {len(pairs_to_score):,}")
        total_candidate_pairs_scored += len(pairs_to_score)

        # Batch Feature Extraction & Scoring
        pair_probabilities: Dict[Tuple[str, str], float] = {}
        t_score = time.time()

        for idx in range(0, len(pairs_to_score), batch_size):
            chunk = pairs_to_score[idx : idx + batch_size]
            X_chunk, _ = extractor.extract_features_for_pairs(chunk)
            probs = model.predict(X_chunk)
            for pair, prob in zip(chunk, probs):
                pair_probabilities[pair] = float(prob)
            
            if (idx // batch_size) % 5 == 0 or idx + batch_size >= len(pairs_to_score):
                print(f"    Scored {min(idx + batch_size, len(pairs_to_score)):,} / {len(pairs_to_score):,} pairs...")

        print(f"  • Scoring completed in {time.time() - t_score:.2f}s ({len(pairs_to_score)/(time.time() - t_score):,.0f} pairs/s)")

        # Filter matches with optimal threshold θ*
        for s1_id in s1_c_ids:
            cands = s1_to_candidates.get(s1_id, [])
            matched = [
                tgt_id for tgt_id in cands
                if pair_probabilities.get((s1_id, tgt_id), 0.0) >= best_thresh
            ]
            s1_to_matches[s1_id] = matched

        # Explicit RAM cleanup
        del s2_country, s3_country, blocker, extractor, pairs_to_score, pair_probabilities, cands_map
        gc.collect()
        print(f"  Partition {country} finished in {time.time() - t_country:.2f}s")

    # 4. Stream Results to Compliant TSV Files in Strict S1 Order
    print("\n" + "=" * 75)
    print("WRITING OFFICIAL SUBMISSION TSVs")
    print("=" * 75)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    matching_out = SUBMISSION_MATCHING_TSV
    candidate_out = SUBMISSION_CANDIDATE_TSV

    print(f"  • Matching Results:  {matching_out}")
    print(f"  • Candidate Pairs:   {candidate_out}")

    singletons_count = 0
    total_predicted_matches = 0

    with open(matching_out, "w", encoding="utf-8") as f_match, \
         open(candidate_out, "w", encoding="utf-8") as f_cand:

        # Official TSV Headers expected by validate_submission.py
        f_match.write("source1_entity_id\tmatched_entity_ids\n")
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")

        for s1_id in all_s1_ids:
            cands = s1_to_candidates.get(s1_id, [])
            matches = s1_to_matches.get(s1_id, [])

            # Candidate TSV
            f_cand.write(f"{s1_id}\t{','.join(cands)}\n")

            # Matching TSV
            if matches:
                f_match.write(f"{s1_id}\t{','.join(matches)}\n")
                total_predicted_matches += len(matches)
            else:
                f_match.write(f"{s1_id}\t\n")
                singletons_count += 1

    print(f"\nInference Summary:")
    print(f"  • Total Test S1 Entities:     {total_s1:,}")
    print(f"  • Total Candidate Pairs:      {total_candidate_pairs_scored:,} (Avg {total_candidate_pairs_scored/total_s1:.1f}/ent)")
    print(f"  • Total Matches Predicted:    {total_predicted_matches:,} (Avg {total_predicted_matches/total_s1:.2f}/ent)")
    print(f"  • Singletons Defended:        {singletons_count:,} ({singletons_count/total_s1*100:.2f}%)")

    # 5. Automated Submission Validation Harness
    print("\n" + "=" * 75)
    print("PHASE 7: RUNNING OFFICIAL SUBMISSION VALIDATOR")
    print("=" * 75)

    val_cmd = [
        sys.executable,
        str(VALIDATION_SCRIPT),
        "--matching", str(matching_out),
        "--candidate", str(candidate_out),
        "--test-dir", str(TEST_DIR),
        "--check-ids",
    ]

    print(f"Executing: {' '.join(val_cmd)}")
    result = subprocess.run(val_cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr)

    if result.returncode == 0:
        print("[SUCCESS] All validation rules passed! Ready for official leaderboard submission.")
    else:
        print(f"[ERROR] Submission validator returned code {result.returncode}")

    return result.returncode


if __name__ == "__main__":
    exit_code = run_full_inference()
    sys.exit(exit_code)
