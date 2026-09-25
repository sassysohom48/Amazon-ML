"""
High-Throughput Test Set Inference & Submission Generation (Amazon ML Challenge 2026).
Runs country-partitioned blocking, extracts pairwise features, applies the calibrated
LightGBM classifier, and outputs validated TSV submissions.
"""

import os
import sys
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
    chunk_size: int = 100000,
    model_path: Path = MODELS_DIR / "lgbm_entity_resolver.txt",
    meta_path: Path = MODELS_DIR / "model_config.json",
):
    """
    Executes full test set inference and generates compliant submission TSVs.
    """
    print("=" * 70)
    print("PHASE 6: FULL TEST SET INFERENCE & SUBMISSION GENERATION")
    print("=" * 70)

    # 1. Load Model & Optimal Threshold
    print(f"Loading trained LightGBM model from {model_path}...")
    model = lgb.Booster(model_file=str(model_path))

    with open(meta_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    best_thresh = config["optimal_threshold"]
    print(f"Loaded model successfully. Applying optimal decision threshold θ* = {best_thresh:.2f}")

    # 2. Load Cleaned Test Datasets
    print("\nLoading cleaned test Parquet datasets...")
    t0 = time.time()
    s1_test = pl.read_parquet(PROCESSED_DIR / "test_source1_cleaned.parquet")
    s2_test = pl.read_parquet(PROCESSED_DIR / "test_source2_cleaned.parquet")
    s3_test = pl.read_parquet(PROCESSED_DIR / "test_source3_cleaned.parquet")
    print(f"Loaded test datasets in {time.time() - t0:.2f}s:")
    print(f"  • Source 1 Test: {len(s1_test):,} entities")
    print(f"  • Source 2 Test: {len(s2_test):,} entities")
    print(f"  • Source 3 Test: {len(s3_test):,} entities")

    # 3. Build Multi-Index Blocker over Test Target Pool (S2 + S3)
    blocker = MultiIndexBlocker(max_candidates=35, max_token_freq=35000)
    blocker.fit(s2_test, s3_test)

    # 4. Register Datasets in FeatureExtractor Lookup
    print("\nRegistering test entities in FeatureExtractor lookup cache...")
    extractor = FeatureExtractor()
    extractor.register_dataset(s1_test)
    extractor.register_dataset(s2_test)
    extractor.register_dataset(s3_test)

    # 5. Process Test S1 Records
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    matching_out = SUBMISSION_MATCHING_TSV
    candidate_out = SUBMISSION_CANDIDATE_TSV

    print(f"\nStreaming predictions to:")
    print(f"  • Matching Results:  {matching_out}")
    print(f"  • Candidate Pairs:   {candidate_out}")

    total_s1 = len(s1_test)
    s1_ids = s1_test["entity_id"].to_list()
    
    # Generate blocking candidates
    cand_map = blocker.block_s1(s1_test)

    print("\nScoring candidate pairs with LightGBM...")
    t_start = time.time()
    
    with open(matching_out, "w", encoding="utf-8") as f_match, \
         open(candidate_out, "w", encoding="utf-8") as f_cand:

        # Write TSV Headers
        f_match.write("source1_entity_id\tmatching_entity_ids\n")
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")

        # Process in batches for fast memory-efficient vector feature extraction
        batch_pairs = []
        batch_s1_indices = []
        
        # Pre-group pairs for scoring
        s1_to_candidates: Dict[str, List[str]] = {}
        for s1_id in s1_ids:
            cands = cand_map.get(s1_id, [])
            s1_to_candidates[s1_id] = cands
            for tgt_id in cands:
                batch_pairs.append((s1_id, tgt_id))

        print(f"Total candidate pairs to score: {len(batch_pairs):,}")

        # Batch feature extraction and prediction
        BATCH_SIZE = 100000
        pair_predictions: Dict[Tuple[str, str], float] = {}

        for i in range(0, len(batch_pairs), BATCH_SIZE):
            chunk = batch_pairs[i:i + BATCH_SIZE]
            X_chunk, _ = extractor.extract_features_for_pairs(chunk)
            probs = model.predict(X_chunk)
            for pair, prob in zip(chunk, probs):
                pair_predictions[pair] = float(prob)
            print(f"  Scored {min(i + BATCH_SIZE, len(batch_pairs)):,} / {len(batch_pairs):,} candidate pairs...")

        # Write output rows in strict S1 row order
        singletons_count = 0
        total_predicted_matches = 0

        for s1_id in s1_ids:
            cands = s1_to_candidates.get(s1_id, [])
            cand_str = ",".join(cands)
            f_cand.write(f"{s1_id}\t{cand_str}\n")

            # Filter matches by calibrated threshold
            matched_targets = [
                tgt_id for tgt_id in cands
                if pair_predictions.get((s1_id, tgt_id), 0.0) >= best_thresh
            ]

            if matched_targets:
                match_str = ",".join(matched_targets)
                f_match.write(f"{s1_id}\t{match_str}\n")
                total_predicted_matches += len(matched_targets)
            else:
                f_match.write(f"{s1_id}\t\n")
                singletons_count += 1

    elapsed = time.time() - t_start
    print(f"\nInference completed in {elapsed:.2f}s ({total_s1/elapsed:,.0f} entities/s).")
    print(f"  • Total S1 Entities:          {total_s1:,}")
    print(f"  • Total Matches Predicted:    {total_predicted_matches:,} (Avg {total_predicted_matches/total_s1:.2f}/entity)")
    print(f"  • Predicted Singletons:       {singletons_count:,} ({singletons_count/total_s1*100:.2f}%)")

    # 6. Validate Submission Files
    print("\n" + "=" * 70)
    print("RUNNING OFFICIAL VALIDATION HARNESS (validate_submission.py)")
    print("=" * 70)

    val_cmd = [
        sys.executable,
        str(VALIDATION_SCRIPT),
        "--matching", str(matching_out),
        "--candidate", str(candidate_out),
        "--test-dir", str(TEST_DIR),
        "--check-ids",
    ]

    result = subprocess.run(val_cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr)

    if result.returncode == 0:
        print("[SUCCESS] Submission passed all validation checks with 100% compliance!")
    else:
        print(f"[FAILED] Submission validation returned non-zero code: {result.returncode}")


if __name__ == "__main__":
    run_full_inference()
