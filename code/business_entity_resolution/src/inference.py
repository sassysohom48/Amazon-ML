"""
High-Throughput Test Set Inference & Submission Generation (Amazon ML Challenge 2026).
Ensembles CatBoost + LightGBM classifiers with RapidFuzz feature extraction,
country-partitioned candidate processing, and calibrated decision thresholding.
"""

import os
import sys
import gc
import time
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
import polars as pl
import lightgbm as lgb
from catboost import CatBoostClassifier

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
    PROJECT_ROOT,
)
from src.blocking import MultiIndexBlocker
from src.feature_engineering import FeatureExtractor
from src.package_submission import create_submission_zip


def run_full_inference(
    batch_size: int = 100000,
    decision_threshold: float = 0.62,
    catboost_weight: float = 0.70,
    lgbm_weight: float = 0.30,
    lgbm_path: Path = MODELS_DIR / "lgbm_entity_resolver.txt",
    catboost_path: Path = MODELS_DIR / "catboost_entity_resolver.cbm",
):
    """
    Executes full test set inference using the CatBoost + LightGBM ensemble.
    Processes country-by-country (France, India, US) to minimize RAM and maximize throughput.
    """
    print("=" * 75)
    print("PHASE 6: FULL TEST SET ENSEMBLE INFERENCE (CATBOOST + LIGHTGBM)")
    print("=" * 75)

    # 1. Load Trained Models
    print(f"Loading trained LightGBM model from: {lgbm_path}")
    lgb_model = lgb.Booster(model_file=str(lgbm_path))

    print(f"Loading trained CatBoost model from: {catboost_path}")
    cb_model = CatBoostClassifier()
    cb_model.load_model(str(catboost_path))

    print(f"[OK] Models loaded. Blending weights: CatBoost = {catboost_weight:.2f}, LightGBM = {lgbm_weight:.2f}")
    print(f"[OK] Calibrated decision threshold: θ* = {decision_threshold:.2f}")

    # 2. Load Test Source 1 in strict original order
    print("\nLoading test Source 1 dataset...")
    t0 = time.time()
    s1_test = pl.read_parquet(PROCESSED_DIR / "test_source1_cleaned.parquet")
    all_s1_ids = s1_test["entity_id"].to_list()
    s1_country_map = dict(zip(s1_test["entity_id"], s1_test["country"]))
    total_s1 = len(all_s1_ids)
    print(f"Loaded {total_s1:,} Test Source 1 entities in {time.time() - t0:.2f}s")

    # 3. Check for existing candidate pairs or prepare blocking
    s1_to_candidates: Dict[str, List[str]] = {}
    candidate_tsv = SUBMISSION_CANDIDATE_TSV
    reusing_candidates = False

    if candidate_tsv.exists() and candidate_tsv.stat().st_size > 10_000_000:
        print(f"\n[FAST PATH] Found pre-computed candidates: {candidate_tsv} ({candidate_tsv.stat().st_size / (1024**2):.1f} MB)")
        print("Loading candidate pairs from TSV to bypass re-indexing...")
        t_cand = time.time()
        with open(candidate_tsv, "r", encoding="utf-8") as f:
            next(f)  # skip header
            for line in f:
                s1, tab, rest = line.partition("\t")
                if not rest:
                    continue
                clist = [x.strip() for x in rest.rstrip("\r\n").split(",") if x.strip()]
                s1_to_candidates[s1] = clist
        print(f"Loaded {len(s1_to_candidates):,} candidate lists in {time.time() - t_cand:.2f}s!")
        reusing_candidates = True

    # Storage for all entity match results
    s1_to_matches: Dict[str, List[str]] = {}
    countries = ["France", "India", "US"]
    total_candidate_pairs_scored = 0

    # 4. Country-by-Country Partitioned Processing
    for country in countries:
        print("\n" + "-" * 75)
        print(f"PROCESSING PARTITION: {country.upper()}")
        print("-" * 75)
        t_country = time.time()

        s1_country = s1_test.filter(pl.col("country") == country)
        s1_c_ids = s1_country["entity_id"].to_list()
        print(f"  • S1 Entities in {country}: {len(s1_c_ids):,}")

        if len(s1_c_ids) == 0:
            continue

        # Load S2 and S3 targets for this country
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

        # Candidates resolution: reuse or block
        if not reusing_candidates:
            print(f"  • Building Multi-Index Blocker for {country} targets...")
            blocker = MultiIndexBlocker(max_candidates=35, max_token_freq=4000)
            blocker.fit(s2_country, s3_country)

            print(f"  • Generating candidate pairs for {len(s1_c_ids):,} entities...")
            t_block = time.time()
            cands_map = blocker.block_s1(s1_country)
            for eid, clist in cands_map.items():
                s1_to_candidates[eid] = clist
            print(f"  • Blocking completed in {time.time() - t_block:.2f}s")
            del blocker

        # Flatten pairs to score and gather unique needed target IDs
        pairs_to_score: List[Tuple[str, str]] = []
        needed_target_ids: Set[str] = set()
        for s1_id in s1_c_ids:
            cands = s1_to_candidates.get(s1_id, [])
            for tgt_id in cands:
                pairs_to_score.append((s1_id, tgt_id))
                needed_target_ids.add(tgt_id)

        print(f"  • Total candidate pairs to score in {country}: {len(pairs_to_score):,} ({len(needed_target_ids):,} unique targets)")
        total_candidate_pairs_scored += len(pairs_to_score)

        # Register only relevant target records in FeatureExtractor (low-RAM!)
        print(f"  • Initializing feature lookup cache for {len(s1_c_ids) + len(needed_target_ids):,} active entities...")
        extractor = FeatureExtractor()
        extractor.register_dataset(s1_country)
        extractor.register_dataset(s2_country, needed_eids=needed_target_ids)
        extractor.register_dataset(s3_country, needed_eids=needed_target_ids)

        # Batched feature extraction and ensemble scoring
        pair_probabilities: Dict[Tuple[str, str], float] = {}
        t_score = time.time()

        for idx in range(0, len(pairs_to_score), batch_size):
            chunk = pairs_to_score[idx : idx + batch_size]
            X_chunk, _ = extractor.extract_features_for_pairs(chunk)

            # Predict with both CatBoost and LightGBM
            p_cb = cb_model.predict_proba(X_chunk)[:, 1]
            p_lgb = lgb_model.predict(X_chunk)
            p_ens = catboost_weight * p_cb + lgbm_weight * p_lgb

            for pair, prob in zip(chunk, p_ens):
                pair_probabilities[pair] = float(prob)

            if (idx // batch_size) % 10 == 0 or idx + batch_size >= len(pairs_to_score):
                progress = min(idx + batch_size, len(pairs_to_score))
                speed = progress / max(time.time() - t_score, 0.01)
                print(f"    Scored {progress:,} / {len(pairs_to_score):,} pairs ({speed:,.0f} pairs/s)...")

        print(f"  • Scoring completed in {time.time() - t_score:.2f}s ({len(pairs_to_score)/(time.time() - t_score):,.0f} pairs/s)")

        # Filter and rank matches with optimal threshold θ* and top-10 cap
        for s1_id in s1_c_ids:
            cands = s1_to_candidates.get(s1_id, [])
            valid_candidates = []
            for tgt_id in cands:
                p = pair_probabilities.get((s1_id, tgt_id), 0.0)
                if p >= decision_threshold:
                    valid_candidates.append((tgt_id, p))

            # Rank by probability descending and cap at 10 matches
            if valid_candidates:
                valid_candidates.sort(key=lambda x: x[1], reverse=True)
                s1_to_matches[s1_id] = [tgt for tgt, _ in valid_candidates[:10]]
            else:
                s1_to_matches[s1_id] = []

        # Cleanup partition memory
        del s2_country, s3_country, extractor, pairs_to_score, pair_probabilities, needed_target_ids
        gc.collect()
        print(f"  Partition {country} finished in {time.time() - t_country:.2f}s")

    # 5. Stream Results to Compliant TSV Files in Strict S1 Order
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

    with open(matching_out, "w", encoding="utf-8") as f_match:
        f_match.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in all_s1_ids:
            matches = s1_to_matches.get(s1_id, [])
            if matches:
                f_match.write(f"{s1_id}\t{','.join(matches)}\n")
                total_predicted_matches += len(matches)
            else:
                f_match.write(f"{s1_id}\t\n")
                singletons_count += 1

    # Write candidates TSV if not already existing
    if not candidate_out.exists():
        with open(candidate_out, "w", encoding="utf-8") as f_cand:
            f_cand.write("source1_entity_id\tcandidate_entity_ids\n")
            for s1_id in all_s1_ids:
                cands = s1_to_candidates.get(s1_id, [])
                f_cand.write(f"{s1_id}\t{','.join(cands)}\n")

    print(f"\nInference Summary:")
    print(f"  • Total Test S1 Entities:     {total_s1:,}")
    print(f"  • Total Candidate Pairs:      {total_candidate_pairs_scored:,} (Avg {total_candidate_pairs_scored/total_s1:.1f}/ent)")
    print(f"  • Total Matches Predicted:    {total_predicted_matches:,} (Avg {total_predicted_matches/total_s1:.2f}/ent)")
    print(f"  • Singletons Defended:        {singletons_count:,} ({singletons_count/total_s1*100:.2f}%)")

    # 6. Automated Submission Validation Harness
    print("\n" + "=" * 75)
    print("PHASE 7: RUNNING OFFICIAL SUBMISSION VALIDATOR & PACKAGER")
    print("=" * 75)

    val_cmd = [
        sys.executable,
        str(VALIDATION_SCRIPT),
        "--matching", str(matching_out),
        "--candidate", str(candidate_out),
        "--test-dir", str(TEST_DIR),
    ]

    print(f"Executing: {' '.join(val_cmd)}")
    result = subprocess.run(val_cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr)

    if result.returncode == 0:
        print("[SUCCESS] All validation rules passed! Building team_submission.zip...")
        create_submission_zip()
        print("\n[COMPLETE] team_submission.zip ready for portal submission.")
    else:
        print(f"[ERROR] Submission validator returned code {result.returncode}")

    return result.returncode


if __name__ == "__main__":
    exit_code = run_full_inference()
    sys.exit(exit_code)
