"""
Phase 6: High-Throughput Test Set Inference & Validated Submission Engine.
Executes the complete end-to-end inference pipeline:
  - High-Recall Multi-Channel Blocker (Phase 3)
  - 8-Family 73-Dimensional Pairwise Feature Extractor with Country IDF (Phase 4)
  - Calibrated Hybrid Model Ensemble (LightGBM + CatBoost) (Phase 5)
  - Multi-Tier Dynamic Thresholding (Country & Source Calibration + Margin Filter)
  - Full Submission TSV Validation & Packaging
"""

import os
import sys
import gc
import time
import json
import subprocess
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional, Any
import numpy as np
import polars as pl
import lightgbm as lgb

# Optional ML library
try:
    import catboost as cb
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

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
from src.country_idf import CountryIDFComputer
from src.blocking import MultiChannelBlocker
from src.feature_engineering import FeatureExtractor, FEATURE_NAMES


def run_full_inference(
    chunk_size: int = 50000,
    batch_size: int = 100000,
    max_k_candidates: int = 50,
    max_score_candidates: int = 35,
    model_dir: Path = MODELS_DIR,
):
    """
    Executes full test set inference and generates compliant submission TSVs.
    Processes country-by-country (France, India, US) in memory-safe 50k entity chunks (< 600 MB RAM)
    with instant per-country disk checkpointing and automatic resumption.
    """
    print("=" * 85)
    print("🚀 PHASE 6: FULL MULTI-SCALE TEST INFERENCE & SUBMISSION GENERATION")
    print("=" * 85)
    t_start = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load Calibrated Models & Decision Configuration
    config_path = model_dir / "phase5_calibration_config.json"
    if not config_path.exists():
        config_path = model_dir / "model_config.json"

    print(f"Loading Phase 5 calibration config from: {config_path.name}")
    with open(config_path, "r", encoding="utf-8") as f:
        config_raw = json.load(f)

    calib = config_raw.get("calibration", config_raw)
    lgb_weight = float(calib.get("lgbm_weight", 1.0))
    cb_weight = float(calib.get("catboost_weight", 0.0))
    global_th = float(calib.get("global_threshold", 0.55))
    country_ths = calib.get("country_thresholds", {"US": 0.55, "India": 0.55, "France": 0.55})
    country_source_ths = calib.get("country_source_thresholds", {})
    margin_gap = float(calib.get("margin_gap_threshold", 0.35))

    print(f"Loaded Calibration Strategy:")
    print(f"  • Model Blend: {lgb_weight*100:.0f}% LightGBM + {cb_weight*100:.0f}% CatBoost")
    print(f"  • Global Fallback Threshold θ*: {global_th:.2f}")
    print(f"  • Country Thresholds: {country_ths}")
    if country_source_ths:
        print(f"  • Country x Source Thresholds: {country_source_ths}")
    print(f"  • Margin Gap Filter: {margin_gap:.2f}")

    # Load LightGBM
    lgb_path = model_dir / "lgbm_entity_resolver.txt"
    print(f"\nLoading LightGBM model from {lgb_path.name}...")
    lgb_model = lgb.Booster(model_file=str(lgb_path))

    # Load CatBoost if present and weight > 0
    cb_model = None
    cb_path = model_dir / "catboost_entity_resolver.cbm"
    if HAS_CATBOOST and cb_weight > 0 and cb_path.exists():
        print(f"Loading CatBoost model from {cb_path.name}...")
        cb_model = cb.CatBoostClassifier()
        cb_model.load_model(str(cb_path))

    # 2. Load Test Source 1 in strict original order
    print("\nLoading Test Source 1 dataset...")
    t0 = time.time()
    cols_to_load = [
        "entity_id", "country", "name_clean", "name_core", "legal_form", "name_tokens", "name_acronym",
        "name_phonetic", "addr_clean", "addr_tokens", "addr_digits", "addr_unit_num", "postal_clean"
    ]
    s1_test = pl.read_parquet(PROCESSED_DIR / "test_source1_cleaned.parquet", columns=cols_to_load)
    all_s1_ids = s1_test["entity_id"].to_list()
    total_s1 = len(all_s1_ids)
    print(f"Loaded {total_s1:,} Test Source 1 entities in {time.time() - t0:.2f}s.")

    # Storage for output predictions across all countries
    s1_to_candidates: Dict[str, List[str]] = {}
    s1_to_matches: Dict[str, List[str]] = {}
    total_candidate_pairs_scored = 0

    countries = ["France", "India", "US"]

    # 3. Country-by-Country Partitioned Processing with Chunking & Checkpointing
    for country in countries:
        print("\n" + "=" * 80)
        print(f"🌍 PROCESSING PARTITION: {country.upper()}")
        print("=" * 80)
        t_country = time.time()

        checkpoint_file = OUTPUT_DIR / f"test_preds_{country}.parquet"

        # Check for completed checkpoint
        if checkpoint_file.exists():
            print(f"  [CHECKPOINT FOUND] Loading existing predictions for {country} from {checkpoint_file.name}...")
            cp_df = pl.read_parquet(checkpoint_file)
            for row in cp_df.iter_rows(named=True):
                eid = str(row["source1_entity_id"])
                c_str = str(row["candidate_ids"] or "")
                m_str = str(row["matched_ids"] or "")
                c_list = [x.strip() for x in c_str.split(",") if x.strip()] if c_str else []
                m_list = [x.strip() for x in m_str.split(",") if x.strip()] if m_str else []
                s1_to_candidates[eid] = c_list
                s1_to_matches[eid] = m_list
                total_candidate_pairs_scored += len(c_list)
            print(f"  [OK] Successfully loaded {len(cp_df):,} entities for {country} from checkpoint.")
            continue

        # Slice country S1
        s1_country = s1_test.filter(pl.col("country") == country)
        s1_c_ids = s1_country["entity_id"].to_list()
        n_s1_country = len(s1_c_ids)
        print(f"  • S1 Entities in {country}: {n_s1_country:,}")

        if n_s1_country == 0:
            continue

        # Load S2 and S3 for this country only
        t_load = time.time()
        s2_country = pl.read_parquet(
            PROCESSED_DIR / "test_source2_cleaned.parquet",
            columns=cols_to_load,
        ).filter(pl.col("country") == country)

        s3_country = pl.read_parquet(
            PROCESSED_DIR / "test_source3_cleaned.parquet",
            columns=cols_to_load,
        ).filter(pl.col("country") == country)

        print(f"  • Loaded Targets in {time.time() - t_load:.2f}s: S2={len(s2_country):,}, S3={len(s3_country):,}")

        # Fit Country IDF Computer
        print(f"  • Fitting Country IDF for {country}...")
        idf_comp = CountryIDFComputer()
        idf_comp.fit_from_dataframes(s2_country, s3_country)

        # Build Multi-Channel Blocker
        print(f"  • Building Multi-Channel Retrieval Blocker for {country} (K={max_k_candidates})...")
        t_block_fit = time.time()
        blocker = MultiChannelBlocker(max_candidates=max_k_candidates)
        blocker.fit(s2_country, s3_country)
        print(f"  • Blocker index built in {time.time() - t_block_fit:.2f}s.")

        extractor = FeatureExtractor(idf_computer=idf_comp)

        # Country threshold parameters
        c_th_default = country_ths.get(country, global_th)
        c_source_dict = country_source_ths.get(country, {})

        country_eids = []
        country_cand_strs = []
        country_match_strs = []

        # Process S1 entities in memory-safe chunks (e.g. 50,000 entities per chunk)
        for chunk_start in range(0, n_s1_country, chunk_size):
            chunk_end = min(chunk_start + chunk_size, n_s1_country)
            s1_chunk_df = s1_country.slice(chunk_start, chunk_end - chunk_start)
            chunk_s1_ids = s1_chunk_df["entity_id"].to_list()
            print(f"\n  [Chunk {chunk_start//chunk_size + 1}] Processing {len(chunk_s1_ids):,} entities ({chunk_start:,} to {chunk_end:,} / {n_s1_country:,})...")

            # 1. Block candidates for chunk
            t_chunk_block = time.time()
            cands_dict = blocker.block_dataframe(s1_chunk_df, max_k=max_k_candidates)
            print(f"    • Blocked {len(chunk_s1_ids):,} entities in {time.time() - t_chunk_block:.2f}s.")

            # 2. Collect flat pairs to score and target IDs for this chunk only
            needed_target_ids = set()
            flat_pairs: List[Tuple[str, str, Dict[str, Any]]] = []

            for s1_id in chunk_s1_ids:
                c_list = cands_dict.get(s1_id, [])
                tgt_list = [c["target_id"] for c in c_list]
                s1_to_candidates[s1_id] = tgt_list
                country_eids.append(s1_id)
                country_cand_strs.append(",".join(tgt_list))
                total_candidate_pairs_scored += len(tgt_list)

                # Score top N candidates with model
                for c in c_list[:max_score_candidates]:
                    tgt_id = c["target_id"]
                    prov_dict = {k: c[k] for k in c if k != "target_id"}
                    flat_pairs.append((s1_id, tgt_id, prov_dict))
                    needed_target_ids.add(tgt_id)

            # 3. Lazy target lookup cache for this chunk only (< 100 MB RAM)
            s2_chunk_targets = s2_country.filter(pl.col("entity_id").is_in(needed_target_ids))
            s3_chunk_targets = s3_country.filter(pl.col("entity_id").is_in(needed_target_ids))

            s1_records = {row["entity_id"]: row for row in s1_chunk_df.iter_rows(named=True)}
            s2_records = {row["entity_id"]: row for row in s2_chunk_targets.iter_rows(named=True)}
            s3_records = {row["entity_id"]: row for row in s3_chunk_targets.iter_rows(named=True)}
            tgt_records = {**s2_records, **s3_records}
            del s2_chunk_targets, s3_chunk_targets, s2_records, s3_records

            # 4. Batch feature extraction and prediction
            s1_scored_candidates: Dict[str, List[Tuple[str, float, str]]] = defaultdict(list)
            t_score = time.time()

            for idx in range(0, len(flat_pairs), batch_size):
                sub_chunk = flat_pairs[idx : idx + batch_size]
                feat_rows = []
                for s1_id, tgt_id, prov in sub_chunk:
                    s1_rec = s1_records.get(s1_id, {})
                    tgt_rec = tgt_records.get(tgt_id, {})
                    feat_rows.append(extractor.extract_pair_features(s1_rec, tgt_rec, prov))

                X_chunk = np.array(feat_rows, dtype=np.float32)
                p_lgb_chunk = lgb_model.predict(X_chunk)

                if cb_model is not None and cb_weight > 0:
                    p_cb_chunk = cb_model.predict_proba(X_chunk)[:, 1]
                    p_chunk = lgb_weight * p_lgb_chunk + cb_weight * p_cb_chunk
                else:
                    p_chunk = p_lgb_chunk

                for (s1_id, tgt_id, _), prob in zip(sub_chunk, p_chunk):
                    tgt_src = "S2" if tgt_id.startswith("S2") else "S3"
                    s1_scored_candidates[s1_id].append((tgt_id, float(prob), tgt_src))

            # 5. Apply thresholds & margin filter for chunk
            for s1_id in chunk_s1_ids:
                scored_list = s1_scored_candidates.get(s1_id, [])
                if not scored_list:
                    s1_to_matches[s1_id] = []
                    country_match_strs.append("")
                    continue

                scored_list.sort(key=lambda x: x[1], reverse=True)
                top_prob = scored_list[0][1]

                matched = []
                for rank_idx, (tgt_id, p, tgt_src) in enumerate(scored_list):
                    req_th = c_source_dict.get(tgt_src, c_th_default)
                    if p >= req_th:
                        if rank_idx > 0 and (top_prob - p) > margin_gap:
                            continue
                        matched.append(tgt_id)

                s1_to_matches[s1_id] = matched
                country_match_strs.append(",".join(matched))

            print(f"    • Scored {len(flat_pairs):,} candidate pairs in {time.time() - t_score:.2f}s ({len(flat_pairs)/(time.time() - t_score):,.0f} pairs/s).")

            # Cleanup chunk memory
            del s1_records, tgt_records, flat_pairs, s1_scored_candidates, cands_dict
            gc.collect()

        # 6. Save checkpoint parquet for this country immediately
        country_checkpoint_df = pl.DataFrame({
            "source1_entity_id": country_eids,
            "candidate_ids": country_cand_strs,
            "matched_ids": country_match_strs,
        })
        country_checkpoint_df.write_parquet(checkpoint_file)
        print(f"\n[OK] Saved country checkpoint to {checkpoint_file.name} ({len(country_checkpoint_df):,} entities).")

        # Free country level memory
        del s2_country, s3_country, blocker, country_eids, country_cand_strs, country_match_strs, country_checkpoint_df
        gc.collect()
        print(f"  Partition {country} finished in {time.time() - t_country:.2f}s.")

    # 4. Stream Results to Compliant TSV Files in Strict S1 Order
    print("\n" + "=" * 85)
    print("WRITING OFFICIAL SUBMISSION TSVs")
    print("=" * 85)

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
    print("\n" + "=" * 85)
    print("RUNNING OFFICIAL SUBMISSION VALIDATOR")
    print("=" * 85)

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

    total_time = time.time() - t_start
    print(f"\n🎉 FULL INFERENCE COMPLETED IN {total_time:.2f}s!")

    if result.returncode == 0:
        print("[SUCCESS] All validation rules passed! Ready for official leaderboard submission.")
    else:
        print(f"[ERROR] Submission validator returned code {result.returncode}")

    return result.returncode


if __name__ == "__main__":
    exit_code = run_full_inference()
    sys.exit(exit_code)
