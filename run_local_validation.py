"""
Local Validation Benchmark & Error Diagnostic Engine (Amazon ML Challenge 2026).
Evaluates your exact Macro F_0.5 score locally against real ground truth BEFORE submission.
Includes:
  1. Stratified 5,000-entity validation split across US and India.
  2. Realistic search pool: All true target matches + 50,000 distractor businesses.
  3. Candidate blocking recall ceiling measurement.
  4. Vectorized 27-feature pairwise scoring using LightGBM.
  5. Systematic threshold sweep (theta in [0.45, 0.85]) to find peak Macro F_0.5.
  6. Country-specific score breakdown (US vs India) and singleton defense audit.
"""

import sys
import time
import math
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple
import numpy as np
import polars as pl
import lightgbm as lgb

# Ensure BER module is discoverable
PROJECT_ROOT = Path(__file__).resolve().parent
CODE_DIR = PROJECT_ROOT / "code" / "business_entity_resolution"
sys.path.insert(0, str(CODE_DIR))

from src.config import TRAIN_S1, TRAIN_S2, TRAIN_S3, TRAIN_GT, MODELS_DIR
from src.text_normalizer import normalize_record
from src.feature_engineering import extract_pairwise_feature_vector, FEATURE_NAMES
from src.evaluator import evaluate_entity_resolution


def run_local_validation(sample_size: int = 5000, n_distractors: int = 50000):
    print("=" * 80)
    print(" AMAZON ML CHALLENGE 2026: LOCAL VALIDATION BENCHMARK (MACRO F_0.5)")
    print("=" * 80)
    t_start = time.time()

    # ------------------------------------------------------------------
    # 1. LOAD STRATIFIED VALIDATION SAMPLE (US + INDIA)
    # ------------------------------------------------------------------
    print(f"\n[1/5] Sampling {sample_size:,} stratified entities from train_source1.tsv...")
    s1_df_raw = pl.read_csv(TRAIN_S1, separator="\t", n_rows=sample_size * 2)

    # Stratified split: 60% US, 40% India
    us_s1 = s1_df_raw.filter(pl.col("country") == "US").slice(0, int(sample_size * 0.60))
    in_s1 = s1_df_raw.filter(pl.col("country") == "India").slice(0, int(sample_size * 0.40))
    val_s1 = pl.concat([us_s1, in_s1])

    s1_ids_list = val_s1["entity_id"].to_list()
    s1_ids_set = set(s1_ids_list)
    s1_countries = dict(zip(val_s1["entity_id"].to_list(), val_s1["country"].to_list()))

    print(f"  • Validation Set: {len(val_s1):,} S1 entities (US: {len(us_s1):,}, India: {len(in_s1):,})")

    # ------------------------------------------------------------------
    # 2. LOAD GROUND TRUTH FOR VALIDATION ENTITIES
    # ------------------------------------------------------------------
    print("\n[2/5] Loading ground truth labels...")
    gt_df = pl.read_csv(TRAIN_GT, separator="\t")
    val_gt_df = gt_df.filter(pl.col("source1_entity_id").is_in(s1_ids_set))

    gt_map: Dict[str, Set[str]] = {}
    all_true_target_ids: Set[str] = set()

    for row in val_gt_df.iter_rows():
        s1_id, matches = row[0], row[1]
        if matches and str(matches).strip():
            m_set = set(str(matches).strip().split(","))
            gt_map[s1_id] = m_set
            all_true_target_ids.update(m_set)
        else:
            gt_map[s1_id] = set()

    # Ensure singletons are recorded as empty sets
    for s1_id in s1_ids_set:
        if s1_id not in gt_map:
            gt_map[s1_id] = set()

    n_singletons = sum(1 for m in gt_map.values() if len(m) == 0)
    total_true_pairs = sum(len(m) for m in gt_map.values())
    print(f"  • True Matches in Ground Truth: {total_true_pairs:,} pairs across {len(gt_map):,} entities")
    print(f"  • Singletons in Ground Truth:   {n_singletons:,} ({n_singletons/len(gt_map)*100:.1f}%)")

    # ------------------------------------------------------------------
    # 3. BUILD SEARCH POOL (ALL TRUE TARGETS + DISTRACTOR BUSINESSES)
    # ------------------------------------------------------------------
    print(f"\n[3/5] Plucking {len(all_true_target_ids):,} true targets + {n_distractors:,} distractors...")
    t_pluck = time.time()
    all_true_list = list(all_true_target_ids)

    # Fast lazy scan to grab all true matches from S2 and S3
    s2_matches = pl.scan_csv(TRAIN_S2, separator="\t").filter(pl.col("entity_id").is_in(all_true_list)).collect()
    s3_matches = pl.scan_csv(TRAIN_S3, separator="\t").filter(pl.col("entity_id").is_in(all_true_list)).collect()

    # Grab distractors to create competition-scale search difficulty
    distractor_half = n_distractors // 2
    s2_dist = pl.read_csv(TRAIN_S2, separator="\t", n_rows=distractor_half)
    s3_dist = pl.read_csv(TRAIN_S3, separator="\t", n_rows=distractor_half)

    target_pool = pl.concat([s2_matches, s3_matches, s2_dist, s3_dist]).unique(subset=["entity_id"])
    print(f"  • Search Pool ready: {len(target_pool):,} records loaded in {time.time() - t_pluck:.2f}s.")

    # Normalize records
    print("  • Normalizing S1 and Target records...")
    t_norm = time.time()

    s1_records = {}
    for row in val_s1.to_dicts():
        eid = row["entity_id"]
        r = normalize_record(eid, row["business_name"], row["business_address"], row["country"])
        s1_records[eid] = {
            "country": r[1],
            "name_clean": r[4],
            "addr_clean": r[5],
            "name_tokens": set(r[6].split()) if r[6] else set(),
            "addr_tokens": set(r[5].split()) if r[5] else set(),
            "postal_digits": set(r[8].split()) if r[8] else set(),
            "digits": set("".join(c for c in r[5] if c.isdigit())),
        }

    target_records = {}
    for row in target_pool.to_dicts():
        eid = row["entity_id"]
        r = normalize_record(eid, row["business_name"], row["business_address"], row["country"])
        target_records[eid] = {
            "country": r[1],
            "name_clean": r[4],
            "addr_clean": r[5],
            "name_tokens": set(r[6].split()) if r[6] else set(),
            "addr_tokens": set(r[5].split()) if r[5] else set(),
            "postal_digits": set(r[8].split()) if r[8] else set(),
            "digits": set("".join(c for c in r[5] if c.isdigit())),
        }

    # ------------------------------------------------------------------
    # 4. COUNTRY-PARTITIONED CANDIDATE BLOCKING
    # ------------------------------------------------------------------
    print("\n[4/5] Executing candidate blocking (Country Partitioned Multi-Index)...")
    t_block = time.time()

    # Partition targets by country
    target_by_country: Dict[str, List[str]] = defaultdict(list)
    for tid, tdata in target_records.items():
        target_by_country[tdata["country"]].append(tid)

    MAX_CANDIDATES = 35
    candidates_map: Dict[str, List[str]] = {}

    for country in ["US", "India"]:
        t_country_ids = target_by_country.get(country, [])
        if not t_country_ids:
            continue

        exact_name_idx = defaultdict(list)
        token_idx = defaultdict(list)
        addr_idx = defaultdict(list)
        postal_idx = defaultdict(list)

        for idx, tid in enumerate(t_country_ids):
            t_data = target_records[tid]
            if t_data["name_clean"]:
                exact_name_idx[t_data["name_clean"]].append(idx)
            for t in t_data["name_tokens"]:
                token_idx[t].append(idx)
            for a in t_data["addr_tokens"]:
                if len(a) >= 3:
                    addr_idx[a].append(idx)
            for p in t_data["postal_digits"]:
                if len(p) >= 4:
                    postal_idx[p].append(idx)

        # IDF weights
        N_c = len(t_country_ids)
        token_weights = {t: math.log(N_c / len(p)) for t, p in token_idx.items()}
        addr_weights = {a: 0.7 * math.log(N_c / len(p)) for a, p in addr_idx.items() if len(p) >= 2}

        # Query country S1 entities
        c_s1_ids = [eid for eid, edata in s1_records.items() if edata["country"] == country]
        for eid in c_s1_ids:
            s_data = s1_records[eid]
            scores = defaultdict(float)

            # Exact Name
            if s_data["name_clean"] in exact_name_idx:
                for idx in exact_name_idx[s_data["name_clean"]]:
                    scores[idx] += 100.0

            # Tokens with IDF
            for t in s_data["name_tokens"]:
                if t in token_weights:
                    delta = token_weights[t]
                    for idx in token_idx[t]:
                        scores[idx] += delta

            # Address tokens
            for a in s_data["addr_tokens"]:
                if a in addr_weights:
                    delta = addr_weights[a]
                    for idx in addr_idx[a]:
                        scores[idx] += delta

            # Postal
            for p in s_data["postal_digits"]:
                if p in postal_idx and len(postal_idx[p]) <= 200:
                    for idx in postal_idx[p]:
                        scores[idx] += 3.0

            if not scores:
                candidates_map[eid] = []
            else:
                top_indices = sorted(scores.keys(), key=lambda i: scores[i], reverse=True)[:MAX_CANDIDATES]
                candidates_map[eid] = [t_country_ids[i] for i in top_indices]

    # Candidate recall audit
    captured = 0
    total_valid = 0
    for eid, true_set in gt_map.items():
        pool_matches = true_set.intersection(set(target_records.keys()))
        total_valid += len(pool_matches)
        cands = set(candidates_map.get(eid, []))
        captured += len(pool_matches.intersection(cands))

    block_recall = (captured / total_valid * 100) if total_valid > 0 else 100.0
    print(f"  • Candidate Blocking Completed in {time.time() - t_block:.2f}s")
    print(f"  • Candidate Recall: {block_recall:.2f}% ({captured:,} / {total_valid:,} true matches preserved in top {MAX_CANDIDATES})")

    # -------------------------------------------------------------
    # 5. FEATURE EXTRACTION & ENSEMBLE PREDICTION (LGBM + CATBOOST)
    # -------------------------------------------------------------
    print("\n[5/5] Extracting RapidFuzz features and scoring with CatBoost + LightGBM Ensemble...")
    model_path = MODELS_DIR / "lgbm_entity_resolver.txt"
    booster = lgb.Booster(model_file=str(model_path))

    cat_path = MODELS_DIR / "catboost_entity_resolver.cbm"
    from catboost import CatBoostClassifier
    cat_model = CatBoostClassifier()
    cat_model.load_model(str(cat_path))

    # Build pairs
    all_pairs: List[Tuple[str, str]] = []
    for eid, cands in candidates_map.items():
        for tid in cands:
            all_pairs.append((eid, tid))

    print(f"  • Total candidate pairs to score: {len(all_pairs):,}")

    # Extract features in vectorized batches
    t_feat = time.time()
    feature_rows = []
    for s1_id, tgt_id in all_pairs:
        s_data = s1_records[s1_id]
        t_data = target_records[tgt_id]
        fv = extract_pairwise_feature_vector(
            s_data["name_clean"], s_data["addr_clean"], s_data["name_tokens"], s_data["addr_tokens"], s_data["postal_digits"], s_data["digits"],
            t_data["name_clean"], t_data["addr_clean"], t_data["name_tokens"], t_data["addr_tokens"], t_data["postal_digits"], t_data["digits"],
        )
        feature_rows.append(fv)

    X_all = np.array(feature_rows, dtype=np.float32)
    t_score = time.time()
    probs_lgb = booster.predict(X_all)
    probs_cat = cat_model.predict_proba(X_all)[:, 1]

    # Weighted Ensemble (70% CatBoost + 30% LightGBM)
    probs_ensemble = 0.70 * probs_cat + 0.30 * probs_lgb

    print(f"  • Ensemble Scored {len(all_pairs):,} pairs in {time.time() - t_score:.2f}s ({len(all_pairs)/(time.time() - t_score):,.0f} pairs/s)")

    # Map probabilities
    pair_prob_map = dict(zip(all_pairs, probs_ensemble))

    # ------------------------------------------------------------------
    # 6. SYSTEMATIC THRESHOLD SWEEP (OPTIMIZING EXACT MACRO F_0.5)
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(" SYSTEMATIC THRESHOLD SWEEP (FINDING OPTIMAL THETA FOR MACRO F_0.5)")
    print("=" * 80)
    print(f" {'Threshold (theta)':<18} | {'Macro F_0.5':<12} | {'Precision':<12} | {'Recall':<12} | {'Singletons Acc':<16}")
    print("-" * 76)

    thresholds = [0.45, 0.50, 0.55, 0.60, 0.63, 0.65, 0.68, 0.70, 0.73, 0.75, 0.80]
    best_thresh = 0.65
    best_f05 = -1.0
    best_metrics = {}

    eval_gt_map = {eid: gt_map[eid].intersection(set(target_records.keys())) for eid in s1_ids_list}

    for th in thresholds:
        preds = {}
        for eid in s1_ids_list:
            cands = candidates_map.get(eid, [])
            matched = {tid for tid in cands if pair_prob_map.get((eid, tid), 0.0) >= th}
            preds[eid] = matched

        m = evaluate_entity_resolution(eval_gt_map, preds, beta=0.5)
        print(f" θ = {th:<14.2f} | {m['macro_f05']:<12.4f} | {m['macro_precision']*100:<10.2f}% | {m['macro_recall']*100:<10.2f}% | {m['singleton_accuracy']*100:<14.2f}%")

        if m["macro_f05"] > best_f05:
            best_f05 = m["macro_f05"]
            best_thresh = th
            best_metrics = m

    # ------------------------------------------------------------------
    # 7. COUNTRY BREAKDOWN & SUMMARY
    # ------------------------------------------------------------------
    # Country-specific metrics at optimal threshold
    best_preds = {eid: {tid for tid in candidates_map.get(eid, []) if pair_prob_map.get((eid, tid), 0.0) >= best_thresh} for eid in s1_ids_list}

    us_gt = {eid: eval_gt_map[eid] for eid in s1_ids_list if s1_countries[eid] == "US"}
    us_preds = {eid: best_preds[eid] for eid in us_gt}
    m_us = evaluate_entity_resolution(us_gt, us_preds, beta=0.5)

    in_gt = {eid: eval_gt_map[eid] for eid in s1_ids_list if s1_countries[eid] == "India"}
    in_preds = {eid: best_preds[eid] for eid in in_gt}
    m_in = evaluate_entity_resolution(in_gt, in_preds, beta=0.5)

    print("\n" + "=" * 80)
    print(" LOCAL VALIDATION SCORECARD & ERROR BREAKDOWN")
    print("=" * 80)
    print(f" • OVERALL MACRO F_0.5 SCORE:    {best_metrics['macro_f05']:.4f} (Optimal θ* = {best_thresh})")
    print(f" • Macro Precision:              {best_metrics['macro_precision']*100:.2f}% (Precision weighted 2x over recall)")
    print(f" • Macro Recall:                 {best_metrics['macro_recall']*100:.2f}%")
    print(f" • Singleton Accuracy:           {best_metrics['singleton_accuracy']*100:.2f}% ({best_metrics['singleton_count']} singletons audited)")
    print("-" * 80)
    print(f" • US Partition F_0.5 Score:     {m_us['macro_f05']:.4f}  (Precision: {m_us['macro_precision']*100:.1f}%, Recall: {m_us['macro_recall']*100:.1f}%)")
    print(f" • India Partition F_0.5 Score:  {m_in['macro_f05']:.4f}  (Precision: {m_in['macro_precision']*100:.1f}%, Recall: {m_in['macro_recall']*100:.1f}%)")
    print("=" * 80)
    print(f" Total Benchmark Time: {time.time() - t_start:.2f}s")
    print("=" * 80)

    return best_metrics, m_us, m_in


if __name__ == "__main__":
    run_local_validation(sample_size=3000, n_distractors=30000)
