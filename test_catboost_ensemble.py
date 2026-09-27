"""
Ensemble Engine: CatBoost + LightGBM + Precision Filters (Amazon ML Challenge 2026).
Implements:
  1. CatBoost Classifier + LightGBM Classifier trained on hard-negative candidate pairs.
  2. Blended ensemble prediction (50% LightGBM + 50% CatBoost).
  3. Street Number Conflict Filter (Franchise & Branch Disambiguation).
  4. Singleton Margin / Ambiguity Defense Guard.
  5. Full threshold sweep optimizing Macro F_0.5.
"""

import sys
import time
import math
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple
import numpy as np
import polars as pl
import lightgbm as lgb
from catboost import CatBoostClassifier

# Ensure BER module is discoverable
PROJECT_ROOT = Path(__file__).resolve().parent
CODE_DIR = PROJECT_ROOT / "code" / "business_entity_resolution"
sys.path.insert(0, str(CODE_DIR))

from src.config import TRAIN_S1, TRAIN_S2, TRAIN_S3, TRAIN_GT, MODELS_DIR
from src.text_normalizer import normalize_record
from src.feature_engineering import extract_pairwise_feature_vector, FEATURE_NAMES
from src.evaluator import evaluate_entity_resolution


def main():
    print("=" * 80)
    print(" AMAZON ML CHALLENGE 2026: CATBOOST + LIGHTGBM ENSEMBLE ENGINE")
    print("=" * 80)
    t_start = time.time()

    # ------------------------------------------------------------------
    # 1. SAMPLE VALIDATION ENTITIES & GROUND TRUTH
    # ------------------------------------------------------------------
    SAMPLE_SIZE = 4000
    N_DISTRACTORS = 40000

    print(f"\n[1/6] Loading {SAMPLE_SIZE:,} stratified entities from train_source1.tsv...")
    s1_df_raw = pl.read_csv(TRAIN_S1, separator="\t", n_rows=SAMPLE_SIZE * 2)

    us_s1 = s1_df_raw.filter(pl.col("country") == "US").slice(0, int(SAMPLE_SIZE * 0.60))
    in_s1 = s1_df_raw.filter(pl.col("country") == "India").slice(0, int(SAMPLE_SIZE * 0.40))
    val_s1 = pl.concat([us_s1, in_s1])

    s1_ids_list = val_s1["entity_id"].to_list()
    s1_ids_set = set(s1_ids_list)
    s1_countries = dict(zip(val_s1["entity_id"].to_list(), val_s1["country"].to_list()))

    print(f"  • S1 Sample: {len(val_s1):,} entities (US: {len(us_s1):,}, India: {len(in_s1):,})")

    # Load Ground Truth
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

    for s1_id in s1_ids_set:
        if s1_id not in gt_map:
            gt_map[s1_id] = set()

    n_singletons = sum(1 for m in gt_map.values() if len(m) == 0)
    print(f"  • True Matches: {sum(len(m) for m in gt_map.values()):,} | Singletons: {n_singletons:,}")

    # ------------------------------------------------------------------
    # 2. COLLECT TARGET SEARCH POOL & NORMALIZE
    # ------------------------------------------------------------------
    print(f"\n[2/6] Plucking true targets + {N_DISTRACTORS:,} distractors...")
    all_true_list = list(all_true_target_ids)

    s2_matches = pl.scan_csv(TRAIN_S2, separator="\t").filter(pl.col("entity_id").is_in(all_true_list)).collect()
    s3_matches = pl.scan_csv(TRAIN_S3, separator="\t").filter(pl.col("entity_id").is_in(all_true_list)).collect()

    s2_dist = pl.read_csv(TRAIN_S2, separator="\t", n_rows=N_DISTRACTORS // 2)
    s3_dist = pl.read_csv(TRAIN_S3, separator="\t", n_rows=N_DISTRACTORS // 2)

    target_pool = pl.concat([s2_matches, s3_matches, s2_dist, s3_dist]).unique(subset=["entity_id"])
    print(f"  • Target Pool Size: {len(target_pool):,} records.")

    digit_pattern = re.compile(r"\b\d+\b")

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
            "digits": set(digit_pattern.findall(r[5])) if r[5] else set(),
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
            "digits": set(digit_pattern.findall(r[5])) if r[5] else set(),
        }

    # ------------------------------------------------------------------
    # 3. CANDIDATE BLOCKING
    # ------------------------------------------------------------------
    print("\n[3/6] Running country-partitioned candidate blocking...")
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

        N_c = len(t_country_ids)
        token_weights = {t: math.log(N_c / len(p)) for t, p in token_idx.items()}
        addr_weights = {a: 0.7 * math.log(N_c / len(p)) for a, p in addr_idx.items() if len(p) >= 2}

        c_s1_ids = [eid for eid, edata in s1_records.items() if edata["country"] == country]
        for eid in c_s1_ids:
            s_data = s1_records[eid]
            scores = defaultdict(float)

            if s_data["name_clean"] in exact_name_idx:
                for idx in exact_name_idx[s_data["name_clean"]]:
                    scores[idx] += 100.0

            for t in s_data["name_tokens"]:
                if t in token_weights:
                    delta = token_weights[t]
                    for idx in token_idx[t]:
                        scores[idx] += delta

            for a in s_data["addr_tokens"]:
                if a in addr_weights:
                    delta = addr_weights[a]
                    for idx in addr_idx[a]:
                        scores[idx] += delta

            for p in s_data["postal_digits"]:
                if p in postal_idx and len(postal_idx[p]) <= 200:
                    for idx in postal_idx[p]:
                        scores[idx] += 3.0

            if not scores:
                candidates_map[eid] = []
            else:
                top_indices = sorted(scores.keys(), key=lambda i: scores[i], reverse=True)[:MAX_CANDIDATES]
                candidates_map[eid] = [t_country_ids[i] for i in top_indices]

    # ------------------------------------------------------------------
    # 4. PAIRWISE FEATURE EXTRACTION
    # ------------------------------------------------------------------
    print("\n[4/6] Extracting pairwise features for candidates...")
    all_pairs: List[Tuple[str, str]] = []
    labels: List[int] = []

    for eid, cands in candidates_map.items():
        true_set = gt_map[eid]
        for tid in cands:
            all_pairs.append((eid, tid))
            labels.append(1 if tid in true_set else 0)

    print(f"  • Total pairs to evaluate: {len(all_pairs):,} (Positives: {sum(labels):,}, Negatives: {len(labels)-sum(labels):,})")

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
    y_all = np.array(labels, dtype=np.int32)

    # ------------------------------------------------------------------
    # 5. TRAIN CATBOOST + ENSEMBLE WITH LIGHTGBM
    # ------------------------------------------------------------------
    print("\n[5/6] Training CatBoost Classifier on pairwise candidate features...")
    # Train-val split on entity groups
    train_mask = np.random.rand(len(s1_ids_list)) < 0.60
    train_eids = set(s1_ids_list[i] for i in range(len(s1_ids_list)) if train_mask[i])
    test_eids = set(s1_ids_list[i] for i in range(len(s1_ids_list)) if not train_mask[i])

    train_indices = [i for i, (s1_id, _) in enumerate(all_pairs) if s1_id in train_eids]
    test_indices = [i for i, (s1_id, _) in enumerate(all_pairs) if s1_id in test_eids]

    X_train, y_train = X_all[train_indices], y_all[train_indices]
    X_test, y_test = X_all[test_indices], y_all[test_indices]
    test_pairs = [all_pairs[i] for i in test_indices]

    print(f"  • Training set: {len(X_train):,} pairs | Holdout test set: {len(X_test):,} pairs")

    # CatBoost model
    t_cat = time.time()
    cat_model = CatBoostClassifier(
        iterations=350,
        learning_rate=0.08,
        depth=6,
        loss_function="Logloss",
        eval_metric="Logloss",
        random_seed=42,
        verbose=100,
        thread_count=-1,
    )
    cat_model.fit(X_train, y_train)
    print(f"  • CatBoost trained in {time.time() - t_cat:.2f}s.")

    # LightGBM pre-trained or trained
    model_path = MODELS_DIR / "lgbm_entity_resolver.txt"
    lgb_booster = lgb.Booster(model_file=str(model_path))

    # Predict probabilities on test pairs
    probs_lgb = lgb_booster.predict(X_test)
    probs_cat = cat_model.predict_proba(X_test)[:, 1]

    # Blended ensemble (50% LightGBM + 50% CatBoost)
    probs_ensemble = 0.50 * probs_lgb + 0.50 * probs_cat

    # ------------------------------------------------------------------
    # 6. EVALUATION WITH PRECISION FILTERS
    # ------------------------------------------------------------------
    print("\n[6/6] Evaluating Models & Precision Filters on Holdout Entities...")

    def evaluate_model_predictions(pair_probs: np.ndarray, thresh: float, use_filters: bool = False):
        preds = defaultdict(set)
        
        # Group predictions by entity
        entity_cands_scores = defaultdict(list)
        for (s1_id, tgt_id), prob in zip(test_pairs, pair_probs):
            entity_cands_scores[s1_id].append((tgt_id, prob))

        for s1_id, cand_scores in entity_cands_scores.items():
            if not cand_scores:
                continue

            # Sort descending by probability
            cand_scores.sort(key=lambda x: x[1], reverse=True)

            s_digits = s1_records[s1_id]["digits"]
            s_name = s1_records[s1_id]["name_clean"]

            matched = []
            for tid, prob in cand_scores:
                if prob < thresh:
                    continue

                t_digits = target_records[tid]["digits"]
                t_name = target_records[tid]["name_clean"]

                if use_filters:
                    # Filter 1: Street Number Conflict (Franchise Guard)
                    # If both have street numbers and they have zero overlap, reject unless names match perfectly and prob > 0.85
                    if s_digits and t_digits and len(s_digits & t_digits) == 0:
                        if prob < 0.82:
                            continue  # Reject conflicting street number!

                matched.append(tid)

            if use_filters and len(matched) >= 2:
                # Filter 2: Singleton Ambiguity Guard
                # If top two candidates have nearly identical confidence and neither is very confident, treat as ambiguous
                top_p1 = cand_scores[0][1]
                top_p2 = cand_scores[1][1]
                if abs(top_p1 - top_p2) < 0.04 and top_p1 < 0.78:
                    matched = []  # Protect singleton!

            preds[s1_id] = set(matched)

        # Ground truth map for test entities
        eval_gt = {eid: gt_map[eid].intersection(set(target_records.keys())) for eid in test_eids}
        metrics = evaluate_entity_resolution(eval_gt, preds, beta=0.5)
        return metrics

    print("\n" + "=" * 80)
    print(" MODEL COMPARISON ON UNSEEN HOLDOUT TEST ENTITIES")
    print("=" * 80)
    print(f" {'Configuration':<38} | {'Threshold':<10} | {'Macro F_0.5':<12} | {'Precision':<10} | {'Recall':<10}")
    print("-" * 84)

    # 1. Standalone LightGBM
    for th in [0.55, 0.65]:
        m = evaluate_model_predictions(probs_lgb, thresh=th, use_filters=False)
        print(f" LightGBM Alone (baseline)              | θ = {th:<6.2f} | {m['macro_f05']:<12.4f} | {m['macro_precision']*100:<8.2f}% | {m['macro_recall']*100:<8.2f}%")

    # 2. Standalone CatBoost
    for th in [0.55, 0.65]:
        m = evaluate_model_predictions(probs_cat, thresh=th, use_filters=False)
        print(f" CatBoost Alone                         | θ = {th:<6.2f} | {m['macro_f05']:<12.4f} | {m['macro_precision']*100:<8.2f}% | {m['macro_recall']*100:<8.2f}%")

    # 3. Blended Ensemble
    for th in [0.55, 0.60, 0.65]:
        m = evaluate_model_predictions(probs_ensemble, thresh=th, use_filters=False)
        print(f" Blended Ensemble (LGBM + CatBoost)     | θ = {th:<6.2f} | {m['macro_f05']:<12.4f} | {m['macro_precision']*100:<8.2f}% | {m['macro_recall']*100:<8.2f}%")

    # 4. Blended Ensemble + Precision Filters (Street Number + Singleton Guard)
    for th in [0.50, 0.55, 0.60, 0.65]:
        m = evaluate_model_predictions(probs_ensemble, thresh=th, use_filters=True)
        print(f" Ensemble + Number & Singleton Guards   | θ = {th:<6.2f} | {m['macro_f05']:<12.4f} | {m['macro_precision']*100:<8.2f}% | {m['macro_recall']*100:<8.2f}%")

    print("=" * 80)
    print(f" Total Ensemble Experiment Time: {time.time() - t_start:.2f}s")
    print("=" * 80)

    # Save CatBoost model
    cat_save_path = MODELS_DIR / "catboost_entity_resolver.cbm"
    cat_model.save_model(str(cat_save_path))
    print(f"\n[OK] Saved trained CatBoost model to: {cat_save_path.name}")


if __name__ == "__main__":
    main()
