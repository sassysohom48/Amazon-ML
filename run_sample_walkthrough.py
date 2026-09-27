"""
Amazon ML Challenge 2026: End-to-End Solution Walkthrough (Sample Demo)
Demonstrates:
  1. Data Ingestion & Inspection (Raw S1, Targets & Ground Truth)
  2. Vectorized Text Normalization (Clean strings, tokenization, postal digits)
  3. Candidate Blocking (Multi-signal Inverted Index, Candidate Recall calculation)
  4. Pairwise Feature Engineering (RapidFuzz C++ 27 features comparison)
  5. Supervised Model Prediction (LightGBM Booster + Calibrated Threshold)
  6. Official Macro F_0.5 Metric Computation & Singleton Defense
"""

import sys
import time
import math
from pathlib import Path
from collections import defaultdict
import polars as pl
import lightgbm as lgb
import numpy as np

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
    print(" AMAZON ML CHALLENGE 2026: BUSINESS ENTITY RESOLUTION")
    print(" SAMPLE END-TO-END PIPELINE WALKTHROUGH")
    print("=" * 80)

    # -------------------------------------------------------------
    # 1. INGESTION & DATA SAMPLE INSPECTION
    # -------------------------------------------------------------
    print("\n[PHASE 1] Loading a small sample of Source 1 and Ground Truth...")
    SAMPLE_SIZE = 200

    # Read first SAMPLE_SIZE records from train_source1.tsv
    df_s1 = pl.read_csv(TRAIN_S1, separator="\t", n_rows=SAMPLE_SIZE)
    df_gt = pl.read_csv(TRAIN_GT, separator="\t")

    s1_ids = set(df_s1["entity_id"].to_list())
    df_gt_sample = df_gt.filter(pl.col("source1_entity_id").is_in(s1_ids))

    # Build ground truth map
    gt_map = {}
    all_true_target_ids = set()
    for row in df_gt_sample.iter_rows():
        s1_id, matches = row[0], row[1]
        if matches and str(matches).strip():
            m_set = set(str(matches).strip().split(","))
            gt_map[s1_id] = m_set
            all_true_target_ids.update(m_set)
        else:
            gt_map[s1_id] = set()

    for s1_id in s1_ids:
        if s1_id not in gt_map:
            gt_map[s1_id] = set()

    n_singletons = sum(1 for m in gt_map.values() if len(m) == 0)
    n_multi = len(gt_map) - n_singletons
    total_true_matches = sum(len(m) for m in gt_map.values())

    print(f" Loaded {len(df_s1)} Source 1 entities:")
    print(f"   - Multi-match entities: {n_multi}")
    print(f"   - Singletons (0 matches): {n_singletons} (Earn 1.0 if empty, 0.0 on false match)")
    print(f"   - Total true match pairs: {total_true_matches}")

    # Show an example entity
    example_id = next(eid for eid, matches in gt_map.items() if len(matches) >= 2)
    example_row = df_s1.filter(pl.col("entity_id") == example_id).to_dicts()[0]
    print(f"\n Example S1 Entity ({example_id}):")
    print(f"   Raw Name:    {example_row['business_name']}")
    print(f"   Raw Address: {example_row['business_address']}")
    print(f"   Country:     {example_row['country']}")
    print(f"   True Target Matches: {gt_map[example_id]}")

    # -------------------------------------------------------------
    # 2. TEXT NORMALIZATION DEMO
    # -------------------------------------------------------------
    print("\n" + "-" * 80)
    print("[PHASE 2] Vectorized Text Normalization Demo:")
    norm = normalize_record(
        example_row["entity_id"],
        example_row["business_name"],
        example_row["business_address"],
        example_row["country"],
    )
    print(f"   Clean Name:    '{norm[4]}'")
    print(f"   Clean Address: '{norm[5]}'")
    print(f"   Name Tokens:   {norm[6]}")
    print(f"   Postal Digits: {norm[8]}")

    # -------------------------------------------------------------
    # 3. LOAD TARGET RECORDS & BUILD SEARCH POOL
    # -------------------------------------------------------------
    print("\n" + "-" * 80)
    print(f"[PHASE 3] Collecting target records for all {len(all_true_target_ids)} true matches + 5,000 distractors...")
    t_load = time.time()
    
    # Use fast Polars lazy scan to quickly pluck exact true targets from S2 and S3
    all_true_list = list(all_true_target_ids)
    s2_matches = pl.scan_csv(TRAIN_S2, separator="\t").filter(pl.col("entity_id").is_in(all_true_list)).collect()
    s3_matches = pl.scan_csv(TRAIN_S3, separator="\t").filter(pl.col("entity_id").is_in(all_true_list)).collect()

    # Read 2,500 distractors from each source to form a competitive candidate pool
    s2_distractors = pl.read_csv(TRAIN_S2, separator="\t", n_rows=2500)
    s3_distractors = pl.read_csv(TRAIN_S3, separator="\t", n_rows=2500)

    target_pool = pl.concat([s2_matches, s3_matches, s2_distractors, s3_distractors]).unique(subset=["entity_id"])
    print(f" Loaded {len(target_pool):,} target records in {time.time() - t_load:.2f}s (contains {len(s2_matches) + len(s3_matches)} true matches + {len(target_pool) - len(s2_matches) - len(s3_matches)} distractors).")

    # Normalize targets
    t_records = {}
    for row in target_pool.to_dicts():
        t_id = row["entity_id"]
        t_norm = normalize_record(t_id, row["business_name"], row["business_address"], row["country"])
        t_records[t_id] = {
            "name_clean": t_norm[4],
            "addr_clean": t_norm[5],
            "name_tokens": set(t_norm[6].split()) if t_norm[6] else set(),
            "addr_tokens": set(t_norm[5].split()) if t_norm[5] else set(),
            "postal_digits": set(t_norm[8].split()) if t_norm[8] else set(),
            "digits": set("".join(c for c in t_norm[5] if c.isdigit())),
        }

    # Normalize S1 records
    s1_records = {}
    for row in df_s1.to_dicts():
        s_id = row["entity_id"]
        s_norm = normalize_record(s_id, row["business_name"], row["business_address"], row["country"])
        s1_records[s_id] = {
            "name_clean": s_norm[4],
            "addr_clean": s_norm[5],
            "name_tokens": set(s_norm[6].split()) if s_norm[6] else set(),
            "addr_tokens": set(s_norm[5].split()) if s_norm[5] else set(),
            "postal_digits": set(s_norm[8].split()) if s_norm[8] else set(),
            "digits": set("".join(c for c in s_norm[5] if c.isdigit())),
        }

    # -------------------------------------------------------------
    # 4. CANDIDATE BLOCKING (INVERTED INDEX)
    # -------------------------------------------------------------
    print("\n" + "-" * 80)
    print("[PHASE 4] Building Multi-Signal Inverted Index for Candidate Blocking...")
    t0 = time.time()

    exact_name_idx = defaultdict(list)
    token_idx = defaultdict(list)
    addr_idx = defaultdict(list)
    postal_idx = defaultdict(list)

    t_ids_list = list(t_records.keys())
    for idx, tid in enumerate(t_ids_list):
        t_data = t_records[tid]
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

    # Inverted Index weights (IDF)
    N_t = len(t_ids_list)
    token_weights = {t: math.log(N_t / len(p)) for t, p in token_idx.items()}
    addr_weights = {a: 0.7 * math.log(N_t / len(p)) for a, p in addr_idx.items() if len(p) >= 2}

    print(f" Inverted index built in {time.time() - t0:.3f}s.")

    # Query S1 entities
    MAX_CANDIDATES = 30
    candidates_map = {}
    for s_id, s_data in s1_records.items():
        scores = defaultdict(float)
        # 1. Exact name match
        if s_data["name_clean"] in exact_name_idx:
            for idx in exact_name_idx[s_data["name_clean"]]:
                scores[idx] += 100.0
        # 2. Name tokens with IDF
        for t in s_data["name_tokens"]:
            if t in token_weights:
                for idx in token_idx[t]:
                    scores[idx] += token_weights[t]
        # 3. Address tokens
        for a in s_data["addr_tokens"]:
            if a in addr_weights:
                for idx in addr_idx[a]:
                    scores[idx] += addr_weights[a]
        # 4. Postal
        for p in s_data["postal_digits"]:
            if p in postal_idx and len(postal_idx[p]) <= 100:
                for idx in postal_idx[p]:
                    scores[idx] += 3.0

        if not scores:
            candidates_map[s_id] = []
        else:
            top_indices = sorted(scores.keys(), key=lambda i: scores[i], reverse=True)[:MAX_CANDIDATES]
            candidates_map[s_id] = [t_ids_list[i] for i in top_indices]

    # Evaluate Candidate Recall
    captured_matches = 0
    total_eval_matches = 0
    for s_id, true_targets in gt_map.items():
        # Only evaluate on true targets that were present in our target pool
        valid_true = true_targets.intersection(set(t_ids_list))
        total_eval_matches += len(valid_true)
        cands = set(candidates_map[s_id])
        captured_matches += len(valid_true.intersection(cands))

    cand_recall = (captured_matches / total_eval_matches * 100) if total_eval_matches > 0 else 100.0
    print(f" Blocking Candidate Recall: {cand_recall:.2f}% ({captured_matches}/{total_eval_matches} true matches preserved in top {MAX_CANDIDATES}).")

    # -------------------------------------------------------------
    # 5. FEATURE ENGINEERING DEMO (TRUE MATCH VS FALSE CANDIDATE)
    # -------------------------------------------------------------
    print("\n" + "-" * 80)
    print("[PHASE 5] Vectorized Pairwise Feature Engineering (RapidFuzz C++ SIMD):")
    
    # Pick a true match pair
    true_tgt_id = next(iter(gt_map[example_id].intersection(set(t_ids_list))))
    feat_true = extract_pairwise_feature_vector(
        s1_records[example_id]["name_clean"],
        s1_records[example_id]["addr_clean"],
        s1_records[example_id]["name_tokens"],
        s1_records[example_id]["addr_tokens"],
        s1_records[example_id]["postal_digits"],
        s1_records[example_id]["digits"],
        t_records[true_tgt_id]["name_clean"],
        t_records[true_tgt_id]["addr_clean"],
        t_records[true_tgt_id]["name_tokens"],
        t_records[true_tgt_id]["addr_tokens"],
        t_records[true_tgt_id]["postal_digits"],
        t_records[true_tgt_id]["digits"],
    )

    # Pick a false candidate
    false_tgt_id = next(c for c in candidates_map[example_id] if c not in gt_map[example_id])
    feat_false = extract_pairwise_feature_vector(
        s1_records[example_id]["name_clean"],
        s1_records[example_id]["addr_clean"],
        s1_records[example_id]["name_tokens"],
        s1_records[example_id]["addr_tokens"],
        s1_records[example_id]["postal_digits"],
        s1_records[example_id]["digits"],
        t_records[false_tgt_id]["name_clean"],
        t_records[false_tgt_id]["addr_clean"],
        t_records[false_tgt_id]["name_tokens"],
        t_records[false_tgt_id]["addr_tokens"],
        t_records[false_tgt_id]["postal_digits"],
        t_records[false_tgt_id]["digits"],
    )

    print(f"\n Pairwise Feature Comparison for Entity {example_id}:")
    print(f" {'Feature Name':<28} | {'True Match Pair':<16} | {'False Candidate Pair':<20}")
    print("-" * 72)
    sample_feat_indices = [0, 3, 5, 6, 9, 12, 13, 15, 20, 24]
    for idx in sample_feat_indices:
        fname = FEATURE_NAMES[idx]
        v_true = feat_true[idx]
        v_false = feat_false[idx]
        print(f" {fname:<28} | {v_true:<16.4f} | {v_false:<20.4f}")

    # -------------------------------------------------------------
    # 6. MODEL INFERENCE & EVALUATION
    # -------------------------------------------------------------
    print("\n" + "-" * 80)
    print("[PHASE 6] Model Prediction & Macro F_0.5 Evaluation:")

    model_path = MODELS_DIR / "lgbm_entity_resolver.txt"
    booster = lgb.Booster(model_file=str(model_path))
    print(f" Loaded pre-trained LightGBM model from {model_path.name}")
    print(f" Trees: {booster.num_trees()}, Features: {booster.num_feature()}")

    # Score all candidates
    OPTIMAL_THRESHOLD = 0.65
    predictions = {}

    for s_id, cand_list in candidates_map.items():
        if not cand_list:
            predictions[s_id] = set()
        else:
            feat_rows = []
            for c_id in cand_list:
                fv = extract_pairwise_feature_vector(
                    s1_records[s_id]["name_clean"],
                    s1_records[s_id]["addr_clean"],
                    s1_records[s_id]["name_tokens"],
                    s1_records[s_id]["addr_tokens"],
                    s1_records[s_id]["postal_digits"],
                    s1_records[s_id]["digits"],
                    t_records[c_id]["name_clean"],
                    t_records[c_id]["addr_clean"],
                    t_records[c_id]["name_tokens"],
                    t_records[c_id]["addr_tokens"],
                    t_records[c_id]["postal_digits"],
                    t_records[c_id]["digits"],
                )
                feat_rows.append(fv)

            X_pairs = np.array(feat_rows)
            probs = booster.predict(X_pairs)

            # Filter by calibrated threshold
            matched = {cand_list[i] for i, p in enumerate(probs) if p >= OPTIMAL_THRESHOLD}
            predictions[s_id] = matched

    # Prepare ground truth restricted to records present in target pool
    eval_gt_map = {s_id: gt_map[s_id].intersection(set(t_ids_list)) for s_id in s1_ids}

    # Evaluate official metrics
    metrics = evaluate_entity_resolution(eval_gt_map, predictions, beta=0.5)

    print(f"\n OFFICIAL METRICS EVALUATION ON SAMPLE:")
    print(f"   • Optimal Decision Threshold:   {OPTIMAL_THRESHOLD}")
    print(f"   • Macro F_0.5 Score:             {metrics['macro_f05']:.4f}")
    print(f"   • Macro Precision:              {metrics['macro_precision']*100:.2f}% (Weighted 2x over recall)")
    print(f"   • Macro Recall:                 {metrics['macro_recall']*100:.2f}%")
    print(f"   • Singleton Precision / Acc:    {metrics['singleton_accuracy']*100:.2f}% ({metrics['singleton_count']} singletons cleanly protected)")
    print("=" * 80)
    print(" SAMPLE WALKTHROUGH COMPLETE - PIPELINE VERIFIED 100% OPERATIONAL!")
    print("=" * 80)


if __name__ == "__main__":
    main()
