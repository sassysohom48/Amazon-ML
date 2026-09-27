"""
Full Scale CatBoost Training Engine (Amazon ML Challenge 2026).
Trains a 500-tree CatBoost classifier on 200,000+ candidate pairs across US and India
with hard negative mining, saving the final model to models/catboost_entity_resolver.cbm.
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
from catboost import CatBoostClassifier

# Ensure BER module is discoverable
PROJECT_ROOT = Path(__file__).resolve().parent
CODE_DIR = PROJECT_ROOT / "code" / "business_entity_resolution"
sys.path.insert(0, str(CODE_DIR))

from src.config import TRAIN_S1, TRAIN_S2, TRAIN_S3, TRAIN_GT, MODELS_DIR
from src.text_normalizer import normalize_record
from src.feature_engineering import extract_pairwise_feature_vector, FEATURE_NAMES


def train_full_catboost():
    print("=" * 80)
    print(" TRAINING HIGH-PRECISION CATBOOST MODEL FOR 90+ LEADERBOARD")
    print("=" * 80)
    t0 = time.time()

    SAMPLE_ENTITIES = 8000
    N_DISTRACTORS = 60000

    print(f"\n[1/4] Sampling {SAMPLE_ENTITIES:,} S1 entities across US & India...")
    s1_raw = pl.read_csv(TRAIN_S1, separator="\t", n_rows=SAMPLE_ENTITIES * 2)

    us_s1 = s1_raw.filter(pl.col("country") == "US").slice(0, int(SAMPLE_ENTITIES * 0.60))
    in_s1 = s1_raw.filter(pl.col("country") == "India").slice(0, int(SAMPLE_ENTITIES * 0.40))
    s1_df = pl.concat([us_s1, in_s1])

    s1_ids = set(s1_df["entity_id"].to_list())

    # Load Ground Truth
    gt_df = pl.read_csv(TRAIN_GT, separator="\t")
    gt_sample = gt_df.filter(pl.col("source1_entity_id").is_in(s1_ids))

    gt_map: Dict[str, Set[str]] = {}
    all_true_tgt_ids: Set[str] = set()

    for row in gt_sample.iter_rows():
        s1_id, matches = row[0], row[1]
        if matches and str(matches).strip():
            m_set = set(str(matches).strip().split(","))
            gt_map[s1_id] = m_set
            all_true_tgt_ids.update(m_set)
        else:
            gt_map[s1_id] = set()

    for s1_id in s1_ids:
        if s1_id not in gt_map:
            gt_map[s1_id] = set()

    total_true = sum(len(m) for m in gt_map.values())
    print(f"  • S1 Entities: {len(s1_df):,} | True Matches: {total_true:,}")

    # Pluck target records
    print(f"\n[2/4] Loading {len(all_true_tgt_ids):,} true targets + {N_DISTRACTORS:,} distractors...")
    all_true_list = list(all_true_tgt_ids)
    s2_m = pl.scan_csv(TRAIN_S2, separator="\t").filter(pl.col("entity_id").is_in(all_true_list)).collect()
    s3_m = pl.scan_csv(TRAIN_S3, separator="\t").filter(pl.col("entity_id").is_in(all_true_list)).collect()

    s2_d = pl.read_csv(TRAIN_S2, separator="\t", n_rows=N_DISTRACTORS // 2)
    s3_d = pl.read_csv(TRAIN_S3, separator="\t", n_rows=N_DISTRACTORS // 2)

    tgt_pool = pl.concat([s2_m, s3_m, s2_d, s3_d]).unique(subset=["entity_id"])
    print(f"  • Search Pool: {len(tgt_pool):,} records.")

    digit_pattern = re.compile(r"\b\d+\b")

    s1_rec = {}
    for r in s1_df.to_dicts():
        eid = r["entity_id"]
        n = normalize_record(eid, r["business_name"], r["business_address"], r["country"])
        s1_rec[eid] = {
            "country": n[1], "name_clean": n[4], "addr_clean": n[5],
            "name_tokens": set(n[6].split()) if n[6] else set(),
            "addr_tokens": set(n[5].split()) if n[5] else set(),
            "postal_digits": set(n[8].split()) if n[8] else set(),
            "digits": set(digit_pattern.findall(n[5])) if n[5] else set(),
        }

    tgt_rec = {}
    for r in tgt_pool.to_dicts():
        eid = r["entity_id"]
        n = normalize_record(eid, r["business_name"], r["business_address"], r["country"])
        tgt_rec[eid] = {
            "country": n[1], "name_clean": n[4], "addr_clean": n[5],
            "name_tokens": set(n[6].split()) if n[6] else set(),
            "addr_tokens": set(n[5].split()) if n[5] else set(),
            "postal_digits": set(n[8].split()) if n[8] else set(),
            "digits": set(digit_pattern.findall(n[5])) if n[5] else set(),
        }

    # Blocking
    print("\n[3/4] Generating hard negative candidate pairs via inverted index...")
    target_by_country = defaultdict(list)
    for tid, tdata in tgt_rec.items():
        target_by_country[tdata["country"]].append(tid)

    pairs: List[Tuple[str, str]] = []
    labels: List[int] = []

    for country in ["US", "India"]:
        t_c_ids = target_by_country.get(country, [])
        if not t_c_ids:
            continue

        exact_idx = defaultdict(list)
        tok_idx = defaultdict(list)
        addr_idx = defaultdict(list)
        post_idx = defaultdict(list)

        for idx, tid in enumerate(t_c_ids):
            td = tgt_rec[tid]
            if td["name_clean"]:
                exact_idx[td["name_clean"]].append(idx)
            for t in td["name_tokens"]:
                tok_idx[t].append(idx)
            for a in td["addr_tokens"]:
                if len(a) >= 3:
                    addr_idx[a].append(idx)
            for p in td["postal_digits"]:
                if len(p) >= 4:
                    post_idx[p].append(idx)

        N_c = len(t_c_ids)
        t_weights = {t: math.log(N_c / len(p)) for t, p in tok_idx.items()}
        a_weights = {a: 0.7 * math.log(N_c / len(p)) for a, p in addr_idx.items() if len(p) >= 2}

        c_s1_ids = [eid for eid, ed in s1_rec.items() if ed["country"] == country]
        for eid in c_s1_ids:
            sd = s1_rec[eid]
            scores = defaultdict(float)

            if sd["name_clean"] in exact_idx:
                for idx in exact_idx[sd["name_clean"]]:
                    scores[idx] += 100.0

            for t in sd["name_tokens"]:
                if t in t_weights:
                    scores_delta = t_weights[t]
                    for idx in tok_idx[t]:
                        scores[idx] += scores_delta

            for a in sd["addr_tokens"]:
                if a in a_weights:
                    scores_delta = a_weights[a]
                    for idx in addr_idx[a]:
                        scores[idx] += scores_delta

            for p in sd["postal_digits"]:
                if p in post_idx and len(post_idx[p]) <= 200:
                    for idx in post_idx[p]:
                        scores[idx] += 3.0

            if scores:
                top_indices = sorted(scores.keys(), key=lambda i: scores[i], reverse=True)[:35]
                true_set = gt_map[eid]
                for idx in top_indices:
                    tid = t_c_ids[idx]
                    pairs.append((eid, tid))
                    labels.append(1 if tid in true_set else 0)

    print(f"  • Extracted {len(pairs):,} candidate pairs (Positives: {sum(labels):,}, Negatives: {len(labels)-sum(labels):,})")

    # Feature matrix
    print("  • Computing 27 RapidFuzz pairwise features...")
    features = []
    for s1_id, tgt_id in pairs:
        s_d = s1_rec[s1_id]
        t_d = tgt_rec[tgt_id]
        fv = extract_pairwise_feature_vector(
            s_d["name_clean"], s_d["addr_clean"], s_d["name_tokens"], s_d["addr_tokens"], s_d["postal_digits"], s_d["digits"],
            t_d["name_clean"], t_d["addr_clean"], t_d["name_tokens"], t_d["addr_tokens"], t_d["postal_digits"], t_d["digits"],
        )
        features.append(fv)

    X = np.array(features, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)

    # Train CatBoost
    print(f"\n[4/4] Training CatBoost Classifier on {len(X):,} pairs (500 iterations)...")
    cat_model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.07,
        depth=6,
        loss_function="Logloss",
        eval_metric="Logloss",
        random_seed=42,
        verbose=100,
        thread_count=-1,
    )
    t_train = time.time()
    cat_model.fit(X, y)
    print(f"  • CatBoost trained in {time.time() - t_train:.2f}s!")

    # Save
    save_path = MODELS_DIR / "catboost_entity_resolver.cbm"
    cat_model.save_model(str(save_path))
    print(f"[SUCCESS] Saved final CatBoost model to: {save_path}")
    print(f"Total pipeline time: {time.time() - t0:.2f}s")


if __name__ == "__main__":
    train_full_catboost()
