"""
LightGBM Model Training, Hard Negative Mining & Macro F0.5 Calibration.
Trains a precision-optimized GBDT classifier and calibrates decision thresholds
specifically on the exact competition macro F0.5 metric with singleton defense.
"""

import os
import sys
import time
import json
import pickle
from pathlib import Path
from typing import Dict, List, Set, Tuple
import numpy as np
import polars as pl
import lightgbm as lgb

# Ensure package import works
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PROCESSED_DIR, MODELS_DIR
from src.feature_engineering import FeatureExtractor, FEATURE_NAMES
from src.evaluator import evaluate_entity_resolution


def build_training_dataset(
    s1_df: pl.DataFrame,
    s2_df: pl.DataFrame,
    s3_df: pl.DataFrame,
    gt_df: pl.DataFrame,
    cand_pairs_path: Path,
    max_samples: int = 400000,
    neg_to_pos_ratio: int = 4,
    random_seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, FeatureExtractor]:
    """
    Constructs balanced training feature matrix (X, y) containing all true positives
    and a balanced sample of hard negatives from candidate pairs.
    """
    print("\n" + "=" * 70)
    print("CONSTRUCTING TRAINING DATASET WITH HARD NEGATIVE MINING")
    print("=" * 70)

    # 1. Build Ground Truth Set
    print("Indexing Ground Truth matches...")
    gt_pairs: Set[Tuple[str, str]] = set()
    for row in gt_df.iter_rows():
        s1_id, matches = row[0], row[1]
        if matches and str(matches).strip():
            for tgt_id in str(matches).strip().split(","):
                if tgt_id.strip():
                    gt_pairs.add((s1_id, tgt_id.strip()))

    print(f"Total True Positive Pairs in GT: {len(gt_pairs):,}")

    # 2. Register datasets in FeatureExtractor
    print("Registering preprocessed datasets in FeatureExtractor lookup cache...")
    extractor = FeatureExtractor()
    extractor.register_dataset(s1_df)
    extractor.register_dataset(s2_df)
    extractor.register_dataset(s3_df)
    print(f"Total registered entities in cache: {len(extractor.entity_lookup):,}")

    # 3. Read Candidate Pairs
    print(f"Loading candidate pairs from {cand_pairs_path}...")
    cand_df = pl.read_parquet(cand_pairs_path)

    pos_pairs: List[Tuple[str, str]] = []
    neg_pairs: List[Tuple[str, str]] = []

    np.random.seed(random_seed)

    for row in cand_df.iter_rows():
        s1_id, cands_str = row[0], row[1]
        if not cands_str:
            continue
        cands = cands_str.split(",")
        for tgt_id in cands:
            pair = (s1_id, tgt_id)
            if pair in gt_pairs:
                pos_pairs.append(pair)
            else:
                neg_pairs.append(pair)

    print(f"Candidate pool composition: {len(pos_pairs):,} Positives, {len(neg_pairs):,} Negatives.")

    # 4. Balanced Negative Subsampling
    n_pos = min(len(pos_pairs), max_samples // (1 + neg_to_pos_ratio))
    n_neg = min(len(neg_pairs), n_pos * neg_to_pos_ratio)

    if len(pos_pairs) > n_pos:
        pos_indices = np.random.choice(len(pos_pairs), size=n_pos, replace=False)
        selected_pos = [pos_pairs[i] for i in pos_indices]
    else:
        selected_pos = pos_pairs

    if len(neg_pairs) > n_neg:
        neg_indices = np.random.choice(len(neg_pairs), size=n_neg, replace=False)
        selected_neg = [neg_pairs[i] for i in neg_indices]
    else:
        selected_neg = neg_pairs

    print(f"Selected training sample: {len(selected_pos):,} Positives (y=1), {len(selected_neg):,} Negatives (y=0).")

    all_pairs = selected_pos + selected_neg
    all_labels = [1] * len(selected_pos) + [0] * len(selected_neg)

    # Shuffle training samples
    shuffle_idx = np.random.permutation(len(all_pairs))
    shuffled_pairs = [all_pairs[i] for i in shuffle_idx]
    shuffled_labels = [all_labels[i] for i in shuffle_idx]

    # 5. Extract Feature Matrix
    X, y = extractor.extract_features_for_pairs(shuffled_pairs, shuffled_labels)
    return X, y, extractor


def train_and_calibrate_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    val_cand_df: pl.DataFrame,
    val_gt_map: Dict[str, Set[str]],
    extractor: FeatureExtractor,
) -> Tuple[lgb.Booster, float, Dict]:
    """
    Trains LightGBM classifier and calibrates decision threshold theta* on validation macro F0.5.
    """
    print("\n" + "=" * 70)
    print("TRAINING LIGHTGBM PAIRWISE RESOLUTION CLASSIFIER")
    print("=" * 70)

    # Prepare LightGBM Dataset
    train_data = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_NAMES)

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "max_depth": 8,
        "min_child_samples": 30,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "verbosity": 1,
        "n_jobs": -1,
        "seed": 42,
    }

    print("Training LightGBM model (300 boosting iterations)...")
    t0 = time.time()
    model = lgb.train(
        params,
        train_data,
        num_boost_round=300,
    )
    print(f"Model training completed in {time.time() - t0:.2f}s.")

    # Feature Importances
    importance = model.feature_importance(importance_type="gain")
    sorted_idx = np.argsort(importance)[::-1]
    print("\nTop 10 Most Predictive Features (Split Gain):")
    for rank, idx in enumerate(sorted_idx[:10], 1):
        print(f"  {rank:2d}. {FEATURE_NAMES[idx]:<28} Gain: {importance[idx]:,.1f}")

    # =========================================================================
    # MACRO F0.5 THRESHOLD CALIBRATION ON VALIDATION SET
    # =========================================================================
    print("\n" + "=" * 70)
    print("SWEEPING THRESHOLDS FOR MACRO F0.5 OPTIMIZATION")
    print("=" * 70)

    # Extract all validation pairs and predict probabilities
    val_pairs = []
    val_pair_map = defaultdict_list = {}
    
    val_s1_list = val_cand_df["source1_entity_id"].to_list()
    val_cands_list = val_cand_df["candidate_entity_ids"].to_list()

    flat_pairs = []
    pair_to_s1 = []
    pair_to_tgt = []

    for s1_id, cands_str in zip(val_s1_list, val_cands_list):
        if not cands_str:
            continue
        for tgt_id in cands_str.split(","):
            flat_pairs.append((s1_id, tgt_id))
            pair_to_s1.append(s1_id)
            pair_to_tgt.append(tgt_id)

    print(f"Extracting features and predicting for {len(flat_pairs):,} validation pairs...")
    X_val, _ = extractor.extract_features_for_pairs(flat_pairs)
    probs = model.predict(X_val)

    # Group predictions by s1_id: s1_id -> list of (tgt_id, prob)
    s1_to_scored_cands: Dict[str, List[Tuple[str, float]]] = {s1_id: [] for s1_id in val_s1_list}
    for s1_id, tgt_id, p in zip(pair_to_s1, pair_to_tgt, probs):
        s1_to_scored_cands[s1_id].append((tgt_id, float(p)))

    # Threshold Sweep
    thresholds = np.arange(0.30, 0.96, 0.05)
    best_f05 = -1.0
    best_thresh = 0.50
    best_metrics = {}

    print(f"\nEvaluating macro F0.5 across {len(thresholds)} threshold candidates...")
    for th in thresholds:
        preds: Dict[str, Set[str]] = {}
        for s1_id, scored_list in s1_to_scored_cands.items():
            matched = {tgt_id for tgt_id, p in scored_list if p >= th}
            preds[s1_id] = matched

        eval_res = evaluate_entity_resolution(val_gt_map, preds, beta=0.5)
        f05 = eval_res["macro_f05"]
        p = eval_res["macro_precision"]
        r = eval_res["macro_recall"]
        print(f"  Threshold θ = {th:.2f} -> Macro F0.5 = {f05:.4f} (Precision: {p:.4f}, Recall: {r:.4f})")

        if f05 > best_f05:
            best_f05 = f05
            best_thresh = float(th)
            best_metrics = eval_res

    print("\n" + "=" * 70)
    print(f"OPTIMAL VALIDATION PERFORMANCE: Threshold θ* = {best_thresh:.2f}")
    print(f"  • Best Macro F0.5:   {best_f05:.4f}")
    print(f"  • Macro Precision:   {best_metrics['macro_precision']:.4f}")
    print(f"  • Macro Recall:      {best_metrics['macro_recall']:.4f}")
    print("=" * 70)

    # Save model and metadata
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "lgbm_entity_resolver.txt"
    meta_path = MODELS_DIR / "model_config.json"

    model.save_model(str(model_path))
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "optimal_threshold": best_thresh,
            "best_macro_f05": best_f05,
            "best_precision": best_metrics["macro_precision"],
            "best_recall": best_metrics["macro_recall"],
            "feature_names": FEATURE_NAMES,
        }, f, indent=2)

    print(f"Model saved to {model_path}")
    print(f"Config saved to {meta_path}")

    return model, best_thresh, best_metrics
