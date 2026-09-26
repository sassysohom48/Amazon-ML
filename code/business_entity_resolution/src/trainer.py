"""
Supervised Model Training & F0.5 Threshold Optimization (Amazon ML Challenge 2026).
Trains an entity-grouped LightGBM GBDT classifier on candidate pairs and hard negatives,
computes feature importance, and sweeps decision thresholds to maximize Macro F0.5.
"""

import time
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
import numpy as np
import polars as pl
import lightgbm as lgb

from .config import (
    FEATURES_DIR, MODELS_DIR, PROCESSED_DIR,
    RANDOM_SEED, BETA
)
from .evaluator import evaluate_entity_resolution

# Exact feature column schema
FEATURE_COLS: List[str] = [
    "candidate_rank",
    "name_levenshtein",
    "name_jaro_winkler",
    "name_token_sort",
    "name_token_set",
    "name_partial_ratio",
    "name_3gram_jaccard",
    "name_len_diff",
    "name_len_ratio",
    "name_exact_match",
    "name_soundex_match",
    "has_s1_address",
    "has_target_address",
    "addr_both_present",
    "addr_token_jaccard",
    "addr_token_overlap",
    "addr_numeric_match",
    "addr_levenshtein",
    "addr_token_sort",
    "is_source2",
    "is_source3",
]


def train_and_optimize_model(
    features_parquet: Optional[Path] = None,
    model_output_path: Optional[Path] = None,
    train_val_split: float = 0.80
) -> Tuple[lgb.LGBMClassifier, float]:
    """
    Trains LightGBM GBDT with hard negatives and performs F0.5 threshold optimization:
    1. Loads the feature table.
    2. Splits data grouped strictly by source1_entity_id (zero entity leakage).
    3. Trains LightGBM with AUC early stopping.
    4. Computes feature importance rankings.
    5. Sweeps decision thresholds tau in [0.40, 0.90] to maximize competition Macro F0.5.
    6. Persists trained model artifact.
    """
    start_total = time.time()
    print("=" * 85)
    print("  AMAZON ML CHALLENGE 2026: PHASE 5 & 6 MODEL TRAINING & THRESHOLD OPTIMIZATION")
    print("=" * 85)

    if features_parquet is None:
        features_parquet = FEATURES_DIR / "val_features.parquet"
    if model_output_path is None:
        model_output_path = MODELS_DIR / "lgbm_model.txt"

    if not features_parquet.exists():
        raise FileNotFoundError(f"Feature table not found at {features_parquet}. Run Step 3 first.")

    print(f"Loading feature matrix from: {features_parquet.name}")
    df_all = pl.read_parquet(features_parquet)
    total_pairs = df_all.height
    print(f"Loaded {total_pairs:,} candidate pairs across {len(FEATURE_COLS)} features.")

    # 1. Entity-Grouped Train/Validation Split (80% Train, 20% Evaluation)
    unique_s1 = df_all["source1_entity_id"].unique().to_list()
    np.random.seed(RANDOM_SEED)
    np.random.shuffle(unique_s1)

    n_train_entities = int(len(unique_s1) * train_val_split)
    train_entities = set(unique_s1[:n_train_entities])
    val_entities = set(unique_s1[n_train_entities:])

    df_train = df_all.filter(pl.col("source1_entity_id").is_in(train_entities))
    df_val = df_all.filter(pl.col("source1_entity_id").is_in(val_entities))

    print(f"\nEntity-Grouped Data Partition:")
    print(f"  • Training:   {len(train_entities):,} S1 entities ({df_train.height:,} candidate pairs)")
    print(f"  • Validation: {len(val_entities):,} S1 entities ({df_val.height:,} candidate pairs)")

    # Prepare numpy arrays
    X_train = df_train.select(FEATURE_COLS).to_numpy()
    y_train = df_train["label"].to_numpy()

    X_val = df_val.select(FEATURE_COLS).to_numpy()
    y_val = df_val["label"].to_numpy()

    print(f"\nClass balance in training: {np.sum(y_train == 1):,} positives / {np.sum(y_train == 0):,} hard negatives")

    # 2. Train LightGBM GBDT Classifier
    print("\nTraining LightGBM Classifier with Histogram Binning & Early Stopping...")
    model = lgb.LGBMClassifier(
        objective="binary",
        metric="auc",
        boosting_type="gbdt",
        learning_rate=0.05,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=50,
        subsample=0.80,
        subsample_freq=1,
        colsample_bytree=0.80,
        n_estimators=1000,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=-1
    )

    t_train_start = time.time()
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=40, verbose=True),
            lgb.log_evaluation(period=50)
        ]
    )
    train_elapsed = time.time() - t_train_start
    print(f"Model training completed in {train_elapsed:.2f}s (Best iteration: {model.best_iteration_})")

    # 3. Feature Importance Analysis
    importances = model.booster_.feature_importance(importance_type="gain")
    importance_pairs = sorted(zip(FEATURE_COLS, importances), key=lambda x: x[1], reverse=True)

    print("\n" + "=" * 60)
    print("TOP 10 MOST INFORMATIVE FEATURES (Gain Importance):")
    print("=" * 60)
    for rank, (feat, gain) in enumerate(importance_pairs[:10], 1):
        print(f"  {rank:>2}. {feat:<25} : {gain:>12,.1f}")
    print("=" * 60)

    # 4. Phase 6: Macro F0.5 Threshold Sweep & Singleton Guardrail
    print("\nExecuting Macro F0.5 Decision Threshold Sweep...")
    val_preds_prob = model.predict_proba(X_val)[:, 1]

    # Load Ground Truth for validation entities
    from .dataset import load_ground_truth
    val_gt_path = PROCESSED_DIR / "val_ground_truth.parquet"
    full_gt = load_ground_truth(val_gt_path)
    val_gt_map = {sid: full_gt.get(sid, set()) for sid in val_entities}

    # Extract aligned ID arrays for validation pairs
    val_s1_ids = df_val["source1_entity_id"].to_list()
    val_target_ids = df_val["target_entity_id"].to_list()

    # Pre-organize pairs for fast threshold evaluations
    entity_pairs: Dict[str, List[Tuple[str, float]]] = {sid: [] for sid in val_entities}
    for sid, tid, prob in zip(val_s1_ids, val_target_ids, val_preds_prob):
        entity_pairs[sid].append((tid, prob))

    thresholds = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.72, 0.75, 0.78, 0.80, 0.85, 0.90]
    sweep_results = []
    best_tau = 0.70
    best_f05 = -1.0

    print("\n" + "=" * 90)
    print(f"{'Threshold (τ)':<16}| {'Macro F0.5':<14}| {'Precision':<14}| {'Recall':<14}| {'Singleton Acc'}")
    print("-" * 90)

    for tau in thresholds:
        preds: Dict[str, Set[str]] = {}
        for sid, cands in entity_pairs.items():
            matched = {tid for tid, prob in cands if prob >= tau}
            preds[sid] = matched

        metrics = evaluate_entity_resolution(val_gt_map, preds, beta=BETA)
        f05 = metrics["macro_f05"]
        p = metrics["macro_precision"]
        r = metrics["macro_recall"]
        s_acc = metrics["singleton_accuracy"]

        sweep_results.append((tau, f05, p, r, s_acc))
        marker = " 🏆 (BEST)" if f05 > best_f05 else ""
        if f05 > best_f05:
            best_f05 = f05
            best_tau = tau

        print(f"{tau:<16.2f}| {f05:<14.4f}| {p:<14.4f}| {r:<14.4f}| {s_acc:<14.4f}{marker}")

    print("=" * 90)
    print(f"\nOptimal Macro F0.5 Threshold: τ* = {best_tau:.2f} (Macro F0.5 = {best_f05:.4f})")

    # 5. Persist Model Artifact and Optimal Threshold
    model.booster_.save_model(str(model_output_path))
    threshold_meta_path = MODELS_DIR / "optimal_threshold.txt"
    with open(threshold_meta_path, "w", encoding="utf-8") as f:
        f.write(f"{best_tau:.4f}\n")

    print(f"✅ Saved trained LightGBM model artifact -> {model_output_path.name}")
    print(f"✅ Saved calibrated threshold ({best_tau:.2f}) -> {threshold_meta_path.name}")

    total_time = time.time() - start_total
    print(f"✅ Phase 5 & 6 Training & Calibration finished in {total_time:.2f} seconds.")
    return model, best_tau


def run_training_pipeline() -> None:
    """Master execution entrypoint for Step 4."""
    train_and_optimize_model()


if __name__ == "__main__":
    run_training_pipeline()
