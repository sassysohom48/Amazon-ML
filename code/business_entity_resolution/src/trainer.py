"""
Supervised Model Training & F0.5 Threshold Optimization (Amazon ML Challenge 2026).
Trains an entity-grouped LightGBM GBDT classifier on candidate pairs and hard negatives,
computes feature importance, and sweeps decision thresholds to maximize Macro F0.5.
"""

import json
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
from .feature_builder import FEATURE_COLS


def train_and_optimize_model(
    features_parquet: Optional[Path] = None,
    model_output_path: Optional[Path] = None,
    train_val_split: float = 0.80
) -> Tuple[lgb.LGBMClassifier, float]:
    """
    Trains LightGBM (and optional CatBoost) GBDT on candidate pairs and hard negatives:
    1. Loads the 25-feature matrix.
    2. Splits data grouped strictly by source1_entity_id (zero entity leakage).
    3. Trains LightGBM with AUC early stopping.
    4. Optionally trains CatBoost and creates a blended ensemble.
    5. Computes feature importance rankings.
    6. Sweeps decision thresholds tau in [0.40, 0.90] globally and per-country to maximize Macro F0.5.
    7. Persists trained model artifacts and optimal threshold configs.
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
    print(f"LightGBM training completed in {train_elapsed:.2f}s (Best iteration: {model.best_iteration_})")

    # Optional CatBoost Blending
    cat_model = None
    try:
        from catboost import CatBoostClassifier
        print("\nTraining CatBoost Classifier for Blended GBDT Ensemble...")
        cat_model = CatBoostClassifier(
            iterations=600,
            learning_rate=0.06,
            depth=6,
            eval_metric="AUC",
            random_seed=RANDOM_SEED,
            thread_count=-1,
            verbose=150
        )
        cat_model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            early_stopping_rounds=40,
            verbose=150
        )
        cat_path = MODELS_DIR / "catboost_model.cbm"
        cat_model.save_model(str(cat_path))
        print(f"✅ Saved trained CatBoost model artifact -> {cat_path.name}")
    except Exception as e:
        print(f"Notice: CatBoost not installed or skipped ({e}). Proceeding with pure LightGBM booster.")

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
    p_lgb = model.predict_proba(X_val)[:, 1]
    if cat_model is not None:
        p_cat = cat_model.predict_proba(X_val)[:, 1]
        val_preds_prob = 0.55 * p_lgb + 0.45 * p_cat
        print("  Blended Ensemble applied: 0.55 * LightGBM + 0.45 * CatBoost")
    else:
        val_preds_prob = p_lgb

    # Load Ground Truth for validation entities
    from .dataset import load_ground_truth
    val_gt_path = PROCESSED_DIR / "val_ground_truth.parquet"
    full_gt = load_ground_truth(val_gt_path)
    val_gt_map = {sid: full_gt.get(sid, set()) for sid in val_entities}

    # Load Country mapping for country-specific threshold tuning
    val_s1_country_df = pl.read_parquet(PROCESSED_DIR / "val_source1.parquet").select(["entity_id", "country"])
    entity_country_map = dict(zip(val_s1_country_df["entity_id"].to_list(), val_s1_country_df["country"].to_list()))

    us_val_entities = {sid for sid in val_entities if entity_country_map.get(sid) == "US"}
    in_val_entities = {sid for sid in val_entities if entity_country_map.get(sid) == "INDIA"}

    us_gt_map = {sid: val_gt_map[sid] for sid in us_val_entities}
    in_gt_map = {sid: val_gt_map[sid] for sid in in_val_entities}

    # Extract aligned ID arrays for validation pairs
    val_s1_ids = df_val["source1_entity_id"].to_list()
    val_target_ids = df_val["target_entity_id"].to_list()

    # Pre-organize pairs for fast threshold evaluations
    entity_pairs: Dict[str, List[Tuple[str, float]]] = {sid: [] for sid in val_entities}
    for sid, tid, prob in zip(val_s1_ids, val_target_ids, val_preds_prob):
        entity_pairs[sid].append((tid, prob))

    thresholds = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.68, 0.70, 0.72, 0.75, 0.78, 0.80, 0.85, 0.90]
    sweep_results = []
    best_tau = 0.70
    best_f05 = -1.0

    best_tau_us = 0.75
    best_f05_us = -1.0
    best_tau_in = 0.70
    best_f05_in = -1.0

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

        # Track per-country optimal
        if us_gt_map:
            us_preds = {sid: preds[sid] for sid in us_val_entities}
            us_m = evaluate_entity_resolution(us_gt_map, us_preds, beta=BETA)
            if us_m["macro_f05"] > best_f05_us:
                best_f05_us = us_m["macro_f05"]
                best_tau_us = tau

        if in_gt_map:
            in_preds = {sid: preds[sid] for sid in in_val_entities}
            in_m = evaluate_entity_resolution(in_gt_map, in_preds, beta=BETA)
            if in_m["macro_f05"] > best_f05_in:
                best_f05_in = in_m["macro_f05"]
                best_tau_in = tau

        print(f"{tau:<16.2f}| {f05:<14.4f}| {p:<14.4f}| {r:<14.4f}| {s_acc:<14.4f}{marker}")

    print("=" * 90)
    print(f"\nGlobal Macro F0.5 Optimal Threshold: τ* = {best_tau:.2f} (Macro F0.5 = {best_f05:.4f})")
    print(f"Country-Specific Calibrations: [US] τ* = {best_tau_us:.2f} ({best_f05_us:.4f}) | [INDIA] τ* = {best_tau_in:.2f} ({best_f05_in:.4f})")

    # 5. Persist Model Artifact and Optimal Thresholds
    model.booster_.save_model(str(model_output_path))
    threshold_meta_path = MODELS_DIR / "optimal_threshold.txt"
    with open(threshold_meta_path, "w", encoding="utf-8") as f:
        f.write(f"{best_tau:.4f}\n")

    # Persist JSON dictionary of country thresholds
    country_thresholds_json = {
        "US": best_tau_us,
        "INDIA": best_tau_in,
        "FRANCE": round((best_tau_us + best_tau_in) / 2.0, 2),  # Midpoint calibration for zero-shot France
        "GLOBAL": best_tau
    }
    threshold_json_path = MODELS_DIR / "optimal_thresholds.json"
    with open(threshold_json_path, "w", encoding="utf-8") as f:
        json.dump(country_thresholds_json, f, indent=2)

    print(f"✅ Saved trained LightGBM model artifact -> {model_output_path.name}")
    print(f"✅ Saved global calibrated threshold ({best_tau:.2f}) -> {threshold_meta_path.name}")
    print(f"✅ Saved country-calibrated thresholds {country_thresholds_json} -> {threshold_json_path.name}")

    total_time = time.time() - start_total
    print(f"✅ Phase 5 & 6 Training & Calibration finished in {total_time:.2f} seconds.")
    return model, best_tau


def run_training_pipeline() -> None:
    """Master execution entrypoint for Step 4."""
    train_and_optimize_model()


if __name__ == "__main__":
    run_training_pipeline()
