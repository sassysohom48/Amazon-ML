"""
Step 5.1 & 5.2: Hybrid Gradient Boosted Model Training & Multi-Tier Calibration Engine.
Trains LightGBM, CatBoost, and optional XGBoost with pure binary_logloss,
extracts gain-based feature importances, and executes multi-tier Macro F0.5
calibration sweeps across:
  - Global decision thresholds theta*
  - Probability ensemble blending (w* * LGBM + (1 - w*) * CatBoost)
  - Country-specific dynamic calibration (US, India, France)
  - Source-specific dynamic calibration (S2 vs S3)
  - Contradiction and margin post-processing rules
"""

import os
import sys
import time
import json
import gc
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional, Any
import numpy as np
import polars as pl
import lightgbm as lgb

# Optional ML libraries with graceful fallbacks
try:
    import catboost as cb
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

# Ensure package import works
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PROCESSED_DIR, MODELS_DIR, OUTPUT_DIR
from src.feature_engineering import FEATURE_NAMES
from src.validation_contract import compute_entity_f05, METRIC_BETA


# =============================================================================
# 1. MODEL TRAINING SUITE (LIGHTGBM, CATBOOST, XGBOOST)
# =============================================================================

def train_lightgbm_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    feature_names: Optional[List[str]] = None,
    params: Optional[Dict[str, Any]] = None,
    num_boost_round: int = 600,
    early_stopping_rounds: int = 50,
    random_seed: int = 42,
) -> Tuple[lgb.Booster, Dict[str, float]]:
    """
    Trains precision-calibrated LightGBM model with binary_logloss.
    """
    print("\n" + "=" * 75)
    print("🚀 TRAINING LIGHTGBM PAIRWISE CLASSIFIER")
    print("=" * 75)
    t0 = time.time()

    f_names = feature_names or FEATURE_NAMES

    default_params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "learning_rate": 0.04,
        "num_leaves": 63,
        "max_depth": 8,
        "min_child_samples": 30,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "verbosity": -1,
        "n_jobs": -1,
        "seed": random_seed,
    }
    if params:
        default_params.update(params)

    train_data = lgb.Dataset(X_train, label=y_train, feature_name=f_names)
    valid_sets = [train_data]
    valid_names = ["train"]

    callbacks = []
    if X_val is not None and y_val is not None:
        val_data = lgb.Dataset(X_val, label=y_val, feature_name=f_names, reference=train_data)
        valid_sets.append(val_data)
        valid_names.append("valid")
        callbacks.append(lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False))
        callbacks.append(lgb.log_evaluation(period=100))
    else:
        callbacks.append(lgb.log_evaluation(period=100))

    print(f"Training on {len(X_train):,} samples with {len(f_names)} features (max {num_boost_round} rounds)...")
    model = lgb.train(
        default_params,
        train_data,
        num_boost_round=num_boost_round,
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=callbacks,
    )

    elapsed = time.time() - t0
    best_iter = model.best_iteration if model.best_iteration > 0 else num_boost_round
    print(f"LightGBM training completed in {elapsed:.2f}s (Best Iteration: {best_iter}).")

    # Feature Importance by Split Gain
    importance_gain = model.feature_importance(importance_type="gain")
    importance_dict = {f_name: float(importance_gain[i]) for i, f_name in enumerate(f_names)}

    sorted_imp = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    print("\nTop 15 Most Predictive Features (LightGBM Split Gain):")
    for rank, (name, gain) in enumerate(sorted_imp[:15], 1):
        print(f"  {rank:2d}. {name:<35} Gain: {gain:>12,.1f}")

    return model, importance_dict


def train_catboost_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    feature_names: Optional[List[str]] = None,
    params: Optional[Dict[str, Any]] = None,
    iterations: int = 600,
    early_stopping_rounds: int = 50,
    random_seed: int = 42,
) -> Tuple[Optional[Any], Optional[Dict[str, float]]]:
    """
    Trains CatBoost model with Logloss objective and oblivious decision trees.
    """
    if not HAS_CATBOOST:
        print("\n[INFO] CatBoost not installed. Skipping CatBoost training.")
        return None, None

    print("\n" + "=" * 75)
    print("🐱 TRAINING CATBOOST PAIRWISE CLASSIFIER")
    print("=" * 75)
    t0 = time.time()

    f_names = feature_names or FEATURE_NAMES

    default_params = {
        "loss_function": "Logloss",
        "eval_metric": "Logloss",
        "iterations": iterations,
        "learning_rate": 0.04,
        "depth": 7,
        "l2_leaf_reg": 3.0,
        "random_seed": random_seed,
        "verbose": 100,
        "thread_count": -1,
    }
    if params:
        default_params.update(params)

    if X_val is not None and y_val is not None:
        default_params["early_stopping_rounds"] = early_stopping_rounds
        eval_set = (X_val, y_val)
    else:
        eval_set = None

    model = cb.CatBoostClassifier(**default_params)
    print(f"Training CatBoost on {len(X_train):,} samples with {len(f_names)} features...")
    model.fit(X_train, y_train, eval_set=eval_set, verbose=100)

    elapsed = time.time() - t0
    print(f"CatBoost training completed in {elapsed:.2f}s.")

    importances = model.get_feature_importance()
    importance_dict = {f_name: float(importances[i]) for i, f_name in enumerate(f_names)}

    sorted_imp = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    print("\nTop 15 Most Predictive Features (CatBoost Importance):")
    for rank, (name, imp) in enumerate(sorted_imp[:15], 1):
        print(f"  {rank:2d}. {name:<35} Importance: {imp:>10.2f}")

    return model, importance_dict


# =============================================================================
# 2. MULTI-TIER MACRO F0.5 THRESHOLD CALIBRATOR
# =============================================================================

class MultiTierCalibrator:
    """
    Scientific calibration suite for entity resolution decision boundaries.
    Optimizes exact Macro F0.5 with strict singleton penalty.
    """

    def __init__(
        self,
        s1_ids: List[str],
        tgt_ids: List[str],
        probs: np.ndarray,
        ground_truth_map: Dict[str, Set[str]],
        countries: Optional[List[str]] = None,
        target_sources: Optional[List[str]] = None,
    ):
        self.s1_ids = s1_ids
        self.tgt_ids = tgt_ids
        self.probs = probs
        self.gt_map = ground_truth_map
        self.countries = countries or ["Unknown"] * len(s1_ids)
        self.target_sources = target_sources or [("S2" if t.startswith("S2") else "S3") for t in tgt_ids]

        # Distinct S1 entities evaluated
        self.all_eval_s1 = list(set(s1_ids) | set(ground_truth_map.keys()))

        # Group candidate predictions by S1 ID
        self.s1_to_candidates: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for i, s1_id in enumerate(s1_ids):
            self.s1_to_candidates[s1_id].append({
                "tgt_id": tgt_ids[i],
                "prob": float(probs[i]),
                "country": self.countries[i],
                "source": self.target_sources[i],
            })

    def evaluate_predictions(self, preds_map: Dict[str, Set[str]]) -> Dict[str, float]:
        """
        Computes exact macro F0.5, macro precision, macro recall, and singleton accuracy.
        """
        f05_list = []
        prec_list = []
        rec_list = []
        singletons_total = 0
        singletons_correct = 0

        for s1_id in self.all_eval_s1:
            true_set = self.gt_map.get(s1_id, set())
            pred_set = preds_map.get(s1_id, set())

            f05, p, r = compute_entity_f05(true_set, pred_set, beta=METRIC_BETA)
            f05_list.append(f05)
            prec_list.append(p)
            rec_list.append(r)

            if not true_set:
                singletons_total += 1
                if not pred_set:
                    singletons_correct += 1

        macro_f05 = float(np.mean(f05_list)) if f05_list else 0.0
        macro_p = float(np.mean(prec_list)) if prec_list else 0.0
        macro_r = float(np.mean(rec_list)) if rec_list else 0.0
        singleton_acc = (singletons_correct / singletons_total) if singletons_total > 0 else 1.0

        return {
            "macro_f05": macro_f05,
            "macro_precision": macro_p,
            "macro_recall": macro_r,
            "singleton_accuracy": singleton_acc,
            "evaluated_entities": len(self.all_eval_s1),
        }

    def sweep_global_threshold(
        self,
        thresholds: np.ndarray = np.arange(0.20, 0.96, 0.01),
        probs_override: Optional[np.ndarray] = None,
    ) -> Tuple[float, Dict[str, float], List[Dict[str, float]]]:
        """
        Sweeps global threshold theta in [0.20, 0.95] to find optimal macro F0.5.
        """
        if probs_override is not None:
            # Rebuild candidate map with new probabilities
            s1_cands = defaultdict(list)
            for i, s1_id in enumerate(self.s1_ids):
                s1_cands[s1_id].append({
                    "tgt_id": self.tgt_ids[i],
                    "prob": float(probs_override[i]),
                    "country": self.countries[i],
                    "source": self.target_sources[i],
                })
        else:
            s1_cands = self.s1_to_candidates

        best_f05 = -1.0
        best_thresh = 0.50
        best_metrics = {}
        curve = []

        for th in thresholds:
            preds: Dict[str, Set[str]] = {}
            for s1_id in self.all_eval_s1:
                c_list = s1_cands.get(s1_id, [])
                matched = {c["tgt_id"] for c in c_list if c["prob"] >= th}
                preds[s1_id] = matched

            res = self.evaluate_predictions(preds)
            f05 = res["macro_f05"]

            curve_point = {
                "threshold": round(float(th), 3),
                "macro_f05": round(f05, 5),
                "macro_precision": round(res["macro_precision"], 5),
                "macro_recall": round(res["macro_recall"], 5),
                "singleton_accuracy": round(res["singleton_accuracy"], 5),
            }
            curve.append(curve_point)

            if f05 > best_f05:
                best_f05 = f05
                best_thresh = float(th)
                best_metrics = res

        return best_thresh, best_metrics, curve

    def sweep_country_thresholds(
        self,
        thresholds: np.ndarray = np.arange(0.25, 0.96, 0.02),
        probs_override: Optional[np.ndarray] = None,
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Independently sweeps optimal thresholds per country: (US, India, France).
        """
        # Determine unique countries
        unique_countries = sorted(list(set(self.countries)))
        country_thresholds = {}
        country_metrics = {}

        if probs_override is not None:
            s1_cands = defaultdict(list)
            for i, s1_id in enumerate(self.s1_ids):
                s1_cands[s1_id].append({
                    "tgt_id": self.tgt_ids[i],
                    "prob": float(probs_override[i]),
                    "country": self.countries[i],
                    "source": self.target_sources[i],
                })
        else:
            s1_cands = self.s1_to_candidates

        # s1_id -> country mapping
        s1_country_map = {}
        for i, s1_id in enumerate(self.s1_ids):
            s1_country_map[s1_id] = self.countries[i]

        for country in unique_countries:
            c_s1 = [s1 for s1 in self.all_eval_s1 if s1_country_map.get(s1, "Unknown") == country]
            if not c_s1:
                country_thresholds[country] = 0.55
                continue

            best_c_f05 = -1.0
            best_c_th = 0.55
            best_c_res = {}

            for th in thresholds:
                f05_list = []
                prec_list = []
                rec_list = []
                for s1_id in c_s1:
                    true_set = self.gt_map.get(s1_id, set())
                    c_list = s1_cands.get(s1_id, [])
                    pred_set = {c["tgt_id"] for c in c_list if c["prob"] >= th}

                    f05, p, r = compute_entity_f05(true_set, pred_set, beta=METRIC_BETA)
                    f05_list.append(f05)
                    prec_list.append(p)
                    rec_list.append(r)

                m_f05 = float(np.mean(f05_list)) if f05_list else 0.0
                if m_f05 > best_c_f05:
                    best_c_f05 = m_f05
                    best_c_th = float(th)
                    best_c_res = {
                        "macro_f05": m_f05,
                        "macro_precision": float(np.mean(prec_list)),
                        "macro_recall": float(np.mean(rec_list)),
                        "count": len(c_s1),
                    }

            country_thresholds[country] = best_c_th
            country_metrics[country] = best_c_res

        # Evaluate combined metric with country-specific thresholds
        combined_preds: Dict[str, Set[str]] = {}
        for s1_id in self.all_eval_s1:
            cntry = s1_country_map.get(s1_id, "Unknown")
            th = country_thresholds.get(cntry, 0.55)
            c_list = s1_cands.get(s1_id, [])
            matched = {c["tgt_id"] for c in c_list if c["prob"] >= th}
            combined_preds[s1_id] = matched

        overall_metrics = self.evaluate_predictions(combined_preds)
        overall_metrics["country_breakdowns"] = country_metrics

        return country_thresholds, overall_metrics

    def sweep_country_source_thresholds(
        self,
        thresholds: np.ndarray = np.arange(0.25, 0.96, 0.03),
        probs_override: Optional[np.ndarray] = None,
    ) -> Tuple[Dict[Tuple[str, str], float], Dict[str, float]]:
        """
        Independently sweeps thresholds per (country, target_source) pair.
        """
        unique_countries = sorted(list(set(self.countries)))
        sources = ["S2", "S3"]

        s1_country_map = {self.s1_ids[i]: self.countries[i] for i in range(len(self.s1_ids))}

        if probs_override is not None:
            s1_cands = defaultdict(list)
            for i, s1_id in enumerate(self.s1_ids):
                s1_cands[s1_id].append({
                    "tgt_id": self.tgt_ids[i],
                    "prob": float(probs_override[i]),
                    "country": self.countries[i],
                    "source": self.target_sources[i],
                })
        else:
            s1_cands = self.s1_to_candidates

        # Grid search or coordinate descent on (th_s2, th_s3) per country
        country_source_thresholds: Dict[str, Dict[str, float]] = {}

        for country in unique_countries:
            c_s1 = [s1 for s1 in self.all_eval_s1 if s1_country_map.get(s1, "Unknown") == country]
            if not c_s1:
                country_source_thresholds[country] = {"S2": 0.55, "S3": 0.60}
                continue

            best_c_f05 = -1.0
            best_pair = (0.55, 0.60)

            for th_s2 in thresholds:
                for th_s3 in thresholds:
                    f05_list = []
                    for s1_id in c_s1:
                        true_set = self.gt_map.get(s1_id, set())
                        c_list = s1_cands.get(s1_id, [])
                        pred_set = set()
                        for c in c_list:
                            req_th = th_s2 if c["source"] == "S2" else th_s3
                            if c["prob"] >= req_th:
                                pred_set.add(c["tgt_id"])

                        f05, _, _ = compute_entity_f05(true_set, pred_set, beta=METRIC_BETA)
                        f05_list.append(f05)

                    m_f05 = float(np.mean(f05_list)) if f05_list else 0.0
                    if m_f05 > best_c_f05:
                        best_c_f05 = m_f05
                        best_pair = (float(th_s2), float(th_s3))

            country_source_thresholds[country] = {
                "S2": best_pair[0],
                "S3": best_pair[1],
            }

        # Combined evaluation
        combined_preds: Dict[str, Set[str]] = {}
        for s1_id in self.all_eval_s1:
            cntry = s1_country_map.get(s1_id, "Unknown")
            c_ths = country_source_thresholds.get(cntry, {"S2": 0.55, "S3": 0.60})
            c_list = s1_cands.get(s1_id, [])
            matched = set()
            for c in c_list:
                req_th = c_ths.get(c["source"], 0.55)
                if c["prob"] >= req_th:
                    matched.add(c["tgt_id"])
            combined_preds[s1_id] = matched

        overall_metrics = self.evaluate_predictions(combined_preds)
        overall_metrics["country_source_thresholds"] = country_source_thresholds

        return country_source_thresholds, overall_metrics

    def evaluate_post_processing_rules(
        self,
        country_thresholds: Dict[str, float],
        margin_gap_threshold: float = 0.35,
        probs_override: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Evaluates margin filtering post-processing:
        If top candidate has high confidence, suppress secondary candidate if P1 - P2 > margin_gap.
        """
        s1_country_map = {self.s1_ids[i]: self.countries[i] for i in range(len(self.s1_ids))}

        if probs_override is not None:
            s1_cands = defaultdict(list)
            for i, s1_id in enumerate(self.s1_ids):
                s1_cands[s1_id].append({
                    "tgt_id": self.tgt_ids[i],
                    "prob": float(probs_override[i]),
                    "country": self.countries[i],
                    "source": self.target_sources[i],
                })
        else:
            s1_cands = self.s1_to_candidates

        filtered_preds: Dict[str, Set[str]] = {}

        for s1_id in self.all_eval_s1:
            cntry = s1_country_map.get(s1_id, "Unknown")
            th = country_thresholds.get(cntry, 0.55)
            c_list = s1_cands.get(s1_id, [])

            if not c_list:
                filtered_preds[s1_id] = set()
                continue

            # Sort by probability descending
            sorted_c = sorted(c_list, key=lambda x: x["prob"], reverse=True)
            top_prob = sorted_c[0]["prob"]

            matched = set()
            for idx, c in enumerate(sorted_c):
                p = c["prob"]
                if p >= th:
                    # If secondary candidate, check margin gap from top candidate
                    if idx > 0 and (top_prob - p) > margin_gap_threshold:
                        continue
                    matched.add(c["tgt_id"])

            filtered_preds[s1_id] = matched

        return self.evaluate_predictions(filtered_preds)


# =============================================================================
# 3. SAVE & LOAD CALIBRATED PHASE 5 ARTIFACTS
# =============================================================================

def save_phase5_artifacts(
    lgbm_model: Optional[lgb.Booster],
    catboost_model: Optional[Any],
    calibration_config: Dict[str, Any],
    experiment_matrix: Dict[str, Any],
    feature_names: List[str] = FEATURE_NAMES,
    output_dir: Path = MODELS_DIR,
):
    """
    Saves trained models, calibration parameters, and experiment results.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Save LightGBM
    if lgbm_model is not None:
        lgb_path = output_dir / "lgbm_entity_resolver.txt"
        lgbm_model.save_model(str(lgb_path))
        print(f"[OK] Saved LightGBM Model to: {lgb_path}")

    # 2. Save CatBoost
    if catboost_model is not None:
        cb_path = output_dir / "catboost_entity_resolver.cbm"
        catboost_model.save_model(str(cb_path))
        print(f"[OK] Saved CatBoost Model to: {cb_path}")

    # 3. Save Calibration Config
    config_path = output_dir / "phase5_calibration_config.json"
    full_config = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "feature_names": feature_names,
        "calibration": calibration_config,
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(full_config, f, indent=2)
    print(f"[OK] Saved Calibration Config to: {config_path}")

    # 4. Save Experiment Matrix
    exp_path = OUTPUT_DIR / "phase5_experiment_matrix.json"
    with open(exp_path, "w", encoding="utf-8") as f:
        json.dump(experiment_matrix, f, indent=2)
    print(f"[OK] Saved Experiment Matrix Report to: {exp_path}")
