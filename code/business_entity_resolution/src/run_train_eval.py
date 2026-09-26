"""
Phase 5: Hybrid Model Training, Multi-Tier Calibration & Out-of-Fold (OOF) Evaluation.
Executes the complete scientific experiment matrix:
  - Exp 5A: Single LightGBM Baseline (Global Sweep)
  - Exp 5B: Single CatBoost Baseline (Global Sweep)
  - Exp 5C: Learned Ensemble Blend (w* * LGBM + (1 - w*) * CatBoost)
  - Exp 5D: Country-Specific Calibration (theta_US*, theta_IN*, theta_FR*)
  - Exp 5E: Country x Target Source Calibration (S2 vs S3)
  - Exp 5F: Contradiction & Margin Post-Processing Gatekeeper
  - Full Loss Attribution Decomposition & Diagnostic Logging
"""

import sys
import os
import time
import json
import gc
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional, Any
import numpy as np
import polars as pl

# Ensure package import works
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PROCESSED_DIR, MODELS_DIR, OUTPUT_DIR, TRAIN_DIR
from src.country_idf import CountryIDFComputer
from src.feature_engineering import FeatureExtractor, FEATURE_NAMES
from src.validation_contract import compute_entity_f05, METRIC_BETA
from src.train_model import (
    train_lightgbm_model,
    train_catboost_model,
    MultiTierCalibrator,
    save_phase5_artifacts,
    HAS_CATBOOST,
)


def load_or_prepare_training_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Loads precomputed training features from train_features.parquet (from Folds 1-4)
    and creates a train/early-stopping validation split.
    """
    train_feat_path = PROCESSED_DIR / "train_features.parquet"
    if not train_feat_path.exists():
        raise FileNotFoundError(
            f"Precomputed {train_feat_path.name} not found! Run run_feature_engineering.py first."
        )

    print(f"\nLoading precomputed training feature dataset from {train_feat_path.name}...")
    t0 = time.time()
    df = pl.read_parquet(train_feat_path)
    print(f"Loaded {len(df):,} training pairs in {time.time() - t0:.2f}s.")

    X_all = df.select(FEATURE_NAMES).to_numpy().astype(np.float32)
    y_all = df["label"].to_numpy().astype(np.int32)
    s1_all = df["source1_entity_id"].to_list()

    n_pos = int((y_all == 1).sum())
    n_neg = int((y_all == 0).sum())
    print(f"  • Total Samples: {len(X_all):,} ({n_pos:,} Positives, {n_neg:,} Negatives, ratio 1:{n_neg/max(n_pos, 1):.1f})")

    # Strict Entity-Level Train / Early-Stopping Split (85% Train, 15% Internal Eval)
    unique_s1 = list(set(s1_all))
    np.random.seed(42)
    np.random.shuffle(unique_s1)
    split_point = int(len(unique_s1) * 0.85)
    train_s1_set = set(unique_s1[:split_point])

    train_mask = np.array([s in train_s1_set for s in s1_all])
    val_mask = ~train_mask

    X_tr, y_tr = X_all[train_mask], y_all[train_mask]
    X_va, y_va = X_all[val_mask], y_all[val_mask]

    print(f"Internal Entity-Disjoint Split: {len(X_tr):,} Train pairs, {len(X_va):,} Early-Stopping Eval pairs.")
    del df, X_all, y_all
    gc.collect()

    return X_tr, y_tr, X_va, y_va, FEATURE_NAMES


def prepare_validation_candidate_features(
    max_val_entities: int = 35000,
    random_seed: int = 42,
) -> Tuple[np.ndarray, List[str], List[str], List[str], List[str], Dict[str, Set[str]]]:
    """
    Extracts features for out-of-fold Fold 0 validation candidate pairs using lazy target lookup.
    Supports both flat pair schemas (with provenance) and grouped candidate schemas.
    """
    print("\n" + "=" * 80)
    print(f"PREPARING OUT-OF-FOLD (FOLD 0) VALIDATION SET ({max_val_entities:,} Entities)")
    print("=" * 80)
    t0 = time.time()

    val_cand_path = PROCESSED_DIR / "val_candidate_pairs.parquet"
    val_gt_path = PROCESSED_DIR / "val_ground_truth.parquet"
    if not val_gt_path.exists():
        val_gt_path = PROCESSED_DIR / "train_ground_truth.parquet"
        if not val_gt_path.exists():
            val_gt_path = TRAIN_DIR / "train_ground_truth.tsv"

    # 1. Load Ground Truth
    print(f"Loading validation ground truth from {val_gt_path.name}...")
    if str(val_gt_path).endswith(".parquet"):
        gt_df = pl.read_parquet(val_gt_path)
    else:
        gt_df = pl.read_csv(val_gt_path, separator="\t")

    s1_col = gt_df.columns[0]
    tgt_col = gt_df.columns[1]
    for c in gt_df.columns:
        if c in ("source1_entity_id", "source_entity_id", "s1_id"):
            s1_col = c
        elif c in ("matched_entity_ids", "matched_entity_id", "target_entity_id", "s2_id", "s3_id"):
            tgt_col = c

    val_gt_map: Dict[str, Set[str]] = defaultdict(set)
    for row in gt_df.iter_rows(named=True):
        s1 = str(row[s1_col]).strip()
        targets_raw = str(row[tgt_col]).strip()
        if targets_raw and targets_raw.lower() not in ("none", "nan", "null", ""):
            for tid in targets_raw.split(","):
                tid_clean = tid.strip()
                if tid_clean and tid_clean.lower() not in ("none", "nan", "null", ""):
                    val_gt_map[s1].add(tid_clean)

    # 2. Load Validation Candidate Pairs
    print(f"Loading candidate pairs from {val_cand_path.name}...")
    val_cand_df = pl.read_parquet(val_cand_path)

    flat_s1 = []
    flat_tgt = []
    flat_prov = []
    needed_tgt_ids = set()
    val_s1_ids = []

    # Check schema: Flat pairs (Phase 3 format) vs Grouped string
    if "target_entity_id" in val_cand_df.columns:
        # Flat schema with provenance
        all_unique_s1 = val_cand_df["source1_entity_id"].unique().to_list()
        if max_val_entities > 0 and len(all_unique_s1) > max_val_entities:
            np.random.seed(random_seed)
            selected_s1 = set(np.random.choice(all_unique_s1, size=max_val_entities, replace=False))
            val_cand_filtered = val_cand_df.filter(pl.col("source1_entity_id").is_in(selected_s1))
            val_s1_ids = list(selected_s1)
        else:
            val_cand_filtered = val_cand_df
            val_s1_ids = all_unique_s1

        prov_cols = [c for c in val_cand_filtered.columns if c not in ("source1_entity_id", "target_entity_id")]

        for row in val_cand_filtered.iter_rows(named=True):
            s1_id = str(row["source1_entity_id"]).strip()
            tgt_id = str(row["target_entity_id"]).strip()
            prov = {c: row[c] for c in prov_cols}
            flat_s1.append(s1_id)
            flat_tgt.append(tgt_id)
            flat_prov.append(prov)
            needed_tgt_ids.add(tgt_id)
    else:
        # Grouped comma-separated candidate string
        cand_col = "candidate_entity_ids" if "candidate_entity_ids" in val_cand_df.columns else val_cand_df.columns[1]
        s1_id_col = "source1_entity_id" if "source1_entity_id" in val_cand_df.columns else val_cand_df.columns[0]

        if max_val_entities > 0 and len(val_cand_df) > max_val_entities:
            np.random.seed(random_seed)
            val_cand_sample = val_cand_df.sample(n=max_val_entities, seed=random_seed)
        else:
            val_cand_sample = val_cand_df

        val_s1_ids = val_cand_sample[s1_id_col].to_list()
        val_cands_str = val_cand_sample[cand_col].to_list()

        for s1_id, c_str in zip(val_s1_ids, val_cands_str):
            if not c_str:
                continue
            for tgt in str(c_str).split(","):
                tgt_clean = tgt.strip()
                if tgt_clean:
                    flat_s1.append(s1_id)
                    flat_tgt.append(tgt_clean)
                    flat_prov.append({})
                    needed_tgt_ids.add(tgt_clean)

    print(f"Evaluated Fold 0 Slice: {len(val_s1_ids):,} entities -> {len(flat_s1):,} candidate pairs ({len(needed_tgt_ids):,} unique targets).")

    # 3. Lazy Target Record Lookup for Validation Pairs (< 150 MB RAM)
    cols_to_load = [
        "entity_id", "country", "name_clean", "name_core", "legal_form", "name_tokens", "name_acronym",
        "name_phonetic", "addr_clean", "addr_tokens", "addr_digits", "addr_unit_num", "postal_clean"
    ]

    print("Loading entity records into fast lookup cache...")
    s1_all = pl.read_parquet(PROCESSED_DIR / "train_source1_cleaned.parquet", columns=cols_to_load)
    s1_val_df = s1_all.filter(pl.col("entity_id").is_in(set(val_s1_ids)))
    del s1_all
    gc.collect()

    s2_all = pl.read_parquet(PROCESSED_DIR / "train_source2_cleaned.parquet", columns=cols_to_load)
    s2_val_df = s2_all.filter(pl.col("entity_id").is_in(needed_tgt_ids))

    s3_all = pl.read_parquet(PROCESSED_DIR / "train_source3_cleaned.parquet", columns=cols_to_load)
    s3_val_df = s3_all.filter(pl.col("entity_id").is_in(needed_tgt_ids))

    # Fit Country IDF Computer
    idf_comp = CountryIDFComputer()
    idf_comp.fit_from_dataframes(s2_all, s3_all)
    del s2_all, s3_all
    gc.collect()

    extractor = FeatureExtractor(idf_computer=idf_comp)
    s1_records = {row["entity_id"]: row for row in s1_val_df.iter_rows(named=True)}
    s2_records = {row["entity_id"]: row for row in s2_val_df.iter_rows(named=True)}
    s3_records = {row["entity_id"]: row for row in s3_val_df.iter_rows(named=True)}
    tgt_records = {**s2_records, **s3_records}
    del s1_val_df, s2_val_df, s3_val_df, s2_records, s3_records
    gc.collect()

    # 4. Extract 73 Features for Validation Pairs
    print(f"Extracting 73 features for {len(flat_s1):,} validation candidate pairs...")
    t_feat = time.time()
    val_features = []
    val_countries = []
    val_sources = []

    for idx, (s1_id, tgt_id, prov) in enumerate(zip(flat_s1, flat_tgt, flat_prov), 1):
        s1_rec = s1_records.get(s1_id, {})
        tgt_rec = tgt_records.get(tgt_id, {})

        feat_vec = extractor.extract_pair_features(s1_rec, tgt_rec, prov)
        val_features.append(feat_vec)
        val_countries.append(str(s1_rec.get("country", "Unknown") or "Unknown"))
        val_sources.append("S2" if tgt_id.startswith("S2") else "S3")

        if idx % 100000 == 0 or idx == len(flat_s1):
            speed = idx / (time.time() - t_feat)
            print(f"  Extracted {idx:,} / {len(flat_s1):,} val pairs ({speed:,.0f} pairs/s)...")

    X_val = np.array(val_features, dtype=np.float32)
    print(f"Validation feature extraction completed in {time.time() - t_feat:.2f}s ({len(X_val)/(time.time() - t_feat):,.0f} pairs/s).")

    # Filter GT map to only evaluated entities
    eval_gt_map = {s1: val_gt_map.get(s1, set()) for s1 in val_s1_ids}

    del s1_records, tgt_records, val_features, flat_prov
    gc.collect()

    return X_val, flat_s1, flat_tgt, val_countries, val_sources, eval_gt_map


def run_phase_5_pipeline(max_val_entities: int = 35000):
    print("=" * 85)
    print("🏆 EXECUTING PHASE 5: HYBRID MODEL TRAINING & OOF CALIBRATION MATRIX")
    print("=" * 85)
    t_total_start = time.time()

    # 1. Load Training Data
    X_tr, y_tr, X_va, y_va, feature_names = load_or_prepare_training_data()

    # 2. Train LightGBM Model
    lgb_model, lgb_importances = train_lightgbm_model(
        X_train=X_tr,
        y_train=y_tr,
        X_val=X_va,
        y_val=y_va,
        feature_names=feature_names,
        num_boost_round=600,
        early_stopping_rounds=50,
    )

    # 3. Train CatBoost Model (if available)
    cb_model, cb_importances = train_catboost_model(
        X_train=X_tr,
        y_train=y_tr,
        X_val=X_va,
        y_val=y_va,
        feature_names=feature_names,
        iterations=600,
        early_stopping_rounds=50,
    )

    del X_tr, y_tr, X_va, y_va
    gc.collect()

    # 4. Prepare Out-of-Fold Validation Set
    X_val, flat_s1, flat_tgt, val_countries, val_sources, eval_gt_map = prepare_validation_candidate_features(
        max_val_entities=max_val_entities
    )

    # 5. Predict Probabilities
    print("\n" + "=" * 80)
    print("PREDICTING VALIDATION PROBABILITIES ACROSS CANDIDATES")
    print("=" * 80)
    t_pred = time.time()

    p_lgb = lgb_model.predict(X_val)
    print(f"LightGBM inference completed for {len(X_val):,} pairs in {time.time() - t_pred:.2f}s.")

    if cb_model is not None:
        t_cb_pred = time.time()
        p_cb = cb_model.predict_proba(X_val)[:, 1]
        print(f"CatBoost inference completed for {len(X_val):,} pairs in {time.time() - t_cb_pred:.2f}s.")
    else:
        p_cb = p_lgb.copy()

    # Initialize Multi-Tier Calibrator
    calibrator = MultiTierCalibrator(
        s1_ids=flat_s1,
        tgt_ids=flat_tgt,
        probs=p_lgb,
        ground_truth_map=eval_gt_map,
        countries=val_countries,
        target_sources=val_sources,
    )

    # =========================================================================
    # 6. EXECUTE THE PHASE 5 EXPERIMENT MATRIX
    # =========================================================================
    print("\n" + "=" * 85)
    print("🔬 RUNNING PHASE 5 SCIENTIFIC EXPERIMENT MATRIX")
    print("=" * 85)
    experiments_log: Dict[str, Any] = {}

    # -------------------------------------------------------------------------
    # Experiment 5A: Single LightGBM Baseline (Global Sweep)
    # -------------------------------------------------------------------------
    print("\n[EXPERIMENT 5A] Single LightGBM Baseline (Global Threshold Sweep):")
    th_5a, res_5a, curve_5a = calibrator.sweep_global_threshold(probs_override=p_lgb)
    print(f"  • Optimal Global Threshold θ* = {th_5a:.2f}")
    print(f"  • Macro F0.5:        {res_5a['macro_f05']:.4f}")
    print(f"  • Macro Precision:   {res_5a['macro_precision']*100:.2f}%")
    print(f"  • Macro Recall:      {res_5a['macro_recall']*100:.2f}%")
    print(f"  • Singleton Defense: {res_5a['singleton_accuracy']*100:.2f}%")
    experiments_log["Exp_5A_LightGBM_Global"] = {
        "description": "Single LightGBM model with global threshold sweep",
        "optimal_threshold": th_5a,
        "metrics": res_5a,
    }

    # -------------------------------------------------------------------------
    # Experiment 5B: Single CatBoost Baseline (Global Sweep)
    # -------------------------------------------------------------------------
    if cb_model is not None:
        print("\n[EXPERIMENT 5B] Single CatBoost Baseline (Global Threshold Sweep):")
        th_5b, res_5b, curve_5b = calibrator.sweep_global_threshold(probs_override=p_cb)
        print(f"  • Optimal Global Threshold θ* = {th_5b:.2f}")
        print(f"  • Macro F0.5:        {res_5b['macro_f05']:.4f}")
        print(f"  • Macro Precision:   {res_5b['macro_precision']*100:.2f}%")
        print(f"  • Macro Recall:      {res_5b['macro_recall']*100:.2f}%")
        print(f"  • Singleton Defense: {res_5b['singleton_accuracy']*100:.2f}%")
        experiments_log["Exp_5B_CatBoost_Global"] = {
            "description": "Single CatBoost model with global threshold sweep",
            "optimal_threshold": th_5b,
            "metrics": res_5b,
        }
    else:
        res_5b = res_5a
        th_5b = th_5a

    # -------------------------------------------------------------------------
    # Experiment 5C: Learned Ensemble Blend (w* * LGBM + (1 - w*) * CatBoost)
    # -------------------------------------------------------------------------
    print("\n[EXPERIMENT 5C] Probability Ensemble Blend Grid Sweep:")
    best_blend_f05 = -1.0
    best_w = 1.0
    best_blend_th = 0.55
    best_blend_res = {}
    best_p_blend = p_lgb

    if cb_model is not None:
        weight_grid = np.linspace(0.0, 1.0, 11)
        for w in weight_grid:
            p_blend = w * p_lgb + (1.0 - w) * p_cb
            th_w, res_w, _ = calibrator.sweep_global_threshold(probs_override=p_blend)
            if res_w["macro_f05"] > best_blend_f05:
                best_blend_f05 = res_w["macro_f05"]
                best_w = float(w)
                best_blend_th = float(th_w)
                best_blend_res = res_w
                best_p_blend = p_blend
        print(f"  • Optimal Ensemble Weight: w* = {best_w:.2f} (LGBM: {best_w*100:.0f}%, CatBoost: {(1-best_w)*100:.0f}%)")
        print(f"  • Optimal Blended Threshold θ* = {best_blend_th:.2f}")
        print(f"  • Blended Macro F0.5: {best_blend_res['macro_f05']:.4f} (Prec: {best_blend_res['macro_precision']*100:.2f}%, Rec: {best_blend_res['macro_recall']*100:.2f}%)")
    else:
        best_w = 1.0
        best_blend_th = th_5a
        best_blend_res = res_5a
        best_p_blend = p_lgb
        print("  • CatBoost not active, ensemble defaults to pure LightGBM.")

    experiments_log["Exp_5C_Ensemble_Blend"] = {
        "description": "Weighted probability ensemble blend",
        "optimal_lgbm_weight": best_w,
        "optimal_threshold": best_blend_th,
        "metrics": best_blend_res,
    }

    # -------------------------------------------------------------------------
    # Experiment 5D: Country-Specific Calibration
    # -------------------------------------------------------------------------
    print("\n[EXPERIMENT 5D] Country-Specific Dynamic Calibration (US vs India vs France):")
    country_ths, res_5d = calibrator.sweep_country_thresholds(probs_override=best_p_blend)
    print("  • Optimal Country Thresholds:")
    for cntry, th in country_ths.items():
        c_stats = res_5d.get("country_breakdowns", {}).get(cntry, {})
        print(f"    - {cntry:<8}: θ* = {th:.2f} | F0.5 = {c_stats.get('macro_f05', 0):.4f} (Prec: {c_stats.get('macro_precision', 0)*100:.2f}%, Rec: {c_stats.get('macro_recall', 0)*100:.2f}%)")
    print(f"  • Overall Macro F0.5 with Country Calibration: {res_5d['macro_f05']:.4f}")

    experiments_log["Exp_5D_Country_Calibration"] = {
        "description": "Independent country threshold calibration",
        "country_thresholds": country_ths,
        "metrics": res_5d,
    }

    # -------------------------------------------------------------------------
    # Experiment 5E: Country x Target Source Calibration (S2 vs S3)
    # -------------------------------------------------------------------------
    print("\n[EXPERIMENT 5E] Country x Target Source Dynamic Calibration (S2 Registry vs S3 Web):")
    country_source_ths, res_5e = calibrator.sweep_country_source_thresholds(probs_override=best_p_blend)
    print("  • Optimal (Country, Source) Thresholds:")
    for cntry, s_dict in country_source_ths.items():
        print(f"    - {cntry:<8}: S2 θ* = {s_dict['S2']:.2f}, S3 θ* = {s_dict['S3']:.2f}")
    print(f"  • Overall Macro F0.5 with (Country x Source) Calibration: {res_5e['macro_f05']:.4f}")

    experiments_log["Exp_5E_Country_Source_Calibration"] = {
        "description": "Country x Target Source (S2/S3) threshold calibration",
        "country_source_thresholds": country_source_ths,
        "metrics": res_5e,
    }

    # -------------------------------------------------------------------------
    # Experiment 5F: Contradiction & Margin Post-Processing Gatekeeper
    # -------------------------------------------------------------------------
    print("\n[EXPERIMENT 5F] Margin Post-Processing Gatekeeper Evaluation:")
    best_margin_gap = 0.35
    res_5f = calibrator.evaluate_post_processing_rules(
        country_thresholds=country_ths,
        margin_gap_threshold=best_margin_gap,
        probs_override=best_p_blend,
    )
    print(f"  • Macro F0.5 with Margin Filter (gap = {best_margin_gap:.2f}): {res_5f['macro_f05']:.4f}")

    experiments_log["Exp_5F_Post_Processing_Gatekeeper"] = {
        "description": "Margin gap candidate suppression post-processing",
        "margin_gap_threshold": best_margin_gap,
        "metrics": res_5f,
    }

    # =========================================================================
    # 7. COMPARATIVE BENCHMARK SUMMARY & LOSS ATTRIBUTION
    # =========================================================================
    print("\n" + "=" * 90)
    print("📊 PHASE 5 SCIENTIFIC BENCHMARK & EXPERIMENT COMPARISON TABLE")
    print("=" * 90)
    print(f"{'EXPERIMENT':<35} | {'MACRO F0.5':<11} | {'PRECISION':<11} | {'RECALL':<10} | {'SINGLETON ACC'}")
    print("-" * 90)
    print(f"{'5A: Single LightGBM (Global θ*)':<35} | {res_5a['macro_f05']:>10.4f}  | {res_5a['macro_precision']*100:>9.2f}%  | {res_5a['macro_recall']*100:>8.2f}%  | {res_5a['singleton_accuracy']*100:>12.2f}%")
    if cb_model is not None:
        print(f"{'5B: Single CatBoost (Global θ*)':<35} | {res_5b['macro_f05']:>10.4f}  | {res_5b['macro_precision']*100:>9.2f}%  | {res_5b['macro_recall']*100:>8.2f}%  | {res_5b['singleton_accuracy']*100:>12.2f}%")
        print(f"{'5C: Learned Ensemble Blend (w*)':<35} | {best_blend_res['macro_f05']:>10.4f}  | {best_blend_res['macro_precision']*100:>9.2f}%  | {best_blend_res['macro_recall']*100:>8.2f}%  | {best_blend_res['singleton_accuracy']*100:>12.2f}%")
    print(f"{'5D: Country Calibration (θ_C*)':<35} | {res_5d['macro_f05']:>10.4f}  | {res_5d['macro_precision']*100:>9.2f}%  | {res_5d['macro_recall']*100:>8.2f}%  | {res_5d['singleton_accuracy']*100:>12.2f}%")
    print(f"{'5E: Country x Source (θ_C,S*)':<35} | {res_5e['macro_f05']:>10.4f}  | {res_5e['macro_precision']*100:>9.2f}%  | {res_5e['macro_recall']*100:>8.2f}%  | {res_5e['singleton_accuracy']*100:>12.2f}%")
    print(f"{'5F: Post-Processing Gatekeeper':<35} | {res_5f['macro_f05']:>10.4f}  | {res_5f['macro_precision']*100:>9.2f}%  | {res_5f['macro_recall']*100:>8.2f}%  | {res_5f['singleton_accuracy']*100:>12.2f}%")
    print("=" * 90)

    # Determine Winning Calibration Strategy
    best_strategy_name = "Exp_5D_Country_Calibration"
    best_val_f05 = res_5d["macro_f05"]
    best_config_meta = {
        "lgbm_weight": best_w,
        "catboost_weight": 1.0 - best_w if cb_model is not None else 0.0,
        "global_threshold": th_5a,
        "country_thresholds": country_ths,
        "country_source_thresholds": country_source_ths,
        "margin_gap_threshold": best_margin_gap,
        "best_macro_f05": best_val_f05,
        "best_precision": res_5d["macro_precision"],
        "best_recall": res_5d["macro_recall"],
        "singleton_accuracy": res_5d["singleton_accuracy"],
    }

    if res_5e["macro_f05"] > best_val_f05:
        best_strategy_name = "Exp_5E_Country_Source_Calibration"
        best_val_f05 = res_5e["macro_f05"]
    if res_5f["macro_f05"] > best_val_f05:
        best_strategy_name = "Exp_5F_Post_Processing_Gatekeeper"
        best_val_f05 = res_5f["macro_f05"]

    print(f"\n🏆 WINNING CONFIGURATION: {best_strategy_name} (Peak Macro F0.5: {best_val_f05:.4f})")

    # =========================================================================
    # 8. SAVE ARTIFACTS & EXPERIMENT LOGS
    # =========================================================================
    save_phase5_artifacts(
        lgbm_model=lgb_model,
        catboost_model=cb_model,
        calibration_config=best_config_meta,
        experiment_matrix=experiments_log,
        feature_names=feature_names,
    )

    total_time = time.time() - t_total_start
    print("\n" + "=" * 85)
    print(f"🎉 PHASE 5 MODEL TRAINING & CALIBRATION SUITE COMPLETED IN {total_time:.2f}s!")
    print("=" * 85)
    return best_config_meta


if __name__ == "__main__":
    n_val = 35000
    for arg in sys.argv[1:]:
        if arg.isdigit():
            n_val = int(arg)
    run_phase_5_pipeline(max_val_entities=n_val)
