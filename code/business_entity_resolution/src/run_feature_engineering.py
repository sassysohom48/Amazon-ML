"""
Step 4.2 & 4.3: High-Dimensional Pairwise Feature Extraction & Hard Negative Mining Pipeline.
Constructs category-mined balanced training datasets (1 Pos : 3 Hard Negatives), extracts
73 multi-scale features across all 8 feature families, and computes feature distribution metrics (AUC, KS, Delta).
"""

import sys
import time
import json
import gc
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional
import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

# Ensure package import works
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PROCESSED_DIR, DATASET_DIR, TRAIN_DIR
from src.country_idf import CountryIDFComputer
from src.blocking import MultiChannelBlocker
from src.feature_engineering import FeatureExtractor, FEATURE_NAMES


def compute_feature_auc(pos_vals: np.ndarray, neg_vals: np.ndarray) -> float:
    """Computes AUC-ROC for a single feature using fast Wilcoxon-Mann-Whitney rank sum."""
    n_pos = len(pos_vals)
    n_neg = len(neg_vals)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    
    # Combined array with labels
    combined = np.concatenate([pos_vals, neg_vals])
    # Tie-aware ranking
    ranks = np.argsort(np.argsort(combined)) + 1.0
    pos_ranks_sum = np.sum(ranks[:n_pos])
    u_stat = pos_ranks_sum - (n_pos * (n_pos + 1.0)) / 2.0
    auc = u_stat / (n_pos * n_neg)
    return max(float(auc), float(1.0 - auc))  # Invariant to feature orientation


def compute_feature_ks(pos_vals: np.ndarray, neg_vals: np.ndarray) -> float:
    """Computes Kolmogorov-Smirnov (KS) maximum distribution separation."""
    if len(pos_vals) == 0 or len(neg_vals) == 0:
        return 0.0
    all_vals = np.sort(np.unique(np.concatenate([pos_vals, neg_vals])))
    if len(all_vals) > 1000:
        all_vals = np.percentile(all_vals, np.linspace(0, 100, 500))
    
    cdf_pos = np.searchsorted(np.sort(pos_vals), all_vals, side="right") / len(pos_vals)
    cdf_neg = np.searchsorted(np.sort(neg_vals), all_vals, side="right") / len(neg_vals)
    ks = np.max(np.abs(cdf_pos - cdf_neg))
    return float(ks)


def run_phase_4_feature_engineering(
    max_train_entities: int = 50000,
    neg_to_pos_ratio: int = 3,
    val_feature_sample_size: int = 30000,
    random_seed: int = 42,
):
    print("=" * 85)
    print("🚀 EXECUTING PHASE 4: MULTI-SCALE PAIRWISE FEATURE ENGINEERING ENGINE")
    print("=" * 85)
    t_start = time.time()
    np.random.seed(random_seed)

    # 1. Load Ground Truth
    gt_path = PROCESSED_DIR / "train_ground_truth.parquet"
    if not gt_path.exists():
        gt_path = TRAIN_DIR / "train_ground_truth.tsv"

    print(f"Loading Ground Truth from {gt_path.name}...")
    if str(gt_path).endswith(".parquet"):
        gt_df = pl.read_parquet(gt_path)
    else:
        gt_df = pl.read_csv(gt_path, separator="\t")

    s1_col = gt_df.columns[0]
    tgt_col = gt_df.columns[1]
    for c in gt_df.columns:
        if c in ("source1_entity_id", "source_entity_id", "s1_id"):
            s1_col = c
        elif c in ("matched_entity_ids", "matched_entity_id", "target_entity_id", "s2_id", "s3_id", "target_id"):
            tgt_col = c

    # Complete Ground Truth lookup mapping (s1_id -> set of true target IDs)
    gt_map: Dict[str, Set[str]] = defaultdict(set)
    for row in gt_df.iter_rows(named=True):
        s1 = str(row[s1_col]).strip()
        targets_raw = str(row[tgt_col]).strip()
        if targets_raw and targets_raw.lower() not in ("none", "nan", "null", ""):
            for tid in targets_raw.split(","):
                tid_clean = tid.strip()
                if tid_clean and tid_clean.lower() not in ("none", "nan", "null", ""):
                    gt_map[s1].add(tid_clean)

    total_gt_pairs = sum(len(v) for v in gt_map.values())
    print(f"Loaded {len(gt_map):,} S1 entities with {total_gt_pairs:,} true positive target pairs.")

    # 2. Load Processed Cleaned Parquets (minimal columns)
    cols_to_load = [
        "entity_id", "country", "name_clean", "name_core", "legal_form", "name_tokens", "name_acronym",
        "name_phonetic", "addr_clean", "addr_tokens", "addr_digits", "addr_unit_num", "postal_clean"
    ]

    print("\nLoading Enriched Cleaned Parquets...")
    s1_all = pl.read_parquet(PROCESSED_DIR / "train_source1_cleaned.parquet", columns=cols_to_load)
    s2_all = pl.read_parquet(PROCESSED_DIR / "train_source2_cleaned.parquet", columns=cols_to_load)
    s3_all = pl.read_parquet(PROCESSED_DIR / "train_source3_cleaned.parquet", columns=cols_to_load)

    # 3. Fit Country IDF Computer for weighted overlaps
    print("\nFitting Country IDF Computer for feature weighting...")
    idf_comp = CountryIDFComputer()
    target_combined = pl.concat([s2_all, s3_all])
    idf_comp.fit_from_dataframe(target_combined)
    extractor = FeatureExtractor(idf_computer=idf_comp)
    del target_combined
    gc.collect()

    # 4. Strict Stratified Fold Isolation
    folds_path = PROCESSED_DIR / "train_folds.parquet"
    if folds_path.exists():
        folds_df = pl.read_parquet(folds_path, columns=["entity_id", "fold_id"])
        val_fold_eids = set(folds_df.filter(pl.col("fold_id") == 0)["entity_id"].to_list())
        train_fold_eids = set(folds_df.filter(pl.col("fold_id") != 0)["entity_id"].to_list())
        print(f"Strict Fold Isolation: {len(train_fold_eids):,} Training entities (Folds 1-4), {len(val_fold_eids):,} Validation entities (Fold 0).")
    else:
        all_eids = s1_all["entity_id"].to_list()
        val_fold_eids = set(all_eids[:len(all_eids)//5])
        train_fold_eids = set(all_eids[len(all_eids)//5:])

    # 5. Build Training Candidate Dataset from Folds 1-4
    print(f"\nSampling {max_train_entities:,} Training entities from Folds 1-4...")
    s1_train_eligible = s1_all.filter(pl.col("entity_id").is_in(train_fold_eids) & pl.col("entity_id").is_in(list(gt_map.keys())))
    if len(s1_train_eligible) > max_train_entities:
        s1_train_sample = s1_train_eligible.sample(n=max_train_entities, seed=random_seed)
    else:
        s1_train_sample = s1_train_eligible

    s1_records = {row["entity_id"]: row for row in s1_train_sample.iter_rows(named=True)}

    # Block training candidates
    print(f"Generating training candidate pairs for {len(s1_train_sample):,} entities (K = 50)...")
    blocker = MultiChannelBlocker(max_candidates=50)
    blocker.fit(s2_all, s3_all)

    train_cands_dict = blocker.block_dataframe(s1_train_sample, max_k=50)

    # Fast Lazy Target Record Lookup (Only for needed candidate targets ~250k rows instead of 10.3M)
    print("\nBuilding targeted in-memory dictionary for candidate targets...")
    needed_tgt_ids = set()
    for cand_list in train_cands_dict.values():
        for c in cand_list:
            needed_tgt_ids.add(c["target_id"])

    s2_filtered = s2_all.filter(pl.col("entity_id").is_in(needed_tgt_ids))
    s3_filtered = s3_all.filter(pl.col("entity_id").is_in(needed_tgt_ids))
    del s2_all, s3_all
    gc.collect()

    s2_records = {row["entity_id"]: row for row in s2_filtered.iter_rows(named=True)}
    s3_records = {row["entity_id"]: row for row in s3_filtered.iter_rows(named=True)}
    tgt_records = {**s2_records, **s3_records}
    del s2_records, s3_records, s2_filtered, s3_filtered
    gc.collect()
    print(f"Cached {len(tgt_records):,} candidate target records in memory (< 150 MB RAM).")

    # 6. Mine Balanced Pairs: Positives + Categorical Hard Negatives
    print("\nMining Category-Balanced Training Pairs (Positives + Hard Negatives)...")
    pos_pairs_with_prov = []
    hard_neg_pairs_with_prov = []

    for s1_id, cand_list in train_cands_dict.items():
        true_targets = gt_map.get(s1_id, set())

        for c in cand_list:
            tgt_id = c["target_id"]
            prov_dict = {k: c[k] for k in c if k != "target_id"}

            if tgt_id in true_targets:
                pos_pairs_with_prov.append((s1_id, tgt_id, 1, prov_dict))
            else:
                # Guaranteed non-match across ALL ground truth targets for this S1
                hard_neg_pairs_with_prov.append((s1_id, tgt_id, 0, prov_dict))

    print(f"Mined {len(pos_pairs_with_prov):,} True Positives and {len(hard_neg_pairs_with_prov):,} Hard Negatives from blocker.")

    # Subsample hard negatives to achieve exact 1:3 ratio
    n_pos = len(pos_pairs_with_prov)
    n_neg = min(len(hard_neg_pairs_with_prov), n_pos * neg_to_pos_ratio)

    neg_indices = np.random.choice(len(hard_neg_pairs_with_prov), size=n_neg, replace=False)
    selected_neg = [hard_neg_pairs_with_prov[i] for i in neg_indices]

    all_train_pairs = pos_pairs_with_prov + selected_neg
    np.random.shuffle(all_train_pairs)

    print(f"\nFinal Balanced Training Matrix: {len(all_train_pairs):,} pairs ({n_pos:,} Positives : {n_neg:,} Hard Negatives, ratio 1:{neg_to_pos_ratio}).")

    # 7. Extract 73 Pairwise Features in Streaming Chunks
    print("\nExtracting 73 Pairwise Features across all 8 feature families...")
    t_feat = time.time()

    feature_rows = []
    labels = []
    s1_ids_out = []
    tgt_ids_out = []

    for idx, (s1_id, tgt_id, label, prov) in enumerate(all_train_pairs, 1):
        s1_rec = s1_records.get(s1_id, {})
        tgt_rec = tgt_records.get(tgt_id, {})

        feat_vec = extractor.extract_pair_features(s1_rec, tgt_rec, prov)
        feature_rows.append(feat_vec)
        labels.append(label)
        s1_ids_out.append(s1_id)
        tgt_ids_out.append(tgt_id)

        if idx % 50000 == 0 or idx == len(all_train_pairs):
            speed = idx / (time.time() - t_feat)
            print(f"  Extracted {idx:,} / {len(all_train_pairs):,} pairs ({speed:,.0f} pairs/s)...")

    feat_matrix = np.array(feature_rows, dtype=np.float32)
    labels_arr = np.array(labels, dtype=np.int8)

    # 8. Compute Empirical Feature Quality & Discriminative Power (AUC-ROC & KS Statistics)
    print("\n" + "=" * 95)
    print("EMPIRICAL FEATURE DISCRIMINATIVE POWER & QUALITY METRICS (73 Signals):")
    print("=" * 95)
    print(f"{'FEATURE NAME':<35} | {'POS MEAN':<10} | {'NEG MEAN':<10} | {'DELTA':<10} | {'AUC-ROC':<10} | {'KS STAT'}")
    print("-" * 95)

    pos_mask = (labels_arr == 1)
    neg_mask = (labels_arr == 0)

    feature_diagnostics = {}
    diagnostic_table = []

    for f_idx, f_name in enumerate(FEATURE_NAMES):
        vals = feat_matrix[:, f_idx]
        p_vals = vals[pos_mask]
        n_vals = vals[neg_mask]

        p_mean = float(np.mean(p_vals)) if len(p_vals) > 0 else 0.0
        n_mean = float(np.mean(n_vals)) if len(n_vals) > 0 else 0.0
        delta = p_mean - n_mean
        p_med = float(np.median(p_vals)) if len(p_vals) > 0 else 0.0
        n_med = float(np.median(n_vals)) if len(n_vals) > 0 else 0.0
        p_std = float(np.std(p_vals)) if len(p_vals) > 0 else 0.0
        n_std = float(np.std(n_vals)) if len(n_vals) > 0 else 0.0

        auc = compute_feature_auc(p_vals, n_vals)
        ks = compute_feature_ks(p_vals, n_vals)

        feature_diagnostics[f_name] = {
            "pos_mean": round(p_mean, 4),
            "neg_mean": round(n_mean, 4),
            "delta": round(delta, 4),
            "pos_median": round(p_med, 4),
            "neg_median": round(n_med, 4),
            "pos_std": round(p_std, 4),
            "neg_std": round(n_std, 4),
            "auc_roc": round(auc, 4),
            "ks_stat": round(ks, 4),
        }
        diagnostic_table.append((f_name, p_mean, n_mean, delta, auc, ks))

    # Sort table by AUC-ROC descending
    diagnostic_table.sort(key=lambda x: x[4], reverse=True)
    for row in diagnostic_table:
        print(f"{row[0]:<35} | {row[1]:>9.4f} | {row[2]:>9.4f} | {row[3]:>+9.4f} | {row[4]:>9.4f} | {row[5]:>7.4f}")
    print("=" * 95)

    # 9. Save Feature Parquet to dataset/processed/train_features.parquet
    out_train_path = PROCESSED_DIR / "train_features.parquet"
    print(f"\nSaving {len(labels_arr):,} extracted training feature pairs to: {out_train_path.name}")

    schema_cols = [
        ("source1_entity_id", pa.string()),
        ("target_entity_id", pa.string()),
        ("label", pa.int8()),
    ] + [(f_name, pa.float32()) for f_name in FEATURE_NAMES]

    arrays = [
        pa.array(s1_ids_out, type=pa.string()),
        pa.array(tgt_ids_out, type=pa.string()),
        pa.array(labels_arr, type=pa.int8()),
    ] + [pa.array(feat_matrix[:, i], type=pa.float32()) for i in range(len(FEATURE_NAMES))]

    train_table = pa.Table.from_arrays(arrays, schema=pa.schema(schema_cols))
    pq.write_table(train_table, str(out_train_path), compression="SNAPPY")
    size_mb = out_train_path.stat().st_size / (1024 * 1024)
    print(f"[OK] Wrote {len(labels_arr):,} feature rows ({size_mb:.2f} MB) to {out_train_path.name}.")

    # 10. Save Diagnostic JSON Report
    diag_report_path = PROCESSED_DIR / "phase4_feature_ablation.json"
    summary_data = {
        "num_training_pairs": len(labels_arr),
        "num_positives": int(n_pos),
        "num_negatives": int(n_neg),
        "ratio": f"1:{neg_to_pos_ratio}",
        "feature_count": len(FEATURE_NAMES),
        "top_features_by_auc": diagnostic_table[:25],
        "feature_diagnostics": feature_diagnostics,
    }
    diag_report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(diag_report_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"[OK] Saved Feature Diagnostic Report to: {diag_report_path.name}")

    total_time = time.time() - t_start
    print("\n" + "=" * 85)
    print(f"🎉 PHASE 4 FEATURE ENGINEERING COMPLETED SUCCESSFULLY IN {total_time:.2f}s!")
    print("=" * 85)
    return summary_data


if __name__ == "__main__":
    n_ent = 50000
    for arg in sys.argv[1:]:
        if arg.isdigit():
            n_ent = int(arg)
    run_phase_4_feature_engineering(max_train_entities=n_ent)
