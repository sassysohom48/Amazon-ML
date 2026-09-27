"""
Step 1.2: Multi-Tier Diagnostic Evaluation Engine (Amazon ML Challenge 2026).
Computes exact Macro F0.5, candidate-recall curves at K=5..150, Oracle blocker ceilings,
loss attribution decomposition, and dumps categorized error diagnostics.
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
import polars as pl
import numpy as np

# Ensure package import works
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PROCESSED_DIR, OUTPUT_DIR
from src.validation_contract import compute_entity_f05, METRIC_BETA


class DiagnosticEvaluator:
    """
    Comprehensive Diagnostic Evaluation Engine.
    Evaluates Candidate Generation (Phase 3) and Matching Models (Phase 4/5).
    """

    def __init__(self, ground_truth_path: Optional[Path] = None):
        if ground_truth_path is None:
            if (PROCESSED_DIR / "train_ground_truth.parquet").exists():
                ground_truth_path = PROCESSED_DIR / "train_ground_truth.parquet"
            elif (TRAIN_DIR / "train_ground_truth.tsv").exists():
                ground_truth_path = TRAIN_DIR / "train_ground_truth.tsv"
            else:
                ground_truth_path = PROCESSED_DIR / "val_ground_truth.parquet"

        self.gt_path = ground_truth_path
        self.ground_truth: Dict[str, Set[str]] = {}
        self._load_ground_truth()

    def _load_ground_truth(self):
        """Loads ground truth into memory mapping: s1_id -> set of true target IDs."""
        print(f"Loading Ground Truth from {self.gt_path}...")
        t0 = time.time()
        
        if str(self.gt_path).endswith(".parquet"):
            df = pl.read_parquet(self.gt_path)
        else:
            df = pl.read_csv(self.gt_path, separator="\t")

        s1_col = "source1_entity_id" if "source1_entity_id" in df.columns else df.columns[0]
        tgt_col = "matched_entity_id" if "matched_entity_id" in df.columns else ("matching_entity_ids" if "matching_entity_ids" in df.columns else df.columns[1])

        sample_val = str(df[tgt_col][0])
        if "," in sample_val or len(sample_val) > 20:
            for s1_id, tgt_str in zip(df[s1_col], df[tgt_col]):
                self.ground_truth[str(s1_id)] = set(x.strip() for x in str(tgt_str).split(",") if x.strip())
        else:
            grouped = df.group_by(s1_col).agg(pl.col(tgt_col))
            for s1_id, tgts in zip(grouped[s1_col], grouped[tgt_col]):
                self.ground_truth[str(s1_id)] = set(str(t) for t in tgts)

        print(f"Loaded {len(self.ground_truth):,} ground truth entities in {time.time() - t0:.2f}s")

    def evaluate_candidate_recall_curve(
        self,
        candidate_map: Dict[str, List[str]],
        k_values: List[int] = [5, 10, 15, 20, 25, 35, 50, 75, 100, 150],
        output_path: Optional[Path] = OUTPUT_DIR / "blocking_recall_curve.parquet",
    ) -> pl.DataFrame:
        """
        Computes Candidate Recall and Candidate Precision across a spectrum of K thresholds.
        """
        print("\n" + "=" * 75)
        print("EVALUATING CANDIDATE RECALL & PRECISION CURVE (Recall@K)")
        print("=" * 75)

        total_s1 = len(candidate_map)
        true_entities = [s1 for s1 in candidate_map if s1 in self.ground_truth and len(self.ground_truth[s1]) > 0]
        total_true_pairs = sum(len(self.ground_truth[s1]) for s1 in true_entities)

        rows = []
        print(f"Evaluated across {len(true_entities):,} non-singleton entities ({total_true_pairs:,} true pairs total):\n")
        print(f"  {'K':>5} | {'Candidate Recall':>18} | {'Pairs Captured':>15} | {'Pairs Missed':>14} | {'Blocker Prec':>14}")
        print("  " + "-" * 75)

        for k in k_values:
            captured_pairs = 0
            total_candidates_generated = 0

            for s1 in true_entities:
                cands_k = set(candidate_map[s1][:k])
                true_k = self.ground_truth[s1]
                captured_pairs += len(cands_k & true_k)
                total_candidates_generated += len(cands_k)

            cand_recall = captured_pairs / total_true_pairs if total_true_pairs > 0 else 0.0
            cand_precision = captured_pairs / total_candidates_generated if total_candidates_generated > 0 else 0.0
            missed = total_true_pairs - captured_pairs

            print(f"  {k:>5} | {cand_recall*100:>17.2f}% | {captured_pairs:>15,} | {missed:>14,} | {cand_precision*100:>13.2f}%")

            rows.append({
                "k": k,
                "candidate_recall": cand_recall,
                "candidate_precision": cand_precision,
                "pairs_captured": captured_pairs,
                "pairs_missed": missed,
                "total_true_pairs": total_true_pairs,
            })

        curve_df = pl.DataFrame(rows)
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            curve_df.write_parquet(output_path, compression="snappy")
            print(f"\n[OK] Exported candidate recall curve to: {output_path}")

        return curve_df

    def evaluate_predictions(
        self,
        predicted_map: Dict[str, List[str]],
        candidate_map: Optional[Dict[str, List[str]]] = None,
        entity_metadata_df: Optional[pl.DataFrame] = None,
        output_error_path: Optional[Path] = OUTPUT_DIR / "error_diagnostics.parquet",
    ) -> Dict:
        """
        Full multi-tier evaluation: Macro F0.5, country breakdowns, oracle ceiling,
        loss attribution decomposition, and categorized error diagnostics dump.
        """
        print("\n" + "=" * 75)
        print("DIAGNOSTIC MODEL EVALUATION & LOSS ATTRIBUTION")
        print("=" * 75)

        all_s1_ids = list(predicted_map.keys())
        total_entities = len(all_s1_ids)

        f05_scores = []
        precision_scores = []
        recall_scores = []

        # Loss Attribution Trackers
        oracle_f05_scores = []
        blocking_misses = 0
        classifier_fn_count = 0
        classifier_fp_count = 0
        singleton_fp_count = 0
        total_singletons = 0
        correct_singletons = 0

        # Detailed Error Records
        error_records = []

        for s1_id in all_s1_ids:
            true_set = self.ground_truth.get(s1_id, set())
            pred_set = set(predicted_map.get(s1_id, []))
            cand_set = set(candidate_map.get(s1_id, [])) if candidate_map else pred_set

            # Primary Entity Score
            f05, prec, rec = compute_entity_f05(true_set, pred_set)
            f05_scores.append(f05)
            precision_scores.append(prec)
            recall_scores.append(rec)

            # Oracle Blocker Score (If classifier was 100% perfect on candidate set)
            oracle_pred = true_set & cand_set
            o_f05, _, _ = compute_entity_f05(true_set, oracle_pred)
            oracle_f05_scores.append(o_f05)

            # Singleton Tracking
            if not true_set:
                total_singletons += 1
                if not pred_set:
                    correct_singletons += 1
                else:
                    singleton_fp_count += len(pred_set)
                    error_records.append({
                        "source1_entity_id": s1_id,
                        "error_type": "SINGLETON_FALSE_POSITIVE",
                        "predicted_ids": ",".join(pred_set),
                        "true_ids": "",
                        "f05_score": 0.0,
                    })

            # Non-singleton error attribution
            if true_set:
                # Blocker Misses
                missed_by_blocker = true_set - cand_set
                if missed_by_blocker:
                    blocking_misses += len(missed_by_blocker)
                    error_records.append({
                        "source1_entity_id": s1_id,
                        "error_type": "BLOCKER_DROP_MISS",
                        "predicted_ids": ",".join(pred_set),
                        "true_ids": ",".join(missed_by_blocker),
                        "f05_score": f05,
                    })

                # Classifier False Negatives (In candidates, but rejected by classifier)
                missed_by_classifier = (true_set & cand_set) - pred_set
                if missed_by_classifier:
                    classifier_fn_count += len(missed_by_classifier)

                # Classifier False Positives (Incorrect merge)
                wrong_merges = pred_set - true_set
                if wrong_merges:
                    classifier_fp_count += len(wrong_merges)
                    error_records.append({
                        "source1_entity_id": s1_id,
                        "error_type": "CLASSIFIER_FALSE_POSITIVE",
                        "predicted_ids": ",".join(wrong_merges),
                        "true_ids": ",".join(true_set),
                        "f05_score": f05,
                    })

        # Aggregated Metrics
        macro_f05 = float(np.mean(f05_scores))
        macro_prec = float(np.mean(precision_scores))
        macro_rec = float(np.mean(recall_scores))
        oracle_macro_f05 = float(np.mean(oracle_f05_scores))
        singleton_acc = (correct_singletons / total_singletons) if total_singletons > 0 else 1.0

        # Loss Attribution Breakdown
        gap_to_perfection = 1.0 - macro_f05
        blocker_gap = 1.0 - oracle_macro_f05
        classifier_gap = oracle_macro_f05 - macro_f05

        summary = {
            "evaluated_entities": total_entities,
            "macro_f05": round(macro_f05, 5),
            "macro_precision": round(macro_prec, 5),
            "macro_recall": round(macro_rec, 5),
            "oracle_blocker_ceiling_f05": round(oracle_macro_f05, 5),
            "singleton_accuracy": round(singleton_acc, 5),
            "loss_attribution": {
                "total_score_deficit": round(gap_to_perfection, 5),
                "delta_lost_to_blocking": round(blocker_gap, 5),
                "delta_lost_to_classification": round(classifier_gap, 5),
            },
            "error_counts": {
                "blocking_dropped_pairs": blocking_misses,
                "classifier_false_negatives": classifier_fn_count,
                "classifier_false_positives": classifier_fp_count,
                "singleton_false_positives": singleton_fp_count,
            }
        }

        # Print Executive Summary
        print(f"  • Macro F0.5 Score:             {macro_f05:.4f}")
        print(f"  • Macro Precision:              {macro_prec*100:.2f}%")
        print(f"  • Macro Recall:                 {macro_rec*100:.2f}%")
        print(f"  • Oracle Blocker Ceiling (Max): {oracle_macro_f05:.4f}")
        print(f"  • Singleton Accuracy:           {singleton_acc*100:.2f}% ({correct_singletons:,}/{total_singletons:,})")
        print("\n[Loss Attribution Decomposition]:")
        print(f"  • Lost to Blocker Drops:        -{blocker_gap:.4f} ({(blocker_gap/gap_to_perfection)*100:.1f}% of total deficit)")
        print(f"  • Lost to Classifier Errors:    -{classifier_gap:.4f} ({(classifier_gap/gap_to_perfection)*100:.1f}% of total deficit)")

        # Export Error Diagnostics Parquet
        if output_error_path and error_records:
            output_error_path.parent.mkdir(parents=True, exist_ok=True)
            err_df = pl.DataFrame(error_records)
            err_df.write_parquet(output_error_path, compression="snappy")
            print(f"\n[OK] Exported {len(err_df):,} diagnostic error records to: {output_error_path}")

        return summary


if __name__ == "__main__":
    evaluator = DiagnosticEvaluator()
    print("DiagnosticEvaluator initialized successfully.")
