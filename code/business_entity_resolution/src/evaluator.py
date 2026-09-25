"""
Evaluation Module for Business Entity Resolution (Amazon ML Challenge 2026).
Computes exact Macro F_0.5 Score (beta=0.5), Precision, Recall, Candidate Recall,
and singleton breakdown.
"""

from typing import Dict, Set, Tuple
import numpy as np


def compute_f_beta(precision: float, recall: float, beta: float = 0.5) -> float:
    """Compute F_beta score with given beta (default 0.5)."""
    if precision + recall == 0:
        return 0.0
    beta_sq = beta ** 2
    return (1.0 + beta_sq) * (precision * recall) / (beta_sq * precision + recall)


def evaluate_entity_resolution(
    ground_truth: Dict[str, Set[str]],
    predictions: Dict[str, Set[str]],
    beta: float = 0.5
) -> Dict[str, float]:
    """
    Computes macro-averaged F_beta, Precision, Recall, and singleton metrics.

    Args:
        ground_truth: Dict mapping source1_entity_id -> set of true matched entity IDs.
        predictions: Dict mapping source1_entity_id -> set of predicted matched entity IDs.
        beta: F-score weight (0.5 for precision-heavy weighting).

    Returns:
        Dict of evaluation metrics.
    """
    total_entities = len(ground_truth)
    if total_entities == 0:
        return {"macro_f05": 0.0, "macro_precision": 0.0, "macro_recall": 0.0}

    f_scores = []
    precisions = []
    recalls = []

    singleton_correct = 0
    singleton_total = 0

    non_singleton_f_scores = []

    for s1_id, true_set in ground_truth.items():
        pred_set = predictions.get(s1_id, set())

        # Singleton Case (0 true matches)
        if len(true_set) == 0:
            singleton_total += 1
            if len(pred_set) == 0:
                # Correctly predicted empty
                f_scores.append(1.0)
                precisions.append(1.0)
                recalls.append(1.0)
                singleton_correct += 1
            else:
                # False positive on singleton
                f_scores.append(0.0)
                precisions.append(0.0)
                recalls.append(1.0)
            continue

        # Non-Singleton Case (> 0 true matches)
        if len(pred_set) == 0:
            # Missed all matches
            f_scores.append(0.0)
            precisions.append(0.0)
            recalls.append(0.0)
            non_singleton_f_scores.append(0.0)
            continue

        tp = len(pred_set & true_set)
        p = tp / len(pred_set)
        r = tp / len(true_set)

        precisions.append(p)
        recalls.append(r)

        if tp == 0:
            f = 0.0
        else:
            f = compute_f_beta(p, r, beta=beta)

        f_scores.append(f)
        non_singleton_f_scores.append(f)

    macro_f05 = float(np.mean(f_scores))
    macro_precision = float(np.mean(precisions))
    macro_recall = float(np.mean(recalls))

    singleton_accuracy = (singleton_correct / singleton_total) if singleton_total > 0 else 1.0
    non_singleton_macro_f05 = float(np.mean(non_singleton_f_scores)) if non_singleton_f_scores else 0.0

    return {
        "macro_f05": round(macro_f05, 5),
        "macro_precision": round(macro_precision, 5),
        "macro_recall": round(macro_recall, 5),
        "singleton_accuracy": round(singleton_accuracy, 5),
        "singleton_count": singleton_total,
        "non_singleton_macro_f05": round(non_singleton_macro_f05, 5),
        "total_evaluated_entities": total_entities,
    }


def evaluate_blocking_recall(
    ground_truth: Dict[str, Set[str]],
    candidates: Dict[str, Set[str]]
) -> Dict[str, float]:
    """
    Evaluates Candidate Generation (Blocking) quality:
    - Candidate Recall (upper bound of recall)
    - Average Candidates per Entity
    - Reduction Ratio
    """
    total_true_pairs = sum(len(ids) for ids in ground_truth.values())
    captured_true_pairs = 0
    total_candidates = sum(len(ids) for ids in candidates.values())
    total_s1 = len(ground_truth)

    for s1_id, true_set in ground_truth.items():
        if not true_set:
            continue
        cand_set = candidates.get(s1_id, set())
        captured_true_pairs += len(true_set & cand_set)

    candidate_recall = (captured_true_pairs / total_true_pairs) if total_true_pairs > 0 else 1.0
    avg_candidates = total_candidates / total_s1 if total_s1 > 0 else 0.0

    return {
        "candidate_recall": round(candidate_recall, 5),
        "captured_true_pairs": captured_true_pairs,
        "total_true_pairs": total_true_pairs,
        "avg_candidates_per_s1": round(avg_candidates, 2),
        "total_candidates": total_candidates,
    }
