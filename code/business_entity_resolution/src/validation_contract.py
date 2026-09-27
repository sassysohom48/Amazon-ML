"""
Step 1.0: Validation Contract & Standardized Metric Suite (Amazon ML Challenge 2026).
Defines canonical protocol, evaluation formulas, leakage rules, and stratification parameters.
"""

import json
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
import numpy as np
import polars as pl

# Canonical Protocol Constants
RANDOM_SEED = 42
NUM_FOLDS = 5
METRIC_BETA = 0.5  # Precision weighted 2x over Recall
CARDINALITY_BINS = ["singleton", "1_to_1", "2_to_5", "6_plus"]
COUNTRY_PARTITIONS = ["US", "India", "France"]

# Contract Metadata Schema
VALIDATION_CONTRACT = {
    "competition": "Amazon ML Challenge 2026 — Multilingual Business Entity Resolution",
    "protocol_version": "2.0_rigorous",
    "random_seed": RANDOM_SEED,
    "num_folds": NUM_FOLDS,
    "primary_metric": "Macro F0.5",
    "metric_formula": "1.25 * Precision * Recall / (0.25 * Precision + Recall)",
    "singleton_rule": "1.0 if empty string predicted; 0.0 if any false positive",
    "country_invariant": "Strict 0% cross-country matching rule",
    "stratification_factors": ["country", "match_cardinality_bin", "has_address"],
    "leakage_protocol": {
        "s1_split": "K-Fold disjoint partition on Source 1 entities",
        "target_pool": "Full target records (S2 + S3) accessible within matching country",
        "evaluation_rule": "Every S1 entity must be scored against its country's target candidate pool",
    },
}


def compute_entity_f05(
    true_matches: Set[str],
    predicted_matches: Set[str],
    beta: float = METRIC_BETA,
) -> Tuple[float, float, float]:
    """
    Computes exact precision, recall, and F_0.5 score for a single S1 entity.
    Strict Singleton Rule: If true_matches is empty:
      - Returns (1.0, 1.0, 1.0) if predicted_matches is empty.
      - Returns (0.0, 0.0, 0.0) if predicted_matches is non-empty (singleton penalty).
    """
    if not true_matches:
        if not predicted_matches:
            return 1.0, 1.0, 1.0
        else:
            return 0.0, 0.0, 0.0

    if not predicted_matches:
        return 0.0, 0.0, 0.0

    tp = len(true_matches & predicted_matches)
    precision = tp / len(predicted_matches)
    recall = tp / len(true_matches)

    if precision + recall == 0 or tp == 0:
        return 0.0, precision, recall

    beta_sq = beta ** 2
    f05 = (1.0 + beta_sq) * (precision * recall) / ((beta_sq * precision) + recall)
    return float(f05), float(precision), float(recall)


def export_validation_contract(output_path: Path):
    """Exports the validation contract specification to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(VALIDATION_CONTRACT, f, indent=2)
    print(f"[OK] Exported Validation Contract to: {output_path}")


if __name__ == "__main__":
    from src.config import PROCESSED_DIR
    export_validation_contract(PROCESSED_DIR / "validation_contract.json")
