"""
Phase 1 Master Runner: Validation Contract, 5-Fold Stratification & Diagnostic Profiler.
Executes all Phase 1 components end-to-end via CLI for Amazon ML Challenge 2026.
"""

import sys
import time
from pathlib import Path

# Ensure package import works
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PROCESSED_DIR
from src.validation_contract import export_validation_contract
from src.split_generator import generate_stratified_folds
from src.profile_ground_truth import profile_ground_truth_signals
from src.french_robustness import generate_french_robustness_benchmark
from src.evaluator import DiagnosticEvaluator


def run_phase1_pipeline():
    print("=" * 80)
    print("🚀 EXECUTING PHASE 1: VALIDATION & DIAGNOSTIC FOUNDATION")
    print("=" * 80)
    t_start = time.time()

    # Step 1.0: Export Validation Contract
    print("\n[1/4] EXPORTING VALIDATION CONTRACT (Step 1.0)...")
    contract_path = PROCESSED_DIR / "validation_contract.json"
    export_validation_contract(contract_path)

    # Step 1.1: Generate 5-Fold Stratified Splits
    print("\n[2/4] GENERATING DETERMINISTIC 5-FOLD STRATIFIED SPLITS (Step 1.1)...")
    folds_path = PROCESSED_DIR / "train_folds.parquet"
    generate_stratified_folds(
        s1_path=PROCESSED_DIR / "train_source1_cleaned.parquet",
        output_path=folds_path,
        num_folds=5,
        random_state=42
    )

    # Step 1.3: Run Ground Truth Signal & Combination Profiler
    print("\n[3/4] PROFILING GROUND TRUTH SIGNALS & COMBINATIONS (Step 1.3)...")
    profile_json = PROCESSED_DIR / "gt_signal_profile.json"
    profile_ground_truth_signals(
        sample_size=50000,
        output_json=profile_json
    )

    # Step 1.4: Generate French Linguistic Robustness Benchmark
    print("\n[4/4] GENERATING FRENCH LINGUISTIC ROBUSTNESS BENCHMARK (Step 1.4)...")
    fr_path = PROCESSED_DIR / "val_synthetic_france.parquet"
    generate_french_robustness_benchmark(
        num_samples=15000,
        output_path=fr_path
    )

    # Initialize Diagnostic Evaluator Verification
    print("\n[OK] Initializing Diagnostic Evaluation Harness...")
    evaluator = DiagnosticEvaluator()

    total_elapsed = time.time() - t_start
    print("\n" + "=" * 80)
    print(f"🎉 PHASE 1 PIPELINE COMPLETED SUCCESSFULLY IN {total_elapsed:.2f}s!")
    print("=" * 80)
    print("Generated Artifacts in dataset/processed/:")
    print(f"  • {contract_path.name}")
    print(f"  • {folds_path.name}")
    print(f"  • {profile_json.name}")
    print(f"  • {fr_path.name}")
    print("\nThe validation foundation is 100% established. Ready for Phase 2!")


if __name__ == "__main__":
    run_phase1_pipeline()
