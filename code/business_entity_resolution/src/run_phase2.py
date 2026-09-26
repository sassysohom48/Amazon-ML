"""
Phase 2 Master Runner: Multi-Representation Normalization & Structured Address Pipeline.
Executes high-throughput stream preprocessing and normalization ablation diagnostics for Amazon ML Challenge 2026.
"""

import sys
import time
from pathlib import Path
import polars as pl

# Ensure package import works
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PROCESSED_DIR
from src.preprocess_datasets import run_full_preprocessing
from src.ablation_normalization import evaluate_normalization_ablation


def run_phase2_pipeline(force_recompute: bool = True):
    print("=" * 80)
    print("🚀 EXECUTING PHASE 2: MULTI-REPRESENTATION NORMALIZATION & PARSING")
    print("=" * 80)
    t_start = time.time()

    # Step 2.1 - 2.3: Streamed Multi-Representation Preprocessing
    print("\n[1/2] RUNNING LOW-MEMORY STREAMING PREPROCESSING (Steps 2.1 - 2.3)...")
    run_full_preprocessing(force_recompute=force_recompute)

    # Step 2.4: Normalization Ablation & Collision Diagnostic Benchmark
    print("\n[2/2] RUNNING NORMALIZATION ABLATION & COLLISION BENCHMARK (Step 2.4)...")
    ablation_json = PROCESSED_DIR / "phase2_normalization_ablation.json"
    evaluate_normalization_ablation(sample_size=100000, output_path=ablation_json)

    total_elapsed = time.time() - t_start
    print("\n" + "=" * 80)
    print(f"🎉 PHASE 2 PIPELINE COMPLETED SUCCESSFULLY IN {total_elapsed:.2f}s!")
    print("=" * 80)
    print("Key Enriched Datasets Created in dataset/processed/:")
    for fname in [
        "train_source1_cleaned.parquet",
        "train_source2_cleaned.parquet",
        "train_source3_cleaned.parquet",
        "test_source1_cleaned.parquet",
        "test_source2_cleaned.parquet",
        "test_source3_cleaned.parquet",
        "phase2_normalization_ablation.json",
    ]:
        p = PROCESSED_DIR / fname
        if p.exists():
            size_mb = p.stat().st_size / (1024 * 1024)
            print(f"  • {fname:<35} ({size_mb:.2f} MB)")
    print("\nMulti-representation foundation is ready for Phase 3 (Candidate Blocker)!")


if __name__ == "__main__":
    force = "--no-force" not in sys.argv
    run_phase2_pipeline(force_recompute=force)
