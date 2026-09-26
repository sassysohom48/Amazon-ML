"""
Phase 3 Master Runner: Multi-Channel Candidate Blocker & Diagnostic Evaluation.
Generates high-recall candidate pools with rich retrieval provenance across US, India, and France.
"""

import sys
import time
from pathlib import Path
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

# Ensure package import works
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PROCESSED_DIR
from src.blocking import MultiChannelBlocker
from src.ablation_blocking import evaluate_blocking_benchmark


def run_phase3_pipeline(
    max_k: int = 50,
    run_benchmark: bool = True,
    generate_val_pairs: bool = True,
    generate_train_pairs: bool = False,
):
    print("=" * 85)
    print("🚀 EXECUTING PHASE 3: MULTI-CHANNEL CANDIDATE BLOCKER & PROVENANCE ENGINE")
    print("=" * 85)
    t_start = time.time()

    # Step 3.1: Candidate Blocker Benchmark & Oracle F0.5 Curve
    if run_benchmark:
        print("\n[1/3] RUNNING CANDIDATE BLOCKER BENCHMARK & ORACLE EVALUATOR (Step 3.1)...")
        benchmark_json = PROCESSED_DIR / "phase3_blocking_ablation.json"
        evaluate_blocking_benchmark(
            sample_size=30000,
            k_list=[5, 10, 15, 20, 25, 35, 50, 75, 100],
            output_json_path=benchmark_json,
        )

    # Step 3.2: Generate Fold 0 Validation Candidate Pairs
    if generate_val_pairs:
        print("\n[2/3] GENERATING FOLD 0 VALIDATION CANDIDATE PAIRS WITH PROVENANCE (Step 3.2)...")
        val_output_path = PROCESSED_DIR / "val_candidate_pairs.parquet"
        
        # Load Fold 0 Entities
        folds_path = PROCESSED_DIR / "train_folds.parquet"
        s1_path = PROCESSED_DIR / "train_source1_cleaned.parquet"
        s2_path = PROCESSED_DIR / "train_source2_cleaned.parquet"
        s3_path = PROCESSED_DIR / "train_source3_cleaned.parquet"

        cols_to_load = [
            "entity_id", "country", "name_core", "name_tokens", "name_acronym",
            "name_phonetic", "addr_clean", "addr_tokens", "addr_digits", "addr_unit_num", "postal_clean"
        ]

        if folds_path.exists():
            print("Loading Fold 0 validation entities...")
            folds_df = pl.read_parquet(folds_path, columns=["entity_id", "fold_id"])
            val_eids = set(folds_df.filter(pl.col("fold_id") == 0)["entity_id"].to_list())
            s1_val = pl.read_parquet(s1_path, columns=cols_to_load).filter(pl.col("entity_id").is_in(val_eids))
        else:
            print("Loading sample validation entities...")
            s1_val = pl.read_parquet(s1_path, columns=cols_to_load).head(100000)

        s2_all = pl.read_parquet(s2_path, columns=cols_to_load)
        s3_all = pl.read_parquet(s3_path, columns=cols_to_load)

        blocker = MultiChannelBlocker(max_candidates=max_k)
        blocker.fit(s2_all, s3_all)

        print(f"\nGenerating candidate pool for {len(s1_val):,} validation entities (K = {max_k})...")
        val_cand_dict = blocker.block_dataframe(s1_val, max_k=max_k)
        
        print("Converting candidate dictionary to Polars DataFrame with provenance...")
        val_cand_df = blocker.convert_candidates_to_dataframe(val_cand_dict)
        val_cand_df.write_parquet(val_output_path, compression="snappy")
        
        size_mb = val_output_path.stat().st_size / (1024 * 1024)
        print(f"[OK] Generated {len(val_cand_df):,} validation candidate pairs ({size_mb:.2f} MB) at: {val_output_path}")

    total_elapsed = time.time() - t_start
    print("\n" + "=" * 85)
    print(f"🎉 PHASE 3 PIPELINE COMPLETED SUCCESSFULLY IN {total_elapsed:.2f}s!")
    print("=" * 85)
    print("Artifacts Created in dataset/processed/:")
    for fname in [
        "phase3_blocking_ablation.json",
        "val_candidate_pairs.parquet",
    ]:
        p = PROCESSED_DIR / fname
        if p.exists():
            mb = p.stat().st_size / (1024 * 1024)
            print(f"  • {fname:<35} ({mb:.2f} MB)")
    print("\nHigh-recall candidate foundation is ready for Phase 4 (Pairwise Feature Engineering)!")


if __name__ == "__main__":
    k = 50
    for arg in sys.argv:
        if arg.startswith("--k="):
            k = int(arg.split("=")[1])
    run_phase3_pipeline(max_k=k)
