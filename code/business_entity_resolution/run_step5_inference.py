"""
Amazon ML Challenge 2026: Execution Entrypoint for Step 5.
Run this script to perform Full Test Inference across US, India, and France,
generate matching_results.tsv and candidate_pairs.tsv, and run validate_submission.py.
"""

import sys
import argparse
from pathlib import Path

# Add current directory to path
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from src.inference import run_full_inference

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 5: Full Test Inference & Submission Generation")
    parser.add_argument("--force", "-f", action="store_true", help="Force recomputation of all countries (ignore existing checkpoints)")
    parser.add_argument("--threshold", "-t", type=float, default=None, help="Decision threshold override (e.g. 0.74)")
    parser.add_argument("--workers", "-w", type=int, default=6, help="Number of parallel worker processes (default: 6)")
    parser.add_argument("--batch-size", "-b", type=int, default=100000, help="Batch size for entity streaming (default: 100,000)")
    args = parser.parse_args()

    print(f"Launching Step 5: Full Test Inference & Submission Generation (force={args.force}, threshold={args.threshold}, workers={args.workers})...")
    run_full_inference(
        threshold_override=args.threshold,
        batch_size=args.batch_size,
        num_workers=args.workers,
        force_recompute=args.force
    )
