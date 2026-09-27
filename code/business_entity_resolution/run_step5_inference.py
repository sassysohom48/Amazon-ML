"""
Amazon ML Challenge 2026: Execution Entrypoint for Step 5.
Run this script to perform Full Test Inference across US, India, and France,
generate matching_results.tsv and candidate_pairs.tsv, and run validate_submission.py.
"""

import sys
from pathlib import Path

# Add current directory to path
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from src.inference import run_full_inference

if __name__ == "__main__":
    force = "--force" in sys.argv or "-f" in sys.argv
    print(f"Launching Step 5: Full Test Inference & Submission Generation (force_recompute={force})...")
    run_full_inference(force_recompute=force)
