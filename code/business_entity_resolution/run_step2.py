"""
Amazon ML Challenge 2026: Execution Entrypoint for Step 2.
Run this script to perform Vectorized Text Normalization, Hybrid Candidate Blocking,
and Validation Candidate Recall Evaluation.
"""

import sys
from pathlib import Path

# Add current directory to path
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from src.blocking import run_blocking_pipeline

if __name__ == "__main__":
    print("Launching Step 2: Vectorized Normalization & Hybrid Candidate Blocking...")
    run_blocking_pipeline(evaluate_on_val=True)
