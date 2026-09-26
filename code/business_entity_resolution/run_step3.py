"""
Amazon ML Challenge 2026: Execution Entrypoint for Step 3.
Run this script to perform Vectorized Pairwise Feature Engineering
on candidate pairs using RapidFuzz C++ SIMD metrics.
"""

import sys
from pathlib import Path

# Add current directory to path
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from src.feature_builder import run_feature_pipeline

if __name__ == "__main__":
    print("Launching Step 3: Vectorized Pairwise Feature Engineering...")
    run_feature_pipeline()
