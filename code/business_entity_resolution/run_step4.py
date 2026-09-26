"""
Amazon ML Challenge 2026: Execution Entrypoint for Step 4.
Run this script to perform Supervised LightGBM Model Training,
Feature Importance Profiling, and Macro F0.5 Decision Threshold Optimization.
"""

import sys
from pathlib import Path

# Add current directory to path
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from src.trainer import run_training_pipeline

if __name__ == "__main__":
    print("Launching Step 4: Supervised Model Training & Threshold Sweep...")
    run_training_pipeline()
