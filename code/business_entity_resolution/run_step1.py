"""
Amazon ML Challenge 2026: Execution Entrypoint for Step 1.
Run this script to perform High-Speed TSV Ingestion, Country Partitioning,
and Stratified Validation Splitting.
"""

import sys
from pathlib import Path

# Add current directory to path
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from src.data_processor import run_pipeline_step1

if __name__ == "__main__":
    print("Launching Step 1: Data Ingestion & Country Partitioning...")
    run_pipeline_step1()
