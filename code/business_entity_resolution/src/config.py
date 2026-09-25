"""
Configuration Module for Business Entity Resolution (Amazon ML Challenge 2026).
Defines paths, AWS S3 buckets, hyperparameters, and feature configurations.
"""

import os
from pathlib import Path

# Intelligent Project Root & Dataset Directory Resolution
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent  # Amazon-ML/

# Check for nested dataset paths (handles both local repo and SageMaker environments)
if (PROJECT_ROOT / "dataset" / "dataset" / "train").exists():
    DATASET_DIR = PROJECT_ROOT / "dataset" / "dataset"
elif (PROJECT_ROOT / "dataset" / "train").exists():
    DATASET_DIR = PROJECT_ROOT / "dataset"
else:
    DATASET_DIR = Path(os.getenv("DATASET_DIR", str(PROJECT_ROOT / "dataset" / "dataset")))

TRAIN_DIR = DATASET_DIR / "train"
TEST_DIR = DATASET_DIR / "test"
OUTPUT_DIR = PROJECT_ROOT / "output"
PROCESSED_DIR = PROJECT_ROOT / "dataset" / "processed"

# Sub-directories for organized artifact management
PARQUET_DIR = PROCESSED_DIR / "parquet"
CANDIDATES_DIR = PROCESSED_DIR / "candidates"
FEATURES_DIR = PROCESSED_DIR / "features"
MODELS_DIR = PROCESSED_DIR / "models"

# Ensure output and processed dirs exist
for directory in [OUTPUT_DIR, PROCESSED_DIR, PARQUET_DIR, CANDIDATES_DIR, FEATURES_DIR, MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Raw TSV Input Paths
TRAIN_S1 = TRAIN_DIR / "train_source1.tsv"
TRAIN_S2 = TRAIN_DIR / "train_source2.tsv"
TRAIN_S3 = TRAIN_DIR / "train_source3.tsv"
TRAIN_GT = TRAIN_DIR / "train_ground_truth.tsv"

TEST_S1 = TEST_DIR / "test_source1.tsv"
TEST_S2 = TEST_DIR / "test_source2.tsv"
TEST_S3 = TEST_DIR / "test_source3.tsv"

# Final Competition Output Paths
OUTPUT_MATCHING = OUTPUT_DIR / "matching_results.tsv"
OUTPUT_CANDIDATES = OUTPUT_DIR / "candidate_pairs.tsv"

# AWS S3 Cloud Configuration (SageMaker & Distributed Scaling)
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
S3_BUCKET = os.getenv("S3_BUCKET", "amazon-sagemaker-720682844180-us-east-1-axi666gcrhqjon")
S3_BASE_PREFIX = "shared/dataset"
S3_RAW_PREFIX = f"{S3_BASE_PREFIX}/raw"
S3_PROCESSED_PREFIX = f"{S3_BASE_PREFIX}/processed"
S3_CANDIDATES_PREFIX = f"{S3_BASE_PREFIX}/candidates"
S3_FEATURES_PREFIX = f"{S3_BASE_PREFIX}/features"
S3_MODELS_PREFIX = f"{S3_BASE_PREFIX}/models"
S3_OUTPUT_PREFIX = f"{S3_BASE_PREFIX}/output"

# Validation Split Settings
VAL_FRACTION = 0.10
RANDOM_SEED = 42

# Candidate Generation & Blocking Settings
MAX_CANDIDATES_PER_S1 = 35
MINHASH_NUM_PERM = 64
MINHASH_THRESHOLD = 0.35
BM25_TOP_K = 20

# Evaluation & Thresholding
BETA = 0.5  # F_0.5 metric: precision weighted 2x over recall
DEFAULT_THRESHOLD = 0.70
SINGLETON_THRESHOLD = 0.55

