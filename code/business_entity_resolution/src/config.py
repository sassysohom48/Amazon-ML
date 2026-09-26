"""
Configuration Module for Business Entity Resolution (Amazon ML Challenge 2026).
Defines paths, AWS S3 buckets, hyperparameters, and feature configurations.
"""

import os
from pathlib import Path

# Base Directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent  # Amazon-ML/
DATASET_DIR = PROJECT_ROOT / "dataset"
DATA_DIR = DATASET_DIR
RAW_DIR = DATASET_DIR / "raw"
TRAIN_DIR = RAW_DIR / "train"
TEST_DIR = RAW_DIR / "test"
OUTPUT_DIR = PROJECT_ROOT / "output"
PROCESSED_DIR = DATASET_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "code" / "business_entity_resolution" / "models"
VALIDATION_SCRIPT = DATASET_DIR / "utils" / "validate_submission.py"
SUBMISSION_MATCHING_TSV = OUTPUT_DIR / "matching_results.tsv"
SUBMISSION_CANDIDATE_TSV = OUTPUT_DIR / "candidate_pairs.tsv"

# Fallback check if raw folder exists or direct train folder
if not TRAIN_DIR.exists() and (DATASET_DIR / "train").exists():
    TRAIN_DIR = DATASET_DIR / "train"
    TEST_DIR = DATASET_DIR / "test"

# Ensure dirs exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Local TSV Paths
TRAIN_S1 = TRAIN_DIR / "train_source1.tsv"
TRAIN_S2 = TRAIN_DIR / "train_source2.tsv"
TRAIN_S3 = TRAIN_DIR / "train_source3.tsv"
TRAIN_GT = TRAIN_DIR / "train_ground_truth.tsv"

TEST_S1 = TEST_DIR / "test_source1.tsv"
TEST_S2 = TEST_DIR / "test_source2.tsv"
TEST_S3 = TEST_DIR / "test_source3.tsv"

OUTPUT_MATCHING = OUTPUT_DIR / "matching_results.tsv"
OUTPUT_CANDIDATES = OUTPUT_DIR / "candidate_pairs.tsv"

# AWS S3 Cloud Configuration
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
S3_BUCKET = os.getenv("S3_BUCKET", "amazon-sagemaker-720682844180-us-east-1-axi666gcrhqjon")
S3_BASE_PREFIX = "shared/dataset"
S3_RAW_PREFIX = f"{S3_BASE_PREFIX}/raw"
S3_PROCESSED_PREFIX = f"{S3_BASE_PREFIX}/processed"
S3_CANDIDATES_PREFIX = f"{S3_BASE_PREFIX}/candidates"
S3_FEATURES_PREFIX = f"{S3_BASE_PREFIX}/features"
S3_MODELS_PREFIX = f"{S3_BASE_PREFIX}/models"
S3_OUTPUT_PREFIX = f"{S3_BASE_PREFIX}/output"

# S3 Full URIs
S3_RAW_TRAIN_URI = f"s3://{S3_BUCKET}/{S3_RAW_PREFIX}/train"
S3_RAW_TEST_URI = f"s3://{S3_BUCKET}/{S3_RAW_PREFIX}/test"
S3_OUTPUT_URI = f"s3://{S3_BUCKET}/{S3_OUTPUT_PREFIX}"

# Validation Split Settings
VAL_FRACTION = 0.10
RANDOM_SEED = 42

# Evaluation & Thresholding
BETA = 0.5  # F_0.5 metric: precision weighted 2x over recall
DEFAULT_THRESHOLD = 0.65
SINGLETON_THRESHOLD = 0.50

# Blocking Settings
MAX_CANDIDATES_PER_S1 = 15
BLOCKING_NGRAM_SIZE = 3
