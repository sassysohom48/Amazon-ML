"""
S3 Dataset & Artifact Synchronization Tool (Amazon ML Challenge 2026).
Easily sync raw datasets, processed parquets, models, and submission files with AWS S3.
"""

import os
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (
    S3_BUCKET,
    PROJECT_ROOT,
    DATASET_DIR,
    PROCESSED_DIR,
    OUTPUT_DIR,
    MODELS_DIR,
    TRAIN_DIR,
    TEST_DIR,
)
from src.aws_utils import (
    get_s3_client,
    upload_file_to_s3,
    download_file_from_s3,
    sync_directory_to_s3,
    list_s3_objects,
)


def pull_raw_from_s3():
    """Download raw train and test TSV files from S3 to local dataset/."""
    print(f"\n--- Pulling Raw Data from s3://{S3_BUCKET}/shared/dataset/raw/ ---")
    s3 = get_s3_client()
    prefix = "shared/dataset/raw/"
    keys = list_s3_objects(prefix)
    if not keys:
        # Try alternate prefix
        keys = list_s3_objects("dataset/raw/")
        prefix = "dataset/raw/"

    print(f"Found {len(keys)} raw data files on S3.")
    for key in keys:
        if key.endswith("/"):
            continue
        rel_path = key[len(prefix):] if key.startswith(prefix) else key
        local_target = DATASET_DIR / rel_path
        download_file_from_s3(key, local_target)


def upload_processed_to_s3():
    """Upload preprocessed parquets to S3."""
    print(f"\n--- Uploading Processed Parquets to s3://{S3_BUCKET}/shared/dataset/processed/ ---")
    sync_directory_to_s3(PROCESSED_DIR, "shared/dataset/processed")


def download_processed_from_s3():
    """Download preprocessed parquets from S3."""
    print(f"\n--- Downloading Processed Parquets from s3://{S3_BUCKET}/shared/dataset/processed/ ---")
    prefix = "shared/dataset/processed/"
    keys = list_s3_objects(prefix)
    print(f"Found {len(keys)} processed files on S3.")
    for key in keys:
        if key.endswith("/"):
            continue
        filename = Path(key).name
        local_target = PROCESSED_DIR / filename
        download_file_from_s3(key, local_target)


def upload_submissions_to_s3():
    """Upload matching_results.tsv and candidate_pairs.tsv to S3."""
    print(f"\n--- Uploading Submissions to s3://{S3_BUCKET}/shared/dataset/output/ ---")
    sync_directory_to_s3(OUTPUT_DIR, "shared/dataset/output")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AWS S3 Sync Tool")
    parser.add_argument("--action", choices=["pull-raw", "push-processed", "pull-processed", "push-output"], required=True)
    args = parser.parse_args()

    if args.action == "pull-raw":
        pull_raw_from_s3()
    elif args.action == "push-processed":
        upload_processed_to_s3()
    elif args.action == "pull-processed":
        download_processed_from_s3()
    elif args.action == "push-output":
        upload_submissions_to_s3()
