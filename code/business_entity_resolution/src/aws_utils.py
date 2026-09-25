"""
AWS Utilities Module for Amazon ML Challenge 2026.
Provides S3 sync, upload, download, and listing functions using boto3.
"""

import os
import sys
from pathlib import Path
from typing import List, Optional
import boto3
from botocore.exceptions import ClientError, NoCredentialsError

from .config import AWS_REGION, S3_BUCKET


def get_s3_client(region_name: str = AWS_REGION):
    """Initialize and return a boto3 S3 client."""
    return boto3.client("s3", region_name=region_name)


def list_s3_objects(prefix: str, bucket: str = S3_BUCKET) -> List[str]:
    """List all object keys under a given prefix in the S3 bucket."""
    s3 = get_s3_client()
    paginator = s3.get_paginator("list_objects_v2")
    keys = []
    try:
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                keys.append(obj["Key"])
    except (NoCredentialsError, ClientError) as e:
        print(f"[AWS Warning] Unable to list S3 objects at s3://{bucket}/{prefix}: {e}")
    return keys


def upload_file_to_s3(local_path: Path, s3_key: str, bucket: str = S3_BUCKET) -> bool:
    """Upload a single file to AWS S3."""
    s3 = get_s3_client()
    try:
        print(f"Uploading {local_path} -> s3://{bucket}/{s3_key}...")
        s3.upload_file(str(local_path), bucket, s3_key)
        print("Upload complete.")
        return True
    except Exception as e:
        print(f"[AWS Error] Failed to upload {local_path} to S3: {e}")
        return False


def download_file_from_s3(s3_key: str, local_path: Path, bucket: str = S3_BUCKET) -> bool:
    """Download a single file from AWS S3."""
    s3 = get_s3_client()
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        print(f"Downloading s3://{bucket}/{s3_key} -> {local_path}...")
        s3.download_file(bucket, s3_key, str(local_path))
        print("Download complete.")
        return True
    except Exception as e:
        print(f"[AWS Error] Failed to download {s3_key} from S3: {e}")
        return False


def sync_directory_to_s3(local_dir: Path, s3_prefix: str, bucket: str = S3_BUCKET):
    """Recursively upload all files from a local directory to an S3 prefix."""
    local_dir = Path(local_dir)
    s3 = get_s3_client()
    for root, _, files in os.walk(local_dir):
        for file in files:
            local_file = Path(root) / file
            rel_path = local_file.relative_to(local_dir).as_posix()
            s3_key = f"{s3_prefix.rstrip('/')}/{rel_path}"
            print(f"Syncing: {rel_path} -> s3://{bucket}/{s3_key}")
            try:
                s3.upload_file(str(local_file), bucket, s3_key)
            except Exception as e:
                print(f"Failed to upload {local_file}: {e}")
