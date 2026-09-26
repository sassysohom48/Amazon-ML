#!/usr/bin/env python3
"""
ML Challenge 2026 — Submission Validator & Integrity Checker.
Validates matching_results.tsv and candidate_pairs.tsv against competition rules.
Supports both raw test TSVs and processed test Parquet for flexible cloud validation.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Set, List, Optional, Tuple

DELIM = "\t"
MATCHING_HEADER = ["source1_entity_id", "matched_entity_ids"]
CANDIDATE_HEADER = ["source1_entity_id", "candidate_entity_ids"]


def load_required_s1_ids(test_dir: str) -> Optional[Set[str]]:
    """Loads all required S1 entity IDs from test_source1.tsv or test_source1_cleaned.parquet."""
    # 1. Try test_source1.tsv in test_dir
    tsv_candidates = [
        os.path.join(test_dir, "test_source1.tsv"),
        os.path.join(test_dir, "raw", "test", "test_source1.tsv"),
        "dataset/raw/test/test_source1.tsv",
        "dataset/test/test_source1.tsv",
    ]
    for p in tsv_candidates:
        if os.path.isfile(p):
            print(f"  Reading required S1 IDs from {p}...")
            with open(p, encoding="utf-8") as f:
                next(f, None)
                return {line.split(DELIM, 1)[0].strip() for line in f if line.strip()}

    # 2. Try Parquet if TSV not found
    parquet_candidates = [
        "dataset/processed/test_source1_cleaned.parquet",
        os.path.join(test_dir, "processed", "test_source1_cleaned.parquet"),
    ]
    for p in parquet_candidates:
        if os.path.isfile(p):
            try:
                import polars as pl
                print(f"  Reading required S1 IDs from {p}...")
                df = pl.read_parquet(p, columns=["entity_id"])
                return set(df["entity_id"].to_list())
            except Exception:
                pass

    return None


def validate_tsv_file(
    path: str,
    expected_header: List[str],
    col_label: str,
    required_s1: Optional[Set[str]],
    errors: List[str],
    warnings: List[str],
) -> Optional[Dict[str, Set[str]]]:
    if not os.path.isfile(path):
        errors.append(f"File not found: {path}")
        return None

    name = os.path.basename(path)
    mapping: Dict[str, Set[str]] = {}
    seen = set()
    dup_rows = set()
    intra_dupes = set()
    self_matches = set()
    wrong_prefix = set()
    n_rows = 0
    empties = 0

    with open(path, "r", encoding="utf-8") as f:
        header = f.readline()
        if not header:
            errors.append(f"{name} is empty.")
            return None

        if DELIM not in header and "," in header:
            errors.append(f"{name}: header is comma-separated instead of tab-separated.")
            return None

        cols = [c.strip().lower() for c in header.rstrip("\r\n").split(DELIM)]
        if cols != expected_header:
            errors.append(f"{name}: unexpected header {cols}. Expected exactly {expected_header}")
            return None

        for line_num, line in enumerate(f, start=2):
            s1, tab, rest = line.partition(DELIM)
            if not tab:
                if s1.strip():
                    errors.append(f"{name}: malformed row at line {line_num}")
                continue

            n_rows += 1
            if s1 in seen:
                dup_rows.add(s1)
            seen.add(s1)

            ids = [x.strip() for x in rest.rstrip("\r\n").split(",") if x.strip()]
            if not ids:
                empties += 1
                mapping[s1] = set()
                continue

            if len(ids) != len(set(ids)):
                intra_dupes.add(s1)

            id_set = set(ids)
            mapping[s1] = id_set

            for mid in id_set:
                if mid.startswith("S1-"):
                    self_matches.add(mid)
                elif not mid.startswith(("S2-", "S3-")):
                    wrong_prefix.add(mid)

    if dup_rows:
        errors.append(f"{name}: contains {len(dup_rows)} duplicate source1_entity_id rows.")
    if intra_dupes:
        errors.append(f"{name}: contains duplicate IDs inside match list for {len(intra_dupes)} entities.")
    if self_matches:
        errors.append(f"{name}: contains S1- IDs (self-matches) for {len(self_matches)} items.")
    if wrong_prefix:
        errors.append(f"{name}: contains IDs without S2-/S3- prefix: {len(wrong_prefix)} items.")

    if required_s1 is not None:
        missing = required_s1 - seen
        extra = seen - required_s1
        if missing:
            errors.append(f"{name}: missing {len(missing)} required S1 entities.")
        if extra:
            errors.append(f"{name}: contains {len(extra)} unknown S1 entities.")
    else:
        if n_rows != 1732544:
            warnings.append(f"{name}: has {n_rows:,} rows (expected 1,732,544 for full test set).")

    print(f"  [OK] {name}: {n_rows:,} rows ({empties:,} singletons, {n_rows - empties:,} with matches).")
    return mapping


def validate_all(matching_path: str, candidate_path: str, test_dir: str = "dataset") -> Tuple[List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []

    print("=" * 75)
    print("SUBMISSION COMPLIANCE VALIDATOR (Amazon ML Challenge 2026)")
    print("=" * 75)

    required_s1 = load_required_s1_ids(test_dir)
    if required_s1:
        print(f"  [OK] Total required S1 entities: {len(required_s1):,}")
    else:
        warnings.append("Could not locate test_source1.tsv; verifying structure and row counts directly.")

    matched_map = validate_tsv_file(matching_path, MATCHING_HEADER, "matched_entity_ids", required_s1, errors, warnings)
    candidate_map = validate_tsv_file(candidate_path, CANDIDATE_HEADER, "candidate_entity_ids", required_s1, errors, warnings)

    # Check subset rule: every match must be in candidates
    if matched_map and candidate_map:
        subset_violations = 0
        for s1_id, matches in matched_map.items():
            cands = candidate_map.get(s1_id, set())
            if not matches.issubset(cands):
                subset_violations += 1

        if subset_violations > 0:
            errors.append(f"Subset Rule Violation: {subset_violations} entities have matches not in candidate_pairs.tsv.")
        else:
            print("  [OK] Subset Invariant Passed: All predicted matches are valid candidates.")

    return errors, warnings


def main():
    parser = argparse.ArgumentParser(description="Submission Validator")
    parser.add_argument("--matching", default="output/matching_results.tsv", help="Path to matching TSV")
    parser.add_argument("--candidate", default="output/candidate_pairs.tsv", help="Path to candidate TSV")
    parser.add_argument("--test-dir", default="dataset", help="Test directory path")
    args = parser.parse_args()

    errors, warnings = validate_all(args.matching, args.candidate, args.test_dir)

    print("\n" + "-" * 75)
    for w in warnings:
        print(f"WARNING: {w}")

    if errors:
        print(f"\n[FAIL] Found {len(errors)} issues:")
        for i, err in enumerate(errors, 1):
            print(f"  {i}. {err}")
        return 1

    print("\n[PASS] All submission checks passed! 100% compliant with competition rules.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
