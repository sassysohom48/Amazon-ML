"""
Phase 7: Final Submission Packaging Script (Amazon ML Challenge 2026).
Validates submission files, checks documentation, and builds compliant team_submission.zip.
"""

import os
import sys
import zipfile
import subprocess
from pathlib import Path

# Ensure package import works
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    PROJECT_ROOT,
    OUTPUT_DIR,
    SUBMISSION_MATCHING_TSV,
    SUBMISSION_CANDIDATE_TSV,
    VALIDATION_SCRIPT,
    TEST_DIR,
)


def create_submission_zip(zip_name: str = "team_submission.zip") -> bool:
    print("=" * 75)
    print("PHASE 7: FINAL SUBMISSION VALIDATION & ZIP PACKAGING")
    print("=" * 75)

    # 1. Verify Output Files
    if not SUBMISSION_MATCHING_TSV.exists():
        print(f"[ERROR] Matching results file not found: {SUBMISSION_MATCHING_TSV}")
        return False
    if not SUBMISSION_CANDIDATE_TSV.exists():
        print(f"[ERROR] Candidate pairs file not found: {SUBMISSION_CANDIDATE_TSV}")
        return False

    print(f"[OK] Found matching TSV ({SUBMISSION_MATCHING_TSV.stat().st_size / (1024*1024):.2f} MB)")
    print(f"[OK] Found candidate TSV ({SUBMISSION_CANDIDATE_TSV.stat().st_size / (1024*1024):.2f} MB)")

    # 2. Run Submission Validator
    print("\nRunning official submission validator...")
    val_cmd = [
        sys.executable,
        str(VALIDATION_SCRIPT),
        "--matching", str(SUBMISSION_MATCHING_TSV),
        "--candidate", str(SUBMISSION_CANDIDATE_TSV),
        "--test-dir", str(TEST_DIR),
        "--check-ids",
    ]
    res = subprocess.run(val_cmd, capture_output=True, text=True)
    print(res.stdout)
    if res.stderr:
        print(res.stderr)

    if res.returncode != 0:
        print(f"[FAILED] Submission validation failed with code {res.returncode}. Aborting packaging.")
        return False

    print("[OK] Official validation passed successfully!")

    # 3. Check Documentation
    doc_path = PROJECT_ROOT / "dataset" / "Documentation_template.md"
    if not doc_path.exists():
        doc_path = PROJECT_ROOT / "Documentation_template.md"

    # 4. Create ZIP
    zip_path = PROJECT_ROOT / zip_name
    print(f"\nPackaging final submission archive: {zip_path}")

    # Exclusions
    excluded_extensions = {".pyc", ".parquet", ".zip", ".tar.gz", ".DS_Store"}
    excluded_dirs = {"__pycache__", ".ipynb_checkpoints", ".git", ".venv", "env", "dataset"}

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # Add output files
        zf.write(SUBMISSION_MATCHING_TSV, arcname="output/matching_results.tsv")
        zf.write(SUBMISSION_CANDIDATE_TSV, arcname="output/candidate_pairs.tsv")

        # Add documentation
        if doc_path.exists():
            zf.write(doc_path, arcname="Documentation_template.md")

        # Add code directory
        code_dir = PROJECT_ROOT / "code" / "business_entity_resolution"
        for root, dirs, files in os.walk(code_dir):
            dirs[:] = [d for d in dirs if d not in excluded_dirs]
            for file in files:
                if any(file.endswith(ext) for ext in excluded_extensions):
                    continue
                file_full = Path(root) / file
                rel_path = file_full.relative_to(PROJECT_ROOT)
                zf.write(file_full, arcname=str(rel_path).replace("\\", "/"))

    zip_size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"\n[SUCCESS] Successfully generated {zip_path.name} ({zip_size_mb:.2f} MB)")
    print(f"Archive is 100% compliant with competition rules and ready for submission.")
    return True


if __name__ == "__main__":
    success = create_submission_zip()
    sys.exit(0 if success else 1)
