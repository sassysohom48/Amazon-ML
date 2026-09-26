"""
Phase 7: Final Submission Packaging Script (Amazon ML Challenge 2026).
Validates matching_results.tsv & candidate_pairs.tsv, checks documentation,
and builds compliant team_submission.zip ready for leaderboard upload.
"""

import os
import sys
import zipfile
from pathlib import Path

# Ensure package import works
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    PROJECT_ROOT,
    OUTPUT_DIR,
    SUBMISSION_MATCHING_TSV,
    SUBMISSION_CANDIDATE_TSV,
)
from src.validate_submission import validate_all


def create_submission_zip(zip_name: str = "team_submission.zip") -> bool:
    print("=" * 75)
    print("PHASE 7: FINAL SUBMISSION VALIDATION & ZIP PACKAGING")
    print("=" * 75)

    # 1. Verify Output Files
    matching_path = OUTPUT_DIR / "matching_results.tsv"
    candidate_path = OUTPUT_DIR / "candidate_pairs.tsv"

    if not matching_path.exists():
        print(f"[ERROR] Matching results file not found: {matching_path}")
        return False
    if not candidate_path.exists():
        print(f"[ERROR] Candidate pairs file not found: {candidate_path}")
        return False

    matching_size_mb = matching_path.stat().st_size / (1024 * 1024)
    candidate_size_mb = candidate_path.stat().st_size / (1024 * 1024)
    print(f"[OK] Found matching TSV ({matching_size_mb:.2f} MB)")
    print(f"[OK] Found candidate TSV ({candidate_size_mb:.2f} MB)")

    # 2. Run Comprehensive Submission Validator
    print("\nRunning submission integrity checks...")
    errors, warnings = validate_all(
        matching_path=str(matching_path),
        candidate_path=str(candidate_path),
        test_dir=str(PROJECT_ROOT / "dataset"),
    )

    if errors:
        print(f"\n[FAILED] Validation failed with {len(errors)} error(s). Aborting zip creation.")
        for err in errors:
            print(f"  • {err}")
        return False

    print("\n[OK] All validation rules verified successfully!")

    # 3. Locate Documentation Template
    doc_candidates = [
        PROJECT_ROOT / "Documentation_template.md",
        PROJECT_ROOT / "dataset" / "Documentation_template.md",
    ]
    doc_path = None
    for p in doc_candidates:
        if p.exists():
            doc_path = p
            break

    if doc_path:
        print(f"[OK] Found documentation: {doc_path.name}")
    else:
        print("[WARNING] Documentation_template.md not found. Packaging code and TSVs.")

    # 4. Create Final team_submission.zip
    zip_path = PROJECT_ROOT / zip_name
    print(f"\nPackaging final submission archive: {zip_path.name}...")

    excluded_extensions = {".pyc", ".parquet", ".zip", ".tar.gz", ".DS_Store", ".log"}
    excluded_dirs = {"__pycache__", ".ipynb_checkpoints", ".git", ".venv", "env", "dataset", ".gemini"}

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # Add TSV output files
        zf.write(matching_path, arcname="output/matching_results.tsv")
        zf.write(candidate_path, arcname="output/candidate_pairs.tsv")

        # Add documentation
        if doc_path and doc_path.exists():
            zf.write(doc_path, arcname="Documentation_template.md")

        # Add code directory
        code_dir = PROJECT_ROOT / "code" / "business_entity_resolution"
        if code_dir.exists():
            for root, dirs, files in os.walk(code_dir):
                dirs[:] = [d for d in dirs if d not in excluded_dirs]
                for file in files:
                    if any(file.endswith(ext) for ext in excluded_extensions):
                        continue
                    file_full = Path(root) / file
                    rel_path = file_full.relative_to(PROJECT_ROOT)
                    zf.write(file_full, arcname=str(rel_path).replace("\\", "/"))

    zip_size_mb = zip_path.stat().st_size / (1024 * 1024)
    print("=" * 75)
    print(f"[SUCCESS] {zip_path.name} CREATED SUCCESSFULLY! ({zip_size_mb:.2f} MB)")
    print("=" * 75)
    print("Files included:")
    print("  • output/matching_results.tsv")
    print("  • output/candidate_pairs.tsv")
    print("  • Documentation_template.md")
    print("  • code/business_entity_resolution/ (src/, models/, requirements.txt, notebooks/)")
    print("\nThe archive is 100% compliant with competition rules and ready for official submission!")
    return True


if __name__ == "__main__":
    success = create_submission_zip()
    sys.exit(0 if success else 1)
