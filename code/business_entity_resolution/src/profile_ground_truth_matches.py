"""
Ground Truth Match Profiler.
Analyzes what similarity signals exist between true (S1, Target) pairs
to determine the exact blocking keys needed to reach >95% recall.
"""

import sys
from pathlib import Path
import polars as pl
from rapidfuzz import fuzz

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import PROCESSED_DIR


def profile_ground_truth(sample_size: int = 10000):
    print("=" * 70)
    print("PROFILING GROUND TRUTH MATCHES TO REACH >95% BLOCKING RECALL")
    print("=" * 70)

    val_s1 = pl.read_parquet(PROCESSED_DIR / "val_source1_cleaned.parquet")
    val_gt = pl.read_parquet(PROCESSED_DIR / "val_ground_truth.parquet")
    train_s2 = pl.read_parquet(PROCESSED_DIR / "train_source2_cleaned.parquet")
    train_s3 = pl.read_parquet(PROCESSED_DIR / "train_source3_cleaned.parquet")

    # Sample true pairs
    true_pairs = []
    for row in val_gt.iter_rows():
        s1_id, matches = row[0], row[1]
        if matches and str(matches).strip():
            for tgt_id in str(matches).strip().split(","):
                if tgt_id.strip():
                    true_pairs.append((s1_id, tgt_id.strip()))
                if len(true_pairs) >= sample_size:
                    break
        if len(true_pairs) >= sample_size:
            break

    print(f"Sampled {len(true_pairs):,} true matching pairs for analysis.")

    # Filter records
    s1_ids = {p[0] for p in true_pairs}
    tgt_ids = {p[1] for p in true_pairs}

    s1_dict = {row["entity_id"]: row for row in val_s1.filter(pl.col("entity_id").is_in(list(s1_ids))).iter_rows(named=True)}
    
    t_targets = pl.concat([
        train_s2.filter(pl.col("entity_id").is_in(list(tgt_ids))),
        train_s3.filter(pl.col("entity_id").is_in(list(tgt_ids)))
    ])
    tgt_dict = {row["entity_id"]: row for row in t_targets.iter_rows(named=True)}

    # Signal Counters
    total = len(true_pairs)
    c_exact_name = 0
    c_shared_name_tok = 0
    c_shared_addr_tok = 0
    c_shared_postal = 0
    c_fuzzy_name_70 = 0
    c_fuzzy_name_50 = 0
    c_name_or_addr_tok = 0
    c_name_or_postal = 0
    c_any_signal = 0
    c_no_signal = 0

    no_signal_samples = []

    for s1_id, tgt_id in true_pairs:
        s1 = s1_dict.get(s1_id)
        tgt = tgt_dict.get(tgt_id)
        if not s1 or not tgt:
            continue

        s1_name = s1.get("name_clean", "") or ""
        tgt_name = tgt.get("name_clean", "") or ""
        s1_addr = s1.get("addr_clean", "") or ""
        tgt_addr = tgt.get("addr_clean", "") or ""

        s1_name_toks = set(s1.get("name_tokens", "").split()) if s1.get("name_tokens") else set()
        tgt_name_toks = set(tgt.get("name_tokens", "").split()) if tgt.get("name_tokens") else set()

        s1_addr_toks = set(s1_addr.split()) if s1_addr else set()
        tgt_addr_toks = set(tgt_addr.split()) if tgt_addr else set()

        s1_postals = set(s1.get("postal_digits", "").split()) if s1.get("postal_digits") else set()
        tgt_postals = set(tgt.get("postal_digits", "").split()) if tgt.get("postal_digits") else set()

        # Check conditions
        exact_name = bool(s1_name and s1_name == tgt_name)
        shared_name_tok = bool(s1_name_toks & tgt_name_toks)
        shared_addr_tok = bool(s1_addr_toks & tgt_addr_toks)
        shared_postal = bool(s1_postals & tgt_postals)
        
        ratio = fuzz.ratio(s1_name, tgt_name) if (s1_name and tgt_name) else 0.0
        fuzzy_70 = ratio >= 70.0
        fuzzy_50 = ratio >= 50.0

        if exact_name:
            c_exact_name += 1
        if shared_name_tok:
            c_shared_name_tok += 1
        if shared_addr_tok:
            c_shared_addr_tok += 1
        if shared_postal:
            c_shared_postal += 1
        if fuzzy_70:
            c_fuzzy_name_70 += 1
        if fuzzy_50:
            c_fuzzy_name_50 += 1

        if shared_name_tok or shared_addr_tok:
            c_name_or_addr_tok += 1
        if shared_name_tok or shared_postal:
            c_name_or_postal += 1

        has_any = (shared_name_tok or shared_addr_tok or shared_postal or fuzzy_50 or exact_name)
        if has_any:
            c_any_signal += 1
        else:
            c_no_signal += 1
            if len(no_signal_samples) < 10:
                no_signal_samples.append((s1, tgt))

    print("\nMATCH SIGNAL COVERAGE (Theoretical Upper Bound):")
    print(f"  • Exact Clean Name Match:          {c_exact_name/total*100:6.2f}% ({c_exact_name:,}/{total:,})")
    print(f"  • Shared Name Token (>= 1):        {c_shared_name_tok/total*100:6.2f}% ({c_shared_name_tok:,}/{total:,})")
    print(f"  • Fuzzy Name Ratio >= 70%:         {c_fuzzy_name_70/total*100:6.2f}% ({c_fuzzy_name_70:,}/{total:,})")
    print(f"  • Fuzzy Name Ratio >= 50%:         {c_fuzzy_name_50/total*100:6.2f}% ({c_fuzzy_name_50:,}/{total:,})")
    print(f"  • Shared Address Token (>= 1):     {c_shared_addr_tok/total*100:6.2f}% ({c_shared_addr_tok:,}/{total:,})")
    print(f"  • Exact Postal / PIN Code Match:   {c_shared_postal/total*100:6.2f}% ({c_shared_postal:,}/{total:,})")
    print("  " + "-" * 60)
    print(f"  • (Name Token OR Postal Match):    {c_name_or_postal/total*100:6.2f}% ({c_name_or_postal:,}/{total:,})")
    print(f"  • (Name Token OR Addr Token):      {c_name_or_addr_tok/total*100:6.2f}% ({c_name_or_addr_tok:,}/{total:,})")
    print(f"  • Total Any Signal:                {c_any_signal/total*100:6.2f}% ({c_any_signal:,}/{total:,})")
    print(f"  • Zero Common Signal:              {c_no_signal/total*100:6.2f}% ({c_no_signal:,}/{total:,})")
    print("=" * 70)

    if no_signal_samples:
        print("\nSample Pairs with Zero Common Signal (Edge Cases):")
        for i, (s1, tgt) in enumerate(no_signal_samples, 1):
            print(f"[{i}] S1:  '{s1['business_name_raw']}' | Addr: '{s1['business_address_raw']}'")
            print(f"    Tgt: '{tgt['business_name_raw']}' | Addr: '{tgt['business_address_raw']}'")


if __name__ == "__main__":
    profile_ground_truth(10000)
