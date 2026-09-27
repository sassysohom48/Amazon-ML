"""
Step 1.3: Ground Truth Signal & Combination Profiler (Amazon ML Challenge 2026).
Analyzes true match pairs to discover empirical signal coverage and multi-key intersections.
Produces actionable blocking guidelines partitioned by country (US, India).
"""

import sys
import time
import json
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict
import polars as pl
from rapidfuzz import fuzz, distance

# Ensure package import works
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PROCESSED_DIR, OUTPUT_DIR


def extract_3grams(text: str) -> Set[str]:
    """Extracts character 3-grams for fast string overlap."""
    if not text or len(text) < 3:
        return {text} if text else set()
    return {text[i:i+3] for i in range(len(text) - 2)}


def simplified_soundex(name: str) -> str:
    """Computes basic phonetic encoding for name prefix matching."""
    if not name:
        return ""
    name = re.sub(r"[^A-Z]", "", name.upper())
    if not name:
        return ""
    mapping = {
        "B": "1", "F": "1", "P": "1", "V": "1",
        "C": "2", "G": "2", "J": "2", "K": "2", "Q": "2", "S": "2", "X": "2", "Z": "2",
        "D": "3", "T": "3",
        "L": "4",
        "M": "5", "N": "5",
        "R": "6",
    }
    first_letter = name[0]
    encoded = [first_letter]
    for char in name[1:]:
        digit = mapping.get(char, "0")
        if digit != "0" and digit != encoded[-1]:
            encoded.append(digit)
    return (("".join(encoded)) + "0000")[:4]


def profile_ground_truth_signals(
    sample_size: int = 50000,
    output_json: Path = PROCESSED_DIR / "gt_signal_profile.json",
) -> Dict:
    """
    Profiles true pairs across individual signals and multi-signal combinations.
    Uses lean, sampled filtering to maintain near-zero memory footprint (< 50MB RAM).
    """
    print("=" * 75)
    print("STEP 1.3: GROUND TRUTH SIGNAL & COMBINATION PROFILER")
    print("=" * 75)
    t0 = time.time()

    # 1. Load Ground Truth First
    gt_path = PROCESSED_DIR / "train_ground_truth.parquet"
    if not gt_path.exists():
        gt_path = PROCESSED_DIR / "val_ground_truth.parquet"
    
    print(f"Reading ground truth from: {gt_path}...")
    gt_df = pl.read_parquet(gt_path)
    
    s1_col = "source1_entity_id" if "source1_entity_id" in gt_df.columns else gt_df.columns[0]
    tgt_col = "matched_entity_id" if "matched_entity_id" in gt_df.columns else gt_df.columns[1]

    # Sample for high-speed multi-metric profiling
    if len(gt_df) > sample_size:
        print(f"Sampling {sample_size:,} true match pairs for detailed profiling...")
        gt_sample = gt_df.sample(sample_size, seed=42)
    else:
        gt_sample = gt_df

    needed_s1_ids = set(gt_sample[s1_col].to_list())
    needed_tgt_ids = set(gt_sample[tgt_col].to_list())
    print(f"Sample contains {len(needed_s1_ids):,} unique S1 IDs and {len(needed_tgt_ids):,} unique Target IDs.")

    # 2. Load Only Sampled Records from S1/S2/S3
    print("\nReading sampled entities from cleaned datasets...")
    cols = ["entity_id", "country", "name_clean", "name_tokens", "addr_clean", "postal_digits"]

    s1_df = pl.read_parquet(PROCESSED_DIR / "train_source1_cleaned.parquet", columns=cols).filter(
        pl.col("entity_id").is_in(needed_s1_ids)
    )
    s2_df = pl.read_parquet(PROCESSED_DIR / "train_source2_cleaned.parquet", columns=cols).filter(
        pl.col("entity_id").is_in(needed_tgt_ids)
    )
    s3_df = pl.read_parquet(PROCESSED_DIR / "train_source3_cleaned.parquet", columns=cols).filter(
        pl.col("entity_id").is_in(needed_tgt_ids)
    )

    print(f"Loaded {len(s1_df):,} S1 and {len(s2_df) + len(s3_df):,} Target records in {time.time() - t0:.2f}s")

    # 3. Build Fast Attribute Lookups for Sampled Entities
    print("Building attribute lookup tables...")
    def build_lookup(df: pl.DataFrame):
        eids = df["entity_id"].to_list()
        countries = df["country"].to_list()
        names = df["name_clean"].to_list()
        tokens = df["name_tokens"].to_list()
        addrs = df["addr_clean"].to_list()
        postals = df["postal_digits"].to_list()
        
        lookup = {}
        for eid, c, n, t, a, p in zip(eids, countries, names, tokens, addrs, postals):
            lookup[eid] = {
                "country": c,
                "name": n if n else "",
                "tokens": set(t.split()) if t else set(),
                "addr": a if a else "",
                "addr_tokens": set(a.split()) if a else set(),
                "postal": set(p.split()) if p else set(),
                "3grams": extract_3grams(n),
                "soundex": simplified_soundex(n),
            }
        return lookup

    lookup_s1 = build_lookup(s1_df)
    lookup_tgt = {**build_lookup(s2_df), **build_lookup(s3_df)}
    print(f"Lookups built for {len(lookup_s1) + len(lookup_tgt):,} entities in {time.time() - t0:.2f}s")

    # 3. Profile Signals per Country
    print("\nEvaluating individual signals & combinations across true pairs...")
    countries = ["US", "India"]
    stats: Dict[str, Dict] = {}

    s1_list = gt_sample[s1_col].to_list()
    tgt_list = gt_sample[tgt_col].to_list()

    for country in countries:
        country_pairs = [
            (s1, tgt) for s1, tgt in zip(s1_list, tgt_list)
            if s1 in lookup_s1 and tgt in lookup_tgt and lookup_s1[s1]["country"] == country
        ]
        total_p = len(country_pairs)
        print(f"\n" + "-" * 75)
        print(f"COUNTRY SIGNAL PROFILE: {country.upper()} ({total_p:,} True Pairs Evaluated)")
        print("-" * 75)

        if total_p == 0:
            continue

        # Individual Counters
        exact_name_cnt = 0
        token_overlap_cnt = 0
        jaccard_05_cnt = 0
        addr_token_overlap_cnt = 0
        postal_exact_cnt = 0
        phonetic_cnt = 0
        ngram3_overlap_cnt = 0
        fuzz_80_cnt = 0

        # Combination Counters (Multi-Key Intersection & Union)
        comb_exact_or_token = 0
        comb_exact_or_token_or_addr = 0
        comb_token_and_postal = 0
        comb_token_and_addr = 0
        comb_phonetic_or_token = 0
        comb_ngram_or_token = 0
        comb_full_union = 0

        for s1_id, tgt_id in country_pairs:
            s1 = lookup_s1[s1_id]
            tgt = lookup_tgt[tgt_id]

            # 1. Exact Name
            is_exact_name = (s1["name"] == tgt["name"]) and len(s1["name"]) > 0
            if is_exact_name:
                exact_name_cnt += 1

            # 2. Token Overlap & Jaccard
            shared_tokens = s1["tokens"] & tgt["tokens"]
            has_token_overlap = len(shared_tokens) > 0
            if has_token_overlap:
                token_overlap_cnt += 1

            union_tokens = s1["tokens"] | tgt["tokens"]
            if union_tokens and (len(shared_tokens) / len(union_tokens)) >= 0.5:
                jaccard_05_cnt += 1

            # 3. Address Token Overlap
            shared_addr_tokens = s1["addr_tokens"] & tgt["addr_tokens"]
            has_addr_token_overlap = len(shared_addr_tokens) > 0
            if has_addr_token_overlap:
                addr_token_overlap_cnt += 1

            # 4. Postal Exact
            shared_postals = s1["postal"] & tgt["postal"]
            has_postal_match = len(shared_postals) > 0
            if has_postal_match:
                postal_exact_cnt += 1

            # 5. Phonetic Match
            has_phonetic_match = (s1["soundex"] == tgt["soundex"]) and len(s1["soundex"]) > 0
            if has_phonetic_match:
                phonetic_cnt += 1

            # 6. 3-gram Overlap
            shared_3grams = s1["3grams"] & tgt["3grams"]
            has_ngram_overlap = len(shared_3grams) >= 2
            if has_ngram_overlap:
                ngram3_overlap_cnt += 1

            # 7. Fuzzy WRatio >= 80
            has_fuzz_80 = fuzz.WRatio(s1["name"], tgt["name"]) >= 80
            if has_fuzz_80:
                fuzz_80_cnt += 1

            # Combinations
            if is_exact_name or has_token_overlap:
                comb_exact_or_token += 1

            if is_exact_name or has_token_overlap or has_addr_token_overlap:
                comb_exact_or_token_or_addr += 1

            if has_token_overlap and has_postal_match:
                comb_token_and_postal += 1

            if has_token_overlap and has_addr_token_overlap:
                comb_token_and_addr += 1

            if has_phonetic_match or has_token_overlap:
                comb_phonetic_or_token += 1

            if has_ngram_overlap or has_token_overlap:
                comb_ngram_or_token += 1

            if is_exact_name or has_token_overlap or has_addr_token_overlap or has_postal_match or has_phonetic_match:
                comb_full_union += 1

        # Format Country Results
        c_stats = {
            "total_pairs_evaluated": total_p,
            "individual_signals": {
                "exact_clean_name": f"{exact_name_cnt / total_p * 100:.2f}%",
                "shared_ge_1_name_token": f"{token_overlap_cnt / total_p * 100:.2f}%",
                "name_token_jaccard_ge_05": f"{jaccard_05_cnt / total_p * 100:.2f}%",
                "shared_ge_1_addr_token": f"{addr_token_overlap_cnt / total_p * 100:.2f}%",
                "shared_postal_pin": f"{postal_exact_cnt / total_p * 100:.2f}%",
                "phonetic_soundex_match": f"{phonetic_cnt / total_p * 100:.2f}%",
                "char_3gram_overlap_ge_2": f"{ngram3_overlap_cnt / total_p * 100:.2f}%",
                "fuzzy_wratio_ge_80": f"{fuzz_80_cnt / total_p * 100:.2f}%",
            },
            "multi_signal_combinations": {
                "Exact_Name OR Name_Token": f"{comb_exact_or_token / total_p * 100:.2f}%",
                "Exact_Name OR Name_Token OR Addr_Token": f"{comb_exact_or_token_or_addr / total_p * 100:.2f}%",
                "Name_Token AND Postal_Match": f"{comb_token_and_postal / total_p * 100:.2f}%",
                "Name_Token AND Addr_Token": f"{comb_token_and_addr / total_p * 100:.2f}%",
                "Phonetic_Name OR Name_Token": f"{comb_phonetic_or_token / total_p * 100:.2f}%",
                "Char_3gram OR Name_Token": f"{comb_ngram_or_token / total_p * 100:.2f}%",
                "TOTAL_RECALL_CEILING (Any Signal)": f"{comb_full_union / total_p * 100:.2f}%",
            }
        }
        stats[country] = c_stats

        # Print Table
        print(f"\n[Individual Signals — {country}]:")
        for k, v in c_stats["individual_signals"].items():
            print(f"  • {k:<30}: {v}")

        print(f"\n[Multi-Key Blocking Combinations — {country}]:")
        for k, v in c_stats["multi_signal_combinations"].items():
            print(f"  • {k:<40}: {v}")

    # Export JSON
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"\n[SUCCESS] Signal & combination profile saved to: {output_json}")
    return stats


if __name__ == "__main__":
    profile_ground_truth_signals()
