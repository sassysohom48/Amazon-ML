"""
Vectorized Pairwise Feature Engineering Module (Amazon ML Challenge 2026).
Computes SIMD-accelerated string metrics, token overlaps, numeric address matches,
and candidate provenance features using RapidFuzz C++ core.
"""

import time
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
import numpy as np
import polars as pl
from rapidfuzz import fuzz, distance

from .config import (
    FEATURES_DIR, CANDIDATES_DIR, PARQUET_DIR, PROCESSED_DIR
)
from .normalizer import clean_text, extract_char_3grams, extract_soundex, RE_DIGITS

# Canonical feature schema produced by this module (25 features)
FEATURE_COLS: List[str] = [
    "candidate_rank",
    "name_levenshtein",
    "name_jaro_winkler",
    "name_token_sort",
    "name_token_set",
    "name_partial_ratio",
    "name_3gram_jaccard",
    "name_len_diff",
    "name_len_ratio",
    "name_exact_match",
    "name_soundex_match",
    "name_head_token_match",
    "name_acronym_match",
    "name_word_count_diff",
    "has_s1_address",
    "has_target_address",
    "addr_both_present",
    "addr_token_jaccard",
    "addr_token_overlap",
    "addr_numeric_match",
    "addr_numeric_conflict",
    "addr_levenshtein",
    "addr_token_sort",
    "is_source2",
    "is_source3",
]


def extract_digit_tokens(text: str) -> Set[str]:
    """Extracts standalone numeric tokens (door numbers, PINs, street numbers)."""
    if not text:
        return set()
    return set(RE_DIGITS.findall(text))


def extract_word_tokens(text: str) -> Set[str]:
    """Extracts lowercase alphabetic word tokens."""
    if not text:
        return set()
    return set(text.lower().split())


def build_pairwise_features(
    s1_ids: List[str],
    s1_names: List[str],
    s1_addrs: List[str],
    target_ids: List[str],
    target_names: List[str],
    target_addrs: List[str],
    ranks: List[int],
    gt_map: Optional[Dict[str, Set[str]]] = None
) -> pl.DataFrame:
    """
    Computes vectorized SIMD pairwise features across candidate pairs:
    - RapidFuzz name similarities (Levenshtein, Jaro-Winkler, Token Sort, Token Set, Partial)
    - Character 3-gram Jaccard, Head token match, and Acronym match
    - Address word-token Jaccard & overlap coefficient
    - Address numeric token intersection & numeric conflict penalty
    - Missingness flags & provenance ranks
    """
    n_pairs = len(s1_ids)
    print(f"Vectorizing features for {n_pairs:,} candidate pairs with RapidFuzz SIMD...")
    t0 = time.time()

    # Pre-allocate feature arrays
    f_name_lev = np.zeros(n_pairs, dtype=np.float32)
    f_name_jw = np.zeros(n_pairs, dtype=np.float32)
    f_name_sort = np.zeros(n_pairs, dtype=np.float32)
    f_name_set = np.zeros(n_pairs, dtype=np.float32)
    f_name_partial = np.zeros(n_pairs, dtype=np.float32)
    f_name_3gram = np.zeros(n_pairs, dtype=np.float32)
    f_name_len_diff = np.zeros(n_pairs, dtype=np.float32)
    f_name_len_ratio = np.zeros(n_pairs, dtype=np.float32)
    f_name_exact = np.zeros(n_pairs, dtype=np.float32)
    f_name_soundex = np.zeros(n_pairs, dtype=np.float32)
    f_name_head = np.zeros(n_pairs, dtype=np.float32)
    f_name_acronym = np.zeros(n_pairs, dtype=np.float32)
    f_name_word_diff = np.zeros(n_pairs, dtype=np.float32)

    f_has_s1_addr = np.zeros(n_pairs, dtype=np.float32)
    f_has_target_addr = np.zeros(n_pairs, dtype=np.float32)
    f_addr_both = np.zeros(n_pairs, dtype=np.float32)
    f_addr_token_jaccard = np.zeros(n_pairs, dtype=np.float32)
    f_addr_token_overlap = np.zeros(n_pairs, dtype=np.float32)
    f_addr_numeric_match = np.zeros(n_pairs, dtype=np.float32)
    f_addr_numeric_conflict = np.zeros(n_pairs, dtype=np.float32)
    f_addr_lev = np.zeros(n_pairs, dtype=np.float32)
    f_addr_sort = np.zeros(n_pairs, dtype=np.float32)

    f_rank = np.array(ranks, dtype=np.float32)
    f_is_s2 = np.zeros(n_pairs, dtype=np.float32)
    f_is_s3 = np.zeros(n_pairs, dtype=np.float32)

    labels = np.zeros(n_pairs, dtype=np.int32) if gt_map is not None else None

    # Compute pairwise metrics
    for i in range(n_pairs):
        n1 = s1_names[i]
        n2 = target_names[i]
        a1 = s1_addrs[i]
        a2 = target_addrs[i]
        tid = target_ids[i]
        sid = s1_ids[i]

        # 1. Name Features
        f_name_lev[i] = fuzz.ratio(n1, n2) / 100.0
        f_name_jw[i] = distance.JaroWinkler.similarity(n1, n2)
        f_name_sort[i] = fuzz.token_sort_ratio(n1, n2) / 100.0
        f_name_set[i] = fuzz.token_set_ratio(n1, n2) / 100.0
        f_name_partial[i] = fuzz.partial_ratio(n1, n2) / 100.0
        f_name_exact[i] = 1.0 if (n1 and n1 == n2) else 0.0

        len1, len2 = len(n1), len(n2)
        f_name_len_diff[i] = abs(len1 - len2)
        f_name_len_ratio[i] = min(len1, len2) / max(len1, len2, 1)

        # Character 3-gram Jaccard
        sh1 = extract_char_3grams(n1)
        sh2 = extract_char_3grams(n2)
        union_sh = len(sh1 | sh2)
        f_name_3gram[i] = (len(sh1 & sh2) / union_sh) if union_sh > 0 else 0.0

        # Phonetic Soundex match on primary token
        w1 = n1.split()[0] if n1 else ""
        w2 = n2.split()[0] if n2 else ""
        sx1 = extract_soundex(w1)
        sx2 = extract_soundex(w2)
        f_name_soundex[i] = 1.0 if (sx1 and sx1 == sx2) else 0.0

        # Head word (primary brand) match
        if w1 and w1 == w2:
            f_name_head[i] = 1.0
        elif w1 and w2 and fuzz.ratio(w1, w2) >= 85:
            f_name_head[i] = 0.85

        # Acronym & Word Count Differences
        words1 = [w for w in n1.split() if len(w) >= 2]
        words2 = [w for w in n2.split() if len(w) >= 2]
        f_name_word_diff[i] = abs(len(words1) - len(words2))
        clean_c1 = n1.replace(" ", "")
        clean_c2 = n2.replace(" ", "")
        acr1 = "".join(w[0] for w in words1) if len(words1) >= 2 else ""
        acr2 = "".join(w[0] for w in words2) if len(words2) >= 2 else ""
        if (clean_c1 and clean_c1 == acr2 and len(clean_c1) >= 2) or (clean_c2 and clean_c2 == acr1 and len(clean_c2) >= 2):
            f_name_acronym[i] = 1.0

        # 2. Address Features
        has_a1 = len(a1) > 0
        has_a2 = len(a2) > 0
        f_has_s1_addr[i] = 1.0 if has_a1 else 0.0
        f_has_target_addr[i] = 1.0 if has_a2 else 0.0
        f_addr_both[i] = 1.0 if (has_a1 and has_a2) else 0.0

        if has_a1 and has_a2:
            tok1 = extract_word_tokens(a1)
            tok2 = extract_word_tokens(a2)
            inter_tok = len(tok1 & tok2)
            union_tok = len(tok1 | tok2)
            f_addr_token_jaccard[i] = (inter_tok / union_tok) if union_tok > 0 else 0.0
            min_tok = min(len(tok1), len(tok2))
            f_addr_token_overlap[i] = (inter_tok / min_tok) if min_tok > 0 else 0.0

            # Standalone digits / PINs
            d1 = extract_digit_tokens(a1)
            d2 = extract_digit_tokens(a2)
            if d1 and d2:
                inter_d = len(d1 & d2)
                f_addr_numeric_match[i] = inter_d / max(len(d1 | d2), 1)
                if inter_d == 0:
                    f_addr_numeric_conflict[i] = 1.0  # Explicit conflicting street number/PIN penalty

            f_addr_lev[i] = fuzz.ratio(a1, a2) / 100.0
            f_addr_sort[i] = fuzz.token_sort_ratio(a1, a2) / 100.0

        # 3. Provenance & Source
        f_is_s2[i] = 1.0 if tid.startswith("S2-") else 0.0
        f_is_s3[i] = 1.0 if tid.startswith("S3-") else 0.0

        # 4. Target Label (if Ground Truth supplied)
        if labels is not None:
            true_set = gt_map.get(sid, set())
            labels[i] = 1 if tid in true_set else 0

    elapsed = time.time() - t0
    rate = int(n_pairs / max(elapsed, 0.001))
    print(f"Computed {n_pairs:,} feature vectors in {elapsed:.2f}s ({rate:,} pairs/sec)!")

    # Assemble Polars DataFrame
    feature_dict = {
        "source1_entity_id": s1_ids,
        "target_entity_id": target_ids,
        "candidate_rank": f_rank,
        "name_levenshtein": f_name_lev,
        "name_jaro_winkler": f_name_jw,
        "name_token_sort": f_name_sort,
        "name_token_set": f_name_set,
        "name_partial_ratio": f_name_partial,
        "name_3gram_jaccard": f_name_3gram,
        "name_len_diff": f_name_len_diff,
        "name_len_ratio": f_name_len_ratio,
        "name_exact_match": f_name_exact,
        "name_soundex_match": f_name_soundex,
        "name_head_token_match": f_name_head,
        "name_acronym_match": f_name_acronym,
        "name_word_count_diff": f_name_word_diff,
        "has_s1_address": f_has_s1_addr,
        "has_target_address": f_has_target_addr,
        "addr_both_present": f_addr_both,
        "addr_token_jaccard": f_addr_token_jaccard,
        "addr_token_overlap": f_addr_token_overlap,
        "addr_numeric_match": f_addr_numeric_match,
        "addr_numeric_conflict": f_addr_numeric_conflict,
        "addr_levenshtein": f_addr_lev,
        "addr_token_sort": f_addr_sort,
        "is_source2": f_is_s2,
        "is_source3": f_is_s3,
    }

    if labels is not None:
        feature_dict["label"] = labels

    return pl.DataFrame(feature_dict)


def run_feature_pipeline(
    candidates_parquet: Optional[Path] = None,
    output_parquet: Optional[Path] = None,
    is_validation: bool = True
) -> Path:
    """
    Executes the end-to-end feature extraction pipeline:
    1. Loads candidate pairs from Stage 3.
    2. Maps entity attributes from Snappy Parquet tables.
    3. Computes 20+ SIMD string & address similarity metrics.
    4. Attaches ground-truth binary labels (1=Match, 0=Hard Negative).
    5. Writes feature dataset to disk.
    """
    start_total = time.time()
    print("=" * 80)
    print("  AMAZON ML CHALLENGE 2026: PHASE 4 VECTORIZED FEATURE ENGINEERING")
    print("=" * 80)

    if candidates_parquet is None:
        candidates_parquet = CANDIDATES_DIR / "val_candidates.parquet"
    if output_parquet is None:
        output_parquet = FEATURES_DIR / "val_features.parquet"

    if not candidates_parquet.exists():
        raise FileNotFoundError(f"Candidates file not found at {candidates_parquet}. Run Step 2 first.")

    print(f"Loading candidate pairs from: {candidates_parquet.name}")
    df_cands = pl.read_parquet(candidates_parquet)
    df_cands = df_cands.filter(pl.col("candidate_entity_ids") != "")
    print(f"Loaded {df_cands.height:,} entities with candidate lists.")

    # 1. Explode candidate pairs into rows
    df_pairs = (
        df_cands
        .with_columns(pl.col("candidate_entity_ids").str.split(","))
        .explode("candidate_entity_ids")
        .rename({"candidate_entity_ids": "target_entity_id"})
        .with_columns(
            pl.int_range(1, pl.len() + 1).over("source1_entity_id").alias("candidate_rank")
        )
    )
    total_pairs = df_pairs.height
    print(f"Total candidate pairs to process: {total_pairs:,}")

    # 2. Load entity attributes
    print("Loading entity attribute registries...")
    val_s1_path = PROCESSED_DIR / "val_source1.parquet"
    df_s1 = pl.read_parquet(val_s1_path).select(["entity_id", "business_name", "business_address", "country"])

    # Load S2 and S3 partitions for required countries
    countries = df_s1["country"].unique().to_list()
    s2_dfs, s3_dfs = [], []
    for c in countries:
        p2 = PARQUET_DIR / f"train_source2_country={c}.parquet"
        p3 = PARQUET_DIR / f"train_source3_country={c}.parquet"
        if p2.exists():
            s2_dfs.append(pl.read_parquet(p2).select(["entity_id", "business_name", "business_address"]))
        if p3.exists():
            s3_dfs.append(pl.read_parquet(p3).select(["entity_id", "business_name", "business_address"]))

    df_targets = pl.concat(s2_dfs + s3_dfs)

    # Filter targets to only those present in candidate pairs (saves memory)
    needed_target_ids = set(df_pairs["target_entity_id"].to_list())
    df_targets = df_targets.filter(pl.col("entity_id").is_in(needed_target_ids))
    print(f"Loaded {df_s1.height:,} S1 records and {df_targets.height:,} unique active target records.")

    # Convert to fast lookup dicts
    s1_lookup: Dict[str, Tuple[str, str]] = {
        row[0]: (clean_text(row[1]), clean_text(row[2])) for row in df_s1.iter_rows()
    }
    target_lookup: Dict[str, Tuple[str, str]] = {
        row[0]: (clean_text(row[1]), clean_text(row[2])) for row in df_targets.iter_rows()
    }

    # Extract aligned arrays
    s1_id_list = df_pairs["source1_entity_id"].to_list()
    target_id_list = df_pairs["target_entity_id"].to_list()
    rank_list = df_pairs["candidate_rank"].to_list()

    s1_names, s1_addrs = [], []
    target_names, target_addrs = [], []

    for sid, tid in zip(s1_id_list, target_id_list):
        n1, a1 = s1_lookup.get(sid, ("", ""))
        n2, a2 = target_lookup.get(tid, ("", ""))
        s1_names.append(n1)
        s1_addrs.append(a1)
        target_names.append(n2)
        target_addrs.append(a2)

    # Load Ground Truth if in validation mode
    gt_map = None
    if is_validation:
        from .dataset import load_ground_truth
        val_gt_path = PROCESSED_DIR / "val_ground_truth.parquet"
        gt_map = load_ground_truth(val_gt_path)

    # 3. Build vectorized features
    df_features = build_pairwise_features(
        s1_ids=s1_id_list,
        s1_names=s1_names,
        s1_addrs=s1_addrs,
        target_ids=target_id_list,
        target_names=target_names,
        target_addrs=target_addrs,
        ranks=rank_list,
        gt_map=gt_map
    )

    # 4. Report statistics & persist
    if "label" in df_features.columns:
        pos_count = df_features.filter(pl.col("label") == 1).height
        neg_count = df_features.filter(pl.col("label") == 0).height
        print(f"\nFeature Matrix Class Balance:")
        print(f"  • Positive Pairs (Ground Truth Matches): {pos_count:,} ({pos_count/total_pairs*100:.2f}%)")
        print(f"  • Hard Negative Pairs (Blocking Near-Misses): {neg_count:,} ({neg_count/total_pairs*100:.2f}%)")
        print(f"  • Imbalance Ratio: 1 : {neg_count / max(pos_count, 1):.1f}")

    df_features.write_parquet(output_parquet, compression="snappy")
    print(f"\n✅ Saved feature table -> {output_parquet.name} ({output_parquet.stat().st_size / (1024*1024):.2f} MB)")

    total_time = time.time() - start_total
    print(f"✅ Phase 4 Feature Engineering finished in {total_time:.2f} seconds.")
    return output_parquet


if __name__ == "__main__":
    run_feature_pipeline()
