"""
Pairwise Feature Engineering Engine (Amazon ML Challenge 2026).
Extracts high-dimensional lexical, structural, numeric, and fuzzy similarity metrics
between S1 entities and candidate S2/S3 records using C++ accelerated RapidFuzz.
"""

import time
import re
from typing import Dict, List, Tuple, Optional
import numpy as np
import polars as pl
from rapidfuzz import fuzz, distance


FEATURE_NAMES = [
    "name_fuzz_ratio",
    "name_fuzz_partial_ratio",
    "name_fuzz_token_sort_ratio",
    "name_fuzz_token_set_ratio",
    "name_fuzz_wratio",
    "name_jaro_winkler",
    "name_exact_match",
    "name_len_diff",
    "name_len_ratio",
    "name_tok_jaccard",
    "name_tok_overlap",
    "name_tok_count_diff",
    "addr_fuzz_ratio",
    "addr_token_set_ratio",
    "addr_partial_ratio",
    "addr_tok_jaccard",
    "addr_tok_overlap",
    "has_addr_both",
    "has_addr_one_missing",
    "has_addr_both_missing",
    "postal_exact_match",
    "postal_both_present",
    "postal_one_missing",
    "street_num_match",
    "name_set_x_addr_jaccard",
    "name_wratio_x_postal",
    "exact_name_missing_addr",
]


def extract_pairwise_feature_vector(
    s1_name: str,
    s1_addr: str,
    s1_tokens: set,
    s1_addr_tokens: set,
    s1_postals: set,
    s1_digits: set,
    tgt_name: str,
    tgt_addr: str,
    tgt_tokens: set,
    tgt_addr_tokens: set,
    tgt_postals: set,
    tgt_digits: set,
) -> List[float]:
    """
    Computes a 27-dimensional feature vector for a single (S1, Target) pair.
    """
    # 1. Fuzzy Name Metrics
    f_ratio = fuzz.ratio(s1_name, tgt_name) / 100.0
    f_partial = fuzz.partial_ratio(s1_name, tgt_name) / 100.0
    f_sort = fuzz.token_sort_ratio(s1_name, tgt_name) / 100.0
    f_set = fuzz.token_set_ratio(s1_name, tgt_name) / 100.0
    f_wratio = fuzz.WRatio(s1_name, tgt_name) / 100.0
    f_jaro = distance.JaroWinkler.similarity(s1_name, tgt_name)
    exact_name = 1.0 if (s1_name and s1_name == tgt_name) else 0.0

    # 2. Name Length & Token Metrics
    l1, l2 = len(s1_name), len(tgt_name)
    len_diff = float(abs(l1 - l2))
    len_ratio = (min(l1, l2) / max(l1, l2, 1)) if (l1 > 0 or l2 > 0) else 0.0

    tok_inter_len = len(s1_tokens & tgt_tokens)
    tok_union_len = len(s1_tokens | tgt_tokens)
    tok_jaccard = (tok_inter_len / tok_union_len) if tok_union_len > 0 else 0.0
    tok_overlap = float(tok_inter_len)
    tok_count_diff = float(abs(len(s1_tokens) - len(tgt_tokens)))

    # 3. Fuzzy Address Metrics
    has_both_addr = 1.0 if (s1_addr and tgt_addr) else 0.0
    has_one_missing_addr = 1.0 if (bool(s1_addr) != bool(tgt_addr)) else 0.0
    has_both_missing_addr = 1.0 if (not s1_addr and not tgt_addr) else 0.0

    if has_both_addr:
        addr_ratio = fuzz.ratio(s1_addr, tgt_addr) / 100.0
        addr_set_ratio = fuzz.token_set_ratio(s1_addr, tgt_addr) / 100.0
        addr_partial = fuzz.partial_ratio(s1_addr, tgt_addr) / 100.0
        addr_inter_len = len(s1_addr_tokens & tgt_addr_tokens)
        addr_union_len = len(s1_addr_tokens | tgt_addr_tokens)
        addr_jaccard = (addr_inter_len / addr_union_len) if addr_union_len > 0 else 0.0
        addr_overlap = float(addr_inter_len)
    else:
        addr_ratio = 0.0
        addr_set_ratio = 0.0
        addr_partial = 0.0
        addr_jaccard = 0.0
        addr_overlap = 0.0

    # 4. Postal & Numeric Metrics
    has_both_postal = 1.0 if (s1_postals and tgt_postals) else 0.0
    has_one_missing_postal = 1.0 if (bool(s1_postals) != bool(tgt_postals)) else 0.0
    postal_match = 1.0 if (s1_postals and tgt_postals and (s1_postals & tgt_postals)) else 0.0
    digit_match = 1.0 if (s1_digits and tgt_digits and (s1_digits & tgt_digits)) else 0.0

    # 5. Cross Interactions
    set_x_addr = f_set * addr_jaccard
    postal_weight = 1.0 if postal_match else (0.75 if has_one_missing_postal else 0.25)
    wratio_x_postal = f_wratio * postal_weight
    exact_missing_addr = 1.0 if (exact_name > 0 and (not s1_addr or not tgt_addr)) else 0.0

    return [
        f_ratio,
        f_partial,
        f_sort,
        f_set,
        f_wratio,
        f_jaro,
        exact_name,
        len_diff,
        len_ratio,
        tok_jaccard,
        tok_overlap,
        tok_count_diff,
        addr_ratio,
        addr_set_ratio,
        addr_partial,
        addr_jaccard,
        addr_overlap,
        has_both_addr,
        has_one_missing_addr,
        has_both_missing_addr,
        postal_match,
        has_both_postal,
        has_one_missing_postal,
        digit_match,
        set_x_addr,
        wratio_x_postal,
        exact_missing_addr,
    ]


class FeatureExtractor:
    """
    High-throughput feature extractor for candidate pairs.
    Pre-caches entity attributes for O(1) attribute lookup and rapid C++ feature computation.
    """

    def __init__(self):
        self.entity_lookup: Dict[str, Tuple[str, str, set, set, set, set]] = {}

    def register_dataset(self, df: pl.DataFrame, needed_eids: Optional[Set[str]] = None):
        """
        Registers a Polars DataFrame into the fast attribute lookup dictionary.
        If needed_eids is provided, only registers rows whose entity_id is in needed_eids.
        """
        eids = df["entity_id"].to_list()
        names = df["name_clean"].to_list()
        addrs_col_name = "addr_clean" if "addr_clean" in df.columns else "address_clean"
        addrs = df[addrs_col_name].to_list()
        tokens = df["name_tokens"].to_list()
        postals = df["postal_digits"].to_list()

        digit_pattern = re.compile(r"\b\d+\b")

        for eid, name, addr, tok_str, post_str in zip(eids, names, addrs, tokens, postals):
            if needed_eids is not None and eid not in needed_eids:
                continue

            name_str = name if name else ""
            addr_str = addr if addr else ""
            name_tok_set = set(tok_str.split()) if tok_str else set()
            addr_tok_set = set(addr_str.split()) if addr_str else set()
            postal_set = set(post_str.split()) if post_str else set()
            digits_set = set(digit_pattern.findall(addr_str)) if addr_str else set()

            self.entity_lookup[eid] = (
                name_str,
                addr_str,
                name_tok_set,
                addr_tok_set,
                postal_set,
                digits_set,
            )

    def extract_features_for_pairs(
        self,
        pairs: List[Tuple[str, str]],
        labels: Optional[List[int]] = None,
        batch_size: int = 50000,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Extracts feature matrix X (and optional label vector y) for a list of (s1_id, target_id) pairs.
        """
        n_pairs = len(pairs)
        if n_pairs == 0:
            return np.empty((0, len(FEATURE_NAMES)), dtype=np.float32), (np.empty(0, dtype=np.int32) if labels is not None else None)

        print(f"Extracting features for {n_pairs:,} candidate pairs...")
        t0 = time.time()

        feature_rows = []
        valid_labels = []

        for idx, (s1_id, tgt_id) in enumerate(pairs):
            if s1_id not in self.entity_lookup or tgt_id not in self.entity_lookup:
                continue

            s1_info = self.entity_lookup[s1_id]
            tgt_info = self.entity_lookup[tgt_id]

            row = extract_pairwise_feature_vector(
                s1_info[0], s1_info[1], s1_info[2], s1_info[3], s1_info[4], s1_info[5],
                tgt_info[0], tgt_info[1], tgt_info[2], tgt_info[3], tgt_info[4], tgt_info[5],
            )
            feature_rows.append(row)

            if labels is not None:
                valid_labels.append(labels[idx])

        X = np.array(feature_rows, dtype=np.float32)
        y = np.array(valid_labels, dtype=np.int32) if labels is not None else None

        elapsed = time.time() - t0
        print(f"Extracted {len(X):,} feature rows in {elapsed:.2f}s ({len(X)/elapsed:,.0f} pairs/s).")
        return X, y
