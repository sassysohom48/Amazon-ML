"""
Dual-Pass High-Recall Hybrid Candidate Blocking Engine (Amazon ML Challenge 2026).
Combines Inverted Token Indexing (BM25/IDF), Canonical Sorted-Token Keys,
Address Locality Anchors, and Targeted Fuzzy Fallback partitioned strictly by country.
"""

import math
import time
from array import array
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
import polars as pl
import numpy as np

from .config import (
    CANDIDATES_DIR, PARQUET_DIR, PROCESSED_DIR,
    MAX_CANDIDATES_PER_S1
)
from .normalizer import (
    clean_text, extract_informative_tokens,
    extract_sorted_token_key, extract_char_4grams,
    extract_address_anchors
)
from .evaluator import evaluate_blocking_recall


class HighRecallCountryIndex:
    """
    Dual-Pass in-memory candidate generator optimized for high recall (>=95%)
    and rapid query throughput (>2,000 entities/sec).
    """

    def __init__(self, country: str, max_candidates: int = MAX_CANDIDATES_PER_S1):
        self.country = country.upper()
        self.max_candidates = max_candidates

        # Target ID registry
        self.target_ids: List[str] = []

        # Compact unsigned integer posting lists
        self.token_postings: Dict[str, array] = defaultdict(lambda: array("I"))
        self.sorted_key_postings: Dict[str, array] = defaultdict(lambda: array("I"))
        self.anchor_postings: Dict[str, array] = defaultdict(lambda: array("I"))
        self.ngram_postings: Dict[str, array] = defaultdict(lambda: array("I"))

        # IDF tables
        self.token_idf: Dict[str, float] = {}
        self.ngram_idf: Dict[str, float] = {}

    def fit_target_pool(self, df_s2: pl.DataFrame, df_s3: pl.DataFrame) -> None:
        """
        Indexes all S2 + S3 target entities across sorted keys, tokens, and address anchors.
        """
        start_time = time.time()
        print(f"\n[{self.country}] Indexing Target Records (S2 + S3)...")

        # Combine S2 and S3 for this country
        combined_df = pl.concat([
            df_s2.select(["entity_id", "business_name", "business_address"]),
            df_s3.select(["entity_id", "business_name", "business_address"])
        ])

        n_targets = combined_df.height
        self.target_ids = combined_df["entity_id"].to_list()
        raw_names = combined_df["business_name"].to_list()
        raw_addrs = combined_df["business_address"].to_list()

        token_df_count: Dict[str, int] = defaultdict(int)
        ngram_df_count: Dict[str, int] = defaultdict(int)

        for idx in range(n_targets):
            c_name = clean_text(raw_names[idx])
            c_addr = clean_text(raw_addrs[idx])

            # Channel 1: Canonical Sorted-Token Key (Word-Order Inversion Invariant)
            sorted_key = extract_sorted_token_key(c_name, max_tokens=3)
            if sorted_key:
                self.sorted_key_postings[sorted_key].append(idx)

            # Channel 2: Distinctive Name Tokens
            tokens = set(extract_informative_tokens(c_name))
            for tok in tokens:
                self.token_postings[tok].append(idx)
                token_df_count[tok] += 1

            # Channel 3: Address Locality & PIN Anchors
            if c_addr:
                anchors = set(extract_address_anchors(c_addr))
                for anchor in anchors:
                    self.anchor_postings[anchor].append(idx)

            # Channel 4: Character 4-Grams (indexed for fallback)
            ngrams = extract_char_4grams(c_name)
            for ng in ngrams:
                self.ngram_postings[ng].append(idx)
                ngram_df_count[ng] += 1

        # Compute BM25 IDF weights for tokens
        for tok, df_val in token_df_count.items():
            self.token_idf[tok] = math.log(1.0 + (n_targets - df_val + 0.5) / (df_val + 0.5))

        # Compute IDF for 4-grams
        for ng, df_val in ngram_df_count.items():
            self.ngram_idf[ng] = math.log(1.0 + (n_targets - df_val + 0.5) / (df_val + 0.5))

        elapsed = time.time() - start_time
        print(f"[{self.country}] Indexed {n_targets:,} target records in {elapsed:.2f}s "
              f"({len(self.sorted_key_postings):,} sorted keys, {len(self.token_postings):,} tokens, "
              f"{len(self.anchor_postings):,} address anchors, {len(self.ngram_postings):,} 4-grams)")

    def generate_candidates_for_s1(
        self,
        df_s1: pl.DataFrame
    ) -> Dict[str, List[str]]:
        """
        Executes dual-pass high-speed candidate retrieval:
        Pass 1: Sorted Key + Distinctive Tokens + Address Locality Anchors.
        Pass 2: Fuzzy 4-Gram fallback only for sparse entities (< 5 candidates).
        """
        start_time = time.time()
        n_s1 = df_s1.height
        print(f"[{self.country}] Generating High-Recall Candidates for {n_s1:,} S1 Entities...")

        s1_ids = df_s1["entity_id"].to_list()
        s1_names = df_s1["business_name"].to_list()
        s1_addrs = df_s1["business_address"].to_list()

        results: Dict[str, List[str]] = {}
        max_token_posting_len = 25000

        for i in range(n_s1):
            s1_id = s1_ids[i]
            c_name = clean_text(s1_names[i])
            c_addr = clean_text(s1_addrs[i])

            candidate_scores: Dict[int, float] = defaultdict(float)

            # --- PASS 1: High Precision Channels ---
            # 1. Canonical Sorted Token Key (+12.0 match bonus)
            sorted_key = extract_sorted_token_key(c_name, max_tokens=3)
            if sorted_key:
                pref_targets = self.sorted_key_postings.get(sorted_key)
                if pref_targets and len(pref_targets) <= 5000:
                    for t_idx in pref_targets:
                        candidate_scores[t_idx] += 12.0

            # 2. Address Anchors (+8.0 match bonus)
            if c_addr:
                anchors = extract_address_anchors(c_addr)
                for anchor in anchors:
                    anc_targets = self.anchor_postings.get(anchor)
                    if anc_targets and len(anc_targets) <= 5000:
                        for t_idx in anc_targets:
                            candidate_scores[t_idx] += 8.0

            # 3. Informative Name Tokens (IDF Weighted)
            tokens = extract_informative_tokens(c_name)
            for tok in tokens:
                idf = self.token_idf.get(tok, 1.0)
                targets = self.token_postings.get(tok)
                if targets and len(targets) <= max_token_posting_len:
                    weight = idf * 2.5
                    for t_idx in targets:
                        candidate_scores[t_idx] += weight

            # --- PASS 2: Targeted Fuzzy 4-Gram Fallback (only for sparse entities) ---
            if len(candidate_scores) < 5:
                ngrams = extract_char_4grams(c_name)
                for ng in ngrams:
                    ng_targets = self.ngram_postings.get(ng)
                    if ng_targets and len(ng_targets) <= 3000:
                        idf_g = self.ngram_idf.get(ng, 1.0)
                        weight = idf_g * 0.5
                        for t_idx in ng_targets:
                            candidate_scores[t_idx] += weight

            if not candidate_scores:
                results[s1_id] = []
                continue

            # Take Top-K candidates sorted by cumulative score
            top_targets = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)[:self.max_candidates]
            results[s1_id] = [self.target_ids[t_idx] for t_idx, _ in top_targets]

        elapsed = time.time() - start_time
        print(f"[{self.country}] Candidates generated in {elapsed:.2f}s ({n_s1 / max(elapsed, 0.001):,.0f} entities/s)")
        return results


def candidates_dict_to_dataframe(cand_dict: Dict[str, List[str]]) -> pl.DataFrame:
    """Converts a {s1_id: [cand_ids]} dict into a structured Polars DataFrame."""
    s1_list = []
    cands_list = []
    for s1_id, cands in cand_dict.items():
        s1_list.append(s1_id)
        cands_list.append(",".join(cands))

    return pl.DataFrame({
        "source1_entity_id": s1_list,
        "candidate_entity_ids": cands_list
    })


def run_blocking_pipeline(evaluate_on_val: bool = True) -> None:
    """
    Executes the full candidate blocking pipeline:
    1. Runs blocking on Validation Split (evaluating recall against Ground Truth).
    2. Persists validation candidates for downstream feature engineering.
    """
    start_total = time.time()
    print("=" * 70)
    print("  AMAZON ML CHALLENGE 2026: PHASE 2 CANDIDATE BLOCKING (DUAL-PASS)")
    print("=" * 70)

    if evaluate_on_val:
        print("\n--- Running Dual-Pass Candidate Blocking on Validation Set ---")
        val_s1_path = PROCESSED_DIR / "val_source1.parquet"
        val_gt_path = PROCESSED_DIR / "val_ground_truth.parquet"

        if val_s1_path.exists() and val_gt_path.exists():
            df_val_s1 = pl.read_parquet(val_s1_path)
            val_candidates: Dict[str, List[str]] = {}

            for country in sorted(df_val_s1["country"].unique().to_list()):
                c_s1 = df_val_s1.filter(pl.col("country") == country)
                df_s2 = pl.read_parquet(PARQUET_DIR / f"train_source2_country={country}.parquet")
                df_s3 = pl.read_parquet(PARQUET_DIR / f"train_source3_country={country}.parquet")

                indexer = HighRecallCountryIndex(country=country, max_candidates=MAX_CANDIDATES_PER_S1)
                indexer.fit_target_pool(df_s2, df_s3)
                c_cands = indexer.generate_candidates_for_s1(c_s1)
                val_candidates.update(c_cands)

            # Load Ground Truth and Score Recall
            from .dataset import load_ground_truth
            gt_map = load_ground_truth(val_gt_path)

            val_cand_sets = {k: set(v) for k, v in val_candidates.items()}
            metrics = evaluate_blocking_recall(gt_map, val_cand_sets)
            print("\n" + "=" * 55)
            print("  VALIDATION CANDIDATE BLOCKING METRICS:")
            print("=" * 55)
            print(f"  • Candidate Recall:        {metrics['candidate_recall'] * 100:.2f}% (Recall Ceiling)")
            print(f"  • True Pairs Captured:     {metrics['captured_true_pairs']:,} / {metrics['total_true_pairs']:,}")
            print(f"  • Avg Candidates per S1:   {metrics['avg_candidates_per_s1']} (Max Cap: {MAX_CANDIDATES_PER_S1})")
            print("=" * 55)

            # Persist validation candidates to disk
            val_cand_df = candidates_dict_to_dataframe(val_candidates)
            val_cand_out = CANDIDATES_DIR / "val_candidates.parquet"
            val_cand_df.write_parquet(val_cand_out, compression="snappy")
            print(f"Saved validation candidate pairs to {val_cand_out.name}")

    total_time = time.time() - start_total
    print(f"\nStep 2 Candidate Blocking finished in {total_time:.2f} seconds.")


if __name__ == "__main__":
    run_blocking_pipeline()
s2 = pl.read_parquet(PARQUET_DIR / f"train_source2_country={country}.parquet")
                df_s3 = pl.read_parquet(PARQUET_DIR / f"train_source3_country={country}.parquet")

                indexer = HighRecallCountryIndex(country=country, max_candidates=MAX_CANDIDATES_PER_S1)
                indexer.fit_target_pool(df_s2, df_s3)
                c_cands = indexer.generate_candidates_for_s1(c_s1)
                val_candidates.update(c_cands)

            # Load Ground Truth and Score Recall
            from .dataset import load_ground_truth
            gt_map = load_ground_truth(val_gt_path)

            val_cand_sets = {k: set(v) for k, v in val_candidates.items()}
            metrics = evaluate_blocking_recall(gt_map, val_cand_sets)
            print("\n" + "=" * 55)
            print("  VALIDATION CANDIDATE BLOCKING METRICS:")
            print("=" * 55)
            print(f"  • Candidate Recall:        {metrics['candidate_recall'] * 100:.2f}% (Recall Ceiling)")
            print(f"  • True Pairs Captured:     {metrics['captured_true_pairs']:,} / {metrics['total_true_pairs']:,}")
            print(f"  • Avg Candidates per S1:   {metrics['avg_candidates_per_s1']} (Max Cap: {MAX_CANDIDATES_PER_S1})")
            print("=" * 55)

            # Persist validation candidates to disk
            val_cand_df = candidates_dict_to_dataframe(val_candidates)
            val_cand_out = CANDIDATES_DIR / "val_candidates.parquet"
            val_cand_df.write_parquet(val_cand_out, compression="snappy")
            print(f"✅ Saved validation candidate pairs to {val_cand_out.name}")

    total_time = time.time() - start_total
    print(f"\n✅ Step 2 Candidate Blocking finished in {total_time:.2f} seconds.")


if __name__ == "__main__":
    run_blocking_pipeline()
