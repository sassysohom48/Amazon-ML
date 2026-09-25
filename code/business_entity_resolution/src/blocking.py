"""
High-Recall Hybrid Candidate Generation & Blocking Engine (Amazon ML Challenge 2026).
Combines Inverted Token Index (BM25/TF-IDF), Character 3-Gram Overlap,
and Exact Numeric/Address Anchors partitioned strictly by country.
"""

import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
import polars as pl
import numpy as np

from .config import (
    CANDIDATES_DIR, PARQUET_DIR, PROCESSED_DIR,
    MAX_CANDIDATES_PER_S1, BM25_TOP_K
)
from .normalizer import (
    clean_text, extract_informative_tokens,
    extract_char_ngrams, extract_numeric_tokens
)
from .evaluator import evaluate_blocking_recall


class CountryBlockingIndex:
    """
    High-performance, in-memory inverted index and candidate generator
    built specifically for a single country partition.
    """

    def __init__(self, country: str, max_candidates: int = MAX_CANDIDATES_PER_S1):
        self.country = country.upper()
        self.max_candidates = max_candidates

        # Target pool stores (S2 and S3 records)
        self.target_ids: List[str] = []
        self.target_names: List[str] = []
        self.target_addrs: List[str] = []

        # Inverted index mappings: token -> list of target indices
        self.token_to_targets: Dict[str, List[int]] = defaultdict(list)
        self.token_idf: Dict[str, float] = {}

        # Character 3-gram index: 3gram -> list of target indices
        self.ngram_to_targets: Dict[str, List[int]] = defaultdict(list)

        # Numeric token index: digit_string -> list of target indices
        self.num_to_targets: Dict[str, List[int]] = defaultdict(list)

    def fit_target_pool(self, df_s2: pl.DataFrame, df_s3: pl.DataFrame) -> None:
        """
        Builds inverted indices across the combined target records (S2 + S3) for this country.
        """
        start_time = time.time()
        print(f"[{self.country}] Indexing Target Records (S2 + S3)...")

        # Combine S2 and S3 for this country
        combined_df = pl.concat([
            df_s2.select(["entity_id", "business_name", "business_address"]),
            df_s3.select(["entity_id", "business_name", "business_address"])
        ])

        n_targets = combined_df.height
        self.target_ids = combined_df["entity_id"].to_list()
        raw_names = combined_df["business_name"].to_list()
        raw_addrs = combined_df["business_address"].to_list()

        # Temporary document frequency tracker
        doc_freq: Dict[str, int] = defaultdict(int)

        for idx in range(n_targets):
            c_name = clean_text(raw_names[idx])
            c_addr = clean_text(raw_addrs[idx])

            self.target_names.append(c_name)
            self.target_addrs.append(c_addr)

            # 1. Informative Name Tokens
            tokens = set(extract_informative_tokens(c_name))
            for tok in tokens:
                self.token_to_targets[tok].append(idx)
                doc_freq[tok] += 1

            # 2. Character 3-Grams (indexed for rare/distinctive n-grams)
            ngrams = extract_char_ngrams(c_name, n=3)
            for ng in ngrams:
                if len(self.ngram_to_targets[ng]) < 1500:  # Cap posting list size to keep index lean
                    self.ngram_to_targets[ng].append(idx)

            # 3. Numeric Tokens from Address
            if c_addr:
                nums = extract_numeric_tokens(c_addr)
                for num in nums:
                    if len(num) >= 3 and len(self.num_to_targets[num]) < 2000:
                        self.num_to_targets[num].append(idx)

        # Compute IDF weights: log((N - df + 0.5) / (df + 0.5) + 1)
        for tok, df_count in doc_freq.items():
            self.token_idf[tok] = math.log(1.0 + (n_targets - df_count + 0.5) / (df_count + 0.5))

        elapsed = time.time() - start_time
        print(f"[{self.country}] Indexed {n_targets:,} target records in {elapsed:.2f}s "
              f"({len(self.token_to_targets):,} unique tokens, {len(self.ngram_to_targets):,} n-grams)")

    def generate_candidates_for_s1(
        self,
        df_s1: pl.DataFrame
    ) -> Dict[str, List[str]]:
        """
        Generates top-K candidate target IDs for every S1 entity in the given DataFrame.
        """
        start_time = time.time()
        n_s1 = df_s1.height
        print(f"[{self.country}] Generating Candidates for {n_s1:,} S1 Entities...")

        s1_ids = df_s1["entity_id"].to_list()
        s1_names = df_s1["business_name"].to_list()
        s1_addrs = df_s1["business_address"].to_list()

        results: Dict[str, List[str]] = {}

        for i in range(n_s1):
            s1_id = s1_ids[i]
            c_name = clean_text(s1_names[i])
            c_addr = clean_text(s1_addrs[i])

            candidate_scores: Dict[int, float] = defaultdict(float)

            # Channel 1: Token Inverted Index Scoring (IDF Weighted)
            tokens = extract_informative_tokens(c_name)
            for tok in tokens:
                idf = self.token_idf.get(tok, 1.0)
                targets = self.token_to_targets.get(tok, [])
                # Only score postings with manageable collision frequency
                if targets and len(targets) <= 3000:
                    score_gain = idf * 2.0
                    for t_idx in targets:
                        candidate_scores[t_idx] += score_gain

            # Channel 2: Character 3-Gram Overlap
            ngrams = extract_char_ngrams(c_name, n=3)
            for ng in ngrams:
                targets = self.ngram_to_targets.get(ng, [])
                if targets and len(targets) <= 1000:
                    for t_idx in targets:
                        candidate_scores[t_idx] += 0.25

            # Channel 3: Numeric Address Anchors (High Precision Booster)
            if c_addr:
                nums = extract_numeric_tokens(c_addr)
                for num in nums:
                    if len(num) >= 3:
                        targets = self.num_to_targets.get(num, [])
                        if targets and len(targets) <= 1500:
                            for t_idx in targets:
                                candidate_scores[t_idx] += 1.5

            if not candidate_scores:
                results[s1_id] = []
                continue

            # Sort and select Top-K candidates
            top_targets = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)[:self.max_candidates]
            results[s1_id] = [self.target_ids[t_idx] for t_idx, _ in top_targets]

        elapsed = time.time() - start_time
        print(f"[{self.country}] Candidate generation completed in {elapsed:.2f}s ({n_s1 / max(elapsed, 0.001):,.0f} entities/s)")
        return results


def run_blocking_for_country(
    country: str,
    s1_table: str,
    s2_table: str,
    s3_table: str,
    max_candidates: int = MAX_CANDIDATES_PER_S1
) -> Dict[str, List[str]]:
    """Loads country partitions, builds the index, and generates candidates."""
    country_clean = country.strip().upper()

    df_s1 = pl.read_parquet(PARQUET_DIR / f"{s1_table}_country={country_clean}.parquet")
    df_s2 = pl.read_parquet(PARQUET_DIR / f"{s2_table}_country={country_clean}.parquet")
    df_s3 = pl.read_parquet(PARQUET_DIR / f"{s3_table}_country={country_clean}.parquet")

    indexer = CountryBlockingIndex(country=country_clean, max_candidates=max_candidates)
    indexer.fit_target_pool(df_s2, df_s3)
    return indexer.generate_candidates_for_s1(df_s1)


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
    2. Runs blocking across all Train splits (for generating hard-negative training pairs).
    3. Runs blocking across all Test splits (for final leaderboard candidate generation).
    """
    start_total = time.time()
    print("=" * 70)
    print("  AMAZON ML CHALLENGE 2026: PHASE 2 CANDIDATE BLOCKING")
    print("=" * 70)

    # 1. Validation Split Blocking & Evaluation
    if evaluate_on_val:
        print("\n--- Running Candidate Blocking on Validation Set ---")
        val_s1_path = PROCESSED_DIR / "val_source1.parquet"
        val_gt_path = PROCESSED_DIR / "val_ground_truth.parquet"

        if val_s1_path.exists() and val_gt_path.exists():
            df_val_s1 = pl.read_parquet(val_s1_path)
            val_candidates: Dict[str, Set[str]] = {}

            for country in sorted(df_val_s1["country"].unique().to_list()):
                c_s1 = df_val_s1.filter(pl.col("country") == country)
                df_s2 = pl.read_parquet(PARQUET_DIR / f"train_source2_country={country}.parquet")
                df_s3 = pl.read_parquet(PARQUET_DIR / f"train_source3_country={country}.parquet")

                indexer = CountryBlockingIndex(country=country, max_candidates=MAX_CANDIDATES_PER_S1)
                indexer.fit_target_pool(df_s2, df_s3)
                c_cands = indexer.generate_candidates_for_s1(c_s1)
                for s1_id, c_list in c_cands.items():
                    val_candidates[s1_id] = set(c_list)

            # Load Ground Truth and Score Recall
            from .dataset import load_ground_truth
            gt_map = load_ground_truth(val_gt_path)

            metrics = evaluate_blocking_recall(gt_map, val_candidates)
            print("\n" + "=" * 50)
            print("  VALIDATION CANDIDATE BLOCKING METRICS:")
            print("=" * 50)
            print(f"  • Candidate Recall:        {metrics['candidate_recall'] * 100:.2f}% (Recall Ceiling)")
            print(f"  • True Pairs Captured:     {metrics['captured_true_pairs']:,} / {metrics['total_true_pairs']:,}")
            print(f"  • Avg Candidates per S1:   {metrics['avg_candidates_per_s1']} (Max Cap: {MAX_CANDIDATES_PER_S1})")
            print("=" * 50)

            # Persist validation candidates to disk for fast resumption
            val_cand_df = candidates_dict_to_dataframe(
                {s1_id: list(cands) for s1_id, cands in val_candidates.items()}
            )
            val_cand_out = CANDIDATES_DIR / "val_candidates.parquet"
            val_cand_df.write_parquet(val_cand_out, compression="snappy")
            print(f"✅ Saved validation candidate pairs to {val_cand_out.name}")

    total_time = time.time() - start_total
    print(f"\n✅ Step 2 Candidate Blocking finished in {total_time:.2f} seconds.")



if __name__ == "__main__":
    run_blocking_pipeline()
