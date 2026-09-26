"""
High-Recall Multi-Channel Candidate Blocking Engine (Amazon ML Challenge 2026).
Combines:
- C1: Canonical / Sorted-Token Keys + Phonetic Soundex Signatures
- C2: Distinctive Name Tokens + Morphological Stemming with BM25-IDF
- C3: High-IDF Boundary Character 3-Grams (^...$)
- C4: Address Locality Anchors (Street number + locality word)
- C5: Standalone Postal PIN Codes (5-6 digits)
Multi-core parallelized and memory-optimized for high throughput (>1,000 ent/s).
"""

import math
import time
import heapq
from array import array
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
import polars as pl
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from .config import (
    CANDIDATES_DIR, PARQUET_DIR, PROCESSED_DIR,
    MAX_CANDIDATES_PER_S1
)
from .normalizer import (
    clean_text, extract_informative_tokens,
    extract_sorted_token_keys, extract_phonetic_keys,
    extract_char_3grams, extract_address_anchors,
    extract_postal_codes
)
from .evaluator import evaluate_blocking_recall, evaluate_blocking_benchmark


class HighRecallCountryIndex:
    """
    Multi-Channel in-memory candidate retrieval engine.
    Engineered for high entity recall (>=92%), high pair recall (>=75-80%),
    and rapid query throughput (>1,000 entities/sec).
    """

    def __init__(self, country: str, max_candidates: int = MAX_CANDIDATES_PER_S1):
        self.country = country.upper()
        self.max_candidates = max_candidates

        # Target ID registry
        self.target_ids: List[str] = []

        # Compact unsigned integer posting lists
        self.c1_sorted_postings: Dict[str, array] = defaultdict(lambda: array("I"))
        self.c2_token_postings: Dict[str, array] = defaultdict(lambda: array("I"))
        self.c3_char3_postings: Dict[str, array] = defaultdict(lambda: array("I"))
        self.c4_addr_postings: Dict[str, array] = defaultdict(lambda: array("I"))
        self.c5_postal_postings: Dict[str, array] = defaultdict(lambda: array("I"))

        # BM25 IDF tables
        self.token_idf: Dict[str, float] = {}
        self.char3_idf: Dict[str, float] = {}

    def fit_target_pool(self, df_s2: pl.DataFrame, df_s3: pl.DataFrame) -> None:
        """
        Indexes all S2 + S3 target entities across all 5 retrieval channels.
        """
        start_time = time.time()
        print(f"\nBuilding High-Recall Multi-Channel Index for country [{self.country}]...")

        combined_df = pl.concat([
            df_s2.select(["entity_id", "business_name", "business_address"]),
            df_s3.select(["entity_id", "business_name", "business_address"])
        ])

        n_targets = combined_df.height
        self.target_ids = combined_df["entity_id"].to_list()
        raw_names = combined_df["business_name"].to_list()
        raw_addrs = combined_df["business_address"].to_list()

        token_df_count: Dict[str, int] = defaultdict(int)
        char3_df_count: Dict[str, int] = defaultdict(int)

        for idx in range(n_targets):
            c_name = clean_text(raw_names[idx])
            c_addr = clean_text(raw_addrs[idx])

            # Channel 1: Canonical / Sorted-Token Keys + Phonetic Keys
            sorted_keys = extract_sorted_token_keys(c_name)
            for sk in sorted_keys:
                self.c1_sorted_postings[sk].append(idx)

            phonetics = extract_phonetic_keys(c_name)
            for ph in phonetics:
                self.c1_sorted_postings[ph].append(idx)

            # Channel 2: Distinctive Name Tokens (with stemming)
            tokens = set(extract_informative_tokens(c_name))
            for tok in tokens:
                self.c2_token_postings[tok].append(idx)
                token_df_count[tok] += 1

            # Channel 3: Boundary-Padded Character 3-Grams
            if c_name:
                shingles = extract_char_3grams(c_name)
                for sh in shingles:
                    self.c3_char3_postings[sh].append(idx)
                    char3_df_count[sh] += 1

            # Channel 4: Address Locality Anchors
            if c_addr:
                anchors = set(extract_address_anchors(c_addr))
                for anchor in anchors:
                    self.c4_addr_postings[anchor].append(idx)

            # Channel 5: Standalone Postal PIN Codes
            if c_addr:
                postals = set(extract_postal_codes(c_addr))
                for pin in postals:
                    self.c5_postal_postings[pin].append(idx)

        # Compute logarithmic BM25 IDF weights for tokens
        for tok, df_val in token_df_count.items():
            self.token_idf[tok] = math.log(1.0 + (n_targets - df_val + 0.5) / (df_val + 0.5))

        # Compute logarithmic BM25 IDF weights for character 3-grams
        for sh, df_val in char3_df_count.items():
            self.char3_idf[sh] = math.log(1.0 + (n_targets - df_val + 0.5) / (df_val + 0.5))

        elapsed = time.time() - start_time
        print(f"    [{self.country}] Multi-Channel Index built for {n_targets:,} records in {elapsed:.2f}s "
              f"(C1 Sorted+Phonetic: {len(self.c1_sorted_postings):,}, "
              f"C2 Tokens: {len(self.c2_token_postings):,}, "
              f"C3 Char3: {len(self.c3_char3_postings):,}, "
              f"C4 AddrTok: {len(self.c4_addr_postings):,}, "
              f"C5 Postal: {len(self.c5_postal_postings):,})")

    def query_single_entity(self, c_name: str, c_addr: str, max_k: int) -> List[str]:
        """Queries the in-memory index for a single pre-cleaned entity."""
        candidate_scores: Dict[int, float] = defaultdict(float)

        # --- Channel 1: Canonical Sorted-Token Keys & Phonetic Keys ---
        sorted_keys = extract_sorted_token_keys(c_name)
        for k_idx, sk in enumerate(sorted_keys):
            targets = self.c1_sorted_postings.get(sk)
            if targets and len(targets) <= 8000:
                weight = 10.0 if k_idx == 0 else 6.0
                for t_idx in targets:
                    candidate_scores[t_idx] += weight

        phonetics = extract_phonetic_keys(c_name)
        for ph in phonetics:
            targets = self.c1_sorted_postings.get(ph)
            if targets and len(targets) <= 6000:
                for t_idx in targets:
                    candidate_scores[t_idx] += 5.0

        # --- Channel 2: Distinctive Name Tokens & Stems (BM25 IDF) ---
        tokens = extract_informative_tokens(c_name)
        for tok in tokens:
            idf = self.token_idf.get(tok, 0.0)
            if idf > 2.0:
                targets = self.c2_token_postings.get(tok)
                if targets and len(targets) <= 15000:
                    weight = idf * 2.2
                    for t_idx in targets:
                        candidate_scores[t_idx] += weight

        # --- Channel 3: Boundary Character 3-Grams (Typo Highway) ---
        if c_name:
            shingles = extract_char_3grams(c_name)
            distinctive_shingles = sorted(
                [sh for sh in shingles if self.char3_idf.get(sh, 0.0) >= 3.5],
                key=lambda x: self.char3_idf[x],
                reverse=True
            )[:4]

            for sh in distinctive_shingles:
                targets = self.c3_char3_postings.get(sh)
                if targets and len(targets) <= 3500:
                    weight = self.char3_idf[sh] * 0.5
                    for t_idx in targets:
                        candidate_scores[t_idx] += weight

        # --- Channel 4: Address Locality Anchors ---
        if c_addr:
            anchors = extract_address_anchors(c_addr)
            for anchor in anchors:
                targets = self.c4_addr_postings.get(anchor)
                if targets and len(targets) <= 5000:
                    for t_idx in targets:
                        candidate_scores[t_idx] += 6.0

        # --- Channel 5: Standalone Postal PIN Codes ---
        if c_addr:
            postals = extract_postal_codes(c_addr)
            for pin in postals:
                targets = self.c5_postal_postings.get(pin)
                if targets and len(targets) <= 5000:
                    for t_idx in targets:
                        candidate_scores[t_idx] += 7.5

        if not candidate_scores:
            return []

        # Use heapq.nlargest for fast C-level Top-K selection (exact same result as sorted()[:max_k])
        top_targets = heapq.nlargest(max_k, candidate_scores.items(), key=lambda x: x[1])
        return [self.target_ids[t_idx] for t_idx, _ in top_targets]

    def generate_candidates_for_s1(
        self,
        df_s1: pl.DataFrame,
        max_k: int = 100,
        num_workers: int = 4
    ) -> Dict[str, List[str]]:
        """
        Queries the multi-channel index for S1 records using multi-threaded execution.
        Parallelizes across worker threads sharing the read-only index memory.
        """
        start_time = time.time()
        n_s1 = df_s1.height
        print(f"[{self.country}] Querying candidates for {n_s1:,} entities (max K = {max_k}, workers = {num_workers})...")

        s1_ids = df_s1["entity_id"].to_list()
        raw_names = df_s1["business_name"].to_list()
        raw_addrs = df_s1["business_address"].to_list()

        # Pre-clean strings once
        clean_names = [clean_text(name) for name in raw_names]
        clean_addrs = [clean_text(addr) for addr in raw_addrs]

        results: Dict[str, List[str]] = {}

        def _worker_task(idx: int) -> Tuple[str, List[str]]:
            sid = s1_ids[idx]
            cands = self.query_single_entity(clean_names[idx], clean_addrs[idx], max_k)
            return sid, cands

        # Execute in parallel threads sharing read-only memory
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            chunk_size = 5000
            for start_idx in range(0, n_s1, chunk_size):
                end_idx = min(start_idx + chunk_size, n_s1)
                batch_indices = range(start_idx, end_idx)
                batch_results = list(executor.map(_worker_task, batch_indices))
                for sid, cands in batch_results:
                    results[sid] = cands

                elapsed_now = time.time() - start_time
                rate = int(end_idx / max(elapsed_now, 0.001))
                print(f"  Blocked {end_idx:,} / {n_s1:,} entities ({rate} ent/s)...")

        elapsed = time.time() - start_time
        final_rate = int(n_s1 / max(elapsed, 0.001))
        print(f"[{self.country}] Candidates generated in {elapsed:.2f}s ({final_rate} entities/s)")
        return results


def candidates_dict_to_dataframe(cand_dict: Dict[str, List[str]], max_k: int = MAX_CANDIDATES_PER_S1) -> pl.DataFrame:
    """Converts a {s1_id: [cand_ids]} dict into a structured Polars DataFrame."""
    s1_list = []
    cands_list = []
    for s1_id, cands in cand_dict.items():
        s1_list.append(s1_id)
        cands_list.append(",".join(cands[:max_k]))

    return pl.DataFrame({
        "source1_entity_id": s1_list,
        "candidate_entity_ids": cands_list
    })


def run_blocking_pipeline(
    evaluate_on_val: bool = True,
    sample_size: Optional[int] = 30000,
    max_k_eval: int = 100
) -> None:
    """Master candidate blocking execution."""
    start_total = time.time()
    print("=" * 80)
    print("  AMAZON ML CHALLENGE 2026: PHASE 3 MULTI-CHANNEL CANDIDATE BLOCKER")
    print("=" * 80)

    val_s1_path = PROCESSED_DIR / "val_source1.parquet"
    val_gt_path = PROCESSED_DIR / "val_ground_truth.parquet"

    if not val_s1_path.exists() or not val_gt_path.exists():
        raise FileNotFoundError("Validation splits not found. Run Step 1 ingestion first.")

    from .dataset import load_ground_truth
    gt_map = load_ground_truth(val_gt_path)

    df_val_s1 = pl.read_parquet(val_s1_path)

    if sample_size and sample_size < df_val_s1.height:
        matched_s1 = [k for k, v in gt_map.items() if len(v) > 0]
        singleton_s1 = [k for k, v in gt_map.items() if len(v) == 0]
        
        np.random.seed(42)
        n_matched = min(len(matched_s1), int(sample_size * 0.70))
        n_single = min(len(singleton_s1), sample_size - n_matched)
        
        sample_s1_set = set(np.random.choice(matched_s1, size=n_matched, replace=False).tolist() +
                            np.random.choice(singleton_s1, size=n_single, replace=False).tolist())
        
        eval_df_s1 = df_val_s1.filter(pl.col("entity_id").is_in(sample_s1_set))
        print(f"Sampled {eval_df_s1.height:,} S1 validation entities ({n_matched:,} with matches, {n_single:,} singletons).")
    else:
        eval_df_s1 = df_val_s1
        print(f"Evaluating across all {eval_df_s1.height:,} validation entities.")

    val_candidates: Dict[str, List[str]] = {}
    country_map: Dict[str, str] = dict(zip(eval_df_s1["entity_id"].to_list(), eval_df_s1["country"].to_list()))

    for country in sorted(eval_df_s1["country"].unique().to_list()):
        c_s1 = eval_df_s1.filter(pl.col("country") == country)
        df_s2 = pl.read_parquet(PARQUET_DIR / f"train_source2_country={country}.parquet")
        df_s3 = pl.read_parquet(PARQUET_DIR / f"train_source3_country={country}.parquet")

        indexer = HighRecallCountryIndex(country=country, max_candidates=max_k_eval)
        indexer.fit_target_pool(df_s2, df_s3)
        c_cands = indexer.generate_candidates_for_s1(c_s1, max_k=max_k_eval, num_workers=4)
        val_candidates.update(c_cands)

    sample_gt_map = {k: gt_map[k] for k in val_candidates if k in gt_map}
    k_steps = [5, 10, 15, 20, 25, 35, 50, 75, 100]
    curve = evaluate_blocking_benchmark(sample_gt_map, val_candidates, k_list=k_steps, country_map=country_map)

    print("\n" + "=" * 95)
    print("CANDIDATE RECALL & ORACLE F0.5 CEILING CURVE (Validation Sample):")
    print("=" * 95)
    print(f"{'K':<6}| {'US PAIR REC':<14}| {'IN PAIR REC':<14}| {'ALL PAIR REC':<15}| {'ENTITY RECALL':<16}| {'ORACLE F0.5'}")
    print("-" * 95)
    for row in curve:
        print(f"{row['k']:<6}| {row['us_pair_recall']*100:>6.2f}%       | {row['in_pair_recall']*100:>6.2f}%       "
              f"| {row['all_pair_recall']*100:>6.2f}%        | {row['entity_recall']*100:>6.2f}%         | {row['oracle_f05']:.4f}")
    print("=" * 95)

    val_cand_df = candidates_dict_to_dataframe(val_candidates, max_k=MAX_CANDIDATES_PER_S1)
    val_cand_out = CANDIDATES_DIR / "val_candidates.parquet"
    val_cand_df.write_parquet(val_cand_out, compression="snappy")
    print(f"\n✅ Saved validation candidates (capped at K={MAX_CANDIDATES_PER_S1}) -> {val_cand_out.name}")

    total_time = time.time() - start_total
    print(f"✅ Phase 3 Candidate Blocking completed in {total_time:.2f} seconds.")


if __name__ == "__main__":
    run_blocking_pipeline()
