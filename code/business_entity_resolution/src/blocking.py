"""
High-Recall Multi-Index Blocking Engine (Amazon ML Challenge 2026).
Combines country partitioning, exact name hashing, name bigrams, IDF-weighted tokens,
distinctive address tokens, and postal code candidate generation.
"""

import time
import heapq
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional
import polars as pl


class CountryInvertedIndexBlocker:
    """
    Inverted Index Blocker for a single country (US, India, France).
    Indexes target records (S2 and S3) across both Name and Address fields.
    """

    def __init__(self, max_candidates: int = 35, max_token_freq: int = 35000):
        self.max_candidates = max_candidates
        self.max_token_freq = max_token_freq

        # Inverted Indexes
        self.exact_name_index = defaultdict(list)
        self.bigram_index = defaultdict(list)
        self.token_index = defaultdict(list)
        self.token_weights = {}
        self.addr_token_index = defaultdict(list)
        self.addr_token_weights = {}
        self.postal_index = defaultdict(list)

        # Target metadata arrays for fast integer-indexed access
        self.target_ids: List[str] = []
        self.target_name_clean: List[str] = []
        self.target_tokens_set: List[Set[str]] = []
        self.target_addr_tokens_set: List[Set[str]] = []
        self.target_postals: List[Set[str]] = []

    def fit(self, target_df: pl.DataFrame):
        """
        Builds multi-field inverted indexes over target records (S2 + S3).
        """
        start_time = time.time()
        n_rows = len(target_df)

        eids = target_df["entity_id"].to_list()
        names = target_df["name_clean"].to_list()
        tokens_col = target_df["name_tokens"].to_list()
        
        addr_col_name = "addr_clean" if "addr_clean" in target_df.columns else "address_clean"
        addrs_col = target_df[addr_col_name].to_list()
        postals_col = target_df["postal_digits"].to_list()

        self.target_ids = eids

        # 1. Exact Name Index & Bigrams
        for idx, (name, tok_str) in enumerate(zip(names, tokens_col)):
            if name:
                self.exact_name_index[name].append(idx)
            
            if tok_str:
                tok_list = tok_str.split()
                for i in range(len(tok_list) - 1):
                    bg = f"{tok_list[i]}_{tok_list[i+1]}"
                    self.bigram_index[bg].append(idx)

        # 2. Token Inverted Index
        raw_token_postings = defaultdict(list)
        for idx, tok_str in enumerate(tokens_col):
            if tok_str:
                for token in set(tok_str.split()):
                    raw_token_postings[token].append(idx)

        for token, postings in raw_token_postings.items():
            p_len = len(postings)
            if p_len <= self.max_token_freq:
                self.token_index[token] = postings
                self.token_weights[token] = 12.0 / (p_len ** 0.35)

        # 3. Address Token Inverted Index (Distinctive words, len >= 3)
        raw_addr_postings = defaultdict(list)
        for idx, addr_str in enumerate(addrs_col):
            if addr_str:
                for at in set(addr_str.split()):
                    if len(at) >= 3:
                        raw_addr_postings[at].append(idx)

        for at, postings in raw_addr_postings.items():
            p_len = len(postings)
            if p_len <= 8000:
                self.addr_token_index[at] = postings
                self.addr_token_weights[at] = 8.0 / (p_len ** 0.35)

        # 4. Postal Code Index
        for idx, post_str in enumerate(postals_col):
            if post_str:
                for p in set(post_str.split()):
                    if len(p) >= 4:
                        self.postal_index[p].append(idx)

        # Filter bigrams and postal postings
        self.bigram_index = {k: v for k, v in self.bigram_index.items() if len(v) <= 10000}
        self.postal_index = {k: v for k, v in self.postal_index.items() if len(v) <= 500}

        elapsed = time.time() - start_time
        print(f"    Indexed {n_rows:,} target records in {elapsed:.2f}s "
              f"({len(self.exact_name_index):,} exact names, {len(self.token_index):,} name tokens, "
              f"{len(self.addr_token_index):,} addr tokens, {len(self.postal_index):,} postal codes).")

    def query_entity(
        self,
        name_clean: str,
        tokens_str: str,
        addr_clean: str,
        postal_str: str
    ) -> List[str]:
        """
        Retrieves top candidate target IDs for a single S1 record.
        """
        s1_tokens = tokens_str.split() if tokens_str else []
        s1_addr_tokens = [at for at in addr_clean.split() if len(at) >= 3] if addr_clean else []
        s1_postals = postal_str.split() if postal_str else []

        candidate_scores = defaultdict(float)

        # Signal 1: Exact Name Match (+100.0 priority)
        if name_clean in self.exact_name_index:
            for idx in self.exact_name_index[name_clean]:
                candidate_scores[idx] += 100.0

        # Signal 2: Name Bigrams (+35.0 priority)
        for i in range(len(s1_tokens) - 1):
            bg = f"{s1_tokens[i]}_{s1_tokens[i+1]}"
            if bg in self.bigram_index:
                for idx in self.bigram_index[bg]:
                    candidate_scores[idx] += 35.0

        # Signal 3: Informative Name Tokens (IDF Weighted, top-4 rarest tokens)
        if s1_tokens:
            sorted_tokens = sorted(
                [t for t in s1_tokens if t in self.token_index],
                key=lambda t: len(self.token_index[t])
            )
            for token in sorted_tokens[:4]:
                w = self.token_weights[token]
                for idx in self.token_index[token]:
                    candidate_scores[idx] += w

        # Signal 4: Distinctive Address Tokens (IDF Weighted, top-3 rarest tokens)
        if s1_addr_tokens:
            sorted_addr_toks = sorted(
                [at for at in s1_addr_tokens if at in self.addr_token_index],
                key=lambda at: len(self.addr_token_index[at])
            )
            for at in sorted_addr_toks[:3]:
                w = self.addr_token_weights[at]
                for idx in self.addr_token_index[at]:
                    candidate_scores[idx] += w

        # Signal 5: Postal Code match
        for postal in s1_postals:
            if postal in self.postal_index:
                for idx in self.postal_index[postal]:
                    candidate_scores[idx] += 2.0

        if not candidate_scores:
            return []

        # Rank candidates by score and pick top K
        if len(candidate_scores) <= self.max_candidates:
            return [self.target_ids[idx] for idx in candidate_scores.keys()]
        
        top_k = heapq.nlargest(self.max_candidates, candidate_scores.items(), key=lambda x: x[1])
        return [self.target_ids[idx] for idx, _ in top_k]


class MultiIndexBlocker:
    """
    Top-level Multi-Index Blocker that partitions data by country (US, India, France)
    and executes high-recall candidate generation.
    """

    def __init__(self, max_candidates: int = 35, max_token_freq: int = 35000):
        self.max_candidates = max_candidates
        self.max_token_freq = max_token_freq
        self.country_blockers: Dict[str, CountryInvertedIndexBlocker] = {}

    def fit(self, s2_df: pl.DataFrame, s3_df: pl.DataFrame):
        """
        Combines S2 and S3, partitions by country, and builds country-specific blockers.
        """
        print("\nBuilding Multi-Index Blockers across S2 and S3...")
        start_time = time.time()

        # Combine S2 and S3 target records
        target_df = pl.concat([s2_df, s3_df])
        total_targets = len(target_df)
        print(f"Total Target Pool (S2 + S3): {total_targets:,} records.")

        countries = target_df["country"].unique().to_list()
        for country in countries:
            print(f"  Building index for country [{country}]...")
            country_target = target_df.filter(pl.col("country") == country)
            blocker = CountryInvertedIndexBlocker(
                max_token_freq=self.max_token_freq,
                max_candidates=self.max_candidates
            )
            blocker.fit(country_target)
            self.country_blockers[country] = blocker

        print(f"All country blockers built successfully in {time.time() - start_time:.2f}s total.")

    def block_s1(self, s1_df: pl.DataFrame) -> Dict[str, List[str]]:
        """
        Generates candidate pairs for a DataFrame of S1 records.
        Returns: Dict mapping source1_entity_id -> list of candidate entity IDs.
        """
        print(f"\nGenerating candidates for {len(s1_df):,} S1 entities...")
        start_time = time.time()

        results: Dict[str, List[str]] = {}
        countries = s1_df["country"].unique().to_list()

        addr_col_name = "addr_clean" if "addr_clean" in s1_df.columns else "address_clean"

        for country in countries:
            if country not in self.country_blockers:
                print(f"[Warning] No target records found for country [{country}]!")
                continue

            blocker = self.country_blockers[country]
            country_s1 = s1_df.filter(pl.col("country") == country)
            print(f"  Blocking {len(country_s1):,} entities for country [{country}]...")

            eids = country_s1["entity_id"].to_list()
            names = country_s1["name_clean"].to_list()
            tokens = country_s1["name_tokens"].to_list()
            addrs = country_s1[addr_col_name].to_list()
            postals = country_s1["postal_digits"].to_list()

            c_start = time.time()
            for eid, name, tok, addr, post in zip(eids, names, tokens, addrs, postals):
                candidates = blocker.query_entity(name, tok, addr, post)
                results[eid] = candidates

            c_elapsed = time.time() - c_start
            print(f"  Finished country [{country}] in {c_elapsed:.2f}s ({len(country_s1)/c_elapsed:,.0f} entities/s).")

        total_elapsed = time.time() - start_time
        print(f"Candidate blocking completed in {total_elapsed:.2f}s ({len(s1_df)/total_elapsed:,.0f} entities/s).")
        return results
