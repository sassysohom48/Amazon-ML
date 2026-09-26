"""
Step 3.3: Top-Level Country-Aware Multi-Channel Candidate Blocker (Amazon ML Challenge 2026).
Performs independent multi-channel candidate generation, set union, retrieval provenance tracking,
and adaptive Top-K reranking across US, India, and France.
"""

import time
import heapq
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from src.country_idf import CountryIDFComputer
from src.blocking_channels import CountryMultiChannelIndex


class MultiChannelBlocker:
    """
    Top-Level Multi-Channel Blocker.
    Partitions targets and queries by Country, merges candidates via Set UNION,
    records retrieval provenance bitmasks, and performs lightweight reranking.
    """

    def __init__(
        self,
        max_candidates: int = 35,
        max_posting_len: int = 400,
        max_candidates_per_channel: int = 25,
    ):
        self.max_candidates = max_candidates
        self.max_posting_len = max_posting_len
        self.max_candidates_per_channel = max_candidates_per_channel

        self.idf_computer = CountryIDFComputer()
        self.country_indexes: Dict[str, CountryMultiChannelIndex] = {}

    def fit(self, s2_df: pl.DataFrame, s3_df: pl.DataFrame):
        """
        Combines S2 + S3 target pool, computes country IDF, and fits multi-channel indexes.
        """
        print("\n" + "=" * 80)
        print("BUILDING PHASE 3 MULTI-CHANNEL RETRIEVAL INDEXES ACROSS S2 + S3 TARGETS")
        print("=" * 80)
        t0 = time.time()

        # Combine S2 and S3 target records
        target_df = pl.concat([s2_df, s3_df])
        total_targets = len(target_df)
        print(f"Total Combined Target Records: {total_targets:,}")

        # 1. Fit Country-Specific IDF Computer
        print("Fitting country-specific Name and Address IDF dictionaries...")
        t_idf = time.time()
        self.idf_computer.fit_from_dataframe(target_df)
        print(f"IDF dictionaries computed in {time.time() - t_idf:.2f}s.")

        # 2. Build Inverted Index per Country Partition
        countries = target_df["country"].unique().to_list()
        for country in countries:
            c_name = str(country) if country is not None else "Unknown"
            print(f"\nBuilding 8-Channel Index for country [{c_name}]...")
            c_target = target_df.filter(pl.col("country") == country)
            c_idx = CountryMultiChannelIndex(
                country=c_name,
                idf_computer=self.idf_computer,
                max_posting_len=self.max_posting_len,
                max_candidates_per_channel=self.max_candidates_per_channel,
            )
            c_idx.fit(c_target)
            self.country_indexes[c_name] = c_idx

        print("\n" + "=" * 80)
        print(f"ALL COUNTRY MULTI-CHANNEL INDEXES BUILT IN {time.time() - t0:.2f}s!")
        print("=" * 80)

    def block_dataframe(
        self,
        s1_df: pl.DataFrame,
        max_k: int = 35,
    ) -> Dict[str, List[Dict[str, any]]]:
        """
        Generates candidate pools with provenance for all S1 entities.
        Returns:
          s1_id -> list of candidate dicts:
            [
              {
                "target_id": str,
                "score": float,
                "c_name_core": int,
                "c_name_token": int,
                "c_name_contain": int,
                "c_acronym": int,
                "c_addr_token": int,
                "c_addr_numeric": int,
                "c_postal": int,
                "c_phonetic": int,
                "num_channels": int,
                "channel_mask": int,
              }, ...
            ]
        """
        print(f"\nQuerying Multi-Channel Blocker for {len(s1_df):,} S1 entities (Top-K = {max_k})...")
        t_start = time.time()
        results: Dict[str, List[Dict[str, any]]] = {}

        countries = s1_df["country"].unique().to_list()

        for country in countries:
            c_name = str(country) if country is not None else "Unknown"
            if c_name not in self.country_indexes:
                print(f"[Warning] No target index found for country [{c_name}]! Skipping.")
                continue

            c_idx = self.country_indexes[c_name]
            country_s1 = s1_df.filter(pl.col("country") == country)
            total_country_s1 = len(country_s1)
            print(f"  Querying [{c_name}]: {total_country_s1:,} entities...")

            eids = country_s1["entity_id"].to_list()
            name_cores = country_s1["name_core"].to_list()
            name_tokens = country_s1["name_tokens"].to_list()
            acronyms = country_s1["name_acronym"].to_list()
            phonetics = country_s1["name_phonetic"].to_list()
            
            addr_cleans = country_s1["addr_clean"].to_list()
            addr_tokens = country_s1["addr_tokens"].to_list()
            addr_digits = country_s1["addr_digits"].to_list()
            addr_units = country_s1["addr_unit_num"].to_list()
            postals = country_s1["postal_clean"].to_list()

            t_c = time.time()
            for i in range(total_country_s1):
                raw_candidates = c_idx.query_channels(
                    name_core=str(name_cores[i]) if name_cores[i] else "",
                    name_tokens_str=str(name_tokens[i]) if name_tokens[i] else "",
                    name_acronym=str(acronyms[i]) if acronyms[i] else "",
                    name_phonetic=str(phonetics[i]) if phonetics[i] else "",
                    addr_clean=str(addr_cleans[i]) if addr_cleans[i] else "",
                    addr_tokens_str=str(addr_tokens[i]) if addr_tokens[i] else "",
                    addr_digits_str=str(addr_digits[i]) if addr_digits[i] else "",
                    addr_unit_num=str(addr_units[i]) if addr_units[i] else "",
                    postal_clean=str(postals[i]) if postals[i] else "",
                )

                if not raw_candidates:
                    results[eids[i]] = []
                    continue

                # Rank candidates by score (Set UNION already performed in query_channels)
                ranked_candidates = sorted(
                    raw_candidates.items(),
                    key=lambda item: item[1]["score"],
                    reverse=True
                )[:max_k]

                cand_list = []
                for tgt_int_idx, info in ranked_candidates:
                    tgt_eid = c_idx.target_ids[tgt_int_idx]
                    
                    # Compute 8-bit channel mask
                    mask = (
                        (info["c_name_core"] << 7) |
                        (info["c_name_token"] << 6) |
                        (info["c_name_contain"] << 5) |
                        (info["c_acronym"] << 4) |
                        (info["c_addr_token"] << 3) |
                        (info["c_addr_numeric"] << 2) |
                        (info["c_postal"] << 1) |
                        (info["c_phonetic"] << 0)
                    )

                    cand_list.append({
                        "target_id": tgt_eid,
                        "score": round(info["score"], 2),
                        "c_name_core": info["c_name_core"],
                        "c_name_token": info["c_name_token"],
                        "c_name_contain": info["c_name_contain"],
                        "c_acronym": info["c_acronym"],
                        "c_addr_token": info["c_addr_token"],
                        "c_addr_numeric": info["c_addr_numeric"],
                        "c_postal": info["c_postal"],
                        "c_phonetic": info["c_phonetic"],
                        "num_channels": info["num_channels"],
                        "channel_mask": mask,
                    })

                results[eids[i]] = cand_list

                if (i + 1) % 50000 == 0 or (i + 1) == total_country_s1:
                    speed = (i + 1) / (time.time() - t_c)
                    print(f"    [{c_name}] Processed {i + 1:,} / {total_country_s1:,} ({speed:,.0f} ent/s)...")

            elapsed_c = time.time() - t_c
            print(f"  Finished country [{c_name}] in {elapsed_c:.2f}s ({total_country_s1/elapsed_c:,.0f} ent/s).")

        total_elapsed = time.time() - t_start
        print(f"Candidate generation completed in {total_elapsed:.2f}s ({len(s1_df)/total_elapsed:,.0f} ent/s).")
        return results

    def convert_candidates_to_dataframe(
        self,
        candidate_dict: Dict[str, List[Dict[str, any]]]
    ) -> pl.DataFrame:
        """
        Flattens the candidate dictionary into an enriched Polars DataFrame with full provenance columns.
        """
        col_s1_id = []
        col_tgt_id = []
        col_score = []
        col_rank = []
        col_c_name_core = []
        col_c_name_token = []
        col_c_name_contain = []
        col_c_acronym = []
        col_c_addr_token = []
        col_c_addr_numeric = []
        col_c_postal = []
        col_c_phonetic = []
        col_num_channels = []
        col_channel_mask = []

        for s1_id, cand_list in candidate_dict.items():
            for rank_idx, cand in enumerate(cand_list, start=1):
                col_s1_id.append(s1_id)
                col_tgt_id.append(cand["target_id"])
                col_score.append(cand["score"])
                col_rank.append(rank_idx)
                col_c_name_core.append(cand["c_name_core"])
                col_c_name_token.append(cand["c_name_token"])
                col_c_name_contain.append(cand["c_name_contain"])
                col_c_acronym.append(cand["c_acronym"])
                col_c_addr_token.append(cand["c_addr_token"])
                col_c_addr_numeric.append(cand["c_addr_numeric"])
                col_c_postal.append(cand["c_postal"])
                col_c_phonetic.append(cand["c_phonetic"])
                col_num_channels.append(cand["num_channels"])
                col_channel_mask.append(cand["channel_mask"])

        return pl.DataFrame({
            "source1_entity_id": col_s1_id,
            "target_entity_id": col_tgt_id,
            "candidate_score": col_score,
            "candidate_rank": col_rank,
            "c_name_core": col_c_name_core,
            "c_name_token": col_c_name_token,
            "c_name_contain": col_c_name_contain,
            "c_acronym": col_c_acronym,
            "c_addr_token": col_c_addr_token,
            "c_addr_numeric": col_c_addr_numeric,
            "c_postal": col_c_postal,
            "c_phonetic": col_c_phonetic,
            "num_channels": col_num_channels,
            "channel_mask": col_channel_mask,
        })
