"""
Step 3.3: High-Recall Streaming Multi-Channel Blocker (Amazon ML Challenge 2026).
Streams candidate generation directly to Parquet in memory-safe chunks (< 250 MB RAM),
recording 9-channel retrieval provenance bitmasks and multi-signal ranking.
"""

import time
import gc
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from src.country_idf import CountryIDFComputer
from src.blocking_channels import CountryMultiChannelIndex


CANDIDATE_SCHEMA = pa.schema([
    ("source1_entity_id", pa.string()),
    ("target_entity_id", pa.string()),
    ("candidate_score", pa.float32()),
    ("candidate_rank", pa.int16()),
    ("c_name_core", pa.int8()),
    ("c_name_token", pa.int8()),
    ("c_name_contain", pa.int8()),
    ("c_char_3gram", pa.int8()),
    ("c_acronym", pa.int8()),
    ("c_addr_token", pa.int8()),
    ("c_addr_numeric", pa.int8()),
    ("c_postal", pa.int8()),
    ("c_phonetic", pa.int8()),
    ("num_channels", pa.int8()),
    ("channel_mask", pa.int16()),
])


class MultiChannelBlocker:
    """
    Top-Level Multi-Channel Candidate Blocker with Streaming Parquet Export.
    """

    def __init__(
        self,
        max_candidates: int = 50,
        max_candidates_per_channel: int = 40,
    ):
        self.max_candidates = max_candidates
        self.max_candidates_per_channel = max_candidates_per_channel

        self.idf_computer = CountryIDFComputer()
        self.country_indexes: Dict[str, CountryMultiChannelIndex] = {}

    def fit(self, s2_df: pl.DataFrame, s3_df: pl.DataFrame):
        """
        Fits multi-channel indexes country-by-country without duplicating datasets in RAM.
        """
        print("\n" + "=" * 80)
        print("BUILDING PHASE 3 MULTI-CHANNEL RETRIEVAL INDEXES ACROSS S2 + S3 TARGETS")
        print("=" * 80)
        t0 = time.time()

        total_targets = len(s2_df) + len(s3_df)
        print(f"Total Combined Target Records: {total_targets:,}")

        # 1. Fit Country-Specific IDF Computer
        print("Fitting country-specific Name and Address IDF dictionaries...")
        t_idf = time.time()
        self.idf_computer.fit_from_dataframes(s2_df, s3_df)
        print(f"IDF dictionaries computed in {time.time() - t_idf:.2f}s.")

        # 2. Build Inverted Index per Country Partition
        countries = list(set(s2_df["country"].unique().to_list()) | set(s3_df["country"].unique().to_list()))
        for country in countries:
            c_name = str(country) if country is not None else "Unknown"
            print(f"\nBuilding High-Recall Multi-Channel Index for country [{c_name}]...")
            s2_c = s2_df.filter(pl.col("country") == country)
            s3_c = s3_df.filter(pl.col("country") == country)
            c_target = pl.concat([s2_c, s3_c])
            del s2_c, s3_c
            gc.collect()

            c_idx = CountryMultiChannelIndex(
                country=c_name,
                idf_computer=self.idf_computer,
                max_candidates_per_channel=self.max_candidates_per_channel,
            )
            c_idx.fit(c_target)
            self.country_indexes[c_name] = c_idx
            del c_target
            gc.collect()

        print("\n" + "=" * 80)
        print(f"ALL COUNTRY MULTI-CHANNEL INDEXES BUILT IN {time.time() - t0:.2f}s!")
        print("=" * 80)

    def query_entity_candidates(
        self,
        country: str,
        name_clean: str,
        name_core: str,
        name_tokens: str,
        name_acronym: str,
        name_phonetic: str,
        addr_clean: str,
        addr_tokens: str,
        addr_digits: str,
        addr_unit_num: str,
        postal_clean: str,
        max_k: int = 50,
    ) -> List[Dict[str, any]]:
        """Queries candidates for a single entity."""
        if country not in self.country_indexes:
            return []

        c_idx = self.country_indexes[country]
        raw_candidates = c_idx.query_channels(
            name_clean=name_clean,
            name_core=name_core,
            name_tokens_str=name_tokens,
            name_acronym=name_acronym,
            name_phonetic=name_phonetic,
            addr_clean=addr_clean,
            addr_tokens_str=addr_tokens,
            addr_digits_str=addr_digits,
            addr_unit_num=addr_unit_num,
            postal_clean=postal_clean,
        )

        if not raw_candidates:
            return []

        ranked_candidates = c_idx.select_top_candidates(raw_candidates, max_k=max_k)

        cand_list = []
        for tgt_int_idx, c in ranked_candidates:
            tgt_eid = c_idx.target_ids[tgt_int_idx]
            mask = c[3]
            cand_list.append({
                "target_id": tgt_eid,
                "score": round(c[0], 2),
                "c_name_core": (mask >> 8) & 1,
                "c_name_token": (mask >> 7) & 1,
                "c_name_contain": (mask >> 6) & 1,
                "c_char_3gram": (mask >> 5) & 1,
                "c_acronym": (mask >> 4) & 1,
                "c_addr_token": (mask >> 3) & 1,
                "c_addr_numeric": (mask >> 2) & 1,
                "c_postal": (mask >> 1) & 1,
                "c_phonetic": mask & 1,
                "num_channels": c[4],
                "channel_mask": mask,
            })
        return cand_list

    def block_dataframe(
        self,
        s1_df: pl.DataFrame,
        max_k: int = 50,
    ) -> Dict[str, List[Dict[str, any]]]:
        """
        In-memory candidate blocking for benchmark evaluation.
        """
        results: Dict[str, List[Dict[str, any]]] = {}
        eids = s1_df["entity_id"].to_list()
        countries = s1_df["country"].to_list()
        name_cleans = s1_df["name_clean"].to_list() if "name_clean" in s1_df.columns else s1_df["name_core"].to_list()
        name_cores = s1_df["name_core"].to_list()
        name_tokens = s1_df["name_tokens"].to_list()
        acronyms = s1_df["name_acronym"].to_list()
        phonetics = s1_df["name_phonetic"].to_list()
        addr_cleans = s1_df["addr_clean"].to_list()
        addr_tokens = s1_df["addr_tokens"].to_list()
        addr_digits = s1_df["addr_digits"].to_list()
        addr_units = s1_df["addr_unit_num"].to_list()
        postals = s1_df["postal_clean"].to_list()

        n_rows = len(s1_df)
        t_start = time.time()
        for i in range(n_rows):
            c_name = str(countries[i]) if countries[i] else "Unknown"
            c_idx = self.country_indexes.get(c_name)
            if c_idx is None:
                results[eids[i]] = []
                continue

            raw_cands = c_idx.query_channels(
                name_clean=str(name_cleans[i]) if name_cleans[i] else "",
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
            if not raw_cands:
                results[eids[i]] = []
            else:
                ranked = c_idx.select_top_candidates(raw_cands, max_k=max_k)
                cand_list = []
                for tgt_int_idx, c in ranked:
                    tgt_eid = c_idx.target_ids[tgt_int_idx]
                    mask = c[3]
                    cand_list.append({
                        "target_id": tgt_eid,
                        "score": round(c[0], 2),
                        "c_name_core": (mask >> 8) & 1,
                        "c_name_token": (mask >> 7) & 1,
                        "c_name_contain": (mask >> 6) & 1,
                        "c_char_3gram": (mask >> 5) & 1,
                        "c_acronym": (mask >> 4) & 1,
                        "c_addr_token": (mask >> 3) & 1,
                        "c_addr_numeric": (mask >> 2) & 1,
                        "c_postal": (mask >> 1) & 1,
                        "c_phonetic": mask & 1,
                        "num_channels": c[4],
                        "channel_mask": mask,
                    })
                results[eids[i]] = cand_list

            if (i + 1) % 25000 == 0 or (i + 1) == n_rows:
                speed = (i + 1) / (time.time() - t_start)
                print(f"  Blocked {i + 1:,} / {n_rows:,} entities ({speed:,.0f} ent/s)...")

        return results

    def block_and_write_streaming_parquet(
        self,
        s1_df: pl.DataFrame,
        output_parquet_path: Path,
        max_k: int = 50,
        batch_size: int = 25000,
    ):
        """
        Streams candidate generation directly to Parquet in small chunks (< 250 MB RAM).
        """
        print(f"\nStreaming Candidate Generation to: {output_parquet_path.name}")
        t_start = time.time()
        output_parquet_path.parent.mkdir(parents=True, exist_ok=True)

        writer = pq.ParquetWriter(str(output_parquet_path), CANDIDATE_SCHEMA, compression="SNAPPY")
        total_entities = len(s1_df)
        total_candidate_pairs = 0

        eids = s1_df["entity_id"].to_list()
        countries = s1_df["country"].to_list()
        name_cleans = s1_df["name_clean"].to_list() if "name_clean" in s1_df.columns else s1_df["name_core"].to_list()
        name_cores = s1_df["name_core"].to_list()
        name_tokens = s1_df["name_tokens"].to_list()
        acronyms = s1_df["name_acronym"].to_list()
        phonetics = s1_df["name_phonetic"].to_list()
        addr_cleans = s1_df["addr_clean"].to_list()
        addr_tokens = s1_df["addr_tokens"].to_list()
        addr_digits = s1_df["addr_digits"].to_list()
        addr_units = s1_df["addr_unit_num"].to_list()
        postals = s1_df["postal_clean"].to_list()

        for start_idx in range(0, total_entities, batch_size):
            end_idx = min(start_idx + batch_size, total_entities)
            
            col_s1_id = []
            col_tgt_id = []
            col_score = []
            col_rank = []
            col_c_name_core = []
            col_c_name_token = []
            col_c_name_contain = []
            col_c_char_3gram = []
            col_c_acronym = []
            col_c_addr_token = []
            col_c_addr_numeric = []
            col_c_postal = []
            col_c_phonetic = []
            col_num_channels = []
            col_channel_mask = []

            for i in range(start_idx, end_idx):
                s1_id = eids[i]
                c_name = str(countries[i]) if countries[i] else "Unknown"
                cands = self.query_entity_candidates(
                    country=c_name,
                    name_clean=str(name_cleans[i]) if name_cleans[i] else "",
                    name_core=str(name_cores[i]) if name_cores[i] else "",
                    name_tokens=str(name_tokens[i]) if name_tokens[i] else "",
                    name_acronym=str(acronyms[i]) if acronyms[i] else "",
                    name_phonetic=str(phonetics[i]) if phonetics[i] else "",
                    addr_clean=str(addr_cleans[i]) if addr_cleans[i] else "",
                    addr_tokens=str(addr_tokens[i]) if addr_tokens[i] else "",
                    addr_digits=str(addr_digits[i]) if addr_digits[i] else "",
                    addr_unit_num=str(addr_units[i]) if addr_units[i] else "",
                    postal_clean=str(postals[i]) if postals[i] else "",
                    max_k=max_k,
                )

                for rank_idx, c in enumerate(cands, start=1):
                    col_s1_id.append(s1_id)
                    col_tgt_id.append(c["target_id"])
                    col_score.append(c["score"])
                    col_rank.append(rank_idx)
                    col_c_name_core.append(c["c_name_core"])
                    col_c_name_token.append(c["c_name_token"])
                    col_c_name_contain.append(c["c_name_contain"])
                    col_c_char_3gram.append(c["c_char_3gram"])
                    col_c_acronym.append(c["c_acronym"])
                    col_c_addr_token.append(c["c_addr_token"])
                    col_c_addr_numeric.append(c["c_addr_numeric"])
                    col_c_postal.append(c["c_postal"])
                    col_c_phonetic.append(c["c_phonetic"])
                    col_num_channels.append(c["num_channels"])
                    col_channel_mask.append(c["channel_mask"])

            if col_s1_id:
                batch_table = pa.Table.from_arrays(
                    [
                        pa.array(col_s1_id, type=pa.string()),
                        pa.array(col_tgt_id, type=pa.string()),
                        pa.array(col_score, type=pa.float32()),
                        pa.array(col_rank, type=pa.int16()),
                        pa.array(col_c_name_core, type=pa.int8()),
                        pa.array(col_c_name_token, type=pa.int8()),
                        pa.array(col_c_name_contain, type=pa.int8()),
                        pa.array(col_c_char_3gram, type=pa.int8()),
                        pa.array(col_c_acronym, type=pa.int8()),
                        pa.array(col_c_addr_token, type=pa.int8()),
                        pa.array(col_c_addr_numeric, type=pa.int8()),
                        pa.array(col_c_postal, type=pa.int8()),
                        pa.array(col_c_phonetic, type=pa.int8()),
                        pa.array(col_num_channels, type=pa.int8()),
                        pa.array(col_channel_mask, type=pa.int16()),
                    ],
                    schema=CANDIDATE_SCHEMA,
                )
                writer.write_table(batch_table)
                total_candidate_pairs += len(col_s1_id)
                del batch_table

            elapsed = time.time() - t_start
            speed = end_idx / elapsed if elapsed > 0 else 0
            print(f"  Streamed {end_idx:,} / {total_entities:,} entities ({total_candidate_pairs:,} candidate pairs) [{speed:,.0f} ent/s]...")
            gc.collect()

        writer.close()
        size_mb = output_parquet_path.stat().st_size / (1024 * 1024)
        print(f"\n[OK] Wrote {total_candidate_pairs:,} candidate pairs to {output_parquet_path.name} ({size_mb:.2f} MB) in {time.time() - t_start:.2f}s.")
