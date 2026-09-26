"""
Step 3.2: 8-Channel Inverted Index Candidate Retrieval Engine.
Implements 8 independent, country-partitioned retrieval channels:
  C1: Exact Core Name (name_core)
  C2: Name Token IDF Postings (name_tokens)
  C3: Name Token Containment / 2-Token Stem (name_containment)
  C4: Bi-Directional Acronym Matching (acronym_bidir)
  C5: Rare Address Token IDF Postings (addr_rare_tokens)
  C6: Numeric Identity Agreement (addr_digits / unit_num + postal)
  C7: Postal & Prefix Geolocation (postal_match)
  C8: Phonetic Locality Anchor (name_phonetic + postal / rare_addr)
"""

import time
import math
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional
import polars as pl

from src.country_idf import CountryIDFComputer


class CountryMultiChannelIndex:
    """
    8-Channel Inverted Index for a single country target partition (S2 + S3).
    Provides channel-isolated candidate retrieval with provenance tracking.
    """

    def __init__(
        self,
        country: str,
        idf_computer: CountryIDFComputer,
        max_posting_len: int = 500,
        max_candidates_per_channel: int = 25,
    ):
        self.country = country
        self.idf_computer = idf_computer
        self.max_posting_len = max_posting_len
        self.max_candidates_per_channel = max_candidates_per_channel

        # Target metadata indexed by integer ID
        self.target_ids: List[str] = []
        self.target_name_cores: List[str] = []
        self.target_name_tokens: List[Set[str]] = []
        self.target_addr_tokens: List[Set[str]] = []
        self.target_postals: List[str] = []

        # 8 Distinct Channel Inverted Indexes
        self.idx_name_core = defaultdict(list)           # C1: name_core -> [target_idx]
        self.idx_name_tokens = defaultdict(list)         # C2: token -> [target_idx]
        self.idx_name_stem = defaultdict(list)           # C3: first 2 tokens stem -> [target_idx]
        self.idx_acronym = defaultdict(list)             # C4: acronym -> [target_idx]
        self.idx_addr_tokens = defaultdict(list)         # C5: addr_token -> [target_idx]
        self.idx_numeric_postal = defaultdict(list)      # C6: (postal_clean, primary_digit) -> [target_idx]
        self.idx_postal_exact = defaultdict(list)        # C7: postal_clean -> [target_idx]
        self.idx_phonetic_postal = defaultdict(list)     # C8: (name_phonetic, postal_prefix) -> [target_idx]

    def fit(self, target_df: pl.DataFrame):
        """
        Builds all 8 channel inverted indexes across target records for this country.
        """
        start_time = time.time()
        n_rows = len(target_df)

        eids = target_df["entity_id"].to_list()
        name_cores = target_df["name_core"].to_list()
        name_tokens_col = target_df["name_tokens"].to_list()
        acronyms = target_df["name_acronym"].to_list()
        phonetics = target_df["name_phonetic"].to_list()
        
        addr_clean_col = target_df["addr_clean"].to_list()
        addr_tokens_col = target_df["addr_tokens"].to_list()
        addr_digits_col = target_df["addr_digits"].to_list()
        addr_units_col = target_df["addr_unit_num"].to_list()
        postals_col = target_df["postal_clean"].to_list()

        self.target_ids = eids
        self.target_name_cores = [str(nc) if nc else "" for nc in name_cores]

        raw_name_tok_postings = defaultdict(list)
        raw_addr_tok_postings = defaultdict(list)

        for idx in range(n_rows):
            n_core = self.target_name_cores[idx]
            n_tok_str = str(name_tokens_col[idx]) if name_tokens_col[idx] else ""
            n_toks = set(n_tok_str.split())
            self.target_name_tokens.append(n_toks)

            a_tok_str = str(addr_tokens_col[idx]) if addr_tokens_col[idx] else ""
            a_toks = set(a_tok_str.split())
            self.target_addr_tokens.append(a_toks)

            postal = str(postals_col[idx]) if postals_col[idx] else ""
            self.target_postals.append(postal)
            postal_pfx = postal[:4] if len(postal) >= 4 else postal

            acro = str(acronyms[idx]) if acronyms[idx] else ""
            phone = str(phonetics[idx]) if phonetics[idx] else ""
            digits_str = str(addr_digits_col[idx]) if addr_digits_col[idx] else ""
            primary_digit = digits_str.split()[0] if digits_str else ""
            unit_num = str(addr_units_col[idx]) if addr_units_col[idx] else ""

            # C1: Exact Core Name
            if n_core and len(n_core) >= 3:
                self.idx_name_core[n_core].append(idx)

            # C2: Raw Name Token Postings
            for t in n_toks:
                if len(t) >= 3:
                    raw_name_tok_postings[t].append(idx)

            # C3: First 2 tokens stem
            n_tok_list = n_tok_str.split()
            if len(n_tok_list) >= 2:
                stem = f"{n_tok_list[0]}_{n_tok_list[1]}"
                self.idx_name_stem[stem].append(idx)

            # C4: Acronym
            if acro and len(acro) >= 2:
                self.idx_acronym[acro].append(idx)

            # C5: Raw Address Token Postings
            for at in a_toks:
                if len(at) >= 3:
                    raw_addr_tok_postings[at].append(idx)

            # C6: Numeric Identity Agreement (Postal + Building/Shop Digit or Unit)
            if postal:
                if primary_digit:
                    self.idx_numeric_postal[(postal, primary_digit)].append(idx)
                if unit_num:
                    self.idx_numeric_postal[(postal, unit_num)].append(idx)

            # C7: Postal Exact
            if postal:
                self.idx_postal_exact[postal].append(idx)

            # C8: Phonetic + Postal Prefix
            if phone and postal_pfx:
                self.idx_phonetic_postal[(phone, postal_pfx)].append(idx)

        # Cap token postings by max_posting_len to prevent broad-stopword explosions
        for t, postings in raw_name_tok_postings.items():
            if len(postings) <= self.max_posting_len * 10:
                self.idx_name_tokens[t] = postings[:self.max_posting_len]

        for at, postings in raw_addr_tok_postings.items():
            if len(postings) <= self.max_posting_len * 10:
                self.idx_addr_tokens[at] = postings[:self.max_posting_len]

        # Cap postal postings
        self.idx_postal_exact = {k: v[:200] for k, v in self.idx_postal_exact.items() if len(v) <= 1000}
        self.idx_phonetic_postal = {k: v[:150] for k, v in self.idx_phonetic_postal.items() if len(v) <= 500}

        elapsed = time.time() - start_time
        print(f"    [{self.country}] Multi-Channel Index built for {n_rows:,} records in {elapsed:.2f}s "
              f"(C1: {len(self.idx_name_core):,}, C2: {len(self.idx_name_tokens):,}, "
              f"C5: {len(self.idx_addr_tokens):,}, C7: {len(self.idx_postal_exact):,}).")

    def query_channels(
        self,
        name_core: str,
        name_tokens_str: str,
        name_acronym: str,
        name_phonetic: str,
        addr_clean: str,
        addr_tokens_str: str,
        addr_digits_str: str,
        addr_unit_num: str,
        postal_clean: str,
    ) -> Dict[int, Dict[str, any]]:
        """
        Queries all 8 independent channels and returns a dictionary of candidate records:
          target_idx -> {
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
          }
        """
        candidates: Dict[int, Dict[str, any]] = defaultdict(lambda: {
            "score": 0.0,
            "c_name_core": 0,
            "c_name_token": 0,
            "c_name_contain": 0,
            "c_acronym": 0,
            "c_addr_token": 0,
            "c_addr_numeric": 0,
            "c_postal": 0,
            "c_phonetic": 0,
            "num_channels": 0,
        })

        s1_ntoks = [t for t in name_tokens_str.split() if len(t) >= 3] if name_tokens_str else []
        s1_atoks = [at for at in addr_tokens_str.split() if len(at) >= 3] if addr_tokens_str else []
        s1_digits = addr_digits_str.split() if addr_digits_str else []
        s1_primary_digit = s1_digits[0] if s1_digits else ""
        postal_pfx = postal_clean[:4] if len(postal_clean) >= 4 else postal_clean

        # =========================================================================
        # C1: Exact Core Name Match (+100.0 priority)
        # =========================================================================
        if name_core and name_core in self.idx_name_core:
            for idx in self.idx_name_core[name_core][:self.max_candidates_per_channel]:
                c = candidates[idx]
                c["score"] += 100.0
                if not c["c_name_core"]:
                    c["c_name_core"] = 1
                    c["num_channels"] += 1

        # =========================================================================
        # C2: Name Token IDF Postings (Top-3 Rarest Tokens)
        # =========================================================================
        if s1_ntoks:
            rarest_ntoks = self.idf_computer.get_rarest_name_tokens(self.country, s1_ntoks, top_n=3)
            for tok, idf_val in rarest_ntoks:
                if tok in self.idx_name_tokens:
                    for idx in self.idx_name_tokens[tok][:self.max_candidates_per_channel]:
                        c = candidates[idx]
                        c["score"] += (idf_val * 4.0)
                        if not c["c_name_token"]:
                            c["c_name_token"] = 1
                            c["num_channels"] += 1

        # =========================================================================
        # C3: Name Token Containment / 2-Token Stem (+40.0 priority)
        # =========================================================================
        if len(s1_ntoks) >= 2:
            stem = f"{s1_ntoks[0]}_{s1_ntoks[1]}"
            if stem in self.idx_name_stem:
                for idx in self.idx_name_stem[stem][:self.max_candidates_per_channel]:
                    c = candidates[idx]
                    c["score"] += 40.0
                    if not c["c_name_contain"]:
                        c["c_name_contain"] = 1
                        c["num_channels"] += 1

        # =========================================================================
        # C4: Bi-Directional Acronym (+30.0 priority, requires address or postal anchor)
        # =========================================================================
        if name_acronym and len(name_acronym) >= 2 and name_acronym in self.idx_acronym:
            for idx in self.idx_acronym[name_acronym][:self.max_candidates_per_channel]:
                # Safe guard: require shared postal OR shared addr token to avoid massive acronym collisions
                target_post = self.target_postals[idx]
                target_atoks = self.target_addr_tokens[idx]
                if (postal_clean and postal_clean == target_post) or (set(s1_atoks) & target_atoks):
                    c = candidates[idx]
                    c["score"] += 35.0
                    if not c["c_acronym"]:
                        c["c_acronym"] = 1
                        c["num_channels"] += 1

        # =========================================================================
        # C5: Rare Address Token IDF Postings (Top-3 Rarest Address Tokens)
        # =========================================================================
        if s1_atoks:
            rarest_atoks = self.idf_computer.get_rarest_addr_tokens(self.country, s1_atoks, top_n=3)
            for atok, idf_val in rarest_atoks:
                if atok in self.idx_addr_tokens:
                    for idx in self.idx_addr_tokens[atok][:self.max_candidates_per_channel]:
                        c = candidates[idx]
                        c["score"] += (idf_val * 3.5)
                        if not c["c_addr_token"]:
                            c["c_addr_token"] = 1
                            c["num_channels"] += 1

        # =========================================================================
        # C6: Numeric Identity Agreement (Postal + Primary Digit or Unit Number)
        # =========================================================================
        if postal_clean:
            if s1_primary_digit:
                key = (postal_clean, s1_primary_digit)
                if key in self.idx_numeric_postal:
                    for idx in self.idx_numeric_postal[key][:self.max_candidates_per_channel]:
                        c = candidates[idx]
                        c["score"] += 25.0
                        if not c["c_addr_numeric"]:
                            c["c_addr_numeric"] = 1
                            c["num_channels"] += 1
            if addr_unit_num:
                key_u = (postal_clean, addr_unit_num)
                if key_u in self.idx_numeric_postal:
                    for idx in self.idx_numeric_postal[key_u][:self.max_candidates_per_channel]:
                        c = candidates[idx]
                        c["score"] += 30.0
                        if not c["c_addr_numeric"]:
                            c["c_addr_numeric"] = 1
                            c["num_channels"] += 1

        # =========================================================================
        # C7: Postal Geolocation (+10.0 priority, requires at least 1 name/addr token overlap)
        # =========================================================================
        if postal_clean and postal_clean in self.idx_postal_exact:
            for idx in self.idx_postal_exact[postal_clean][:self.max_candidates_per_channel]:
                # Anchor requirement: must have at least 1 name token or 1 address token overlap
                if (set(s1_ntoks) & self.target_name_tokens[idx]) or (set(s1_atoks) & self.target_addr_tokens[idx]):
                    c = candidates[idx]
                    c["score"] += 15.0
                    if not c["c_postal"]:
                        c["c_postal"] = 1
                        c["num_channels"] += 1

        # =========================================================================
        # C8: Phonetic Locality Anchor (+20.0 priority)
        # =========================================================================
        if name_phonetic and postal_pfx:
            key_ph = (name_phonetic, postal_pfx)
            if key_ph in self.idx_phonetic_postal:
                for idx in self.idx_phonetic_postal[key_ph][:self.max_candidates_per_channel]:
                    c = candidates[idx]
                    c["score"] += 20.0
                    if not c["c_phonetic"]:
                        c["c_phonetic"] = 1
                        c["num_channels"] += 1

        # Multi-channel redundancy bonus: each extra confirming channel adds superlinear confidence
        for idx, c in candidates.items():
            if c["num_channels"] >= 2:
                c["score"] += (c["num_channels"] * 15.0)

        return candidates
