"""
Step 3.2: High-Recall Multi-Channel Inverted Index Engine (Amazon ML Challenge 2026).
Engineered for ultra-low memory (< 350 MB) and >= 95% Candidate Recall.
Includes domain suffix stripping, concatenated name indexing, rare token posting traversal,
and multi-modal tiered candidate selection.
"""

import time
import math
import array
import re
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional
import polars as pl

from src.country_idf import CountryIDFComputer


DOMAIN_SUFFIXES = (
    ".com", ".org", ".net", ".in", ".co.in", ".co", ".io", ".ai", ".fr",
    ".biz", ".info", ".gov", ".edu", ".us", ".uk", ".de", ".eu",
    " com", " org", " net", " co in", " in", " co", " io", " ai", " fr",
    " biz", " info", " gov", " edu", " us"
)


def strip_domain_suffix(text: str) -> str:
    """Strips web domain extensions from names (e.g. 'summithealth com' -> 'summithealth')."""
    if not text:
        return ""
    text_clean = text.strip()
    for suffix in DOMAIN_SUFFIXES:
        if text_clean.endswith(suffix) and len(text_clean) > len(suffix) + 2:
            return text_clean[:-len(suffix)].strip()
    return text_clean


def extract_char_3grams(text: str) -> List[str]:
    """Extracts character 3-grams from normalized text."""
    if not text or len(text) < 3:
        return []
    clean = text.replace(" ", "")
    if len(clean) < 3:
        return []
    return [clean[i:i+3] for i in range(len(clean) - 2)]


class CountryMultiChannelIndex:
    """
    High-Recall 9-Channel Inverted Index backed by raw C-level uint32 arrays.
    Deep posting lists with ultra-low memory (< 350 MB RAM per country partition).
    """

    def __init__(
        self,
        country: str,
        idf_computer: CountryIDFComputer,
        max_posting_len: int = 1500,
        max_candidates_per_channel: int = 50,
    ):
        self.country = country
        self.idf_computer = idf_computer
        self.max_posting_len = max_posting_len
        self.max_candidates_per_channel = max_candidates_per_channel

        # Target IDs indexed by integer ID
        self.target_ids: List[str] = []
        self.target_postals: List[str] = []

        # Raw 32-bit unsigned int arrays (C-level efficiency)
        self.idx_name_core: Dict[str, array.array] = defaultdict(lambda: array.array('I'))
        self.idx_name_tokens: Dict[str, array.array] = defaultdict(lambda: array.array('I'))
        self.idx_name_char3: Dict[str, array.array] = defaultdict(lambda: array.array('I'))
        self.idx_name_stem: Dict[str, array.array] = defaultdict(lambda: array.array('I'))
        self.idx_acronym: Dict[str, array.array] = defaultdict(lambda: array.array('I'))
        self.idx_addr_tokens: Dict[str, array.array] = defaultdict(lambda: array.array('I'))
        self.idx_numeric_postal: Dict[Tuple[str, str], array.array] = defaultdict(lambda: array.array('I'))
        self.idx_postal_exact: Dict[str, array.array] = defaultdict(lambda: array.array('I'))
        self.idx_phonetic_postal: Dict[Tuple[str, str], array.array] = defaultdict(lambda: array.array('I'))

    def fit(self, target_df: pl.DataFrame):
        """
        Builds high-recall inverted indexes using raw C-level uint32 arrays.
        """
        start_time = time.time()
        n_rows = len(target_df)

        eids = target_df["entity_id"].to_list()
        name_cores = target_df["name_core"].to_list()
        name_tokens_col = target_df["name_tokens"].to_list()
        name_cleans_col = target_df["name_clean"].to_list() if "name_clean" in target_df.columns else name_cores
        acronyms = target_df["name_acronym"].to_list()
        phonetics = target_df["name_phonetic"].to_list()
        
        addr_tokens_col = target_df["addr_tokens"].to_list()
        addr_digits_col = target_df["addr_digits"].to_list()
        addr_units_col = target_df["addr_unit_num"].to_list()
        postals_col = target_df["postal_clean"].to_list()

        self.target_ids = eids
        self.target_postals = [str(p) if p else "" for p in postals_col]
        max_p = self.max_posting_len

        for idx in range(n_rows):
            n_core = str(name_cores[idx]) if name_cores[idx] else ""
            n_clean = str(name_cleans_col[idx]) if name_cleans_col[idx] else ""
            n_tok_str = str(name_tokens_col[idx]) if name_tokens_col[idx] else ""
            a_tok_str = str(addr_tokens_col[idx]) if addr_tokens_col[idx] else ""
            postal = self.target_postals[idx]
            postal_pfx = postal[:4] if len(postal) >= 4 else postal

            acro = str(acronyms[idx]) if acronyms[idx] else ""
            phone = str(phonetics[idx]) if phonetics[idx] else ""
            digits_str = str(addr_digits_col[idx]) if addr_digits_col[idx] else ""
            primary_digit = digits_str.split()[0] if digits_str else ""
            unit_num = str(addr_units_col[idx]) if addr_units_col[idx] else ""

            # Domain-stripped forms
            n_core_no_dom = strip_domain_suffix(n_core)
            n_clean_no_dom = strip_domain_suffix(n_clean)
            n_concat = n_clean_no_dom.replace(" ", "")

            # 1. Exact Core Name & Concatenated Name Forms
            if n_core and len(n_core) >= 3:
                p = self.idx_name_core[n_core]
                if len(p) < max_p:
                    p.append(idx)

            if n_core_no_dom and n_core_no_dom != n_core and len(n_core_no_dom) >= 3:
                p = self.idx_name_core[n_core_no_dom]
                if len(p) < max_p:
                    p.append(idx)

            if n_concat and len(n_concat) >= 4 and n_concat != n_core and n_concat != n_core_no_dom:
                p = self.idx_name_core[n_concat]
                if len(p) < max_p:
                    p.append(idx)

            # 2. Name Tokens (Deep Postings)
            all_name_toks = set()
            if n_tok_str:
                all_name_toks.update(n_tok_str.split())
            if n_clean_no_dom:
                all_name_toks.update([t for t in n_clean_no_dom.split() if len(t) >= 3])

            for t in all_name_toks:
                if len(t) >= 3:
                    p = self.idx_name_tokens[t]
                    if len(p) < max_p:
                        p.append(idx)

            # 3. First 2 tokens stem
            n_tok_list = n_tok_str.split() if n_tok_str else []
            if len(n_tok_list) >= 2:
                stem = f"{n_tok_list[0]}_{n_tok_list[1]}"
                p = self.idx_name_stem[stem]
                if len(p) < max_p:
                    p.append(idx)

            # 4. Character 3-grams for Brand Names
            if n_clean_no_dom and len(n_clean_no_dom) >= 4:
                char3_list = extract_char_3grams(n_clean_no_dom)
                for c3 in set(char3_list):
                    p = self.idx_name_char3[c3]
                    if len(p) < 600:
                        p.append(idx)

            # 5. Acronym
            if acro and len(acro) >= 2:
                p = self.idx_acronym[acro]
                if len(p) < 600:
                    p.append(idx)

            # 6. Address Token Postings (Deep Postings)
            if a_tok_str:
                for at in set(a_tok_str.split()):
                    if len(at) >= 3:
                        p = self.idx_addr_tokens[at]
                        if len(p) < max_p:
                            p.append(idx)

            # 7. Numeric Identity (Postal + Building/Shop Digit or Unit)
            if postal:
                if primary_digit:
                    p = self.idx_numeric_postal[(postal, primary_digit)]
                    if len(p) < 500:
                        p.append(idx)
                if unit_num:
                    p = self.idx_numeric_postal[(postal, unit_num)]
                    if len(p) < 500:
                        p.append(idx)

                # Postal Exact
                p_post = self.idx_postal_exact[postal]
                if len(p_post) < 600:
                    p_post.append(idx)

            # 8. Phonetic + Postal Prefix
            if phone and postal_pfx:
                p = self.idx_phonetic_postal[(phone, postal_pfx)]
                if len(p) < 500:
                    p.append(idx)

        elapsed = time.time() - start_time
        print(f"    [{self.country}] Multi-Channel Index built for {n_rows:,} records in {elapsed:.2f}s "
              f"(C1 Core/Concat: {len(self.idx_name_core):,}, C2 Tokens: {len(self.idx_name_tokens):,}, "
              f"Char3: {len(self.idx_name_char3):,}, C5 AddrTok: {len(self.idx_addr_tokens):,}, C7 Postal: {len(self.idx_postal_exact):,}).")

    def query_channels(
        self,
        name_clean: str,
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
        Queries all channels with rarity-aware dynamic posting list traversal.
        """
        candidates: Dict[int, Dict[str, any]] = defaultdict(lambda: {
            "score": 0.0,
            "name_score": 0.0,
            "addr_score": 0.0,
            "c_name_core": 0,
            "c_name_token": 0,
            "c_name_contain": 0,
            "c_char_3gram": 0,
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

        name_core_no_dom = strip_domain_suffix(name_core)
        name_concat = strip_domain_suffix(name_clean).replace(" ", "")

        # C1: Exact Core Name Match & Concatenated Name Form (+100.0 priority)
        if name_core and name_core in self.idx_name_core:
            for idx in self.idx_name_core[name_core]:
                c = candidates[idx]
                c["score"] += 100.0
                c["name_score"] += 100.0
                if not c["c_name_core"]:
                    c["c_name_core"] = 1
                    c["num_channels"] += 1

        if name_concat and len(name_concat) >= 4 and name_concat in self.idx_name_core:
            for idx in self.idx_name_core[name_concat]:
                c = candidates[idx]
                c["score"] += 95.0
                c["name_score"] += 95.0
                if not c["c_name_core"]:
                    c["c_name_core"] = 1
                    c["num_channels"] += 1

        if name_core_no_dom and name_core_no_dom != name_core and name_core_no_dom in self.idx_name_core:
            for idx in self.idx_name_core[name_core_no_dom]:
                c = candidates[idx]
                c["score"] += 90.0
                c["name_score"] += 90.0
                if not c["c_name_core"]:
                    c["c_name_core"] = 1
                    c["num_channels"] += 1

        # C2: Name Token IDF Postings (IDF-aware dynamic posting list traversal)
        if s1_ntoks:
            rarest_ntoks = self.idf_computer.get_rarest_name_tokens(self.country, s1_ntoks, top_n=6)
            for tok, idf_val in rarest_ntoks:
                if tok in self.idx_name_tokens:
                    posting = self.idx_name_tokens[tok]
                    limit = None if idf_val >= 3.0 else (400 if idf_val >= 2.0 else 100)
                    targets = posting if limit is None else posting[:limit]
                    w = idf_val * 6.0
                    for idx in targets:
                        c = candidates[idx]
                        c["score"] += w
                        c["name_score"] += w
                        if not c["c_name_token"]:
                            c["c_name_token"] = 1
                            c["num_channels"] += 1

        # C3: Name Token Containment / 2-Token Stem (+45.0 priority)
        if len(s1_ntoks) >= 2:
            stem = f"{s1_ntoks[0]}_{s1_ntoks[1]}"
            if stem in self.idx_name_stem:
                for idx in self.idx_name_stem[stem][:250]:
                    c = candidates[idx]
                    c["score"] += 45.0
                    c["name_score"] += 45.0
                    if not c["c_name_contain"]:
                        c["c_name_contain"] = 1
                        c["num_channels"] += 1

        # C4: Character 3-grams for Brand Names (bridges spelling noise like Lakshmi/Laxmi)
        if name_clean and len(name_clean) >= 4:
            c3_list = extract_char_3grams(name_clean)
            for c3 in c3_list[:8]:
                if c3 in self.idx_name_char3:
                    for idx in self.idx_name_char3[c3][:60]:
                        c = candidates[idx]
                        c["score"] += 8.0
                        c["name_score"] += 8.0
                        if not c["c_char_3gram"]:
                            c["c_char_3gram"] = 1
                            c["num_channels"] += 1

        # C5: Bi-Directional Acronym (+35.0 priority)
        if name_acronym and len(name_acronym) >= 2 and name_acronym in self.idx_acronym:
            for idx in self.idx_acronym[name_acronym][:150]:
                target_post = self.target_postals[idx]
                if postal_clean and postal_clean == target_post:
                    c = candidates[idx]
                    c["score"] += 35.0
                    c["name_score"] += 20.0
                    c["addr_score"] += 15.0
                    if not c["c_acronym"]:
                        c["c_acronym"] = 1
                        c["num_channels"] += 1

        # C6: Rare Address Token IDF Postings (IDF-aware dynamic posting list traversal)
        if s1_atoks:
            rarest_atoks = self.idf_computer.get_rarest_addr_tokens(self.country, s1_atoks, top_n=6)
            for atok, idf_val in rarest_atoks:
                if atok in self.idx_addr_tokens:
                    posting = self.idx_addr_tokens[atok]
                    limit = None if idf_val >= 3.0 else (400 if idf_val >= 2.0 else 100)
                    targets = posting if limit is None else posting[:limit]
                    w = idf_val * 5.5
                    for idx in targets:
                        c = candidates[idx]
                        c["score"] += w
                        c["addr_score"] += w
                        if not c["c_addr_token"]:
                            c["c_addr_token"] = 1
                            c["num_channels"] += 1

        # C7: Numeric Identity Agreement (Postal + Primary Digit or Unit Number)
        if postal_clean:
            if s1_primary_digit:
                key = (postal_clean, s1_primary_digit)
                if key in self.idx_numeric_postal:
                    for idx in self.idx_numeric_postal[key][:200]:
                        c = candidates[idx]
                        c["score"] += 35.0
                        c["addr_score"] += 35.0
                        if not c["c_addr_numeric"]:
                            c["c_addr_numeric"] = 1
                            c["num_channels"] += 1
            if addr_unit_num:
                key_u = (postal_clean, addr_unit_num)
                if key_u in self.idx_numeric_postal:
                    for idx in self.idx_numeric_postal[key_u][:200]:
                        c = candidates[idx]
                        c["score"] += 40.0
                        c["addr_score"] += 40.0
                        if not c["c_addr_numeric"]:
                            c["c_addr_numeric"] = 1
                            c["num_channels"] += 1

        # C8: Postal Geolocation (+15.0 priority)
        if postal_clean and postal_clean in self.idx_postal_exact:
            for idx in self.idx_postal_exact[postal_clean][:60]:
                c = candidates[idx]
                c["score"] += 15.0
                c["addr_score"] += 15.0
                if not c["c_postal"]:
                    c["c_postal"] = 1
                    c["num_channels"] += 1

        # C9: Phonetic Locality Anchor (+25.0 priority)
        if name_phonetic and postal_pfx:
            key_ph = (name_phonetic, postal_pfx)
            if key_ph in self.idx_phonetic_postal:
                for idx in self.idx_phonetic_postal[key_ph][:150]:
                    c = candidates[idx]
                    c["score"] += 25.0
                    c["name_score"] += 15.0
                    c["addr_score"] += 10.0
                    if not c["c_phonetic"]:
                        c["c_phonetic"] = 1
                        c["num_channels"] += 1

        # Multi-Channel & Cross-Modal confirmation bonus
        for idx, c in candidates.items():
            if c["num_channels"] >= 2:
                c["score"] += (c["num_channels"] * 25.0)
            if c["name_score"] > 0 and c["addr_score"] > 0:
                c["score"] += 30.0

        return candidates

    def select_top_candidates(
        self,
        candidates: Dict[int, Dict[str, any]],
        max_k: int = 50,
    ) -> List[Tuple[int, Dict[str, any]]]:
        """
        Tiered / Quota-based candidate selection guaranteeing inclusion of:
          - Composite Multi-Channel matches (Tier 1: 60% quota)
          - Address-Centric matches for targets with empty names (Tier 2: 20% quota)
          - Name-Centric matches for targets with empty addresses (Tier 3: 20% quota)
        """
        if not candidates:
            return []
        if len(candidates) <= max_k:
            return sorted(candidates.items(), key=lambda x: x[1]["score"], reverse=True)

        sorted_by_score = sorted(candidates.items(), key=lambda x: x[1]["score"], reverse=True)
        selected_indices = set()
        selected_list = []

        # Tier 1: Top Composite Candidates (60% quota)
        n_tier1 = max(1, int(max_k * 0.60))
        for idx, c in sorted_by_score[:n_tier1]:
            selected_indices.add(idx)
            selected_list.append((idx, c))

        remaining = [item for item in sorted_by_score if item[0] not in selected_indices]

        # Tier 2: Top Address Candidates (20% quota - captures empty-name true targets)
        n_tier2 = max(1, int(max_k * 0.20))
        sorted_by_addr = sorted(remaining, key=lambda x: x[1]["addr_score"], reverse=True)
        for idx, c in sorted_by_addr[:n_tier2]:
            if c["addr_score"] >= 15.0:
                selected_indices.add(idx)
                selected_list.append((idx, c))

        remaining = [item for item in sorted_by_score if item[0] not in selected_indices]

        # Tier 3: Top Name Candidates (20% quota - captures empty-address true targets)
        n_tier3 = max_k - len(selected_list)
        sorted_by_name = sorted(remaining, key=lambda x: x[1]["name_score"], reverse=True)
        for idx, c in sorted_by_name[:n_tier3]:
            if c["name_score"] >= 15.0:
                selected_indices.add(idx)
                selected_list.append((idx, c))

        # Fill remaining slots up to max_k
        if len(selected_list) < max_k:
            remaining = [item for item in sorted_by_score if item[0] not in selected_indices]
            for idx, c in remaining[:(max_k - len(selected_list))]:
                selected_list.append((idx, c))

        selected_list.sort(key=lambda x: x[1]["score"], reverse=True)
        return selected_list
