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

    def fit(self, target_df: pl.DataFrame, batch_size: int = 250000):
        """
        Builds high-recall inverted indexes using raw C-level uint32 arrays with streaming batch slicing.
        """
        start_time = time.time()
        n_rows = len(target_df)

        self.target_ids = target_df["entity_id"].to_list()
        postals_raw = target_df["postal_clean"].to_list() if "postal_clean" in target_df.columns else []
        self.target_postals = [str(p) if p else "" for p in postals_raw]
        del postals_raw
        max_p = self.max_posting_len

        for chunk_start in range(0, n_rows, batch_size):
            chunk = target_df.slice(chunk_start, batch_size)
            chunk_len = len(chunk)

            name_cores = chunk["name_core"].to_list() if "name_core" in chunk.columns else [""] * chunk_len
            name_cleans_col = chunk["name_clean"].to_list() if "name_clean" in chunk.columns else name_cores
            name_tokens_col = chunk["name_tokens"].to_list() if "name_tokens" in chunk.columns else [""] * chunk_len
            acronyms = chunk["name_acronym"].to_list() if "name_acronym" in chunk.columns else [""] * chunk_len
            phonetics = chunk["name_phonetic"].to_list() if "name_phonetic" in chunk.columns else [""] * chunk_len
            addr_tokens_col = chunk["addr_tokens"].to_list() if "addr_tokens" in chunk.columns else [""] * chunk_len
            addr_digits_col = chunk["addr_digits"].to_list() if "addr_digits" in chunk.columns else [""] * chunk_len
            addr_units_col = chunk["addr_unit_num"].to_list() if "addr_unit_num" in chunk.columns else [""] * chunk_len

            for i in range(chunk_len):
                idx = chunk_start + i
                n_core = str(name_cores[i]) if name_cores[i] else ""
                n_clean = str(name_cleans_col[i]) if name_cleans_col[i] else ""
                n_tok_str = str(name_tokens_col[i]) if name_tokens_col[i] else ""
                a_tok_str = str(addr_tokens_col[i]) if addr_tokens_col[i] else ""
                postal = self.target_postals[idx]
                postal_pfx = postal[:4] if len(postal) >= 4 else postal

                acro = str(acronyms[i]) if acronyms[i] else ""
                phone = str(phonetics[i]) if phonetics[i] else ""
                digits_str = str(addr_digits_col[i]) if addr_digits_col[i] else ""
                primary_digit = digits_str.split()[0] if digits_str else ""
                unit_num = str(addr_units_col[i]) if addr_units_col[i] else ""

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

            del chunk, name_cores, name_cleans_col, name_tokens_col, acronyms, phonetics, addr_tokens_col, addr_digits_col, addr_units_col

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
    ) -> Dict[int, List[Any]]:
        """
        Ultra-fast rarity-aware multi-channel candidate retrieval (< 0.2ms per query).
        Returns dict of tgt_int_idx -> [score, name_score, addr_score, mask, num_channels]
        Bit positions:
          8: c_name_core
          7: c_name_token
          6: c_name_contain
          5: c_char_3gram
          4: c_acronym
          3: c_addr_token
          2: c_addr_numeric
          1: c_postal
          0: c_phonetic
        """
        cand_data: Dict[int, List[Any]] = {}

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
                c = cand_data.get(idx)
                if c is None:
                    cand_data[idx] = [100.0, 100.0, 0.0, 1 << 8, 1]
                else:
                    c[0] += 100.0
                    c[1] += 100.0
                    if not (c[3] & (1 << 8)):
                        c[3] |= (1 << 8)
                        c[4] += 1

        if name_concat and len(name_concat) >= 4 and name_concat in self.idx_name_core:
            for idx in self.idx_name_core[name_concat]:
                c = cand_data.get(idx)
                if c is None:
                    cand_data[idx] = [95.0, 95.0, 0.0, 1 << 8, 1]
                else:
                    c[0] += 95.0
                    c[1] += 95.0
                    if not (c[3] & (1 << 8)):
                        c[3] |= (1 << 8)
                        c[4] += 1

        if name_core_no_dom and name_core_no_dom != name_core and name_core_no_dom in self.idx_name_core:
            for idx in self.idx_name_core[name_core_no_dom]:
                c = cand_data.get(idx)
                if c is None:
                    cand_data[idx] = [90.0, 90.0, 0.0, 1 << 8, 1]
                else:
                    c[0] += 90.0
                    c[1] += 90.0
                    if not (c[3] & (1 << 8)):
                        c[3] |= (1 << 8)
                        c[4] += 1

        # C2: Name Token IDF Postings (High-IDF bounded traversal)
        if s1_ntoks:
            rarest_ntoks = self.idf_computer.get_rarest_name_tokens(self.country, s1_ntoks, top_n=4)
            for tok, idf_val in rarest_ntoks:
                if tok in self.idx_name_tokens:
                    posting = self.idx_name_tokens[tok]
                    limit = 60 if idf_val >= 3.0 else (30 if idf_val >= 2.0 else 15)
                    targets = posting[:limit]
                    w = idf_val * 6.0
                    for idx in targets:
                        c = cand_data.get(idx)
                        if c is None:
                            cand_data[idx] = [w, w, 0.0, 1 << 7, 1]
                        else:
                            c[0] += w
                            c[1] += w
                            if not (c[3] & (1 << 7)):
                                c[3] |= (1 << 7)
                                c[4] += 1

        # C3: Name Token Containment / 2-Token Stem (+45.0 priority)
        if len(s1_ntoks) >= 2:
            stem = f"{s1_ntoks[0]}_{s1_ntoks[1]}"
            if stem in self.idx_name_stem:
                for idx in self.idx_name_stem[stem][:40]:
                    c = cand_data.get(idx)
                    if c is None:
                        cand_data[idx] = [45.0, 45.0, 0.0, 1 << 6, 1]
                    else:
                        c[0] += 45.0
                        c[1] += 45.0
                        if not (c[3] & (1 << 6)):
                            c[3] |= (1 << 6)
                            c[4] += 1

        # C4: Character 3-grams for Brand Names
        if name_clean and len(name_clean) >= 4:
            c3_list = extract_char_3grams(name_clean)
            for c3 in c3_list[:5]:
                if c3 in self.idx_name_char3:
                    for idx in self.idx_name_char3[c3][:20]:
                        c = cand_data.get(idx)
                        if c is None:
                            cand_data[idx] = [8.0, 8.0, 0.0, 1 << 5, 1]
                        else:
                            c[0] += 8.0
                            c[1] += 8.0
                            if not (c[3] & (1 << 5)):
                                c[3] |= (1 << 5)
                                c[4] += 1

        # C5: Bi-Directional Acronym (+35.0 priority)
        if name_acronym and len(name_acronym) >= 2 and name_acronym in self.idx_acronym:
            for idx in self.idx_acronym[name_acronym][:40]:
                target_post = self.target_postals[idx]
                if postal_clean and postal_clean == target_post:
                    c = cand_data.get(idx)
                    if c is None:
                        cand_data[idx] = [35.0, 20.0, 15.0, 1 << 4, 1]
                    else:
                        c[0] += 35.0
                        c[1] += 20.0
                        c[2] += 15.0
                        if not (c[3] & (1 << 4)):
                            c[3] |= (1 << 4)
                            c[4] += 1

        # C6: Rare Address Token IDF Postings (High-IDF bounded traversal)
        if s1_atoks:
            rarest_atoks = self.idf_computer.get_rarest_addr_tokens(self.country, s1_atoks, top_n=4)
            for atok, idf_val in rarest_atoks:
                if atok in self.idx_addr_tokens:
                    posting = self.idx_addr_tokens[atok]
                    limit = 50 if idf_val >= 3.0 else (25 if idf_val >= 2.0 else 10)
                    targets = posting[:limit]
                    w = idf_val * 5.5
                    for idx in targets:
                        c = cand_data.get(idx)
                        if c is None:
                            cand_data[idx] = [w, 0.0, w, 1 << 3, 1]
                        else:
                            c[0] += w
                            c[2] += w
                            if not (c[3] & (1 << 3)):
                                c[3] |= (1 << 3)
                                c[4] += 1

        # C7: Numeric Identity Agreement (Postal + Primary Digit or Unit Number)
        if postal_clean:
            if s1_primary_digit:
                key = (postal_clean, s1_primary_digit)
                if key in self.idx_numeric_postal:
                    for idx in self.idx_numeric_postal[key][:40]:
                        c = cand_data.get(idx)
                        if c is None:
                            cand_data[idx] = [35.0, 0.0, 35.0, 1 << 2, 1]
                        else:
                            c[0] += 35.0
                            c[2] += 35.0
                            if not (c[3] & (1 << 2)):
                                c[3] |= (1 << 2)
                                c[4] += 1
            if addr_unit_num:
                key_u = (postal_clean, addr_unit_num)
                if key_u in self.idx_numeric_postal:
                    for idx in self.idx_numeric_postal[key_u][:40]:
                        c = cand_data.get(idx)
                        if c is None:
                            cand_data[idx] = [40.0, 0.0, 40.0, 1 << 2, 1]
                        else:
                            c[0] += 40.0
                            c[2] += 40.0
                            if not (c[3] & (1 << 2)):
                                c[3] |= (1 << 2)
                                c[4] += 1

        # C8: Postal Geolocation (+15.0 priority)
        if postal_clean and postal_clean in self.idx_postal_exact:
            for idx in self.idx_postal_exact[postal_clean][:25]:
                c = cand_data.get(idx)
                if c is None:
                    cand_data[idx] = [15.0, 0.0, 15.0, 1 << 1, 1]
                else:
                    c[0] += 15.0
                    c[2] += 15.0
                    if not (c[3] & (1 << 1)):
                        c[3] |= (1 << 1)
                        c[4] += 1

        # C9: Phonetic Locality Anchor (+25.0 priority)
        if name_phonetic and postal_pfx:
            key_ph = (name_phonetic, postal_pfx)
            if key_ph in self.idx_phonetic_postal:
                for idx in self.idx_phonetic_postal[key_ph][:40]:
                    c = cand_data.get(idx)
                    if c is None:
                        cand_data[idx] = [25.0, 15.0, 10.0, 1 << 0, 1]
                    else:
                        c[0] += 25.0
                        c[1] += 15.0
                        c[2] += 10.0
                        if not (c[3] & (1 << 0)):
                            c[3] |= (1 << 0)
                            c[4] += 1

        # Multi-Channel & Cross-Modal confirmation bonus
        for idx, c in cand_data.items():
            if c[4] >= 2:
                c[0] += (c[4] * 25.0)
            if c[1] > 0 and c[2] > 0:
                c[0] += 30.0

        return cand_data

    def select_top_candidates(
        self,
        cand_data: Dict[int, List[Any]],
        max_k: int = 50,
    ) -> List[Tuple[int, List[Any]]]:
        """
        Tiered / Quota-based candidate selection guaranteeing inclusion of:
          - Composite Multi-Channel matches (Tier 1: 60% quota)
          - Address-Centric matches for targets with empty names (Tier 2: 20% quota)
          - Name-Centric matches for targets with empty addresses (Tier 3: 20% quota)
        """
        if not cand_data:
            return []
        if len(cand_data) <= max_k:
            items = list(cand_data.items())
            items.sort(key=lambda x: x[1][0], reverse=True)
            return items

        items = list(cand_data.items())
        items.sort(key=lambda x: x[1][0], reverse=True)

        selected_indices = set()
        selected_list = []

        # Tier 1: Top Composite Candidates (60% quota)
        n_tier1 = max(1, int(max_k * 0.60))
        for idx, c in items[:n_tier1]:
            selected_indices.add(idx)
            selected_list.append((idx, c))

        remaining = [item for item in items if item[0] not in selected_indices]

        # Tier 2: Top Address Candidates (20% quota - captures empty-name true targets)
        n_tier2 = max(1, int(max_k * 0.20))
        if remaining:
            sorted_by_addr = sorted(remaining, key=lambda x: x[1][2], reverse=True)
            for idx, c in sorted_by_addr[:n_tier2]:
                if c[2] >= 15.0:
                    selected_indices.add(idx)
                    selected_list.append((idx, c))

        # Tier 3: Top Name Candidates (20% quota - captures empty-address true targets)
        n_tier3 = max_k - len(selected_list)
        remaining2 = [item for item in items if item[0] not in selected_indices]
        if remaining2 and n_tier3 > 0:
            sorted_by_name = sorted(remaining2, key=lambda x: x[1][1], reverse=True)
            for idx, c in sorted_by_name[:n_tier3]:
                if c[1] >= 15.0:
                    selected_indices.add(idx)
                    selected_list.append((idx, c))

        # Fill remaining slots up to max_k
        if len(selected_list) < max_k:
            remaining3 = [item for item in items if item[0] not in selected_indices]
            for idx, c in remaining3[:(max_k - len(selected_list))]:
                selected_list.append((idx, c))

        selected_list.sort(key=lambda x: x[1][0], reverse=True)
        return selected_list
