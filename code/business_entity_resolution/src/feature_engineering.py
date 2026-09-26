"""
Step 4.1 & Phase 5.5: High-Dimensional Multi-Scale Pairwise Feature Engineering Engine.
Extracts 8 comprehensive feature families (82 discriminative signals) between S1 entities and candidate S2/S3 targets:
  - F1: Name Similarity & Asymmetric Directional Containment (17 features)
  - F2: Legal Form & Algorithmic Acronym Disentanglement (7 features)
  - F3: Phonetic, Sub-Word N-Grams & Consonant Skeletons (6 features)
  - F4: Address Hierarchy, Structured Locality, Landmarks & Contradictions (16 features)
  - F5: Pure Missingness Indicators & Unthresholded Conditional Similarities (9 features)
  - F6: 9-Channel Blocker Provenance & Channel Co-Occurrence (15 features)
  - F7: Multi-Tier Hierarchical Exact Match Anchors & Cross-Interactions (7 features)
  - F8: Source System & Country Context (5 features)
"""

import math
from typing import Dict, List, Set, Tuple, Optional
import polars as pl
from rapidfuzz import fuzz, distance

from src.country_idf import CountryIDFComputer


FEATURE_NAMES: List[str] = [
    # Family 1: Name Similarity & Asymmetric Containment (17 features)
    "name_clean_fuzz_ratio",
    "name_clean_token_sort_ratio",
    "name_clean_token_set_ratio",
    "name_clean_wratio",
    "name_clean_jaro_winkler",
    "name_core_fuzz_ratio",
    "name_core_token_set_ratio",
    "name_core_jaro_winkler",
    "name_core_exact_match",
    "name_concat_exact_match",
    "name_len_diff_ratio",
    "name_first_token_similarity",
    "name_last_token_similarity",
    "name_containment_s1_in_tgt",
    "name_containment_tgt_in_s1",
    "name_idf_weighted_overlap",
    "name_token_jaccard",

    # Family 2: Legal Form & Algorithmic Acronym Disentanglement (7 features)
    "legal_form_match",
    "legal_form_both_present",
    "legal_form_conflict",
    "acronym_exact_match",
    "acronym_in_name",
    "acronym_initials_s1_in_tgt",
    "acronym_initials_tgt_in_s1",

    # Family 3: Phonetic, Sub-Word & Consonant Skeletons (6 features)
    "phonetic_code_match",
    "phonetic_similarity",
    "char_3gram_jaccard",
    "char_4gram_jaccard",
    "name_consonant_skeleton_sim",
    "name_consonant_skeleton_exact",

    # Family 4: Address Hierarchy, Structured Locality & Contradictions (16 features)
    "addr_clean_fuzz_ratio",
    "addr_clean_token_set_ratio",
    "addr_clean_partial_ratio",
    "addr_token_jaccard",
    "addr_token_overlap_count",
    "addr_idf_weighted_overlap",
    "addr_locality_jaccard",
    "addr_landmark_jaccard",
    "postal_exact_match",
    "postal_prefix_match",
    "postal_conflict",
    "unit_num_match",
    "unit_num_conflict",
    "house_num_match",
    "house_num_conflict",
    "addr_digits_jaccard",

    # Family 5: Pure Missingness Indicators & Conditional Signals (9 features)
    "target_name_missing",
    "target_addr_missing",
    "target_postal_missing",
    "source_addr_missing",
    "source_postal_missing",
    "has_both_addr",
    "has_both_postals",
    "name_sim_when_target_addr_missing",
    "addr_sim_when_target_name_missing",

    # Family 6: 9-Channel Blocker Provenance & Channel Synergy (15 features)
    "blocker_candidate_score",
    "blocker_candidate_rank",
    "blocker_reciprocal_rank",
    "c_name_core",
    "c_name_token",
    "c_name_contain",
    "c_char_3gram",
    "c_acronym",
    "c_addr_token",
    "c_addr_numeric",
    "c_postal",
    "c_phonetic",
    "num_channels",
    "c_name_and_addr_hit",
    "c_phonetic_and_postal_hit",

    # Family 7: Multi-Tier Hierarchical Anchors & Interactions (7 features)
    "name_core_x_addr_jaccard",
    "name_core_x_postal_match",
    "addr_fuzz_x_postal_match",
    "three_way_consistency",
    "exact_core_name_and_postal",
    "exact_core_name_and_house",
    "exact_core_name_and_locality",

    # Family 8: Source System & Country Context (5 features)
    "is_target_s2",
    "is_target_s3",
    "is_country_india",
    "is_country_us",
    "is_country_france",
]


# Structured Locality & Landmark Keywords for Non-Destructive Parsing
LOCALITY_KEYWORDS: Set[str] = {
    "nagar", "colony", "layout", "phase", "sector", "sec", "vihar", "enclave",
    "society", "soc", "mohalla", "puram", "wadi", "pada", "baug", "bazaar",
    "bazar", "mandi", "market", "park", "village", "gram", "taluka", "dist",
    "district", "suburb", "hills", "gardens", "heights", "town", "ward",
}

LANDMARK_KEYWORDS: Set[str] = {
    "opp", "opposite", "nr", "near", "adj", "adjacent", "behind", "bh", "bd",
    "beside", "front", "above", "below", "temple", "mandir", "masjid", "church",
    "station", "stn", "railway", "bus", "stand", "stop", "depot", "bridge",
    "cross", "circle", "chowk", "corner", "gate", "hospital", "school", "college",
    "petrol", "pump", "bank",
}

VOWELS_SET: Set[str] = set("aeiouyAEIOUY ")


def extract_consonant_skeleton(text: str) -> str:
    """
    Extracts vowel-stripped, deduplicated consonant skeleton for transliteration invariance.
    e.g. 'Venkateshwara' -> 'vnktshwr', 'Venkateswara' -> 'vnktswr'
    """
    if not text:
        return ""
    consonants = [c for c in text if c not in VOWELS_SET and c.isalnum()]
    dedup = []
    for c in consonants:
        if not dedup or dedup[-1] != c:
            dedup.append(c)
    return "".join(dedup).lower()


def match_acronym_initials(acro: str, tokens: List[str]) -> float:
    """
    Algorithmic multi-token acronym match: checks if acronym characters match
    the initials of consecutive tokens without requiring hardcoded dictionaries.
    """
    if not acro or len(acro) < 2 or not tokens:
        return 0.0
    initials = "".join(t[0] for t in tokens if t).lower()
    return 1.0 if acro.lower() in initials else 0.0


def extract_char_ngrams(text: str, n: int = 3) -> Set[str]:
    """Extracts character n-grams from normalized text."""
    if not text or len(text) < n:
        return set()
    clean = text.replace(" ", "")
    if len(clean) < n:
        return set()
    return {clean[i:i+n] for i in range(len(clean) - n + 1)}


def get_country_postal_prefix(postal: str, country: str) -> str:
    """Returns country-aware postal prefix (3 digits for India PIN, 3 digits for US ZIP, 2 for FR)."""
    if not postal:
        return ""
    clean = postal.strip()
    if country == "India":
        return clean[:3] if len(clean) >= 3 else clean
    elif country == "US":
        return clean[:3] if len(clean) >= 3 else clean
    elif country == "France":
        return clean[:2] if len(clean) >= 2 else clean
    return clean[:3] if len(clean) >= 3 else clean


class FeatureExtractor:
    """
    High-Performance Pairwise Feature Extractor across all 8 feature families (82 signals).
    """

    def __init__(self, idf_computer: Optional[CountryIDFComputer] = None):
        self.idf_computer = idf_computer

    def extract_pair_features(
        self,
        s1_record: Dict[str, any],
        tgt_record: Dict[str, any],
        provenance: Optional[Dict[str, any]] = None,
    ) -> List[float]:
        """
        Computes the complete 82-dimensional feature vector for a single (S1, Target) pair.
        """
        country = str(s1_record.get("country", "") or "")
        tgt_id = str(tgt_record.get("entity_id", "") or "")

        # S1 strings & parsed attributes
        s1_name_clean = str(s1_record.get("name_clean", "") or "")
        s1_name_core = str(s1_record.get("name_core", "") or "")
        s1_legal = str(s1_record.get("legal_form", "") or "NONE")
        s1_acro = str(s1_record.get("name_acronym", "") or "")
        s1_phone = str(s1_record.get("name_phonetic", "") or "")
        s1_addr_clean = str(s1_record.get("addr_clean", "") or "")
        s1_postal = str(s1_record.get("postal_clean", "") or "")
        s1_unit = str(s1_record.get("addr_unit_num", "") or "")
        s1_digits_str = str(s1_record.get("addr_digits", "") or "")
        s1_primary_digit = s1_digits_str.split()[0] if s1_digits_str else ""

        s1_tok_str = str(s1_record.get("name_tokens", "") or "")
        s1_tokens = set(s1_tok_str.split()) if s1_tok_str else set(s1_name_clean.split())
        s1_tok_list = s1_tok_str.split() if s1_tok_str else s1_name_clean.split()
        s1_first_tok = s1_tok_list[0] if s1_tok_list else ""
        s1_last_tok = s1_tok_list[-1] if s1_tok_list else ""
        s1_name_concat = s1_name_clean.replace(" ", "")

        s1_atok_str = str(s1_record.get("addr_tokens", "") or "")
        s1_atoks = set(s1_atok_str.split()) if s1_atok_str else set(s1_addr_clean.split())
        s1_digits = set(s1_digits_str.split()) if s1_digits_str else set()

        # Target strings & parsed attributes
        tgt_name_clean = str(tgt_record.get("name_clean", "") or "")
        tgt_name_core = str(tgt_record.get("name_core", "") or "")
        tgt_legal = str(tgt_record.get("legal_form", "") or "NONE")
        tgt_acro = str(tgt_record.get("name_acronym", "") or "")
        tgt_phone = str(tgt_record.get("name_phonetic", "") or "")
        tgt_addr_clean = str(tgt_record.get("addr_clean", "") or "")
        tgt_postal = str(tgt_record.get("postal_clean", "") or "")
        tgt_unit = str(tgt_record.get("addr_unit_num", "") or "")
        tgt_digits_str = str(tgt_record.get("addr_digits", "") or "")
        tgt_primary_digit = tgt_digits_str.split()[0] if tgt_digits_str else ""

        tgt_tok_str = str(tgt_record.get("name_tokens", "") or "")
        tgt_tokens = set(tgt_tok_str.split()) if tgt_tok_str else set(tgt_name_clean.split())
        tgt_tok_list = tgt_tok_str.split() if tgt_tok_str else tgt_name_clean.split()
        tgt_first_tok = tgt_tok_list[0] if tgt_tok_list else ""
        tgt_last_tok = tgt_tok_list[-1] if tgt_tok_list else ""
        tgt_name_concat = tgt_name_clean.replace(" ", "")

        tgt_atok_str = str(tgt_record.get("addr_tokens", "") or "")
        tgt_atoks = set(tgt_atok_str.split()) if tgt_atok_str else set(tgt_addr_clean.split())
        tgt_digits = set(tgt_digits_str.split()) if tgt_digits_str else set()

        prov = provenance or {}

        # =========================================================================
        # Family 1: Name Similarity & Asymmetric Containment (17 features)
        # =========================================================================
        name_clean_fuzz = fuzz.ratio(s1_name_clean, tgt_name_clean) / 100.0 if (s1_name_clean and tgt_name_clean) else 0.0
        name_clean_sort = fuzz.token_sort_ratio(s1_name_clean, tgt_name_clean) / 100.0 if (s1_name_clean and tgt_name_clean) else 0.0
        name_clean_set = fuzz.token_set_ratio(s1_name_clean, tgt_name_clean) / 100.0 if (s1_name_clean and tgt_name_clean) else 0.0
        name_clean_wratio = fuzz.WRatio(s1_name_clean, tgt_name_clean) / 100.0 if (s1_name_clean and tgt_name_clean) else 0.0
        name_clean_jw = distance.JaroWinkler.similarity(s1_name_clean, tgt_name_clean) if (s1_name_clean and tgt_name_clean) else 0.0

        name_core_fuzz = fuzz.ratio(s1_name_core, tgt_name_core) / 100.0 if (s1_name_core and tgt_name_core) else 0.0
        name_core_set = fuzz.token_set_ratio(s1_name_core, tgt_name_core) / 100.0 if (s1_name_core and tgt_name_core) else 0.0
        name_core_jw = distance.JaroWinkler.similarity(s1_name_core, tgt_name_core) if (s1_name_core and tgt_name_core) else 0.0

        name_core_exact = 1.0 if (s1_name_core and s1_name_core == tgt_name_core) else 0.0
        name_concat_exact = 1.0 if (s1_name_concat and len(s1_name_concat) >= 4 and s1_name_concat == tgt_name_concat) else 0.0

        l1, l2 = len(s1_name_clean), len(tgt_name_clean)
        name_len_diff_ratio = (abs(l1 - l2) / max(l1, l2, 1)) if (l1 > 0 or l2 > 0) else 0.0

        name_first_tok_sim = fuzz.ratio(s1_first_tok, tgt_first_tok) / 100.0 if (s1_first_tok and tgt_first_tok) else 0.0
        name_last_tok_sim = fuzz.ratio(s1_last_tok, tgt_last_tok) / 100.0 if (s1_last_tok and tgt_last_tok) else 0.0

        name_inter = s1_tokens & tgt_tokens
        name_union = s1_tokens | tgt_tokens
        name_contain_s1_in_tgt = (len(name_inter) / len(s1_tokens)) if s1_tokens else 0.0
        name_contain_tgt_in_s1 = (len(name_inter) / len(tgt_tokens)) if tgt_tokens else 0.0
        name_tok_jaccard = (len(name_inter) / len(name_union)) if name_union else 0.0

        if self.idf_computer and name_union:
            sum_inter = sum(self.idf_computer.get_name_token_idf(country, t) for t in name_inter)
            sum_union = sum(self.idf_computer.get_name_token_idf(country, t) for t in name_union)
            name_idf_overlap = (sum_inter / sum_union) if sum_union > 0 else 0.0
        else:
            name_idf_overlap = name_tok_jaccard

        # =========================================================================
        # Family 2: Legal Form & Algorithmic Acronym Disentanglement (7 features)
        # =========================================================================
        has_s1_legal = s1_legal not in ("NONE", "")
        has_tgt_legal = tgt_legal not in ("NONE", "")
        legal_match = 1.0 if (has_s1_legal and has_tgt_legal and s1_legal == tgt_legal) else 0.0
        legal_both_present = 1.0 if (has_s1_legal and has_tgt_legal) else 0.0
        legal_conflict = 1.0 if (has_s1_legal and has_tgt_legal and s1_legal != tgt_legal) else 0.0

        acro_match = 1.0 if (s1_acro and tgt_acro and s1_acro == tgt_acro) else 0.0
        acro_in_name = 1.0 if ((s1_acro and s1_acro in tgt_tokens) or (tgt_acro and tgt_acro in s1_tokens)) else 0.0

        # Algorithmic Initial Matching (No Hardcoded Dictionary)
        acro_init_s1_in_tgt = match_acronym_initials(s1_acro, tgt_tok_list)
        acro_init_tgt_in_s1 = match_acronym_initials(tgt_acro, s1_tok_list)

        # =========================================================================
        # Family 3: Phonetic, Sub-Word & Consonant Skeletons (6 features)
        # =========================================================================
        phone_match = 1.0 if (s1_phone and tgt_phone and s1_phone == tgt_phone) else 0.0
        phone_sim = (fuzz.ratio(s1_phone, tgt_phone) / 100.0) if (s1_phone and tgt_phone) else 0.0

        s1_c3 = extract_char_ngrams(s1_name_core, 3)
        tgt_c3 = extract_char_ngrams(tgt_name_core, 3)
        c3_inter = s1_c3 & tgt_c3
        c3_union = s1_c3 | tgt_c3
        c3_jaccard = (len(c3_inter) / len(c3_union)) if c3_union else 0.0

        s1_c4 = extract_char_ngrams(s1_name_core, 4)
        tgt_c4 = extract_char_ngrams(tgt_name_core, 4)
        c4_inter = s1_c4 & tgt_c4
        c4_union = s1_c4 | tgt_c4
        c4_jaccard = (len(c4_inter) / len(c4_union)) if c4_union else 0.0

        # Consonant Skeleton Transliteration Feature
        s1_skel = extract_consonant_skeleton(s1_name_core)
        tgt_skel = extract_consonant_skeleton(tgt_name_core)
        consonant_skel_sim = (fuzz.ratio(s1_skel, tgt_skel) / 100.0) if (s1_skel and tgt_skel) else 0.0
        consonant_skel_exact = 1.0 if (s1_skel and s1_skel == tgt_skel) else 0.0

        # =========================================================================
        # Family 4: Address Hierarchy, Structured Locality & Contradictions (16 features)
        # =========================================================================
        has_s1_addr = bool(s1_addr_clean)
        has_tgt_addr = bool(tgt_addr_clean)
        has_both_addr_val = 1.0 if (has_s1_addr and has_tgt_addr) else 0.0

        # Locality and Landmark sub-token analysis
        s1_loc = s1_atoks & LOCALITY_KEYWORDS
        tgt_loc = tgt_atoks & LOCALITY_KEYWORDS
        loc_union = s1_loc | tgt_loc
        addr_loc_jaccard = (len(s1_loc & tgt_loc) / len(loc_union)) if loc_union else 0.0

        s1_land = s1_atoks & LANDMARK_KEYWORDS
        tgt_land = tgt_atoks & LANDMARK_KEYWORDS
        land_union = s1_land | tgt_land
        addr_land_jaccard = (len(s1_land & tgt_land) / len(land_union)) if land_union else 0.0

        if has_both_addr_val:
            addr_fuzz = fuzz.ratio(s1_addr_clean, tgt_addr_clean) / 100.0
            addr_set = fuzz.token_set_ratio(s1_addr_clean, tgt_addr_clean) / 100.0
            addr_part = fuzz.partial_ratio(s1_addr_clean, tgt_addr_clean) / 100.0
            ainter = s1_atoks & tgt_atoks
            aunion = s1_atoks | tgt_atoks
            addr_jaccard = (len(ainter) / len(aunion)) if aunion else 0.0
            addr_overlap_cnt = float(len(ainter))
            if self.idf_computer and aunion:
                asum_inter = sum(self.idf_computer.get_addr_token_idf(country, at) for at in ainter)
                asum_union = sum(self.idf_computer.get_addr_token_idf(country, at) for at in aunion)
                addr_idf_overlap = (asum_inter / asum_union) if asum_union > 0 else 0.0
            else:
                addr_idf_overlap = addr_jaccard
        else:
            addr_fuzz = 0.0
            addr_set = 0.0
            addr_part = 0.0
            addr_jaccard = 0.0
            addr_overlap_cnt = 0.0
            addr_idf_overlap = 0.0

        has_s1_post = bool(s1_postal)
        has_tgt_post = bool(tgt_postal)
        has_both_post_val = 1.0 if (has_s1_post and has_tgt_post) else 0.0

        postal_exact = 1.0 if (has_both_post_val and s1_postal == tgt_postal) else 0.0
        s1_pfx = get_country_postal_prefix(s1_postal, country)
        tgt_pfx = get_country_postal_prefix(tgt_postal, country)
        postal_pfx_match = 1.0 if (s1_pfx and tgt_pfx and s1_pfx == tgt_pfx) else 0.0
        postal_conflict = 1.0 if (has_both_post_val and s1_postal != tgt_postal) else 0.0

        has_s1_u = bool(s1_unit)
        has_tgt_u = bool(tgt_unit)
        unit_match = 1.0 if (has_s1_u and has_tgt_u and s1_unit == tgt_unit) else 0.0
        unit_conflict = 1.0 if (has_s1_u and has_tgt_u and s1_unit != tgt_unit) else 0.0

        has_s1_dig = bool(s1_primary_digit)
        has_tgt_dig = bool(tgt_primary_digit)
        house_num_match = 1.0 if (has_s1_dig and has_tgt_dig and s1_primary_digit == tgt_primary_digit) else 0.0
        house_num_conflict = 1.0 if (has_s1_dig and has_tgt_dig and s1_primary_digit != tgt_primary_digit) else 0.0

        d_inter = s1_digits & tgt_digits
        d_union = s1_digits | tgt_digits
        addr_digits_jaccard = (len(d_inter) / len(d_union)) if d_union else 0.0

        # =========================================================================
        # Family 5: Pure Missingness Indicators & Conditional Signals (9 features)
        # =========================================================================
        tgt_name_missing = 1.0 if not tgt_name_clean else 0.0
        tgt_addr_missing = 1.0 if not tgt_addr_clean else 0.0
        tgt_post_missing = 1.0 if not tgt_postal else 0.0
        s1_addr_missing = 1.0 if not s1_addr_clean else 0.0
        s1_post_missing = 1.0 if not s1_postal else 0.0

        name_sim_when_tgt_addr_missing = name_clean_fuzz if tgt_addr_missing else 0.0
        addr_sim_when_tgt_name_missing = addr_fuzz if tgt_name_missing else 0.0

        # =========================================================================
        # Family 6: 9-Channel Blocker Provenance & Channel Synergy (15 features)
        # =========================================================================
        b_score = float(prov.get("candidate_score", 0.0))
        b_rank = float(prov.get("candidate_rank", 60))
        b_recip_rank = 1.0 / b_rank if b_rank > 0 else 0.0

        c_name_core = float(prov.get("c_name_core", 0))
        c_name_token = float(prov.get("c_name_token", 0))
        c_name_contain = float(prov.get("c_name_contain", 0))
        c_char_3gram = float(prov.get("c_char_3gram", 0))
        c_acronym = float(prov.get("c_acronym", 0))
        c_addr_token = float(prov.get("c_addr_token", 0))
        c_addr_numeric = float(prov.get("c_addr_numeric", 0))
        c_postal = float(prov.get("c_postal", 0))
        c_phonetic = float(prov.get("c_phonetic", 0))
        num_channels = float(prov.get("num_channels", 0))

        c_name_and_addr = 1.0 if ((c_name_core or c_name_token) and (c_addr_token or c_addr_numeric)) else 0.0
        c_phone_and_post = 1.0 if (c_phonetic and c_postal) else 0.0

        # =========================================================================
        # Family 7: Multi-Tier Hierarchical Anchors & Interactions (7 features)
        # =========================================================================
        name_x_addr = name_core_set * addr_jaccard
        name_x_post = name_core_jw * postal_exact
        addr_x_post = addr_fuzz * postal_exact
        three_way = name_core_jw * addr_fuzz * postal_exact

        # Multi-Tier Exact Anchors
        exact_core_and_postal = 1.0 if (name_core_exact == 1.0 and postal_exact == 1.0) else 0.0
        exact_core_and_house = 1.0 if (name_core_exact == 1.0 and house_num_match == 1.0) else 0.0
        exact_core_and_loc = 1.0 if (name_core_exact == 1.0 and bool(s1_loc & tgt_loc)) else 0.0

        # =========================================================================
        # Family 8: Source System & Country Context (5 features)
        # =========================================================================
        is_s2 = 1.0 if tgt_id.startswith("S2") else 0.0
        is_s3 = 1.0 if tgt_id.startswith("S3") else 0.0
        is_in = 1.0 if country == "India" else 0.0
        is_us = 1.0 if country == "US" else 0.0
        is_fr = 1.0 if country == "France" else 0.0

        return [
            # F1: Name
            name_clean_fuzz,
            name_clean_sort,
            name_clean_set,
            name_clean_wratio,
            name_clean_jw,
            name_core_fuzz,
            name_core_set,
            name_core_jw,
            name_core_exact,
            name_concat_exact,
            name_len_diff_ratio,
            name_first_tok_sim,
            name_last_tok_sim,
            name_contain_s1_in_tgt,
            name_contain_tgt_in_s1,
            name_idf_overlap,
            name_tok_jaccard,

            # F2: Legal & Acronym
            legal_match,
            legal_both_present,
            legal_conflict,
            acro_match,
            acro_in_name,
            acro_init_s1_in_tgt,
            acro_init_tgt_in_s1,

            # F3: Phonetic & Sub-Word & Consonant Skeletons
            phone_match,
            phone_sim,
            c3_jaccard,
            c4_jaccard,
            consonant_skel_sim,
            consonant_skel_exact,

            # F4: Address Hierarchy, Locality & Contradictions
            addr_fuzz,
            addr_set,
            addr_part,
            addr_jaccard,
            addr_overlap_cnt,
            addr_idf_overlap,
            addr_loc_jaccard,
            addr_land_jaccard,
            postal_exact,
            postal_pfx_match,
            postal_conflict,
            unit_match,
            unit_conflict,
            house_num_match,
            house_num_conflict,
            addr_digits_jaccard,

            # F5: Missingness
            tgt_name_missing,
            tgt_addr_missing,
            tgt_post_missing,
            s1_addr_missing,
            s1_post_missing,
            has_both_addr_val,
            has_both_post_val,
            name_sim_when_tgt_addr_missing,
            addr_sim_when_tgt_name_missing,

            # F6: Provenance
            b_score,
            b_rank,
            b_recip_rank,
            c_name_core,
            c_name_token,
            c_name_contain,
            c_char_3gram,
            c_acronym,
            c_addr_token,
            c_addr_numeric,
            c_postal,
            c_phonetic,
            num_channels,
            c_name_and_addr,
            c_phone_and_post,

            # F7: Multi-Tier Anchors & Interactions
            name_x_addr,
            name_x_post,
            addr_x_post,
            three_way,
            exact_core_and_postal,
            exact_core_and_house,
            exact_core_and_loc,

            # F8: Source & Country
            is_s2,
            is_s3,
            is_in,
            is_us,
            is_fr,
        ]
