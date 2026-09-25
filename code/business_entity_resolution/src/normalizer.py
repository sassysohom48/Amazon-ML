"""
Vectorized Multilingual Normalization & Representation Module (Amazon ML Challenge 2026).
High-speed Unicode normalization, diacritic stripping, sorted token keys,
and address locality anchor extraction for US, India, and France.
"""

import re
import unicodedata
from typing import List, Set, Tuple, Optional, Dict
import polars as pl

# Statistical non-distinctive words across corporate business names
CORP_STOPWORDS: Set[str] = {
    "the", "a", "an", "and", "of", "in", "for", "on", "at", "by", "to", "with",
    "inc", "incorporated", "corp", "corporation", "co", "company", "llc", "ltd",
    "limited", "pvt", "private", "sa", "sarl", "sas", "eurl", "sci", "enterprise",
    "enterprises", "service", "services", "center", "centre", "group", "international",
    "global", "solutions", "holdings", "management", "india", "usa", "us", "france"
}

# Generic address descriptors to filter when creating anchors
ADDR_STOPWORDS: Set[str] = {
    "st", "street", "rd", "road", "ave", "avenue", "blvd", "boulevard", "dr", "drive",
    "ln", "lane", "ct", "court", "unit", "apt", "apartment", "ste", "suite", "flr", "floor",
    "bldg", "building", "near", "opp", "opposite", "behind", "door", "flat", "no", "h"
}

# Pre-compiled regex patterns for speed
RE_NON_ALPHANUM = re.compile(r"[^a-z0-9\s]")
RE_WHITESPACE = re.compile(r"\s+")
RE_DIGITS = re.compile(r"\b\d{2,7}\b")


def strip_accents(text: str) -> str:
    """Fast diacritic stripping (e.g. é -> e, ô -> o, ç -> c)."""
    if not text:
        return ""
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(c for c in normalized if not unicodedata.combining(c)).lower()


def clean_text(text: str) -> str:
    """
    Standardizes text: replaces symbols, removes diacritics,
    strips non-alphanumeric chars, collapses whitespace.
    """
    if not text:
        return ""
    text = str(text).lower()
    text = text.replace("&", " and ").replace("@", " at ")
    text = strip_accents(text)
    text = RE_NON_ALPHANUM.sub(" ", text)
    return RE_WHITESPACE.sub(" ", text).strip()


def extract_informative_tokens(text_clean: str, min_len: int = 2) -> List[str]:
    """Extracts distinctive tokens for inverted index blocking."""
    if not text_clean:
        return []
    tokens = text_clean.split()
    return [t for t in tokens if len(t) >= min_len and t not in CORP_STOPWORDS]


def extract_sorted_token_key(text_clean: str, max_tokens: int = 3) -> str:
    """
    Extracts alphabetically sorted significant tokens (e.g. 'clinic_mumbai_producer').
    Solves word-order inversion / transposition across sources.
    """
    tokens = [t for t in text_clean.split() if len(t) >= 2 and t not in CORP_STOPWORDS]
    if not tokens:
        return text_clean[:10] if text_clean else ""
    sorted_tokens = sorted(tokens[:max_tokens])
    return "_".join(sorted_tokens)


def extract_char_4grams(text_clean: str) -> Set[str]:
    """Generates character 4-gram shingles for typo-tolerant fuzzy matching."""
    if not text_clean:
        return set()
    compact = text_clean.replace(" ", "")
    if len(compact) < 4:
        return {compact} if compact else set()
    return {compact[i:i + 4] for i in range(len(compact) - 4 + 1)}


def extract_numeric_tokens(address_text: str) -> List[str]:
    """Extracts standalone numeric tokens (house/unit numbers, postal codes)."""
    if not address_text:
        return []
    return RE_DIGITS.findall(address_text.lower())


def extract_address_anchors(addr_clean: str) -> List[str]:
    """
    Extracts multi-token address anchors:
    - (street_number + locality_word) e.g. '1795_westchester', '797_lake'
    - (postal_code / 5-6 digit numbers) e.g. '75008', '400080'
    - (locality word pairs)
    """
    if not addr_clean:
        return []
    tokens = addr_clean.split()
    anchors = []

    # 1. Extract standalone 5-6 digit postal codes (high precision)
    for tok in tokens:
        if tok.isdigit() and len(tok) in (5, 6):
            anchors.append(f"pin_{tok}")

    # 2. Extract digit + adjacent informative word anchors
    for i, tok in enumerate(tokens):
        if tok.isdigit() and len(tok) >= 2:
            # Pair with next non-stopword token
            if i + 1 < len(tokens):
                next_t = tokens[i + 1]
                if len(next_t) >= 3 and next_t not in ADDR_STOPWORDS and next_t not in CORP_STOPWORDS:
                    anchors.append(f"{tok}_{next_t}")
            # Pair with previous non-stopword token
            if i > 0:
                prev_t = tokens[i - 1]
                if len(prev_t) >= 3 and prev_t not in ADDR_STOPWORDS and prev_t not in CORP_STOPWORDS:
                    anchors.append(f"{prev_t}_{tok}")

    return anchors
