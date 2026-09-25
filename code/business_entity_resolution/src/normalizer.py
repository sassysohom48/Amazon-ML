"""
Vectorized Multilingual Normalization & Representation Module (Amazon ML Challenge 2026).
High-speed Unicode normalization, diacritic stripping, token shingling,
and address/numeric anchor extraction for US, India, and France.
"""

import re
import unicodedata
from typing import List, Set, Tuple, Optional, Dict
import polars as pl

# Statistical stopword tokens that are non-distinctive across entity names
CORP_STOPWORDS: Set[str] = {
    "the", "a", "an", "and", "of", "in", "for", "on", "at", "by", "to", "with",
    "inc", "incorporated", "corp", "corporation", "co", "company", "llc", "ltd",
    "limited", "pvt", "private", "sa", "sarl", "sas", "eurl", "sci", "enterprise",
    "enterprises", "service", "services", "center", "centre", "group", "international",
    "global", "solutions", "holdings", "management", "india", "usa", "us", "france"
}

# Regex pre-compilation for maximum speed
RE_NON_ALPHANUM = re.compile(r"[^a-z0-9\s]")
RE_WHITESPACE = re.compile(r"\s+")
RE_DIGITS = re.compile(r"\b\d{2,7}\b")
RE_WORDS = re.compile(r"\b[a-z0-9]{2,}\b")


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


def extract_name_prefix_key(text_clean: str, n_tokens: int = 2) -> str:
    """Extracts first N significant words as a compound prefix key."""
    tokens = [t for t in text_clean.split() if t not in CORP_STOPWORDS]
    if not tokens:
        return text_clean[:8] if text_clean else ""
    return "_".join(tokens[:n_tokens])


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
    Extracts address anchor tuples: combinations of numeric tokens with adjacent words
    (e.g. '1795_westchester', '2100_cameron').
    """
    if not addr_clean:
        return []
    tokens = addr_clean.split()
    anchors = []
    for i, tok in enumerate(tokens):
        if tok.isdigit() and len(tok) >= 2:
            # Pair with next non-stopword token if available
            if i + 1 < len(tokens) and len(tokens[i + 1]) >= 3 and tokens[i + 1] not in CORP_STOPWORDS:
                anchors.append(f"{tok}_{tokens[i + 1]}")
            # Pair with previous token
            elif i > 0 and len(tokens[i - 1]) >= 3 and tokens[i - 1] not in CORP_STOPWORDS:
                anchors.append(f"{tokens[i - 1]}_{tok}")
            else:
                anchors.append(tok)
    return anchors
