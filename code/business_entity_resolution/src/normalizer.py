"""
Vectorized Multilingual Normalization & Representation Module (Amazon ML Challenge 2026).
High-speed Unicode normalization, diacritic stripping, token shingling,
and address numeric anchor extraction for US, India, and France.
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
    """Extracts non-stopword tokens for inverted index blocking."""
    if not text_clean:
        return []
    tokens = text_clean.split()
    return [t for t in tokens if len(t) >= min_len and t not in CORP_STOPWORDS]


def extract_char_ngrams(text_clean: str, n: int = 3) -> Set[str]:
    """Generates character n-gram shingles for typo-tolerant fuzzy matching."""
    if not text_clean:
        return set()
    compact = text_clean.replace(" ", "")
    if len(compact) < n:
        return {compact} if compact else set()
    return {compact[i:i + n] for i in range(len(compact) - n + 1)}


def extract_numeric_tokens(address_text: str) -> Set[str]:
    """Extracts standalone numeric tokens (house/unit numbers, postal codes)."""
    if not address_text:
        return set()
    return set(RE_DIGITS.findall(address_text.lower()))


def batch_normalize_dataframe(df: pl.DataFrame) -> pl.DataFrame:
    """
    Vectorized batch normalization of a Polars DataFrame containing business_name
    and business_address columns.
    """
    exprs = []
    if "business_name" in df.columns:
        exprs.append(
            pl.col("business_name")
            .str.to_lowercase()
            .str.replace_all(r"&", " and ")
            .str.replace_all(r"@", " at ")
            .str.replace_all(r"[^a-zA-Z0-9\s]", " ")
            .str.replace_all(r"\s+", " ")
            .str.strip_chars()
            .fill_null("")
            .alias("name_clean")
        )

    if "business_address" in df.columns:
        exprs.append(
            pl.col("business_address")
            .str.to_lowercase()
            .str.replace_all(r"&", " and ")
            .str.replace_all(r"@", " at ")
            .str.replace_all(r"[^a-zA-Z0-9\s]", " ")
            .str.replace_all(r"\s+", " ")
            .str.strip_chars()
            .fill_null("")
            .alias("addr_clean")
        )

    return df.with_columns(exprs)
