"""
Multilingual Text Normalization & Cleaning Module (Amazon ML Challenge 2026).
Handles diacritic normalization, legal suffix standardization, address component
cleaning, and token extraction for US, India, and France entities.
"""

import re
import unicodedata
from typing import List, Tuple, Optional, Set

# Legal suffix mapping for standardization across English and French entities
LEGAL_SUFFIX_MAP = {
    # English / US / India
    r"\bincorporated\b": "inc",
    r"\binc\b\.?": "inc",
    r"\bcorporation\b": "corp",
    r"\bcorp\b\.?": "corp",
    r"\bcompany\b": "co",
    r"\bco\b\.?": "co",
    r"\blimited liability company\b": "llc",
    r"\bllc\b\.?": "llc",
    r"\bl\.l\.c\b\.?": "llc",
    r"\bprivate limited\b": "pvt ltd",
    r"\bpvt\b\.?\s*\bltd\b\.?": "pvt ltd",
    r"\bp\b\.?\s*\bltd\b\.?": "pvt ltd",
    r"\blimited\b": "ltd",
    r"\bltd\b\.?": "ltd",
    r"\bpublic limited\b": "pub ltd",
    r"\benterprises\b": "ent",
    r"\benterprise\b": "ent",
    r"\bservices\b": "svc",
    r"\bservice\b": "svc",
    r"\bassociates\b": "assoc",
    r"\bassoc\b\.?": "assoc",
    # French Legal Forms
    r"\bsociete anonyme\b": "sa",
    r"\bs\.a\b\.?": "sa",
    r"\bsarl\b\.?": "sarl",
    r"\bs\.a\.r\.l\b\.?": "sarl",
    r"\bsas\b\.?": "sas",
    r"\bs\.a\.s\b\.?": "sas",
    r"\beurl\b\.?": "eurl",
    r"\bsci\b\.?": "sci",
}

# Address Road & Structure abbreviations
ADDRESS_ABBREV_MAP = {
    r"\bstreet\b": "st",
    r"\bst\b\.?": "st",
    r"\broad\b": "rd",
    r"\brd\b\.?": "rd",
    r"\bavenue\b": "ave",
    r"\bave\b\.?": "ave",
    r"\bboulevard\b": "blvd",
    r"\bblvd\b\.?": "blvd",
    r"\blane\b": "ln",
    r"\bln\b\.?": "ln",
    r"\bdrive\b": "dr",
    r"\bdr\b\.?": "dr",
    r"\bcourt\b": "ct",
    r"\bct\b\.?": "ct",
    r"\bhighway\b": "hwy",
    r"\bhwy\b\.?": "hwy",
    r"\bapartment\b": "apt",
    r"\bapt\b\.?": "apt",
    r"\bsuite\b": "ste",
    r"\bste\b\.?": "ste",
    r"\bfloor\b": "flr",
    r"\bflr\b\.?": "flr",
    r"\bnear\b": "near",
    r"\bopposite\b": "opp",
    r"\bopp\b\.?": "opp",
    r"\bdoor no\b\.?": "no",
    r"\bflat no\b\.?": "no",
    r"\bh no\b\.?": "no",
    r"\bhouse no\b\.?": "no",
}

# Stopwords for candidate blocking (words that appear too frequently to be distinctive)
NAME_STOPWORDS = {
    "the", "a", "an", "and", "of", "in", "for", "on", "at", "by", "to", "with",
    "inc", "corp", "co", "llc", "ltd", "pvt", "sa", "sarl", "sas", "company",
    "enterprise", "enterprises", "service", "services", "center", "centre",
    "group", "international", "global", "solutions", "holdings", "management",
    "india", "usa", "us", "france"
}


def strip_accents_and_normalize(text: str) -> str:
    """Strip Unicode accents and normalize characters to ascii-compatible form."""
    if not text:
        return ""
    # Normalize unicode to decomposed form and remove non-spacing marks
    normalized = unicodedata.normalize("NFKD", text)
    stripped = "".join(c for c in normalized if not unicodedata.combining(c))
    return stripped.lower()


def clean_text_basic(text: str) -> str:
    """Basic text cleanup: replace punctuation with spaces and collapse whitespace."""
    if not text or not str(text).strip():
        return ""
    text = str(text)
    # Replace & with and, @ with at
    text = text.replace("&", " and ").replace("@", " at ")
    # Strip diacritics and lowercase
    text = strip_accents_and_normalize(text)
    # Replace non-alphanumeric characters with space
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    # Collapse multiple whitespaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_business_name(name: str) -> str:
    """Full normalization for business name."""
    clean = clean_text_basic(name)
    if not clean:
        return ""
    # Apply legal suffix standardization
    for pattern, replacement in LEGAL_SUFFIX_MAP.items():
        clean = re.sub(pattern, replacement, clean)
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean


def normalize_business_address(address: str) -> str:
    """Full normalization for business address."""
    clean = clean_text_basic(address)
    if not clean:
        return ""
    # Apply address abbreviation standardization
    for pattern, replacement in ADDRESS_ABBREV_MAP.items():
        clean = re.sub(pattern, replacement, clean)
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean


def extract_informative_tokens(name_clean: str, min_len: int = 3) -> List[str]:
    """Extract distinct, non-stopword tokens from cleaned business name."""
    tokens = name_clean.split()
    return [t for t in tokens if len(t) >= min_len and t not in NAME_STOPWORDS]


def extract_core_name_prefix(name_clean: str, n_tokens: int = 2) -> str:
    """Extract first n significant tokens as a core blocking prefix."""
    tokens = [t for t in name_clean.split() if t not in NAME_STOPWORDS]
    return " ".join(tokens[:n_tokens]) if tokens else name_clean[:10]


def extract_digits(text: str) -> Set[str]:
    """Extract all standalone or continuous digit strings (PIN codes, house numbers)."""
    if not text:
        return set()
    return set(re.findall(r"\b\d{3,6}\b", text))
