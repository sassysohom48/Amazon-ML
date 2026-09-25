"""
Multilingual Text Normalization & Cleaning Module (Amazon ML Challenge 2026).
Optimized for high-speed multi-core batch processing across millions of entities.
"""

import re
import unicodedata
from typing import List, Tuple, Optional, Set

# Pre-compiled Legal Suffix Regexes
LEGAL_PATTERNS = [
    # English / US / India
    (re.compile(r"\bincorporated\b"), "inc"),
    (re.compile(r"\binc\b\.?"), "inc"),
    (re.compile(r"\bcorporation\b"), "corp"),
    (re.compile(r"\bcorp\b\.?"), "corp"),
    (re.compile(r"\bcompany\b"), "co"),
    (re.compile(r"\bco\b\.?"), "co"),
    (re.compile(r"\blimited liability company\b"), "llc"),
    (re.compile(r"\bllc\b\.?"), "llc"),
    (re.compile(r"\bl\.l\.c\b\.?"), "llc"),
    (re.compile(r"\bprivate limited\b"), "pvt ltd"),
    (re.compile(r"\bpvt\b\.?\s*\bltd\b\.?"), "pvt ltd"),
    (re.compile(r"\bp\b\.?\s*\bltd\b\.?"), "pvt ltd"),
    (re.compile(r"\blimited\b"), "ltd"),
    (re.compile(r"\bltd\b\.?"), "ltd"),
    (re.compile(r"\bpublic limited\b"), "pub ltd"),
    (re.compile(r"\benterprises\b"), "ent"),
    (re.compile(r"\benterprise\b"), "ent"),
    (re.compile(r"\bservices\b"), "svc"),
    (re.compile(r"\bservice\b"), "svc"),
    (re.compile(r"\bassociates\b"), "assoc"),
    (re.compile(r"\bassoc\b\.?"), "assoc"),
    # French Legal Forms
    (re.compile(r"\bsociete anonyme\b"), "sa"),
    (re.compile(r"\bs\.a\b\.?"), "sa"),
    (re.compile(r"\bsarl\b\.?"), "sarl"),
    (re.compile(r"\bs\.a\.r\.l\b\.?"), "sarl"),
    (re.compile(r"\bsas\b\.?"), "sas"),
    (re.compile(r"\bs\.a\.s\b\.?"), "sas"),
    (re.compile(r"\beurl\b\.?"), "eurl"),
    (re.compile(r"\bsci\b\.?"), "sci"),
]

# Pre-compiled Address Regexes
ADDRESS_PATTERNS = [
    (re.compile(r"\bstreet\b"), "st"),
    (re.compile(r"\bst\b\.?"), "st"),
    (re.compile(r"\broad\b"), "rd"),
    (re.compile(r"\brd\b\.?"), "rd"),
    (re.compile(r"\bavenue\b"), "ave"),
    (re.compile(r"\bave\b\.?"), "ave"),
    (re.compile(r"\bboulevard\b"), "blvd"),
    (re.compile(r"\bblvd\b\.?"), "blvd"),
    (re.compile(r"\blane\b"), "ln"),
    (re.compile(r"\bln\b\.?"), "ln"),
    (re.compile(r"\bdrive\b"), "dr"),
    (re.compile(r"\bdr\b\.?"), "dr"),
    (re.compile(r"\bcourt\b"), "ct"),
    (re.compile(r"\bct\b\.?"), "ct"),
    (re.compile(r"\bhighway\b"), "hwy"),
    (re.compile(r"\bhwy\b\.?"), "hwy"),
    (re.compile(r"\bapartment\b"), "apt"),
    (re.compile(r"\bapt\b\.?"), "apt"),
    (re.compile(r"\bsuite\b"), "ste"),
    (re.compile(r"\bste\b\.?"), "ste"),
    (re.compile(r"\bfloor\b"), "flr"),
    (re.compile(r"\bflr\b\.?"), "flr"),
    (re.compile(r"\bnear\b"), "near"),
    (re.compile(r"\bopposite\b"), "opp"),
    (re.compile(r"\bopp\b\.?"), "opp"),
    (re.compile(r"\bdoor no\b\.?"), "no"),
    (re.compile(r"\bflat no\b\.?"), "no"),
    (re.compile(r"\bh no\b\.?"), "no"),
    (re.compile(r"\bhouse no\b\.?"), "no"),
]

# Stopwords for candidate blocking
NAME_STOPWORDS = {
    "the", "a", "an", "and", "of", "in", "for", "on", "at", "by", "to", "with",
    "inc", "corp", "co", "llc", "ltd", "pvt", "sa", "sarl", "sas", "company",
    "enterprise", "enterprises", "service", "services", "center", "centre",
    "group", "international", "global", "solutions", "holdings", "management",
    "india", "usa", "us", "france"
}

RE_NON_ALPHANUM = re.compile(r"[^a-z0-9\s]")
RE_WHITESPACE = re.compile(r"\s+")
RE_DIGITS = re.compile(r"\b\d{3,6}\b")


def strip_accents_and_normalize(text: str) -> str:
    """Strip Unicode accents and normalize characters to ascii-compatible form."""
    if not text:
        return ""
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(c for c in normalized if not unicodedata.combining(c)).lower()


def clean_text_basic(text: str) -> str:
    """Basic text cleanup: replace punctuation with spaces and collapse whitespace."""
    if not text or not str(text).strip():
        return ""
    text = str(text).replace("&", " and ").replace("@", " at ")
    text = strip_accents_and_normalize(text)
    text = RE_NON_ALPHANUM.sub(" ", text)
    return RE_WHITESPACE.sub(" ", text).strip()


def normalize_business_name(name: str) -> str:
    """Full normalization for business name with legal suffix standardization."""
    clean = clean_text_basic(name)
    if not clean:
        return ""
    for pattern, replacement in LEGAL_PATTERNS:
        clean = pattern.sub(replacement, clean)
    return RE_WHITESPACE.sub(" ", clean).strip()


def normalize_business_address(address: str) -> str:
    """Full normalization for business address with abbreviation standardization."""
    clean = clean_text_basic(address)
    if not clean:
        return ""
    for pattern, replacement in ADDRESS_PATTERNS:
        clean = pattern.sub(replacement, clean)
    return RE_WHITESPACE.sub(" ", clean).strip()


def extract_informative_tokens(name_clean: str, min_len: int = 3) -> str:
    """Extract distinct, non-stopword tokens as space-separated string."""
    tokens = [t for t in name_clean.split() if len(t) >= min_len and t not in NAME_STOPWORDS]
    # Keep unique in order
    seen = set()
    unique_tokens = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            unique_tokens.append(t)
    return " ".join(unique_tokens)


def extract_core_name_prefix(name_clean: str, n_tokens: int = 2) -> str:
    """Extract first n significant tokens as a core blocking stem."""
    tokens = [t for t in name_clean.split() if t not in NAME_STOPWORDS]
    return " ".join(tokens[:n_tokens]) if tokens else name_clean[:10]


def extract_postal_digits(text: str) -> str:
    """Extract all standalone 3-6 digit sequences as space-separated string."""
    if not text:
        return ""
    matches = RE_DIGITS.findall(text)
    return " ".join(matches) if matches else ""


def normalize_record(
    eid: str, bname: str, baddr: str, country: str
) -> Tuple[str, str, str, str, str, str, str, str, str, int]:
    """
    Normalizes a single record and returns all enriched features as a tuple:
    (entity_id, country, bname_raw, baddr_raw, name_clean, addr_clean,
     name_tokens, core_stem, postal_digits, has_address)
    """
    bname_str = str(bname) if bname is not None else ""
    baddr_str = str(baddr) if baddr is not None else ""
    has_address = 1 if baddr_str.strip() else 0

    name_clean = normalize_business_name(bname_str)
    addr_clean = normalize_business_address(baddr_str)
    name_tokens = extract_informative_tokens(name_clean)
    core_stem = extract_core_name_prefix(name_clean)
    postal_digits = extract_postal_digits(baddr_str)

    return (
        eid,
        country,
        bname_str,
        baddr_str,
        name_clean,
        addr_clean,
        name_tokens,
        core_stem,
        postal_digits,
        has_address,
    )
