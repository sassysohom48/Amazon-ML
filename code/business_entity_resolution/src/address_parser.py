"""
Step 2.2: Structured Address Parser & Normalizer (Amazon ML Challenge 2026).
Extracts structured components from unstructured multilingual addresses across US, India, and France:
- addr_clean: normalized address string with standardized abbreviations & diacritic stripping
- postal_clean: canonical 5-digit (US/FR) or 6-digit (IN) postal/PIN code
- addr_unit_num: shop / flat / suite / unit / door number if present
- addr_digits: sequence of standalone numerical digits (building numbers, pin, block)
- addr_tokens: distinctive, non-stopword geographic/landmark/street tokens
"""

import re
import unicodedata
from typing import Dict, List, Tuple, Optional, Set

# Address standardization patterns (Multilingual: US, IN, FR)
ADDRESS_ABBREVIATIONS = [
    # English / US / India Street Types
    (re.compile(r"\b(street|str|st\.)\b", re.IGNORECASE), "st"),
    (re.compile(r"\b(road|rd\.)\b", re.IGNORECASE), "rd"),
    (re.compile(r"\b(avenue|ave\.)\b", re.IGNORECASE), "ave"),
    (re.compile(r"\b(boulevard|blvd\.)\b", re.IGNORECASE), "blvd"),
    (re.compile(r"\b(lane|ln\.)\b", re.IGNORECASE), "ln"),
    (re.compile(r"\b(drive|dr\.)\b", re.IGNORECASE), "dr"),
    (re.compile(r"\b(court|ct\.)\b", re.IGNORECASE), "ct"),
    (re.compile(r"\b(highway|hwy\.)\b", re.IGNORECASE), "hwy"),
    (re.compile(r"\b(building|bldg\.)\b", re.IGNORECASE), "bldg"),
    (re.compile(r"\b(floor|flr\.)\b", re.IGNORECASE), "flr"),

    # French Address Types
    (re.compile(r"\b(rue|r\.)\b", re.IGNORECASE), "rue"),
    (re.compile(r"\b(boulevard|bd\.|bld\.)\b", re.IGNORECASE), "bd"),
    (re.compile(r"\b(avenue|av\.)\b", re.IGNORECASE), "av"),
    (re.compile(r"\b(allee|all\.)\b", re.IGNORECASE), "allee"),
    (re.compile(r"\b(impasse|imp\.)\b", re.IGNORECASE), "imp"),
    (re.compile(r"\b(chemin|che\.)\b", re.IGNORECASE), "chemin"),
    (re.compile(r"\b(place|pl\.)\b", re.IGNORECASE), "place"),
    (re.compile(r"\b(zone\s*industrielle|z\.i\.|zi)\b", re.IGNORECASE), "zi"),
    (re.compile(r"\b(zone\s*d\s*activite|z\.a\.|za)\b", re.IGNORECASE), "za"),

    # Indian Landmark & Locality Indicators
    (re.compile(r"\b(opposite|opp\.)\b", re.IGNORECASE), "opp"),
    (re.compile(r"\b(near|nr\.)\b", re.IGNORECASE), "near"),
    (re.compile(r"\b(behind|bh\.)\b", re.IGNORECASE), "behind"),
    (re.compile(r"\b(beside|adj\.)\b", re.IGNORECASE), "beside"),
    (re.compile(r"\b(nagar|ngr\.)\b", re.IGNORECASE), "nagar"),
    (re.compile(r"\b(colony|clny\.)\b", re.IGNORECASE), "colony"),
    (re.compile(r"\b(marg|mrg\.)\b", re.IGNORECASE), "marg"),
    (re.compile(r"\b(chowk|chwk\.)\b", re.IGNORECASE), "chowk"),
    (re.compile(r"\b(bazaar|bazar)\b", re.IGNORECASE), "bazar"),
]

# Unit / Suite / Shop / Flat patterns
UNIT_PATTERNS = [
    re.compile(r"\b(?:shop\s*no|flat\s*no|plot\s*no|door\s*no|house\s*no|h\.?\s*no|d\.?\s*no|apt\s*no|suite|ste|apt|apartment|flat|shop|unit|door|plot|house|no)\.?\s*[:#\.\-]?\s*([a-z0-9\-/]+)\b", re.IGNORECASE),
    re.compile(r"#\s*([a-z0-9\-/]+)", re.IGNORECASE),
]

# Postal Code Regexes
RE_POSTAL_IN = re.compile(r"\b([1-9][0-9]{5})\b")  # India 6-digit PIN code (110001 - 999999)
RE_POSTAL_FR_US = re.compile(r"\b([0-9]{5})(?:-[0-9]{4})?\b")  # US 5-digit ZIP or FR 5-digit code
RE_ALL_DIGIT_BLOCKS = re.compile(r"\b\d{1,8}\b")

# Geographic / Address stopwords for distinctive token extraction
ADDR_STOPWORDS = {
    "the", "a", "an", "and", "of", "in", "for", "on", "at", "by", "to", "with",
    "st", "rd", "ave", "blvd", "ln", "dr", "ct", "hwy", "bldg", "flr", "apt", "ste",
    "street", "road", "avenue", "lane", "drive", "court", "highway", "building", "floor",
    "rue", "bd", "av", "allee", "imp", "chemin", "place", "zi", "za", "cedex",
    "opp", "near", "behind", "beside", "nagar", "colony", "marg", "chowk", "bazar",
    "india", "usa", "us", "france", "box", "po", "pobox"
}

RE_NON_ALPHANUM = re.compile(r"[^a-z0-9\s]")
RE_WHITESPACE = re.compile(r"\s+")


def strip_accents_nfkd(text: str) -> str:
    """Safely decomposes Unicode accents (NFKD) and lowercases characters."""
    if not text:
        return ""
    normalized = unicodedata.normalize("NFKD", str(text))
    return "".join(c for c in normalized if not unicodedata.combining(c)).lower()


def clean_address_basic(raw_address: str) -> str:
    """
    Standard address cleanup: NFKD normalized, abbreviations standardized,
    punctuation to spaces, whitespace collapsed.
    """
    if not raw_address or not str(raw_address).strip():
        return ""
    
    text = str(raw_address).replace("&", " and ").replace("@", " at ").replace("#", " # ")
    text = strip_accents_nfkd(text)
    
    for pattern, replacement in ADDRESS_ABBREVIATIONS:
        text = pattern.sub(f" {replacement} ", text)
        
    text = RE_NON_ALPHANUM.sub(" ", text)
    return RE_WHITESPACE.sub(" ", text).strip()


def extract_postal_code(raw_address: str, country: str = "") -> str:
    """
    Extracts the canonical postal code based on country or regex pattern matching.
    India -> 6 digits, US/France -> 5 digits.
    """
    if not raw_address:
        return ""
    
    text = str(raw_address)
    country_upper = str(country).upper()

    if country_upper in ("INDIA", "IN"):
        m = RE_POSTAL_IN.search(text)
        if m:
            return m.group(1)
    elif country_upper in ("FRANCE", "FR", "UNITED STATES", "US", "USA"):
        m = RE_POSTAL_FR_US.search(text)
        if m:
            return m.group(1)
            
    # Country-agnostic fallback
    m_in = RE_POSTAL_IN.search(text)
    if m_in:
        return m_in.group(1)
    m_5 = RE_POSTAL_FR_US.search(text)
    if m_5:
        return m_5.group(1)
        
    return ""


def extract_unit_number(raw_address: str) -> str:
    """Extracts apartment / suite / flat / shop number if present."""
    if not raw_address:
        return ""
    
    for pattern in UNIT_PATTERNS:
        match = pattern.search(raw_address)
        if match:
            unit = match.group(1).lower().strip()
            # Clean non-alphanumeric except hyphen
            unit = re.sub(r"[^a-z0-9\-]", "", unit)
            if unit:
                return unit
    return ""


def extract_all_digits(raw_address: str) -> str:
    """Extracts all standalone digit sequences as space-separated string."""
    if not raw_address:
        return ""
    matches = RE_ALL_DIGIT_BLOCKS.findall(str(raw_address))
    return " ".join(matches) if matches else ""


def extract_informative_address_tokens(addr_clean: str, min_len: int = 3) -> str:
    """Extracts distinctive, non-stopword geographic/street tokens."""
    tokens = [t for t in addr_clean.split() if len(t) >= min_len and t not in ADDR_STOPWORDS]
    seen = set()
    unique = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    return " ".join(unique)


def parse_address_record(raw_address: str, country: str = "") -> Dict[str, any]:
    """
    Main entry point for structured address parsing.
    Returns:
      addr_clean: normalized full address string
      postal_clean: canonical postal code
      addr_unit_num: shop/flat/suite identifier
      addr_digits: sequence of all digit blocks
      addr_tokens: distinctive locality/street tokens
      has_address: 1 if address is non-empty, 0 otherwise
    """
    raw_str = str(raw_address).strip() if raw_address is not None else ""
    if not raw_str or raw_str.lower() in ("nan", "none", "null"):
        return {
            "addr_clean": "",
            "postal_clean": "",
            "addr_unit_num": "",
            "addr_digits": "",
            "addr_tokens": "",
            "has_address": 0,
        }

    addr_clean = clean_address_basic(raw_str)
    postal_clean = extract_postal_code(raw_str, country)
    addr_unit_num = extract_unit_number(raw_str)
    addr_digits = extract_all_digits(raw_str)
    addr_tokens = extract_informative_address_tokens(addr_clean)

    return {
        "addr_clean": addr_clean,
        "postal_clean": postal_clean,
        "addr_unit_num": addr_unit_num,
        "addr_digits": addr_digits,
        "addr_tokens": addr_tokens,
        "has_address": 1,
    }


if __name__ == "__main__":
    test_cases = [
        ("123 Main Street, Suite 400, New York, NY 10001", "US"),
        ("Shop No 14, Ground Floor, MG Road, Indiranagar, Bangalore 560038", "India"),
        ("15 Rue de Rivoli, 75004 Paris, France", "France"),
        ("Plot 42-B, Sector 18, Electronic City, Bengaluru, Karnataka 560100", "IN"),
        ("None", "US"),
    ]
    print("Structured Address Parser Test Cases:")
    for addr, cty in test_cases:
        res = parse_address_record(addr, cty)
        print(f"\nRaw: {addr} [{cty}]")
        for k, v in res.items():
            print(f"  {k:<15}: {v}")
