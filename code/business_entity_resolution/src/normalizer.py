"""
Vectorized Multilingual Normalization & Representation Module (Amazon ML Challenge 2026).
High-speed Unicode normalization, diacritic stripping, sorted token keys,
phonetic Soundex keys, stemmed tokens, character 3-grams, and postal/address anchors.
"""

import re
import unicodedata
from typing import List, Set, Tuple, Optional, Dict
import polars as pl

# Statistical non-distinctive words across corporate business names (US, India, France)
CORP_STOPWORDS: Set[str] = {
    "the", "a", "an", "and", "of", "in", "for", "on", "at", "by", "to", "with",
    "inc", "incorporated", "corp", "corporation", "co", "company", "llc", "ltd",
    "limited", "pvt", "private", "sa", "sarl", "sas", "sasu", "eurl", "sci", "snc", "selarl",
    "ets", "etablissement", "etablissements", "ste", "societe", "cie", "compagnie",
    "enterprise", "enterprises", "service", "services", "center", "centre", "group",
    "international", "global", "solutions", "holdings", "management",
    "cabinet", "atelier", "association", "boutique", "magasin",
    "india", "usa", "us", "france", "fr", "paris"
}

# Generic address descriptors to filter when creating anchors (US, India, France)
ADDR_STOPWORDS: Set[str] = {
    "st", "street", "rd", "road", "ave", "avenue", "blvd", "boulevard", "dr", "drive",
    "ln", "lane", "ct", "court", "unit", "apt", "apartment", "ste", "suite", "flr", "floor",
    "bldg", "building", "near", "opp", "opposite", "behind", "door", "flat", "no", "h",
    # French address keywords to prevent generic anchor collisions (e.g. 12_rue)
    "rue", "route", "impasse", "allee", "place", "voie", "chemin", "passage",
    "cours", "quai", "residence", "res", "bat", "batiment", "cedex",
    "arrondissement", "boite", "postale", "bp"
}

# Corporate suffix stem mappings to harmonize variations (e.g. advisory -> advis)
STEM_SUFFIXES = [
    ("industries", "industr"), ("industry", "industr"), ("industrial", "industr"),
    ("enterprises", "enterpris"), ("enterprise", "enterpris"),
    ("technologies", "technolog"), ("technology", "technolog"),
    ("solutions", "solut"), ("solution", "solut"),
    ("services", "servic"), ("service", "servic"),
    ("consulting", "consult"), ("consultants", "consult"), ("consultant", "consult"),
    ("advisory", "advis"), ("advisors", "advis"), ("advisor", "advis"),
    ("logistics", "logist"), ("logistic", "logist"),
    ("holdings", "hold"), ("holding", "hold"),
    ("properties", "propert"), ("property", "propert"),
    ("bakeries", "baker"), ("bakery", "baker"),
    ("jewellers", "jewel"), ("jewelers", "jewel"), ("jewellery", "jewel"), ("jewelry", "jewel"),
    ("etablissements", "etabliss"), ("etablissement", "etabliss"),
    ("societes", "societ"), ("societe", "societ")
]

# Fast Soundex mapping table
SOUNDEX_MAP: Dict[str, str] = {
    "b": "1", "f": "1", "p": "1", "v": "1",
    "c": "2", "g": "2", "j": "2", "k": "2", "q": "2", "s": "2", "x": "2", "z": "2",
    "d": "3", "t": "3",
    "l": "4",
    "m": "5", "n": "5",
    "r": "6"
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


def stem_token(token: str) -> str:
    """Harmonizes common morphological and corporate suffix variants."""
    for full_suffix, stem in STEM_SUFFIXES:
        if token == full_suffix or token.endswith(full_suffix):
            return stem
    return token


def extract_soundex(word: str) -> str:
    """
    Computes standard Soundex code (e.g. 'Orelee' -> 'O640', 'Orlees' -> 'O642').
    Captures phonetic typos and transliteration noise.
    """
    if not word or not word[0].isalpha():
        return ""
    word = word.lower()
    code = [word[0].upper()]
    prev = SOUNDEX_MAP.get(word[0], "")
    for char in word[1:]:
        digit = SOUNDEX_MAP.get(char, "")
        if digit:
            if digit != prev:
                code.append(digit)
                if len(code) == 4:
                    break
            prev = digit
        else:
            prev = ""
    return "".join(code).ljust(4, "0")[:4]


def extract_informative_tokens(text_clean: str, min_len: int = 2) -> List[str]:
    """
    Extracts distinctive tokens for inverted index blocking.
    Includes both original distinctive tokens and their stemmed variants.
    """
    if not text_clean:
        return []
    tokens = text_clean.split()
    informative = [t for t in tokens if len(t) >= min_len and t not in CORP_STOPWORDS]
    if not informative:
        # Fallback for short names consisting solely of stopwords
        informative = [t for t in tokens if len(t) >= min_len]
    
    # Add stemmed variants if they differ from original
    augmented = list(informative)
    for tok in informative:
        st = stem_token(tok)
        if st != tok and len(st) >= min_len:
            augmented.append(st)
    return augmented


def extract_sorted_token_keys(text_clean: str) -> List[str]:
    """
    Extracts canonical alphabetically sorted token keys.
    Returns:
    - Primary key: sorted combination of all significant tokens (up to 4)
    - Secondary prefix key: sorted combination of first 2 significant tokens
    Solves word-order inversion across noisy sources.
    """
    if not text_clean:
        return []
    tokens = [t for t in text_clean.split() if len(t) >= 2 and t not in CORP_STOPWORDS]
    if not tokens:
        tokens = [t for t in text_clean.split() if len(t) >= 2]
    if not tokens:
        return [text_clean[:12]] if text_clean else []

    sorted_all = sorted(set(tokens))
    keys = []

    # Primary key: up to 4 distinctive sorted tokens
    primary_key = "_".join(sorted_all[:4])
    if primary_key:
        keys.append(primary_key)

    # Secondary prefix key: first 2 sorted tokens
    if len(sorted_all) >= 2:
        prefix_key = "_".join(sorted_all[:2])
        if prefix_key != primary_key:
            keys.append(prefix_key)

    return keys


def extract_phonetic_keys(text_clean: str) -> List[str]:
    """
    Extracts phonetic Soundex keys across the most significant tokens.
    Handles severe spelling variations (e.g. 'Orelee' vs 'Orlees').
    """
    if not text_clean:
        return []
    tokens = [t for t in text_clean.split() if len(t) >= 2 and t not in CORP_STOPWORDS]
    if not tokens:
        tokens = text_clean.split()
    if not tokens:
        return []

    # Take Soundex of primary significant tokens
    phonetics = [extract_soundex(t) for t in tokens[:2]]
    valid = [p for p in phonetics if p]
    if not valid:
        return []
    
    # Combined 2-token phonetic signature e.g. "ph_O642_B260"
    return [f"ph_{'_'.join(valid)}"]


def extract_sorted_token_key(text_clean: str, max_tokens: int = 3) -> str:
    """Backward-compatible single sorted key extractor."""
    keys = extract_sorted_token_keys(text_clean)
    return keys[0] if keys else ""


def extract_char_3grams(text_clean: str) -> Set[str]:
    """
    Generates boundary-padded character 3-gram shingles for typo-tolerant fuzzy matching.
    Boundary markers ('^' and '$') ensure prefix and suffix alignment.
    """
    if not text_clean:
        return set()
    compact = f"^{text_clean.replace(' ', '')}$"
    if len(compact) < 3:
        return {compact} if compact else set()
    return {compact[i:i + 3] for i in range(len(compact) - 3 + 1)}


def extract_char_4grams(text_clean: str) -> Set[str]:
    """Retained for backward compatibility: generates character 4-gram shingles."""
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


def extract_postal_codes(addr_clean: str) -> List[str]:
    """Extracts standalone 5-6 digit postal codes (ZIP / PIN / Code Postal)."""
    if not addr_clean:
        return []
    tokens = addr_clean.split()
    return [f"pin_{tok}" for tok in tokens if tok.isdigit() and len(tok) in (5, 6)]


def extract_address_anchors(addr_clean: str) -> List[str]:
    """
    Extracts multi-token address anchors:
    - (street_number + locality_word) e.g. '1795_westchester', '797_lake'
    - (locality_word + street_number)
    """
    if not addr_clean:
        return []
    tokens = addr_clean.split()
    anchors = []

    # Extract digit + adjacent informative word anchors
    for i, tok in enumerate(tokens):
        if tok.isdigit() and len(tok) >= 1:
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
