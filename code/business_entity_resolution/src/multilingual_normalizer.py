"""
Step 2.1: Multi-Representation Multilingual Name Normalizer (Amazon ML Challenge 2026).
Produces parallel representations (name_clean, name_core, legal_form, name_acronym,
name_phonetic, name_tokens) without destructive in-place data loss.
"""

import re
import unicodedata
from typing import Dict, List, Tuple, Optional, Set

# Legal Form Categories & Extraction Regexes
LEGAL_FORM_PATTERNS = [
    # Private Limited (India / UK / Commonwealth)
    (re.compile(r"\b(pvt\s*ltd|private\s*limited|p\s*ltd|pvt\s*limited)\b", re.IGNORECASE), "PVT_LTD"),
    (re.compile(r"\b(public\s*limited|pub\s*ltd)\b", re.IGNORECASE), "PUB_LTD"),
    (re.compile(r"\b(limited|ltd)\b\.?", re.IGNORECASE), "LTD"),
    
    # LLC / Limited Liability Company (US)
    (re.compile(r"\b(limited\s*liability\s*company|l\.l\.c\.|llc)\b", re.IGNORECASE), "LLC"),
    
    # Corporation / Incorporated (US / Global)
    (re.compile(r"\b(incorporated|inc\.|inc)\b", re.IGNORECASE), "INC"),
    (re.compile(r"\b(corporation|corp\.|corp)\b", re.IGNORECASE), "CORP"),
    
    # French Legal Forms (France / European)
    (re.compile(r"\b(societe\s*anonyme|s\.a\.|sa)\b", re.IGNORECASE), "SA"),
    (re.compile(r"\b(societe\s*a\s*responsabilite\s*limitee|s\.a\.r\.l\.|sarl)\b", re.IGNORECASE), "SARL"),
    (re.compile(r"\b(societe\s*par\s*actions\s*simplifiee|s\.a\.s\.|sas)\b", re.IGNORECASE), "SAS"),
    (re.compile(r"\b(entreprise\s*unipersonnelle|eurl)\b", re.IGNORECASE), "EURL"),
    (re.compile(r"\b(societe\s*civile\s*immobiliere|sci)\b", re.IGNORECASE), "SCI"),
    (re.compile(r"\b(societe|soc\.|soc)\b", re.IGNORECASE), "SOCIETE"),

    # Partnerships & Others
    (re.compile(r"\b(limited\s*liability\s*partnership|llp)\b", re.IGNORECASE), "LLP"),
    (re.compile(r"\b(and\s*company|and\s*co|&\s*co|&\s*company)\b", re.IGNORECASE), "PARTNERSHIP"),
    (re.compile(r"\b(and\s*sons|&\s*sons|and\s*brothers|&\s*bros|bros)\b", re.IGNORECASE), "FAMILY_BUSINESS"),
]

# Stopwords for informative token extraction
NAME_STOPWORDS = {
    "the", "a", "an", "and", "of", "in", "for", "on", "at", "by", "to", "with",
    "pvt", "ltd", "llc", "inc", "corp", "co", "sa", "sarl", "sas", "llp", "eurl", "sci",
    "private", "limited", "company", "corporation", "incorporated", "societe",
    "services", "service", "enterprises", "enterprise", "solutions", "international",
    "global", "group", "holdings", "management", "india", "usa", "us", "france"
}

RE_NON_ALPHANUM = re.compile(r"[^a-z0-9\s]")
RE_WHITESPACE = re.compile(r"\s+")


def strip_accents_nfkd(text: str) -> str:
    """Safely decomposes Unicode accents (NFKD) and lowercases characters."""
    if not text:
        return ""
    normalized = unicodedata.normalize("NFKD", str(text))
    return "".join(c for c in normalized if not unicodedata.combining(c)).lower()


def clean_name_basic(raw_name: str) -> str:
    """
    Standard clean name: NFKD normalized, punctuation to spaces, whitespace collapsed.
    Preserves all original words and tokens safely without deletions.
    """
    if not raw_name or not str(raw_name).strip():
        return ""
    text = str(raw_name).replace("&", " and ").replace("@", " at ")
    text = strip_accents_nfkd(text)
    text = RE_NON_ALPHANUM.sub(" ", text)
    return RE_WHITESPACE.sub(" ", text).strip()


def extract_legal_form(name_clean: str) -> Tuple[str, str]:
    """
    Extracts the categorical legal form tag and returns (core_name, legal_form_tag).
    Strips trailing and leading corporate suffixes without destroying the brand stem.
    """
    if not name_clean:
        return "", "NONE"

    detected_form = "NONE"
    core_name = name_clean

    for pattern, tag in LEGAL_FORM_PATTERNS:
        match = pattern.search(core_name)
        if match:
            detected_form = tag
            # Remove legal suffix from core name
            core_name = pattern.sub(" ", core_name)
            break

    core_name = RE_WHITESPACE.sub(" ", core_name).strip()
    return core_name if core_name else name_clean, detected_form


def extract_acronym(name_clean: str) -> str:
    """
    Extracts first-letter acronym if business name has 2 or more distinct words.
    Example: 'Tata Consultancy Services' -> 'tcs', 'State Bank of India' -> 'sbi'.
    """
    words = [w for w in name_clean.split() if w not in {"and", "of", "the", "in", "for", "at", "to", "by", "a", "an"}]
    if len(words) >= 2:
        return "".join(w[0] for w in words if w and w[0].isalnum())
    return ""


def compute_double_metaphone(name: str) -> str:
    """
    Computes simplified Double Metaphone / Soundex representation for phonetic matching.
    """
    if not name:
        return ""
    clean_alpha = re.sub(r"[^A-Z]", "", name.upper())
    if not clean_alpha:
        return ""
    mapping = {
        "B": "1", "F": "1", "P": "1", "V": "1",
        "C": "2", "G": "2", "J": "2", "K": "2", "Q": "2", "S": "2", "X": "2", "Z": "2",
        "D": "3", "T": "3",
        "L": "4",
        "M": "5", "N": "5",
        "R": "6",
    }
    first_char = clean_alpha[0]
    encoded = [first_char]
    for char in clean_alpha[1:]:
        code = mapping.get(char, "0")
        if code != "0" and code != encoded[-1]:
            encoded.append(code)
    return (("".join(encoded)) + "0000")[:4]


def extract_informative_name_tokens(name_clean: str, min_len: int = 3) -> str:
    """Extracts distinctive, non-stopword tokens as space-separated string."""
    tokens = [t for t in name_clean.split() if len(t) >= min_len and t not in NAME_STOPWORDS]
    seen = set()
    unique = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    return " ".join(unique)


def normalize_business_name_record(raw_name: str, country: str = "") -> Dict[str, str]:
    """
    Main entry point for multi-representation name normalization.
    Returns all 6 parallel representations for a single business entity.
    """
    name_clean = clean_name_basic(raw_name)
    name_core, legal_form = extract_legal_form(name_clean)
    name_tokens = extract_informative_name_tokens(name_clean)
    name_acronym = extract_acronym(name_clean)
    name_phonetic = compute_double_metaphone(name_core if name_core else name_clean)

    return {
        "name_clean": name_clean,
        "name_core": name_core,
        "legal_form": legal_form,
        "name_tokens": name_tokens,
        "name_acronym": name_acronym,
        "name_phonetic": name_phonetic,
    }


if __name__ == "__main__":
    test_cases = [
        "Tata Consultancy Services Pvt. Ltd.",
        "L.L.Bean Inc.",
        "Societe Generale S.A.R.L.",
        "Laxmi Jewellers & Sons",
        "Lakshmi Jewellers Private Limited",
        "State Bank of India",
        "Cafe de la Paix",
    ]
    print("Multi-Representation Name Normalizer Test Cases:")
    for tc in test_cases:
        res = normalize_business_name_record(tc)
        print(f"\nRaw: {tc}")
        for k, v in res.items():
            print(f"  {k:<15}: {v}")
