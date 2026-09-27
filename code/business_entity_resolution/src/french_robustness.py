"""
Step 1.4: French Linguistic Robustness Benchmark Generator (Amazon ML Challenge 2026).
Stress tests normalization, tokenization, and blocking against realistic French legal
suffixes, street morphology, diacritics, and 5-digit postal code patterns.
"""

import sys
import re
import random
from pathlib import Path
from typing import Dict, List, Set, Tuple
import polars as pl

# Ensure package import works
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PROCESSED_DIR, RANDOM_SEED


# French Linguistic Transformation Rules
FRENCH_CORP_SUFFIXES = {
    r"\b(llc|l\.l\.c\.)\b": "sarl",
    r"\b(inc|incorporated|inc\.)\b": "sa",
    r"\b(corp|corporation|corp\.)\b": "sas",
    r"\b(ltd|limited|ltd\.)\b": "societe",
    r"\b(co|company)\b": "cie",
}

FRENCH_STREET_TYPES = {
    r"\b(street|st|st\.)\b": "rue",
    r"\b(avenue|ave|ave\.)\b": "avenue",
    r"\b(boulevard|blvd|blvd\.)\b": "boulevard",
    r"\b(road|rd|rd\.)\b": "route",
    r"\b(lane|ln|ln\.)\b": "allee",
    r"\b(drive|dr|dr\.)\b": "impasse",
    r"\b(way|wy)\b": "chemin",
    r"\b(place|pl)\b": "place",
}

FRENCH_POSTAL_PREFIXES = ["75001", "75008", "69002", "13001", "31000", "44000", "06000", "67000", "33000", "59000"]

DIACRITIC_MAP = {
    "e": ["é", "è", "ê", "ë"],
    "a": ["à", "â"],
    "c": ["ç"],
    "i": ["î", "ï"],
    "o": ["ô"],
    "u": ["ù", "û"],
}


def apply_french_transformations(
    name: str,
    addr: str,
    rng: random.Random,
    diacritic_prob: float = 0.35,
) -> Tuple[str, str, str]:
    """
    Applies French corporate syntax, street morphology, and realistic accent insertions.
    """
    name_fr = name.lower() if name else ""
    addr_fr = addr.lower() if addr else ""

    # 1. Transform Corporate Suffixes
    for pattern, replacement in FRENCH_CORP_SUFFIXES.items():
        if re.search(pattern, name_fr):
            name_fr = re.sub(pattern, replacement, name_fr)
            break

    # 2. Transform Street Types
    for pattern, replacement in FRENCH_STREET_TYPES.items():
        if re.search(pattern, addr_fr):
            addr_fr = re.sub(pattern, replacement, addr_fr)

    # 3. Inject Realistic French Accents
    if rng.random() < diacritic_prob:
        chars = list(name_fr)
        for idx, char in enumerate(chars):
            if char in DIACRITIC_MAP and rng.random() < 0.25:
                chars[idx] = rng.choice(DIACRITIC_MAP[char])
        name_fr = "".join(chars)

    # 4. Generate French 5-digit postal code
    postal_fr = rng.choice(FRENCH_POSTAL_PREFIXES)

    return name_fr, addr_fr, postal_fr


def generate_french_robustness_benchmark(
    num_samples: int = 15000,
    output_path: Path = PROCESSED_DIR / "val_synthetic_france.parquet",
    random_seed: int = RANDOM_SEED,
) -> pl.DataFrame:
    """
    Generates the French Linguistic Robustness Benchmark dataset.
    """
    print("=" * 75)
    print("STEP 1.4: GENERATING FRENCH LINGUISTIC ROBUSTNESS BENCHMARK")
    print("=" * 75)

    rng = random.Random(random_seed)

    # Load a stratified sample from validation
    val_s1 = pl.read_parquet(PROCESSED_DIR / "val_source1_cleaned.parquet")
    sample_df = val_s1.sample(min(num_samples, len(val_s1)), seed=random_seed)

    print(f"Applying French morphology transformations to {len(sample_df):,} entity records...")

    eids = sample_df["entity_id"].to_list()
    names = sample_df["name_clean"].to_list()
    addrs_col = "addr_clean" if "addr_clean" in sample_df.columns else "address_clean"
    addrs = sample_df[addrs_col].to_list()

    transformed_names = []
    transformed_addrs = []
    transformed_tokens = []
    transformed_postals = []

    for name, addr in zip(names, addrs):
        n_fr, a_fr, p_fr = apply_french_transformations(name, addr, rng)
        transformed_names.append(n_fr)
        transformed_addrs.append(a_fr)
        transformed_tokens.append(" ".join(n_fr.split()))
        transformed_postals.append(p_fr)

    fr_df = pl.DataFrame({
        "entity_id": [f"FR_SYN_{eid}" for eid in eids],
        "country": ["France"] * len(eids),
        "name_clean": transformed_names,
        "addr_clean": transformed_addrs,
        "name_tokens": transformed_tokens,
        "postal_digits": transformed_postals,
        "has_address": [1 if len(a) > 0 else 0 for a in transformed_addrs],
        "original_entity_id": eids,
    })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fr_df.write_parquet(output_path, compression="snappy")
    print(f"[SUCCESS] Exported {len(fr_df):,} French benchmark records to: {output_path}")

    return fr_df


if __name__ == "__main__":
    generate_french_robustness_benchmark()
