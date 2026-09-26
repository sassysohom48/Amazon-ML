"""
Step 3.1: Country-Aware Inverted Document Frequency (IDF) Computer.
Computes mathematically rigorous, country-partitioned IDF dictionaries for name and address tokens:
IDF(t) = ln((N + 1) / (df(t) + 1)) + 1.0
"""

import math
from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple
import polars as pl


class CountryIDFComputer:
    """
    Computes and caches country-specific token IDF scores across target pools (S2 + S3).
    Ensures rare local tokens (e.g. 'andheri' in India, 'rivoli' in France) receive high discriminating weight.
    """

    def __init__(self, min_token_len: int = 3, max_df_fraction: float = 0.25):
        self.min_token_len = min_token_len
        self.max_df_fraction = max_df_fraction
        
        # country -> token -> idf_score
        self.name_idf: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.addr_idf: Dict[str, Dict[str, float]] = defaultdict(dict)
        
        # country -> total_documents
        self.country_doc_counts: Dict[str, int] = defaultdict(int)

    def fit_from_dataframe(self, target_df: pl.DataFrame):
        """
        Computes country-specific name and address IDF dictionaries from S2 + S3 target pool.
        """
        countries = target_df["country"].unique().to_list()

        for country in countries:
            c_name = str(country) if country is not None else "Unknown"
            c_df = target_df.filter(pl.col("country") == country)
            n_docs = len(c_df)
            self.country_doc_counts[c_name] = n_docs

            # 1. Name Token Document Frequencies
            name_tokens_list = c_df["name_tokens"].to_list()
            name_df_counts = Counter()
            for tok_str in name_tokens_list:
                if tok_str:
                    for t in set(tok_str.split()):
                        if len(t) >= self.min_token_len:
                            name_df_counts[t] += 1

            max_allowed_df = int(n_docs * self.max_df_fraction)
            for t, df_val in name_df_counts.items():
                if df_val <= max_allowed_df:
                    idf = math.log((n_docs + 1.0) / (df_val + 1.0)) + 1.0
                    self.name_idf[c_name][t] = idf

            # 2. Address Token Document Frequencies
            addr_tokens_list = c_df["addr_tokens"].to_list()
            addr_df_counts = Counter()
            for addr_str in addr_tokens_list:
                if addr_str:
                    for at in set(addr_str.split()):
                        if len(at) >= self.min_token_len:
                            addr_df_counts[at] += 1

            for at, df_val in addr_df_counts.items():
                if df_val <= max_allowed_df:
                    idf = math.log((n_docs + 1.0) / (df_val + 1.0)) + 1.0
                    self.addr_idf[c_name][at] = idf

    def get_name_token_idf(self, country: str, token: str) -> float:
        return self.name_idf.get(country, {}).get(token, 1.0)

    def get_addr_token_idf(self, country: str, token: str) -> float:
        return self.addr_idf.get(country, {}).get(token, 1.0)

    def get_rarest_name_tokens(self, country: str, tokens: List[str], top_n: int = 3) -> List[Tuple[str, float]]:
        """Returns top_n rarest name tokens sorted by descending IDF."""
        scored = [(t, self.get_name_token_idf(country, t)) for t in tokens if len(t) >= self.min_token_len]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_n]

    def get_rarest_addr_tokens(self, country: str, tokens: List[str], top_n: int = 3) -> List[Tuple[str, float]]:
        """Returns top_n rarest address tokens sorted by descending IDF."""
        scored = [(t, self.get_addr_token_idf(country, t)) for t in tokens if len(t) >= self.min_token_len]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_n]
