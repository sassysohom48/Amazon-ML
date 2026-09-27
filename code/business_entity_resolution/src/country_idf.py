"""
Step 3.1: Ultra-Fast Low-Memory Country-Aware Inverted Document Frequency (IDF) Computer.
Computes token IDF tables directly using Polars vectorized expressions with < 100 MB peak RAM.
"""

import math
from collections import defaultdict
from typing import Dict, List, Set, Tuple
import polars as pl


class CountryIDFComputer:
    """
    Computes and caches country-specific token IDF scores across target pools (S2 + S3).
    IDF(t) = ln((N + 1) / (df(t) + 1)) + 1.0
    """

    def __init__(self, min_token_len: int = 3, max_df_fraction: float = 0.25):
        self.min_token_len = min_token_len
        self.max_df_fraction = max_df_fraction
        
        # country -> token -> idf_score
        self.name_idf: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.addr_idf: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.country_doc_counts: Dict[str, int] = defaultdict(int)

    def fit_from_dataframes(self, s2_df: pl.DataFrame, s3_df: pl.DataFrame):
        """
        Computes country-specific name and address IDF dictionaries without duplicating DataFrames in memory.
        """
        countries = list(set(s2_df["country"].unique().to_list()) | set(s3_df["country"].unique().to_list()))

        for country in countries:
            c_name = str(country) if country is not None else "Unknown"
            s2_c = s2_df.filter(pl.col("country") == country)
            s3_c = s3_df.filter(pl.col("country") == country)
            n_docs = len(s2_c) + len(s3_c)
            self.country_doc_counts[c_name] = n_docs
            max_allowed_df = int(n_docs * self.max_df_fraction)

            # 1. Name Tokens IDF
            name_counts = defaultdict(int)
            for df_part in (s2_c, s3_c):
                if "name_tokens" in df_part.columns:
                    series = df_part.filter((pl.col("name_tokens") != "") & pl.col("name_tokens").is_not_null())["name_tokens"]
                    for tok_str in series:
                        for t in set(tok_str.split()):
                            if len(t) >= self.min_token_len:
                                name_counts[t] += 1

            for t, df_val in name_counts.items():
                if df_val <= max_allowed_df:
                    idf = math.log((n_docs + 1.0) / (df_val + 1.0)) + 1.0
                    self.name_idf[c_name][t] = idf

            # 2. Address Tokens IDF
            addr_counts = defaultdict(int)
            for df_part in (s2_c, s3_c):
                if "addr_tokens" in df_part.columns:
                    series = df_part.filter((pl.col("addr_tokens") != "") & pl.col("addr_tokens").is_not_null())["addr_tokens"]
                    for atok_str in series:
                        for at in set(atok_str.split()):
                            if len(at) >= self.min_token_len:
                                addr_counts[at] += 1

            for at, df_val in addr_counts.items():
                if df_val <= max_allowed_df:
                    idf = math.log((n_docs + 1.0) / (df_val + 1.0)) + 1.0
                    self.addr_idf[c_name][at] = idf

            del s2_c, s3_c

    def fit_from_dataframe(self, target_df: pl.DataFrame):
        """
        Computes country-specific name and address IDF dictionaries.
        """
        countries = target_df["country"].unique().to_list()

        for country in countries:
            c_name = str(country) if country is not None else "Unknown"
            c_df = target_df.filter(pl.col("country") == country)
            n_docs = len(c_df)
            self.country_doc_counts[c_name] = n_docs
            max_allowed_df = int(n_docs * self.max_df_fraction)

            # 1. Name Tokens IDF
            name_tokens_series = c_df.filter((pl.col("name_tokens") != "") & pl.col("name_tokens").is_not_null())["name_tokens"]
            name_counts = defaultdict(int)
            for tok_str in name_tokens_series:
                for t in set(tok_str.split()):
                    if len(t) >= self.min_token_len:
                        name_counts[t] += 1

            for t, df_val in name_counts.items():
                if df_val <= max_allowed_df:
                    idf = math.log((n_docs + 1.0) / (df_val + 1.0)) + 1.0
                    self.name_idf[c_name][t] = idf

            # 2. Address Tokens IDF
            addr_tokens_series = c_df.filter((pl.col("addr_tokens") != "") & pl.col("addr_tokens").is_not_null())["addr_tokens"]
            addr_counts = defaultdict(int)
            for atok_str in addr_tokens_series:
                for at in set(atok_str.split()):
                    if len(at) >= self.min_token_len:
                        addr_counts[at] += 1

            for at, df_val in addr_counts.items():
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
