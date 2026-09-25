"""
Multi-Core Parallel Inverted Index Blocker (Amazon ML Challenge 2026).
Uses 8-12 CPU worker processes with shared read-only index to block 88k entities in < 25s.
"""

import os
import sys
import time
from collections import defaultdict
from functools import partial
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Set, Tuple
import polars as pl
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import PROCESSED_DIR
from src.evaluator import evaluate_blocking_recall

# Global index references for child processes (zero-copy memory sharing)
G_EXACT_NAME = {}
G_TOKEN_IDX = {}
G_TOKEN_WEIGHTS = {}
G_STEM_IDX = {}
G_POSTAL_IDX = {}
G_TARGET_IDS = []
G_TARGET_TOKENS = []


def init_worker(exact_name, token_idx, token_weights, stem_idx, postal_idx, target_ids, target_tokens):
    global G_EXACT_NAME, G_TOKEN_IDX, G_TOKEN_WEIGHTS, G_STEM_IDX, G_POSTAL_IDX, G_TARGET_IDS, G_TARGET_TOKENS
    G_EXACT_NAME = exact_name
    G_TOKEN_IDX = token_idx
    G_TOKEN_WEIGHTS = token_weights
    G_STEM_IDX = stem_idx
    G_POSTAL_IDX = postal_idx
    G_TARGET_IDS = target_ids
    G_TARGET_TOKENS = target_tokens


def score_single_entity(args: Tuple[str, str, str, str, str, int]) -> Tuple[str, List[str]]:
    eid, name_clean, tokens_str, stem_str, postal_str, max_cand = args

    s1_tokens = set(tokens_str.split()) if tokens_str else set()
    s1_postals = set(postal_str.split()) if postal_str else set()

    candidate_scores = defaultdict(float)

    # 1. Exact Name Match (+100.0)
    if name_clean in G_EXACT_NAME:
        for idx in G_EXACT_NAME[name_clean]:
            candidate_scores[idx] += 100.0

    # 2. Informative Token Inverted Index (IDF Weighted)
    for token in s1_tokens:
        if token in G_TOKEN_IDX:
            w = G_TOKEN_WEIGHTS[token]
            for idx in G_TOKEN_IDX[token]:
                candidate_scores[idx] += w

    # 3. Core Stem Prefix Match (+15.0)
    if stem_str and stem_str in G_STEM_IDX:
        for idx in G_STEM_IDX[stem_str]:
            candidate_scores[idx] += 15.0

    # 4. Postal Code match with at least 1 shared token (+20.0)
    for postal in s1_postals:
        if postal in G_POSTAL_IDX:
            for idx in G_POSTAL_IDX[postal]:
                if s1_tokens & G_TARGET_TOKENS[idx]:
                    candidate_scores[idx] += 20.0

    if not candidate_scores:
        return eid, []

    # Sort and pick top K
    sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)[:max_cand]
    return eid, [G_TARGET_IDS[idx] for idx, _ in sorted_candidates]


def benchmark_parallel_blocking(max_candidates: int = 30, max_token_freq: int = 25000):
    n_cores = min(cpu_count(), 10)
    print(f"Running Parallel Blocking Benchmark on India with {n_cores} CPU cores (K={max_candidates}, max_freq={max_token_freq})...")

    # Load data
    val_s1 = pl.read_parquet(PROCESSED_DIR / "val_source1_cleaned.parquet").filter(pl.col("country") == "India")
    train_s2 = pl.read_parquet(PROCESSED_DIR / "train_source2_cleaned.parquet").filter(pl.col("country") == "India")
    train_s3 = pl.read_parquet(PROCESSED_DIR / "train_source3_cleaned.parquet").filter(pl.col("country") == "India")
    val_gt = pl.read_parquet(PROCESSED_DIR / "val_ground_truth.parquet")

    india_s1_ids = set(val_s1["entity_id"].to_list())
    val_gt_map = {}
    for row in val_gt.iter_rows():
        s1_id, matches = row[0], row[1]
        if s1_id in india_s1_ids and matches and str(matches).strip():
            val_gt_map[s1_id] = set(str(matches).strip().split(","))

    targets = pl.concat([train_s2, train_s3])
    t_eids = targets["entity_id"].to_list()
    t_names = targets["name_clean"].to_list()
    t_tokens = targets["name_tokens"].to_list()
    t_stems = targets["core_stem"].to_list()
    t_postals = targets["postal_digits"].to_list()

    print(f"Building master inverted index over {len(targets):,} target records...")
    t0 = time.time()

    exact_name_idx = defaultdict(list)
    raw_token_idx = defaultdict(list)
    stem_idx = defaultdict(list)
    postal_idx = defaultdict(list)
    target_tokens_set = []

    for idx, (name, tok_str, stem, post_str) in enumerate(zip(t_names, t_tokens, t_stems, t_postals)):
        if name:
            exact_name_idx[name].append(idx)

        tok_set = set(tok_str.split()) if tok_str else set()
        target_tokens_set.append(tok_set)

        for t in tok_set:
            raw_token_idx[t].append(idx)

        if stem and len(stem) >= 3:
            stem_idx[stem].append(idx)

        postals = post_str.split() if post_str else []
        for p in postals:
            if len(p) >= 4:
                postal_idx[p].append(idx)

    # Filter tokens by max_token_freq and compute IDF weights
    token_idx = {}
    token_weights = {}
    for t, postings in raw_token_idx.items():
        if len(postings) <= max_token_freq:
            token_idx[t] = postings
            token_weights[t] = 10.0 / (len(postings) ** 0.3)

    print(f"Index build completed in {time.time() - t0:.2f}s:")
    print(f"  • Exact Names:     {len(exact_name_idx):,}")
    print(f"  • Indexed Tokens:  {len(token_idx):,} (filtered at {max_token_freq:,})")
    print(f"  • Stems:           {len(stem_idx):,}")
    print(f"  • Postal Codes:    {len(postal_idx):,}")

    # Prepare S1 query items
    s1_eids = val_s1["entity_id"].to_list()
    s1_names = val_s1["name_clean"].to_list()
    s1_tokens = val_s1["name_tokens"].to_list()
    s1_stems = val_s1["core_stem"].to_list()
    s1_postals = val_s1["postal_digits"].to_list()

    query_items = [
        (eid, name, tok, stem, post, max_candidates)
        for eid, name, tok, stem, post in zip(s1_eids, s1_names, s1_tokens, s1_stems, s1_postals)
    ]

    print(f"\nLaunching parallel querying across {n_cores} worker processes...")
    t1 = time.time()

    with Pool(
        processes=n_cores,
        initializer=init_worker,
        initargs=(exact_name_idx, token_idx, token_weights, stem_idx, postal_idx, t_eids, target_tokens_set),
    ) as pool:
        results = pool.map(score_single_entity, query_items, chunksize=1000)

    elapsed = time.time() - t1
    print(f"Parallel candidate blocking completed in {elapsed:.2f}s ({len(s1_eids)/elapsed:,.0f} entities/s)!")

    cand_map = {eid: set(cands) for eid, cands in results}
    metrics = evaluate_blocking_recall(val_gt_map, cand_map)

    print("\n" + "=" * 70)
    print(f"PARALLEL BLOCKING EVALUATION (India Validation, K={max_candidates}):")
    print(f"  • Candidate Recall:        {metrics['candidate_recall']*100:.2f}%")
    print(f"  • Captured True Matches:   {metrics['captured_true_pairs']:,} / {metrics['total_true_pairs']:,}")
    print(f"  • Missed Matches:          {metrics['total_true_pairs'] - metrics['captured_true_pairs']:,}")
    print(f"  • Avg Candidates per S1:   {metrics['avg_candidates_per_s1']:.2f}")
    print("=" * 70)


if __name__ == "__main__":
    benchmark_parallel_blocking(max_candidates=30, max_token_freq=25000)
