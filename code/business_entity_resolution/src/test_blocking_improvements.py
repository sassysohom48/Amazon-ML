"""
Test Blocker Optimization:
1. Increase max_token_freq or use IDF ranking rather than hard cutoff.
2. Index 2-token prefixes / bigrams.
3. Index address tokens / city / postal codes.
4. Increase candidate cap from 15 to 25 (or keep all exact matches + top tokens).
"""

import time
from collections import defaultdict
from typing import Dict, List, Set
import polars as pl
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import PROCESSED_DIR
from src.evaluator import evaluate_blocking_recall


def test_blocking_improvements():
    print("Testing improved blocking keys on India split (88k entities)...")
    
    val_s1 = pl.read_parquet(PROCESSED_DIR / "val_source1_cleaned.parquet").filter(pl.col("country") == "India")
    train_s2 = pl.read_parquet(PROCESSED_DIR / "train_source2_cleaned.parquet").filter(pl.col("country") == "India")
    train_s3 = pl.read_parquet(PROCESSED_DIR / "train_source3_cleaned.parquet").filter(pl.col("country") == "India")
    val_gt = pl.read_parquet(PROCESSED_DIR / "val_ground_truth.parquet")

    # Filter GT for India S1s
    india_s1_ids = set(val_s1["entity_id"].to_list())
    val_gt_map = {}
    for row in val_gt.iter_rows():
        s1_id, matches = row[0], row[1]
        if s1_id in india_s1_ids and matches and str(matches).strip():
            val_gt_map[s1_id] = set(str(matches).strip().split(","))

    print(f"India Validation: {len(val_s1):,} S1 entities, {sum(len(v) for v in val_gt_map.values()):,} true matches.")

    # Build target indexes
    targets = pl.concat([train_s2, train_s3])
    t_eids = targets["entity_id"].to_list()
    t_names = targets["name_clean"].to_list()
    t_tokens = targets["name_tokens"].to_list()
    t_addrs = targets["addr_clean"].to_list()

    print(f"Target pool: {len(targets):,} records.")

    # 1. Exact name index: name -> list of target ids
    exact_name_idx = defaultdict(list)
    # 2. Token index: token -> list of target ids
    token_idx = defaultdict(list)
    # 3. Bigram index: (tok1, tok2) -> list of target ids
    bigram_idx = defaultdict(list)

    for idx, (eid, name, tok_str, addr) in enumerate(zip(t_eids, t_names, t_tokens, t_addrs)):
        if name:
            exact_name_idx[name].append(eid)
        toks = tok_str.split() if tok_str else []
        for t in toks:
            token_idx[t].append(eid)
        if len(toks) >= 2:
            bigram = f"{toks[0]}_{toks[1]}"
            bigram_idx[bigram].append(eid)

    print(f"Indexed: {len(exact_name_idx):,} exact names, {len(token_idx):,} tokens, {len(bigram_idx):,} bigrams.")

    # Query validation entities
    s1_eids = val_s1["entity_id"].to_list()
    s1_names = val_s1["name_clean"].to_list()
    s1_tokens = val_s1["name_tokens"].to_list()

    MAX_CANDIDATES = 20
    MAX_POSTINGS_TO_CONSIDER = 50000

    cand_map = {}
    t0 = time.time()

    for eid, name, tok_str in zip(s1_eids, s1_names, s1_tokens):
        candidates = set()

        # Key 1: Exact name
        if name and name in exact_name_idx:
            candidates.update(exact_name_idx[name][:MAX_CANDIDATES])

        toks = tok_str.split() if tok_str else []

        # Key 2: Bigram prefix
        if len(toks) >= 2:
            bigram = f"{toks[0]}_{toks[1]}"
            if bigram in bigram_idx:
                candidates.update(bigram_idx[bigram][:MAX_CANDIDATES])

        # Key 3: Rare/informative tokens (sort tokens by postings length)
        token_postings = []
        for t in toks:
            if t in token_idx:
                p_len = len(token_idx[t])
                if p_len < MAX_POSTINGS_TO_CONSIDER:
                    token_postings.append((p_len, t))

        token_postings.sort(key=lambda x: x[0])  # rarest tokens first!

        # Add candidates from the top 2 rarest tokens
        for p_len, t in token_postings[:2]:
            candidates.update(token_idx[t][:MAX_CANDIDATES])
            if len(candidates) >= MAX_CANDIDATES:
                break

        cand_map[eid] = candidates

    elapsed = time.time() - t0
    print(f"Queried {len(s1_eids):,} entities in {elapsed:.2f}s ({len(s1_eids)/elapsed:,.0f} entities/s).")

    # Evaluate recall
    metrics = evaluate_blocking_recall(val_gt_map, cand_map)
    print("\n" + "=" * 70)
    print(f"Candidate Recall on India: {metrics['candidate_recall']*100:.2f}%")
    print(f"Captured True Matches:     {metrics['captured_true_pairs']:,} / {metrics['total_true_pairs']:,}")
    print(f"Avg Candidates per S1:     {metrics['avg_candidates_per_s1']:.2f}")
    print("=" * 70)


if __name__ == "__main__":
    test_blocking_improvements()
