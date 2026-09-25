"""
Experiment with Blocking Keys to maximize Candidate Recall.
Tests:
1. All Token Inverted Index
2. Core Stem / 4-char Prefix
3. Postal Code Match
4. Address Token Match
5. Character 3-gram Match
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


def run_experiment():
    print("Loading India validation data...")
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
    t_addrs = targets["addr_clean"].to_list()
    t_postals = targets["postal_digits"].to_list()
    t_stems = targets["core_stem"].to_list()

    print(f"Building comprehensive indexes over {len(targets):,} targets...")
    exact_name_idx = defaultdict(list)
    token_idx = defaultdict(list)
    prefix4_idx = defaultdict(list)
    postal_idx = defaultdict(list)
    addr_tok_idx = defaultdict(list)

    for eid, name, tok_str, addr_str, post_str, stem in zip(t_eids, t_names, t_tokens, t_addrs, t_postals, t_stems):
        if name:
            exact_name_idx[name].append(eid)
            # Prefix of length 4 of the full name
            if len(name) >= 4:
                prefix4_idx[name[:4]].append(eid)

        toks = tok_str.split() if tok_str else []
        for t in toks:
            token_idx[t].append(eid)

        postals = post_str.split() if post_str else []
        for p in postals:
            if len(p) >= 4:
                postal_idx[p].append(eid)

        addr_toks = addr_str.split() if addr_str else []
        for at in addr_toks:
            if len(at) >= 4:
                addr_tok_idx[at].append(eid)

    print(f"Index summary:")
    print(f"  • Exact Names:     {len(exact_name_idx):,}")
    print(f"  • Unique Tokens:   {len(token_idx):,}")
    print(f"  • 4-Char Prefixes: {len(prefix4_idx):,}")
    print(f"  • Postal Codes:    {len(postal_idx):,}")
    print(f"  • Address Tokens:  {len(addr_tok_idx):,}")

    s1_eids = val_s1["entity_id"].to_list()
    s1_names = val_s1["name_clean"].to_list()
    s1_tokens = val_s1["name_tokens"].to_list()
    s1_addrs = val_s1["addr_clean"].to_list()
    s1_postals = val_s1["postal_digits"].to_list()

    MAX_CANDIDATES = 25
    MAX_POSTINGS = 100000

    print(f"\nQuerying {len(s1_eids):,} S1 entities...")
    t0 = time.time()
    cand_map = {}

    for eid, name, tok_str, addr_str, post_str in zip(s1_eids, s1_names, s1_tokens, s1_addrs, s1_postals):
        candidates = set()

        # 1. Exact Name match (all exact matches!)
        if name and name in exact_name_idx:
            candidates.update(exact_name_idx[name][:MAX_CANDIDATES])

        # 2. Token matches (all tokens with < MAX_POSTINGS, prioritize shorter postings)
        toks = tok_str.split() if tok_str else []
        token_postings = []
        for t in toks:
            if t in token_idx:
                p_len = len(token_idx[t])
                if p_len < MAX_POSTINGS:
                    token_postings.append((p_len, t))

        token_postings.sort(key=lambda x: x[0])

        for p_len, t in token_postings:
            # Add up to 10 candidates per token
            candidates.update(token_idx[t][:10])
            if len(candidates) >= MAX_CANDIDATES:
                break

        # 3. Postal Code Match (if postal code has < 500 records, add them)
        postals = post_str.split() if post_str else []
        for p in postals:
            if p in postal_idx and len(postal_idx[p]) < 500:
                candidates.update(postal_idx[p][:10])
                if len(candidates) >= MAX_CANDIDATES:
                    break

        # 4. If still few candidates, use 4-char prefix
        if len(candidates) < 5 and name and len(name) >= 4:
            p4 = name[:4]
            if p4 in prefix4_idx and len(prefix4_idx[p4]) < 2000:
                candidates.update(prefix4_idx[p4][:10])

        cand_map[eid] = candidates

    elapsed = time.time() - t0
    print(f"Candidate retrieval completed in {elapsed:.2f}s ({len(s1_eids)/elapsed:,.0f} entities/s).")

    metrics = evaluate_blocking_recall(val_gt_map, cand_map)
    print("\n" + "=" * 70)
    print(f"Candidate Recall on India: {metrics['candidate_recall']*100:.2f}%")
    print(f"Captured True Matches:     {metrics['captured_true_pairs']:,} / {metrics['total_true_pairs']:,}")
    print(f"Missed True Matches:       {metrics['total_true_pairs'] - metrics['captured_true_pairs']:,}")
    print(f"Avg Candidates per S1:     {metrics['avg_candidates_per_s1']:.2f}")
    print("=" * 70)


if __name__ == "__main__":
    run_experiment()
