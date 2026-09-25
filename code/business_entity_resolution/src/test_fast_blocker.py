"""
Fast Multi-Key Blocker with Direct Set-Union Candidate Retrieval.
Evaluates speed and candidate recall on India validation split.
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


def test_fast_blocker(max_candidates: int = 25):
    print(f"Testing Fast Multi-Key Blocker on India Validation Split (max_candidates={max_candidates})...")
    
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

    print(f"Building fast multi-key inverted indexes over {len(targets):,} targets...")
    t0 = time.time()

    exact_name_idx = defaultdict(list)
    bigram_idx = defaultdict(list)
    token_idx = defaultdict(list)
    postal_idx = defaultdict(list)

    for eid, name, tok_str, post_str in zip(t_eids, t_names, t_tokens, t_postals):
        if name:
            exact_name_idx[name].append(eid)

        toks = tok_str.split() if tok_str else []
        for t in toks:
            token_idx[t].append(eid)

        # Index all consecutive bigrams
        for i in range(len(toks) - 1):
            bigram = f"{toks[i]}_{toks[i+1]}"
            bigram_idx[bigram].append(eid)

        postals = post_str.split() if post_str else []
        for p in postals:
            if len(p) >= 4:
                postal_idx[p].append(eid)

    # Filter out overly massive postings lists
    MAX_BIGRAM_POSTINGS = 15000
    MAX_TOKEN_POSTINGS = 20000
    MAX_POSTAL_POSTINGS = 500

    filtered_bigram_idx = {k: v for k, v in bigram_idx.items() if len(v) <= MAX_BIGRAM_POSTINGS}
    filtered_token_idx = {k: v for k, v in token_idx.items() if len(v) <= MAX_TOKEN_POSTINGS}
    filtered_postal_idx = {k: v for k, v in postal_idx.items() if len(v) <= MAX_POSTAL_POSTINGS}

    print(f"Indexes built in {time.time() - t0:.2f}s:")
    print(f"  • Exact Names:  {len(exact_name_idx):,}")
    print(f"  • Bigrams:      {len(filtered_bigram_idx):,}")
    print(f"  • Tokens:       {len(filtered_token_idx):,}")
    print(f"  • Postal Codes: {len(filtered_postal_idx):,}")

    # Query entities
    s1_eids = val_s1["entity_id"].to_list()
    s1_names = val_s1["name_clean"].to_list()
    s1_tokens = val_s1["name_tokens"].to_list()
    s1_postals = val_s1["postal_digits"].to_list()

    print(f"\nQuerying {len(s1_eids):,} validation S1 entities...")
    t1 = time.time()
    cand_map = {}

    for eid, name, tok_str, post_str in zip(s1_eids, s1_names, s1_tokens, s1_postals):
        candidates = set()

        # Key 1: Exact name match (all exact matches)
        if name and name in exact_name_idx:
            candidates.update(exact_name_idx[name][:max_candidates])

        toks = tok_str.split() if tok_str else []

        # Key 2: All Bigrams
        if len(toks) >= 2:
            for i in range(len(toks) - 1):
                bg = f"{toks[i]}_{toks[i+1]}"
                if bg in filtered_bigram_idx:
                    candidates.update(filtered_bigram_idx[bg][:max_candidates])
                    if len(candidates) >= max_candidates:
                        break

        # Key 3: Rarest tokens (sort by postings list length)
        if len(candidates) < max_candidates:
            token_postings = []
            for t in toks:
                if t in filtered_token_idx:
                    token_postings.append((len(filtered_token_idx[t]), t))

            token_postings.sort(key=lambda x: x[0])  # rarest first

            for _, t in token_postings[:3]:
                candidates.update(filtered_token_idx[t][:max_candidates])
                if len(candidates) >= max_candidates:
                    break

        # Key 4: Postal code fallback if few candidates
        if len(candidates) < 5:
            postals = post_str.split() if post_str else []
            for p in postals:
                if p in filtered_postal_idx:
                    candidates.update(filtered_postal_idx[p][:10])
                    if len(candidates) >= max_candidates:
                        break

        cand_map[eid] = candidates

    t_query = time.time() - t1
    print(f"Candidate retrieval completed in {t_query:.2f}s ({len(s1_eids)/t_query:,.0f} entities/s).")

    metrics = evaluate_blocking_recall(val_gt_map, cand_map)
    print("\n" + "=" * 70)
    print(f"CANDIDATE BLOCKING METRICS (India Validation, K={max_candidates}):")
    print(f"  • Candidate Recall:        {metrics['candidate_recall']*100:.2f}%")
    print(f"  • Captured True Matches:   {metrics['captured_true_pairs']:,} / {metrics['total_true_pairs']:,}")
    print(f"  • Missed True Matches:     {metrics['total_true_pairs'] - metrics['captured_true_pairs']:,}")
    print(f"  • Avg Candidates per S1:   {metrics['avg_candidates_per_s1']:.2f}")
    print("=" * 70)


if __name__ == "__main__":
    test_fast_blocker(25)
